#!/usr/bin/env python
"""
BioTwin.X — CMCL Training & Evaluation Pipeline
================================================
Trains the Cross-Modal Contrastive Learning dual encoder and produces the
artefacts consumed by the inference engine and the dashboard.

Protocol
--------
* **Patient-level splits only.** Every window of a patient lives in the same
  fold as that patient. Splitting windows would leak the subject's baseline
  physiology into the test set and inflate every metric.
* **5-fold cross-validation** for the headline numbers, reported as
  mean ± std across folds — far more honest than a single lucky split.
* **Out-of-fold calibration.** Probability calibration is fitted on the
  out-of-fold predictions, never on the model's own training predictions.
* **Ablations.** The script also trains the *static-only* and *temporal-only*
  variants so the README can quote the measured contribution of each modality
  rather than asserting it.

Usage
-----
    python scripts/train_pipeline.py --profile demo --folds 5 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from biotwin.calibration import (  # noqa: E402
    TemperatureScaler,
    brier_score,
    expected_calibration_error,
)
from biotwin.cmcl import BioTwinCMCL, nt_xent  # noqa: E402
from biotwin.config import MODEL_DIR, PROFILES, ModelConfig  # noqa: E402
from biotwin.dataio import (  # noqa: E402
    EVENT_INDEX,
    CohortBundle,
    build_tensors,
)
from biotwin.features import TEMPORAL_CHANNELS, WINDOW_CONTEXT_FEATURES  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

EVENT_NAMES = list(EVENT_INDEX.keys())
RNG = np.random.default_rng(20260214)

#: Number of times each positive window is repeated inside an epoch's pool.
#: Oversampling the minority class is essential because the positive rate is
#: ~5 %; the resulting base-rate distortion is undone by the out-of-fold
#: temperature+bias calibrator.
POS_OVERSAMPLE_REPS = 2.0


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
class Scaler:
    """Robust, NaN-aware standardisation fitted on the training fold only."""

    def __init__(self) -> None:
        self.time_mean: np.ndarray | None = None
        self.time_std: np.ndarray | None = None
        self.static_mean: np.ndarray | None = None
        self.static_std: np.ndarray | None = None

    def fit(self, Xt: np.ndarray, Xs: np.ndarray) -> "Scaler":
        with np.errstate(invalid="ignore"):
            self.time_mean = np.nanmean(Xt, axis=(0, 2))
            self.time_std = np.nanstd(Xt, axis=(0, 2))
            self.static_mean = np.nanmean(Xs, axis=0)
            self.static_std = np.nanstd(Xs, axis=0)
        self.time_mean = np.nan_to_num(self.time_mean, nan=0.0)
        self.time_std = np.nan_to_num(self.time_std, nan=1.0)
        self.time_std[self.time_std < 1e-6] = 1.0
        self.static_mean = np.nan_to_num(self.static_mean, nan=0.0)
        self.static_std = np.nan_to_num(self.static_std, nan=1.0)
        self.static_std[self.static_std < 1e-6] = 1.0
        return self

    def transform_time(self, X: np.ndarray) -> np.ndarray:
        z = (X - self.time_mean[:, None]) / self.time_std[:, None]
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def transform_static(self, X: np.ndarray) -> np.ndarray:
        z = (X - self.static_mean) / self.static_std
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def to_dict(self) -> dict[str, np.ndarray]:
        return {"time_mean": self.time_mean, "time_std": self.time_std,
                "static_mean": self.static_mean, "static_std": self.static_std}


# --------------------------------------------------------------------------- #
# Batch construction
# --------------------------------------------------------------------------- #
def make_batches(indices: np.ndarray, y_event: np.ndarray, batch_size: int,
                 rng: np.random.Generator,
                 pos_oversample: float = POS_OVERSAMPLE_REPS) -> list[np.ndarray]:
    """Shuffled mini-batches with mild positive oversampling.

    Oversampling is applied at the *index* level (positives appear
    ``pos_oversample`` times more often). The induced base-rate shift is
    removed afterwards by the temperature+bias calibrator, which is fitted on
    out-of-fold predictions.

    Unlike a "one window per patient per batch" scheme, this keeps every
    epoch at full data volume; the false negatives that arise from several
    windows of the same patient sharing a batch are masked inside the
    contrastive loss (see :func:`biotwin.cmcl.nt_xent`).
    """
    pos = indices[y_event[indices].max(axis=1) > 0]
    neg = indices[y_event[indices].max(axis=1) == 0]
    reps = int(round(pos_oversample))
    pool = np.concatenate([pos] * reps + [neg])
    rng.shuffle(pool)
    return [pool[i:i + batch_size] for i in range(0, len(pool), batch_size)]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    from sklearn.metrics import (average_precision_score, f1_score,
                                 precision_score, recall_score, roc_auc_score)

    if len(np.unique(y)) < 2:
        return {"auc": float("nan"), "f1": float("nan"), "ap": float("nan"),
                "precision": float("nan"), "recall": float("nan"), "n_pos": int(y.sum())}
    pred = (p >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y, p)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "ap": float(average_precision_score(y, p)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "n_pos": int(y.sum()),
    }


def macro_auc(y_event: np.ndarray, probs: np.ndarray) -> float:
    vals = []
    for k in range(y_event.shape[1]):
        if len(np.unique(y_event[:, k])) < 2:
            continue
        from sklearn.metrics import roc_auc_score
        vals.append(roc_auc_score(y_event[:, k], probs[:, k]))
    return float(np.mean(vals)) if vals else float("nan")


def strip_window_context(Xs: np.ndarray) -> np.ndarray:
    """Zero the trailing window-context block of every static vector.

    The static branch of the fusion model legitimately sees 20 causal
    telemetry summaries alongside the clinical/genomic architecture. A
    *static-only* ablation must not — otherwise it would still be reading the
    wearable stream and the ablation would prove nothing.
    """
    X = Xs.copy()
    k = len(WINDOW_CONTEXT_FEATURES)
    if k and X.shape[1] >= k:
        X[:, -k:] = 0.0
    return X


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import f1_score
    best_t, best_f = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 19):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), float(f)
    return best_t, best_f


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_fold(Xt: np.ndarray, Xs: np.ndarray, y_event: np.ndarray, y_risk: np.ndarray,
               y_ttx: np.ndarray, ttx_mask: np.ndarray, patient_index: np.ndarray,
               tr_idx: np.ndarray, va_idx: np.ndarray, cfg: ModelConfig,
               epochs: int, seed: int, modality: str = "fusion",
               verbose: bool = False) -> tuple[BioTwinCMCL, Scaler, dict]:
    """Train one fold. ``modality`` ∈ {fusion, static_only, temporal_only}."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    if modality == "static_only":
        # keep the ablation honest: no telemetry-derived context in the static branch
        Xs_eff = strip_window_context(Xs)
    else:
        Xs_eff = Xs
    scaler = Scaler().fit(Xt[tr_idx], Xs_eff[tr_idx])
    Xt_t = torch.from_numpy(scaler.transform_time(Xt)).float()
    Xs_t = torch.from_numpy(scaler.transform_static(Xs_eff)).float()
    y_e = torch.from_numpy(y_event).float()
    y_r = torch.from_numpy(y_risk).float()
    y_t = torch.from_numpy(y_ttx).float()
    m_t = torch.from_numpy(ttx_mask).float()
    pid_t = torch.from_numpy(patient_index).long()

    static_dim = Xs.shape[1]
    model = BioTwinCMCL(static_dim=static_dim, n_channels=Xt.shape[1], cfg=cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # The number of optimisation steps per epoch is driven by the *oversampled*
    # pool size, not by |train windows| / batch_size — otherwise the cosine
    # schedule would decay far too slowly and the model would never anneal.
    _pos = int((y_event[tr_idx].max(axis=1) > 0).sum())
    _neg = int(len(tr_idx) - _pos)
    pool_size = _pos * POS_OVERSAMPLE_REPS + _neg
    steps_per_epoch = max(1, int(np.ceil(pool_size / cfg.batch_size)))
    total_steps = steps_per_epoch * epochs
    warm = min(cfg.warmup_steps, max(1, total_steps // 10))

    def lr_lambda(step: int) -> float:
        if step < warm:
            return (step + 1) / max(warm, 1)
        p = (step - warm) / max(total_steps - warm, 1)
        return 0.5 * (1.0 + np.cos(np.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_auc, best_state, best_epoch = -np.inf, None, 0
    history = []

    for epoch in range(epochs):
        model.train()
        batches = make_batches(tr_idx, y_event, cfg.batch_size, rng)
        epoch_loss = 0.0
        for bidx in batches:
            gi = bidx            # make_batches already yields *global* window indices
            xs = Xs_t[gi]
            xt = Xt_t[gi]
            ye = y_e[gi]
            yr = y_r[gi]
            yt = y_t[gi]
            mt = m_t[gi]
            pid = pid_t[gi]

            out = model(xs, xt, archetype_dropout=cfg.archetype_dropout)
            if modality == "static_only":
                fused = out["z_static"]
                out = dict(out)
                out["z_fused"] = fused
                out["event_logits"] = model.event_head(fused)
                out["risk_logit"] = model.risk_head(fused).squeeze(-1)
            elif modality == "temporal_only":
                fused = out["z_temporal"]
                out = dict(out)
                out["z_fused"] = fused
                out["event_logits"] = model.event_head(fused)
                out["risk_logit"] = model.risk_head(fused).squeeze(-1)

            loss, parts = model.compute_loss(out, ye, yt, yr, mt,
                                              pid_a=pid, pid_b=pid)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            epoch_loss += float(loss.detach())

        if len(va_idx):
            probs, prisk = predict(model, scaler, Xt[va_idx], Xs[va_idx],
                                   modality=modality)
            auc = macro_auc(y_event[va_idx], probs)
        else:
            auc = float("nan")
        history.append({"epoch": epoch, "loss": epoch_loss / max(len(batches), 1),
                        "val_macro_auc": auc})
        if np.isfinite(auc) and auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and epoch % 5 == 0:
            print(f"    epoch {epoch:>3d}  loss {epoch_loss / max(len(batches), 1):.4f}"
                  f"  val_macro_auc {auc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, {"best_val_auc": best_auc, "best_epoch": best_epoch,
                            "epochs_run": epochs,
                            "steps_per_epoch": steps_per_epoch,
                            "history": history}


@torch.no_grad()
def predict(model: BioTwinCMCL, scaler: Scaler, Xt: np.ndarray,
            Xs: np.ndarray, modality: str = "fusion") -> tuple[np.ndarray, np.ndarray]:
    """Forward pass. ``modality`` mirrors the training-time branch override so
    the ablations measure what they claim to measure."""
    model.eval()
    xt = torch.from_numpy(scaler.transform_time(Xt)).float()
    Xs_eff = strip_window_context(Xs) if modality == "static_only" else Xs
    xs = torch.from_numpy(scaler.transform_static(Xs_eff)).float()
    out = model(xs, xt)
    if modality == "static_only":
        fused = out["z_static"]
    elif modality == "temporal_only":
        fused = out["z_temporal"]
    else:
        fused = out["z_fused"]
    probs = torch.sigmoid(model.event_head(fused)).numpy()
    prisk = torch.sigmoid(model.risk_head(fused).squeeze(-1)).numpy()
    return probs, prisk


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def kfold_patient_splits(patient_ids: Sequence[str], k: int,
                         seed: int = 20260214) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    ids = list(patient_ids)
    rng.shuffle(ids)
    folds: list[list[str]] = [[] for _ in range(k)]
    for i, p in enumerate(ids):
        folds[i % k].append(p)
    return folds


def run_cv(bundle: CohortBundle, tensors, cfg: ModelConfig, k: int, epochs: int,
           modality: str = "fusion", verbose: bool = True) -> dict:
    ids = bundle.patient_ids()
    folds = kfold_patient_splits(ids, k)
    pid_index = {p: i for i, p in enumerate(ids)}

    oof_probs = np.zeros_like(tensors.y_event)
    oof_risk = np.zeros(len(tensors.y_event))
    fold_metrics = []
    used = np.zeros(len(tensors.y_event), dtype=bool)

    for f, val_patients in enumerate(folds):
        tr_patients = [p for p in ids if p not in val_patients]
        tr_idx = np.where(np.isin(tensors.patient_index, [pid_index[p] for p in tr_patients]))[0]
        va_idx = np.where(np.isin(tensors.patient_index, [pid_index[p] for p in val_patients]))[0]
        used[va_idx] = True

        model, scaler, info = train_fold(
            tensors.X_time, tensors.X_static, tensors.y_event, tensors.y_risk,
            tensors.y_ttx, tensors.ttx_mask, tensors.patient_index,
            tr_idx, va_idx, cfg, epochs, seed=1000 + f, modality=modality,
            verbose=verbose)

        probs, prisk = predict(model, scaler, tensors.X_time[va_idx],
                               tensors.X_static[va_idx], modality=modality)
        oof_probs[va_idx] = probs
        oof_risk[va_idx] = prisk

        m = {
            "fold": f,
            "n_val_patients": len(val_patients),
            "macro_auc": macro_auc(tensors.y_event[va_idx], probs),
            "any_event_auc": binary_metrics(
                tensors.y_event[va_idx].max(axis=1), prisk)["auc"],
        }
        for k2, name in enumerate(EVENT_NAMES):
            m[f"auc_{name}"] = binary_metrics(tensors.y_event[va_idx][:, k2],
                                              probs[:, k2])["auc"]
        fold_metrics.append(m)
        if verbose:
            print(f"  fold {f + 1}/{k}: macro AUC {m['macro_auc']:.4f} | "
                  + " ".join(f"{name.split('_')[0]} {m[f'auc_{name}']:.3f}"
                             for name in EVENT_NAMES))

    used_idx = np.where(used)[0]
    oof_probs = oof_probs[used_idx]
    oof_risk = oof_risk[used_idx]
    y = tensors.y_event[used_idx]
    y_any = y.max(axis=1)

    per_class = {}
    for k2, name in enumerate(EVENT_NAMES):
        t, f1 = best_f1_threshold(y[:, k2], oof_probs[:, k2])
        bm = binary_metrics(y[:, k2], oof_probs[:, k2], threshold=t)
        bm["best_threshold"] = t
        per_class[name] = bm

    any_t, any_f1 = best_f1_threshold(y_any, oof_risk)
    any_m = binary_metrics(y_any, oof_risk, threshold=any_t)
    any_m["best_threshold"] = any_t
    any_m["ece"] = expected_calibration_error(oof_risk, y_any)
    any_m["brier"] = brier_score(oof_risk, y_any)

    return {
        "modality": modality,
        "folds": fold_metrics,
        "macro_auc_mean": float(np.nanmean([m["macro_auc"] for m in fold_metrics])),
        "macro_auc_std": float(np.nanstd([m["macro_auc"] for m in fold_metrics])),
        "any_event": any_m,
        "per_class": per_class,
        "oof_probs": oof_probs,
        "oof_risk": oof_risk,
        "oof_index": used_idx,
    }


# --------------------------------------------------------------------------- #
# Tabular twin (XGBoost) for the ablation table
# --------------------------------------------------------------------------- #
def train_tabular_twin(bundle: CohortBundle, tensors,
                       variant: str = "static") -> dict:
    """XGBoost on the tabular layer — the 'table stakes' baseline.

    ``variant='static'``  → clinical + genomic architecture only (what an
    EHR-only risk model can see).
    ``variant='full'``    → additionally the engineered window-context
    summaries (still no raw waveform / time-series access).
    """
    from sklearn.metrics import roc_auc_score
    try:
        from xgboost import XGBClassifier
    except Exception:
        return {"available": False}

    X = tensors.X_static
    if variant == "static":
        X = strip_window_context(X)

    ids = bundle.patient_ids()
    pid_index = {p: i for i, p in enumerate(ids)}
    folds = kfold_patient_splits(ids, 5)
    oof = np.zeros(len(tensors.y_event))
    used = np.zeros(len(tensors.y_event), dtype=bool)
    for f, val_patients in enumerate(folds):
        tr_patients = [p for p in ids if p not in val_patients]
        tr_idx = np.where(np.isin(tensors.patient_index, [pid_index[p] for p in tr_patients]))[0]
        va_idx = np.where(np.isin(tensors.patient_index, [pid_index[p] for p in val_patients]))[0]
        used[va_idx] = True
        clf = XGBClassifier(n_estimators=350, max_depth=4, learning_rate=0.06,
                            subsample=0.85, colsample_bytree=0.8, reg_lambda=2.0,
                            min_child_weight=5, eval_metric="logloss",
                            random_state=0, n_jobs=4)
        clf.fit(X[tr_idx], tensors.y_event[tr_idx].max(axis=1))
        oof[va_idx] = clf.predict_proba(X[va_idx])[:, 1]
    idx = np.where(used)[0]
    y = tensors.y_event[idx].max(axis=1)
    return {
        "available": True,
        "variant": variant,
        "auc": float(roc_auc_score(y, oof[idx])) if len(np.unique(y)) > 1 else float("nan"),
        "note": ("XGBoost over clinical + genomic architecture only (EHR-equivalent)."
                 if variant == "static" else
                 "XGBoost over clinical + genomic + engineered window summaries."),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="BioTwin.X CMCL training pipeline")
    ap.add_argument("--profile", default="demo", choices=sorted(PROFILES))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--seed", type=int, default=20260214)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 78)
    print("  BioTwin.X · Cross-Modal Contrastive Learning — training pipeline")
    print("=" * 78)

    bundle = CohortBundle.load(args.profile)
    print(f"▸ cohort      : {len(bundle.patient_ids())} patients "
          f"({bundle.profile.describe()})")
    t0 = time.perf_counter()
    tensors = build_tensors(bundle)
    print(f"▸ windows     : {len(tensors):,} labelled windows "
          f"({int(tensors.y_event.sum())} positive) [{time.perf_counter() - t0:.1f}s]")

    cfg = ModelConfig(batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    print(f"▸ model       : {cfg.describe()}")

    print("\n▸ 5-fold patient-level cross-validation — full CMCL fusion")
    cv = run_cv(bundle, tensors, cfg, args.folds, args.epochs, "fusion")
    print(f"  → macro ROC-AUC {cv['macro_auc_mean']:.4f} ± {cv['macro_auc_std']:.4f}")
    for name, m in cv["per_class"].items():
        print(f"    {name:<24s} AUC {m['auc']:.4f}  F1 {m['f1']:.4f}  "
              f"AP {m['ap']:.4f}  (n+ {m['n_pos']})")
    print(f"  → any-event  AUC {cv['any_event']['auc']:.4f}  "
          f"F1 {cv['any_event']['f1']:.4f}  ECE {cv['any_event']['ece']:.4f}")

    results: dict[str, Any] = {
        "profile": args.profile,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "config": asdict(cfg),
        "folds": args.folds,
        "epochs": args.epochs,
        "cv": {k: v for k, v in cv.items() if k not in ("oof_probs", "oof_risk", "oof_index")},
    }

    if not args.skip_ablation:
        print("\n▸ ablation — temporal branch only (identical training budget)")
        cv_t = run_cv(bundle, tensors, cfg, args.folds, args.epochs, "temporal_only")
        print(f"  → macro ROC-AUC {cv_t['macro_auc_mean']:.4f} ± {cv_t['macro_auc_std']:.4f}")
        results["ablation_temporal_only"] = {
            "macro_auc_mean": cv_t["macro_auc_mean"],
            "macro_auc_std": cv_t["macro_auc_std"],
            "per_class_auc": {k: v["auc"] for k, v in cv_t["per_class"].items()},
            "any_event_auc": cv_t["any_event"]["auc"],
        }
        print("▸ ablation — static/genomic branch only (no telemetry context)")
        cv_s = run_cv(bundle, tensors, cfg, args.folds, args.epochs, "static_only")
        print(f"  → macro ROC-AUC {cv_s['macro_auc_mean']:.4f} ± {cv_s['macro_auc_std']:.4f}")
        results["ablation_static_only"] = {
            "macro_auc_mean": cv_s["macro_auc_mean"],
            "macro_auc_std": cv_s["macro_auc_std"],
            "per_class_auc": {k: v["auc"] for k, v in cv_s["per_class"].items()},
            "any_event_auc": cv_s["any_event"]["auc"],
        }
        print("▸ baseline — XGBoost tabular twin (clinical + genomic only)")
        tab_s = train_tabular_twin(bundle, tensors, "static")
        results["baseline_xgboost_static"] = tab_s
        print(f"  → any-event AUC {tab_s.get('auc', float('nan')):.4f}")
        print("▸ baseline — XGBoost + engineered window summaries")
        tab_f = train_tabular_twin(bundle, tensors, "full")
        results["baseline_xgboost_full"] = tab_f
        print(f"  → any-event AUC {tab_f.get('auc', float('nan')):.4f}")

    # ---------------- final model on the full cohort ------------------- #
    print("\n▸ final fit on the full cohort (for the shipped checkpoint)")
    all_idx = np.arange(len(tensors))
    final_model, final_scaler, info = train_fold(
        tensors.X_time, tensors.X_static, tensors.y_event, tensors.y_risk,
        tensors.y_ttx, tensors.ttx_mask, tensors.patient_index,
        all_idx, np.array([], dtype=int), cfg, args.epochs, seed=args.seed,
        modality="fusion", verbose=False)
    print(f"  → {info['epochs_run']} epochs "
          f"({info['steps_per_epoch']} steps/epoch) trained on all "
          f"{len(tensors):,} windows")

    # ---------------- calibration on out-of-fold predictions ------------- #
    print("▸ fitting probability calibration on out-of-fold predictions")
    oof_idx = cv["oof_index"]
    calibrators: dict[str, Any] = {}
    oof_any = cv["oof_risk"]
    y_any = tensors.y_event[oof_idx].max(axis=1)
    logit_any = np.log(np.clip(oof_any, 1e-6, 1 - 1e-6) /
                       np.clip(1 - oof_any, 1e-6, 1))
    calibrators["risk"] = TemperatureScaler().fit(logit_any, y_any).to_dict()
    for k2, name in enumerate(EVENT_NAMES):
        p = cv["oof_probs"][:, k2]
        lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        calibrators[f"event_{k2}"] = TemperatureScaler().fit(
            lg, tensors.y_event[oof_idx][:, k2]).to_dict()

    # ---------------- persist ------------------------------------------- #
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODEL_DIR / f"biotwin_cmcl_{args.profile}.pt"
    torch.save({
        "state_dict": final_model.state_dict(),
        "config": asdict(cfg),
        "static_dim": tensors.X_static.shape[1],
        "n_channels": tensors.X_time.shape[1],
        "scaler": final_scaler.to_dict(),
        "calibrators": calibrators,
        "metrics": {
            "macro_auc_mean": cv["macro_auc_mean"],
            "macro_auc_std": cv["macro_auc_std"],
            "any_event": cv["any_event"],
            "per_class": cv["per_class"],
        },
        "profile": args.profile,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
    }, ckpt_path)
    print(f"  → checkpoint: {ckpt_path}")

    results["checkpoint"] = str(ckpt_path)
    results["calibrators"] = calibrators
    out = ROOT / "data" / "artifacts" / f"training_report_{args.profile}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"  → report    : {out}")
    print("\n✔ done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
