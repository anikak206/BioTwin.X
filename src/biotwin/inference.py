"""
BioTwin.X — Clinical Inference Engine
=======================================
The production-facing façade of the platform.

Responsibilities
----------------
1. **Assemble** the two modalities for a patient at a point in time:
   the immutable static architecture (clinical + genomic) and the trailing
   12-hour window of asynchronously resampled telemetry.
2. **Infer** the Clinical Risk Index (0-100), per-event probabilities, the
   predicted time-to-event and a calibrated confidence.
3. **Explain** the score with auditable, evidence-grounded attributions.
4. **Translate** the score into ranked, urgency-tagged clinical actions.

Backend strategy (fail-safe by design)
--------------------------------------
``cmcl``
    The trained PyTorch dual-encoder. Preferred.
``surrogate``
    A calibrated gradient-boosted surrogate over the same engineered feature
    space, fitted in seconds. Used automatically when no checkpoint is
    available — the platform *degrades in accuracy, never in availability*,
    because a clinical console that shows nothing is worse than one that
    shows a slightly less sharp score.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np
import pandas as pd

from .calibration import TemperatureScaler, brier_score, expected_calibration_error
from .config import (
    DEFAULT_PROFILE,
    MODEL_DIR,
    PROFILES,
    SLO_F1,
    SLO_LATENCY_MS,
    SLO_ROC_AUC,
    DatasetProfile,
    ModelConfig,
)
from .dataio import (
    EVENT_INDEX,
    CohortBundle,
    build_tensors,
    patient_features,
    patient_split,
)
from .explain import (
    Attribution,
    Explanation,
    analytic_attributions,
    build_narrative,
    interaction_share_from_attributions,
)
from .features import TEMPORAL_CHANNELS, WINDOW_CONTEXT_FEATURES
from .risk import ClinicalAction, Insight, build_actions, build_insights, tier_for

# --------------------------------------------------------------------------- #
# Risk-index composition weights (documented in README §6.3)
# --------------------------------------------------------------------------- #
W_EVENT = 0.55
W_CLASS = 0.25
W_STRAIN = 0.12
W_GENOME = 0.08


@dataclass
class PatientReport:
    """Everything the clinician console needs for one patient at one instant."""

    patient_id: str
    as_of: pd.Timestamp
    risk_index: float
    tier: str
    confidence: float
    event_probs: dict[str, float]
    dominant_event: str
    predicted_ttx_hours: float
    insights: list[Insight] = field(default_factory=list)
    actions: list[ClinicalAction] = field(default_factory=list)
    explanation: Explanation | None = None
    context: dict[str, float] = field(default_factory=dict)
    static: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    backend: str = "cmcl"
    coverage: float = 0.0
    baseline_rmssd: float | None = None
    nocturnal_rmssd: float | None = None

    @property
    def urgency(self) -> str:
        return self.actions[0].urgency if self.actions else "OPTIMISE"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["as_of"] = self.as_of.isoformat()
        d["insights"] = [i.to_dict() for i in self.insights]
        d["actions"] = [a.to_dict() for a in self.actions]
        if self.explanation:
            d["explanation"] = {
                "attributions": [asdict(a) for a in self.explanation.attributions],
                "interaction_share": self.explanation.interaction_share,
                "modality_share": self.explanation.modality_share,
                "narrative": self.explanation.narrative,
            }
        return d


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class BioTwinEngine:
    """Cross-Modal Contrastive Digital Twin — clinical inference engine."""

    def __init__(self, profile: str = DEFAULT_PROFILE, model_path: Path | None = None,
                 allow_surrogate: bool = True, verbose: bool = False) -> None:
        self.profile: DatasetProfile = PROFILES.get(profile, PROFILES[DEFAULT_PROFILE])
        self.bundle = CohortBundle.load(self.profile)
        self.model_path = Path(model_path) if model_path else (MODEL_DIR / f"biotwin_cmcl_{self.profile.name}.pt")
        self.verbose = verbose
        self.backend: str = "rule-fallback"
        self.metrics: dict[str, Any] = {}

        self._model = None
        self._torch = None
        self._scaler: dict[str, np.ndarray] = {}
        self._calibrators: dict[str, TemperatureScaler] = {}
        self._surrogate: dict[str, Any] = {}
        self._config = ModelConfig()

        self._try_load_checkpoint()
        if self.backend != "cmcl" and allow_surrogate:
            self._fit_surrogate()

    # ------------------------------------------------------------------ #
    # Backend selection
    # ------------------------------------------------------------------ #
    def _try_load_checkpoint(self) -> None:
        if not self.model_path.exists():
            if self.verbose:
                print(f"[BioTwin.X] no checkpoint at {self.model_path} — "
                      f"falling back to {'surrogate' if True else 'rules'}")
            return
        try:
            import torch
            from .cmcl import BioTwinCMCL
        except Exception:  # pragma: no cover
            return

        ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
        cfg = ModelConfig(**ckpt["config"]) if ckpt.get("config") else ModelConfig()
        self._config = cfg
        model = BioTwinCMCL(
            static_dim=int(ckpt["static_dim"]),
            n_channels=int(ckpt.get("n_channels", len(TEMPORAL_CHANNELS))),
            cfg=cfg,
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self._model = model
        self._torch = torch
        self._scaler = {k: np.asarray(v) for k, v in ckpt["scaler"].items()}
        self._calibrators = {
            k: TemperatureScaler.from_dict(v) for k, v in ckpt.get("calibrators", {}).items()
        }
        self.metrics = ckpt.get("metrics", {})
        self.backend = "cmcl"

    def _fit_surrogate(self) -> None:
        """Fit a calibrated tabular surrogate over the same feature space."""
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
        except Exception:  # pragma: no cover
            return

        t0 = time.perf_counter()
        tensors = build_tensors(self.bundle)
        tr_ids, va_ids = patient_split(self.bundle, val_frac=0.25)
        idx = {p: i for i, p in enumerate(self.bundle.patient_ids())}
        m_tr = np.isin(tensors.patient_index, [idx[p] for p in tr_ids])

        X = tensors.X_static
        models: dict[str, Any] = {}
        calibs: dict[str, TemperatureScaler] = {}
        for name, col in list(EVENT_INDEX.items()) + [("any_event", None)]:
            y = (tensors.y_event.max(axis=1) if col is None else tensors.y_event[:, col])
            if y[m_tr].sum() < 5 or len(np.unique(y[m_tr])) < 2:
                continue
            clf = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06, max_depth=4,
                min_samples_leaf=20, l2_regularization=1.0, random_state=0)
            clf.fit(X[m_tr], y[m_tr])
            models[name] = clf

            # calibrate on the held-out patient split
            m_va = ~m_tr
            if m_va.sum() and y[m_va].sum() > 0:
                logit = np.log(np.clip(clf.predict_proba(X[m_va])[:, 1], 1e-6, 1 - 1e-6)
                               / np.clip(1 - clf.predict_proba(X[m_va])[:, 1], 1e-6, 1))
                cal = TemperatureScaler().fit(logit, y[m_va])
            else:
                cal = TemperatureScaler()
            calibs[name] = cal

        self._surrogate = {"models": models, "calibrators": calibs}
        self.backend = "surrogate"
        self.metrics = {
            "backend": "surrogate",
            "fit_seconds": round(time.perf_counter() - t0, 2),
            "note": "Calibrated gradient-boosted surrogate over the CMCL feature space.",
        }

    # ------------------------------------------------------------------ #
    # Feature assembly
    # ------------------------------------------------------------------ #
    def _window_features(self, patient_id: str, end_idx: int | None = None,
                         horizon_bins: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        """Return ``(time_tensor, static_vector, context_dict)`` for a window."""
        feats = patient_features(patient_id, self.profile)
        grid = feats.grid
        n = len(grid)
        hb = horizon_bins or int(self.profile.horizon_hours * 60 / self.profile.grid_minutes)
        end = n if end_idx is None else min(end_idx, n)
        start = max(0, end - hb)
        tensor = grid[list(TEMPORAL_CHANNELS)].to_numpy(dtype=np.float32)[start:end].T
        ctx_row = feats.window_context.iloc[end - 1]
        ctx = {k: (float(v) if pd.notna(v) else float("nan"))
               for k, v in ctx_row[list(WINDOW_CONTEXT_FEATURES)].items()}
        static_vec = np.concatenate([
            self.bundle.static_base_vector(patient_id),
            self.bundle.static_context_vector(ctx_row),
        ]).astype(np.float32)
        return tensor, static_vec, ctx

    def _standardise_time(self, tensor: np.ndarray) -> np.ndarray:
        if "time_mean" not in self._scaler:
            return tensor
        mean = self._scaler["time_mean"][:, None]
        std = self._scaler["time_std"][:, None]
        t = (tensor - mean) / std
        return np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    def _standardise_static(self, vec: np.ndarray) -> np.ndarray:
        if "static_mean" not in self._scaler:
            return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        mean = self._scaler["static_mean"]
        std = self._scaler["static_std"]
        v = (vec - mean) / std
        return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------ #
    # Core inference
    # ------------------------------------------------------------------ #
    def _forward(self, tensors: np.ndarray, statics: np.ndarray) -> dict[str, np.ndarray]:
        """Run the active backend over a batch of windows."""
        if self.backend == "cmcl":
            torch = self._torch
            with torch.no_grad():
                xs = torch.from_numpy(self._standardise_static(statics)).float()
                xt = torch.from_numpy(self._standardise_time(tensors)).float()
                out = self._model(xs, xt)
                return {
                    "event_logits": out["event_logits"].numpy(),
                    "risk_logit": out["risk_logit"].numpy(),
                    "ttx": out["ttx"].numpy(),
                    "z_static": out["z_static"].numpy(),
                    "z_temporal": out["z_temporal"].numpy(),
                    "z_fused": out["z_fused"].numpy(),
                }
        # surrogate
        preds: dict[str, np.ndarray] = {}
        for name, clf in self._surrogate.get("models", {}).items():
            p = clf.predict_proba(statics)[:, 1]
            cal = self._surrogate["calibrators"].get(name)
            if cal is not None:
                logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) /
                               np.clip(1 - p, 1e-6, 1))
                p = cal.transform(logit)
            preds[name] = p
        event_prob = np.stack([preds.get(k, np.zeros(len(statics)))
                               for k in EVENT_INDEX], axis=1)
        any_prob = preds.get("any_event", event_prob.max(axis=1))
        risk_logit = np.log(np.clip(any_prob, 1e-6, 1 - 1e-6) /
                            np.clip(1 - any_prob, 1e-6, 1))
        return {
            "event_probs": event_prob,
            "risk_logit": risk_logit,
            "ttx": np.full(len(statics), np.log1p(12.0)),
            "z_static": np.zeros((len(statics), 128), dtype=np.float32),
            "z_temporal": np.zeros((len(statics), 128), dtype=np.float32),
            "z_fused": np.zeros((len(statics), 128), dtype=np.float32),
        }

    # -- risk-index composition ---------------------------------------- #
    @staticmethod
    def _strain(ctx: dict[str, float]) -> float:
        def g(k: str) -> float:
            v = ctx.get(k, float("nan"))
            try:
                v = float(v)
            except (TypeError, ValueError):
                return float("nan")
            return v if np.isfinite(v) else float("nan")

        dev = g("nhr_rmssd_dev_baseline")
        lf_hf = g("lf_hf_mean")
        hypo = g("glu_hypo_minutes")
        gslope = g("glu_slope")
        cov = g("coverage_mean")

        s = 0.0
        if np.isfinite(dev):
            s += 0.45 * max(0.0, -dev)
        if np.isfinite(lf_hf):
            s += 0.25 * max(0.0, 1.0 - min(lf_hf, 3.0) / 3.0)
        if np.isfinite(hypo):
            s += 0.20 * min(hypo / 60.0, 1.0)
        if np.isfinite(gslope):
            s += 0.10 * max(0.0, -gslope / 3.0)
        if np.isfinite(cov):
            s *= (0.6 + 0.4 * min(cov, 1.0))
        return float(np.clip(s, 0.0, 1.0))

    @staticmethod
    def _genome_norm(prs_total_z: float) -> float:
        return float(1.0 / (1.0 + np.exp(-1.2 * float(prs_total_z))))

    def clinical_risk_index(self, p_event: float, p_class: float, strain: float,
                            g_norm: float) -> float:
        raw = (W_EVENT * p_event + W_CLASS * p_class
               + W_STRAIN * strain + W_GENOME * g_norm)
        return float(100.0 * np.clip(raw, 0.0, 1.0))

    def _calibrated_event_probs(self, logits: np.ndarray) -> np.ndarray:
        """Per-class probabilities with out-of-fold temperature calibration.

        Accepts either a single ``(n_classes,)`` logit vector or a batch
        ``(n_windows, n_classes)`` matrix. Each class has its own temperature,
        so the transform is applied column-wise and vectorised.
        """
        z = np.asarray(logits, dtype=np.float64)
        single = z.ndim == 1
        if single:
            z = z[None, :]
        out = np.empty_like(z)
        for k in range(z.shape[1]):
            cal = self._calibrators.get(f"event_{k}")
            if cal is not None:
                out[:, k] = cal.transform(z[:, k])
            else:
                out[:, k] = 1.0 / (1.0 + np.exp(-np.clip(z[:, k], -60, 60)))
        return out[0] if single else out

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def score_patient(self, patient_id: str, end_idx: int | None = None) -> PatientReport:
        """Score one patient at the given instant (default: the latest)."""
        t0 = time.perf_counter()

        tensor, static_vec, ctx = self._window_features(patient_id, end_idx)
        out = self._forward(tensor[None, ...], static_vec[None, ...])

        if self.backend == "cmcl":
            logits = out["event_logits"][0]
            probs = self._calibrated_event_probs(logits)
            risk_logit = out["risk_logit"][0]
            rcal = self._calibrators.get("risk")
            p_event = float(rcal.transform(np.array([risk_logit]))[0]) if rcal else float(
                1.0 / (1.0 + np.exp(-risk_logit)))
            ttx = float(np.expm1(max(out["ttx"][0], 0.0)))
        else:
            probs = out["event_probs"][0]
            p_event = float(out["risk_logit"][0])
            p_event = float(1.0 / (1.0 + np.exp(-p_event)))
            ttx = float(np.expm1(max(out["ttx"][0], 0.0)))

        p_class = float(np.max(probs))
        event_probs = {k: float(v) for k, v in zip(EVENT_INDEX.keys(), probs)}
        dominant = max(event_probs, key=event_probs.get) if event_probs else "none"

        static_meta = self._static_metadata(patient_id)
        strain = self._strain(ctx)
        g_norm = self._genome_norm(float(static_meta.get("prs_raw_z", 0.0)))
        risk = self.clinical_risk_index(p_event, p_class, strain, g_norm)
        tier = tier_for(risk)

        coverage = float(ctx.get("coverage_mean", 0.0) or 0.0)
        if not np.isfinite(coverage):
            coverage = 0.0
        p_vec = np.array(list(event_probs.values()), dtype=np.float64)
        p_vec = p_vec / max(p_vec.sum(), 1e-9)
        entropy = float(-(p_vec * np.log(p_vec + 1e-12)).sum())
        conf_model = float(np.clip(1.0 - entropy / np.log(len(p_vec) + 1e-9), 0, 1))
        confidence = float(np.clip(0.5 * coverage + 0.5 * conf_model, 0.0, 1.0))

        atts = analytic_attributions(ctx, static_meta, risk)
        inter_share, groups = interaction_share_from_attributions(atts)
        expl = Explanation(
            attributions=atts,
            interaction_share=inter_share,
            modality_share=groups,
            narrative=build_narrative(atts, risk, tier),
        )

        actions = build_actions(ctx, static_meta, risk, event_probs, ttx)
        insights = build_insights(ctx, static_meta, risk, event_probs)

        latency = (time.perf_counter() - t0) * 1000.0
        feats = patient_features(patient_id, self.profile)
        as_of = feats.grid["ts"].iloc[-1]

        return PatientReport(
            patient_id=patient_id,
            as_of=as_of,
            risk_index=round(risk, 2),
            tier=tier,
            confidence=round(confidence, 3),
            event_probs={k: round(v, 4) for k, v in event_probs.items()},
            dominant_event=dominant,
            predicted_ttx_hours=round(ttx, 2),
            insights=insights,
            actions=actions,
            explanation=expl,
            context=ctx,
            static=static_meta,
            latency_ms=round(latency, 2),
            backend=self.backend,
            coverage=round(coverage, 3),
            baseline_rmssd=_baseline_rmssd(ctx),
            nocturnal_rmssd=_first_finite(ctx, "nhr_rmssd_mean"),
        )

    def score_cohort(self) -> pd.DataFrame:
        """Score every patient in the cohort (the triage board)."""
        rows = []
        for pid in self.bundle.patient_ids():
            rep = self.score_patient(pid)
            row = {
                "patient_id": pid,
                "risk_index": rep.risk_index,
                "tier": rep.tier,
                "confidence": rep.confidence,
                "dominant_event": rep.dominant_event,
                "p_silent_ischemia": rep.event_probs.get("silent_ischemia", 0.0),
                "p_glycemic_crash": rep.event_probs.get("glycemic_crash", 0.0),
                "p_autonomic_arrhythmia": rep.event_probs.get("autonomic_arrhythmia", 0.0),
                "pred_ttx_h": rep.predicted_ttx_hours,
                "urgency": rep.urgency,
                "age": rep.static.get("age"),
                "sex": rep.static.get("sex"),
                "bmi": rep.static.get("bmi"),
                "prs_raw_z": rep.static.get("prs_raw_z"),
                "prs_cvd_z": rep.static.get("prs_cvd_z"),
                "prs_t2d_z": rep.static.get("prs_t2d_z"),
                "dominant_snp": rep.static.get("dominant_snp"),
                "dominant_gene": rep.static.get("dominant_gene"),
                "nocturnal_rmssd": rep.context.get("nhr_rmssd_mean"),
                "rmssd_dev_baseline": rep.context.get("nhr_rmssd_dev_baseline"),
                "glu_min": rep.context.get("glu_min"),
                "coverage": rep.coverage,
                "latency_ms": rep.latency_ms,
                "as_of": rep.as_of,
                "backend": rep.backend,
            }
            row["headline"] = (rep.actions[0].title if rep.actions else "")
            rows.append(row)
        return pd.DataFrame(rows).sort_values("risk_index", ascending=False).reset_index(drop=True)

    def risk_series(self, patient_id: str, hours: float = 48.0,
                    step_minutes: float = 30.0) -> pd.DataFrame:
        """Risk index over a trailing period (for the trend chart)."""
        feats = patient_features(patient_id, self.profile)
        grid = feats.grid
        hb = int(self.profile.horizon_hours * 60 / self.profile.grid_minutes)
        step = max(1, int(step_minutes / self.profile.grid_minutes))
        n = len(grid)
        span = int(hours * 60 / self.profile.grid_minutes)
        ends = list(range(max(hb, n - span), n + 1, step))
        if not ends:
            ends = [n]

        tensors, statics, stamps, ctxs = [], [], [], []
        base_static = self.bundle.static_base_vector(patient_id)
        for end in ends:
            start = max(0, end - hb)
            tensors.append(grid[list(TEMPORAL_CHANNELS)].to_numpy(dtype=np.float32)[start:end].T)
            ctx_row = feats.window_context.iloc[end - 1]
            statics.append(np.concatenate([
                base_static,
                self.bundle.static_context_vector(ctx_row),
            ]).astype(np.float32))
            stamps.append(grid["ts"].iloc[end - 1])
            ctxs.append(ctx_row)

        Xt = np.asarray(tensors, dtype=np.float32)
        Xs = np.asarray(statics, dtype=np.float32)
        out = self._forward(Xt, Xs)

        if self.backend == "cmcl":
            probs = self._calibrated_event_probs(out["event_logits"])
            rcal = self._calibrators.get("risk")
            p_event = (rcal.transform(out["risk_logit"]) if rcal
                       else 1.0 / (1.0 + np.exp(-out["risk_logit"])))
        else:
            probs = out["event_probs"]
            p_event = 1.0 / (1.0 + np.exp(-out["risk_logit"]))

        rows = []
        for i, ts in enumerate(stamps):
            ctx = {k: (float(v) if pd.notna(v) else float("nan"))
                   for k, v in ctxs[i][list(WINDOW_CONTEXT_FEATURES)].items()}
            static_meta = self._static_metadata(patient_id)
            strain = self._strain(ctx)
            g_norm = self._genome_norm(float(static_meta.get("prs_raw_z", 0.0)))
            risk = self.clinical_risk_index(float(p_event[i]), float(probs[i].max()),
                                            strain, g_norm)
            rows.append({
                "ts": ts,
                "risk_index": round(risk, 2),
                "p_silent_ischemia": round(float(probs[i][0]), 4),
                "p_glycemic_crash": round(float(probs[i][1]), 4),
                "p_autonomic_arrhythmia": round(float(probs[i][2]), 4),
                "hrv_rmssd": ctx.get("nhr_rmssd_mean"),
                "hr": ctx.get("hr_mean"),
                "glucose": ctx.get("glu_mean"),
                "lf_hf": ctx.get("lf_hf_mean"),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _static_metadata(self, patient_id: str) -> dict[str, Any]:
        row = self.bundle.row(patient_id)
        meta = {
            "patient_id": patient_id,
            "age": float(row["age"]),
            "sex": row["sex"],
            "bmi": float(row["bmi"]),
            "hba1c": float(row["hba1c"]),
            "systolic_bp": float(row["systolic_bp"]),
            "ldl": float(row["ldl"]),
            "prs_raw_z": float(row["prs_raw_z"]),
            "prs_cvd_z": float(row["prs_cvd_z"]),
            "prs_t2d_z": float(row["prs_t2d_z"]),
            "prs_aut_z": float(row["prs_aut_z"]),
            "metformin": int(row["metformin"]),
            "statin": int(row["statin"]),
        }
        meta.update(self.bundle.genomic_metadata(patient_id))
        return meta

    # -- diagnostics ---------------------------------------------------- #
    def benchmark_latency(self, n: int = 40, warmup: int = 5) -> dict[str, float]:
        """Measure end-to-end inference latency (SLO: < 120 ms)."""
        ids = self.bundle.patient_ids()
        if not ids:
            return {}
        for i in range(warmup):
            self.score_patient(ids[i % len(ids)])
        samples = []
        for i in range(n):
            t0 = time.perf_counter()
            self.score_patient(ids[i % len(ids)])
            samples.append((time.perf_counter() - t0) * 1000.0)
        arr = np.asarray(samples)
        return {
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_ms": float(arr.max()),
            "slo_ms": SLO_LATENCY_MS,
            "slo_met": bool(np.percentile(arr, 95) < SLO_LATENCY_MS),
        }


def _first_finite(d: dict[str, float], key: str) -> float | None:
    v = d.get(key)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _baseline_rmssd(ctx: dict[str, float]) -> float | None:
    """Recover the patient's personal nocturnal RMSSD baseline.

    ``nhr_rmssd_dev_baseline`` is the *relative* deviation of the current
    nocturnal RMSSD from the patient's own trailing baseline, i.e.
    ``dev = current / baseline - 1``. The clinician needs the absolute pair
    (baseline vs. current), so we invert it here rather than showing the same
    number twice.
    """
    current = _first_finite(ctx, "nhr_rmssd_mean")
    dev = _first_finite(ctx, "nhr_rmssd_dev_baseline")
    if current is None:
        return None
    if dev is None:
        return current
    denom = 1.0 + float(np.clip(dev, -0.95, 4.0))
    return float(current / denom)
