"""
BioTwin.X — Data Access Layer
==============================
Centralised, cached access to the cohort, the telemetry streams, the derived
feature grids and the assembled modelling tensors.

Everything is keyed by a *dataset profile* so that a ``demo`` cohort and a
``research`` cohort can coexist on disk without collision, and every expensive
derivation (feature grids, window tensors) is memoised to Parquet.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import (
    ARTIFACT_DIR,
    CACHE_DIR,
    COHORT_DIR,
    DATA_DIR,
    DEFAULT_PROFILE,
    FEATURE_DIR,
    MODEL_DIR,
    PROFILES,
    RAW_DIR,
    DatasetProfile,
)
from .features import (
    DERIVED_CLINICAL,
    STATIC_CLINICAL,
    TEMPORAL_CHANNELS,
    WINDOW_CONTEXT_FEATURES,
    TelemetryFeatures,
    Window,
    build_windows,
    context_matrix,
    extract_features,
    window_tensor,
)
from .genomics import SNP_PANEL, PANEL_BY_RSID

# --------------------------------------------------------------------------- #
# Small IO helpers
# --------------------------------------------------------------------------- #
def _hash(*parts: Any) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:12]


def _generator_module():
    """Import the root-level ``synthetic_generator`` module lazily.

    It lives at the project root (not inside the package) so that it can be
    executed as a standalone CLI; we add the root to ``sys.path`` on demand.
    """
    import importlib
    import sys

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module("synthetic_generator")


def ensure_cohort(profile: DatasetProfile | str = DEFAULT_PROFILE,
                  force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (patients_static, patient_events) for a profile, generating if missing.

    Cohort tables are **profile-keyed** (``data/cohorts/<profile>_*.csv``).
    A single shared CSV would let a cached telemetry file from one profile be
    paired with another profile's static cohort — a silent, catastrophic
    data-integrity bug.
    """
    prof = PROFILES[profile] if isinstance(profile, str) else profile
    static_path = COHORT_DIR / f"{prof.name}_patients_static.csv"
    events_path = COHORT_DIR / f"{prof.name}_patient_events.csv"

    need = force or not static_path.exists() or not events_path.exists()
    if need:
        gen_mod = _generator_module()
        gen = gen_mod.SyntheticCohortGenerator(profile)
        gen.generate_all(persist=True, verbose=False)

    static = pd.read_csv(static_path)
    events = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame(
        columns=["patient_id", "event_type", "event_ts"])
    return static, events


def load_telemetry(patient_id: str, profile: DatasetProfile | str = DEFAULT_PROFILE) -> pd.DataFrame:
    """Load the raw asynchronous telemetry for one patient."""
    prof = PROFILES[profile] if isinstance(profile, str) else profile
    path = CACHE_DIR / f"tele_{prof.name}_{patient_id}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    raw = RAW_DIR / f"{prof.name}_{patient_id}.parquet"
    if raw.exists():
        tele = pd.read_parquet(raw)
        tele.to_parquet(path, index=False, compression="zstd")
        return tele
    # Regenerate on demand (deterministic).
    static, _ = ensure_cohort(prof)
    row = static[static["patient_id"] == patient_id].iloc[0]
    gen_mod = _generator_module()
    tele = gen_mod.SyntheticCohortGenerator(prof).generate_telemetry(
        gen_mod._row_to_profile(row))
    tele.to_parquet(path, index=False, compression="zstd")
    return tele


def _row_to_profile(row: pd.Series):
    """Rehydrate a :class:`PatientProfile` from a static CSV row."""
    gen_mod = _generator_module()
    PatientProfile = gen_mod.PatientProfile
    gt = {k[3:]: int(v) for k, v in row.items() if k.startswith("gt_")}
    ts = row.get("event_ts")
    return PatientProfile(
        patient_id=row["patient_id"], age=row["age"], sex=row["sex"], bmi=row["bmi"],
        smoking=int(row["smoking"]), alcohol_units_wk=row["alcohol_units_wk"],
        sleep_hours=row["sleep_hours"], resting_hr=row["resting_hr"],
        systolic_bp=row["systolic_bp"], diastolic_bp=row["diastolic_bp"],
        ldl=row["ldl"], hdl=row["hdl"], triglycerides=row["triglycerides"],
        hba1c=row["hba1c"], fasting_glucose=row["fasting_glucose"],
        antihypertensive=int(row["antihypertensive"]), statin=int(row["statin"]),
        metformin=int(row["metformin"]), family_history_cv=int(row["family_history_cv"]),
        family_history_t2d=int(row["family_history_t2d"]),
        prs_raw=row["prs_raw"], prs_raw_z=row["prs_raw_z"], prs_t2d_z=row["prs_t2d_z"],
        prs_cvd_z=row["prs_cvd_z"], prs_dys_z=row["prs_dys_z"], prs_aut_z=row["prs_aut_z"],
        primary_event=row["primary_event"],
        event_ts=pd.Timestamp(ts) if isinstance(ts, str) and ts else None,
        ttx_hours=row["ttx_hours"], event_in_horizon=int(row["event_in_horizon"]),
        latent_risk=row["latent_risk"], genotypes=gt,
        sleep_offset_h=row.get("sleep_offset_h", 0.0) or 0.0,
        respiratory_rate=row.get("respiratory_rate", 14.5) or 14.5,
        events=[],
    )


# --------------------------------------------------------------------------- #
# Feature-grid cache
# --------------------------------------------------------------------------- #
def patient_features(patient_id: str, profile: DatasetProfile | str = DEFAULT_PROFILE,
                     use_cache: bool = True) -> TelemetryFeatures:
    """Extract (and memoise) the 5-minute feature grid for one patient."""
    prof = PROFILES[profile] if isinstance(profile, str) else profile
    cache_path = FEATURE_DIR / f"grid_{prof.name}_{patient_id}.parquet"
    ctx_path = FEATURE_DIR / f"ctx_{prof.name}_{patient_id}.parquet"

    if use_cache and cache_path.exists() and ctx_path.exists():
        grid = pd.read_parquet(cache_path)
        ctx = pd.read_parquet(ctx_path)
        return TelemetryFeatures(grid=grid, context_names=WINDOW_CONTEXT_FEATURES,
                                 window_context=ctx)

    tele = load_telemetry(patient_id, prof)
    feats = extract_features(tele, prof)
    feats.grid.to_parquet(cache_path, index=False, compression="zstd")
    feats.window_context.to_parquet(ctx_path, index=False, compression="zstd")
    return feats


# --------------------------------------------------------------------------- #
# Cohort bundle
# --------------------------------------------------------------------------- #
@dataclass
class CohortBundle:
    """Everything the model and the dashboard need about a cohort."""

    static: pd.DataFrame
    events: pd.DataFrame
    profile: DatasetProfile
    genotype_stats: dict[str, tuple[float, float]]

    # -- construction ---------------------------------------------------- #
    @classmethod
    def load(cls, profile: DatasetProfile | str = DEFAULT_PROFILE,
             force_regen: bool = False) -> "CohortBundle":
        prof = PROFILES[profile] if isinstance(profile, str) else profile
        static, events = ensure_cohort(prof, force=force_regen)
        gt_cols = [c for c in static.columns if c.startswith("gt_")]
        stats: dict[str, tuple[float, float]] = {}
        for c in gt_cols:
            v = static[c].to_numpy(dtype=np.float64)
            sd = float(v.std())
            stats[c] = (float(v.mean()), sd if sd > 1e-8 else 1.0)
        return cls(static=static, events=events, profile=prof, genotype_stats=stats)

    # -- per-patient helpers --------------------------------------------- #
    def patient_ids(self) -> list[str]:
        return self.static["patient_id"].tolist()

    def row(self, patient_id: str) -> pd.Series:
        return self.static.loc[self.static["patient_id"] == patient_id].iloc[0]

    def events_for(self, patient_id: str) -> list[tuple[str, pd.Timestamp]]:
        sub = self.events[self.events["patient_id"] == patient_id]
        return [(r.event_type, pd.Timestamp(r.event_ts)) for r in sub.itertuples()]

    def genomic_metadata(self, patient_id: str) -> dict[str, Any]:
        row = self.row(patient_id)
        best_rsid, best = None, -1e9
        for locus in SNP_PANEL:
            dosage = int(row.get(f"gt_{locus.rsid}", 0))
            contrib = dosage * locus.effect_weight
            if contrib > best:
                best, best_rsid = contrib, locus.rsid
        locus = PANEL_BY_RSID[best_rsid] if best_rsid else SNP_PANEL[0]
        return {
            "dominant_snp": locus.rsid,
            "dominant_gene": locus.gene,
            "dominant_trait": locus.trait,
            "dominant_dosage": int(row.get(f"gt_{locus.rsid}", 0)),
            "dominant_effect": locus.effect_weight,
            "dominant_freq": locus.population_freq,
        }

    # -- static vector ---------------------------------------------------- #
    def clinical_vector(self, patient_id: str) -> np.ndarray:
        row = self.row(patient_id)
        vals = []
        for name in STATIC_CLINICAL:
            v = row.get(name, 0)
            if name == "sex_male":
                vals.append(1.0 if row.get("sex") == "M" else 0.0)
            elif name == "smoking":
                vals.append(float(v))
            else:
                vals.append(float(v))
        return np.asarray(vals, dtype=np.float32)

    def genomic_vector(self, patient_id: str) -> np.ndarray:
        row = self.row(patient_id)
        dosage = []
        for locus in SNP_PANEL:
            d = float(row.get(f"gt_{locus.rsid}", 0))
            mu, sd = self.genotype_stats.get(f"gt_{locus.rsid}", (0.0, 1.0))
            dosage.append((d - mu) / sd)
        prs = [float(row.get(f"prs_{k}_z", 0.0)) for k in ("raw", "t2d", "cvd", "dys", "aut")]
        meta = self.genomic_metadata(patient_id)
        dominant = []
        for locus in SNP_PANEL:
            d = float(row.get(f"gt_{locus.rsid}", 0))
            mu, sd = self.genotype_stats.get(f"gt_{locus.rsid}", (0.0, 1.0))
            dominant.append(((d - mu) / sd) * locus.effect_weight)
        return np.asarray(prs + dosage + [float(np.max(dominant))], dtype=np.float32)

    def derived_vector(self, patient_id: str) -> np.ndarray:
        row = self.row(patient_id)
        bmi = float(row["bmi"])
        hba1c = float(row["hba1c"])
        tg = float(row["triglycerides"])
        hdl = float(row["hdl"])
        sbp = float(row["systolic_bp"])
        age = float(row["age"])
        prs_cvd = float(row.get("prs_cvd_z", 0.0))
        prs_t2d = float(row.get("prs_t2d_z", 0.0))
        ir = (bmi - 22.0) / 12.0 + (hba1c - 5.2) / 2.5
        athero = tg / max(hdl, 1e-3) - bmi / 25.0
        metsyn = float((sbp >= 130) + (tg >= 150) + (hdl < 40 or hba1c >= 5.6)
                       + (bmi >= 30) >= 3)
        return np.asarray([
            ir, athero, metsyn,
            age * prs_cvd / 100.0,
            bmi * prs_t2d / 100.0,
        ], dtype=np.float32)

    def static_vector(self, patient_id: str,
                      ctx_row: pd.Series | None = None) -> np.ndarray:
        parts = [self.clinical_vector(patient_id),
                 self.genomic_vector(patient_id),
                 self.derived_vector(patient_id)]
        if ctx_row is not None:
            parts.append(ctx_row[list(WINDOW_CONTEXT_FEATURES)].to_numpy(dtype=np.float32))
        else:
            parts.append(np.zeros(len(WINDOW_CONTEXT_FEATURES), dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def static_base_vector(self, patient_id: str) -> np.ndarray:
        """Patient-invariant part of the static vector (no window context).

        Hoisted out of the window loop: the clinical + genomic architecture of
        a patient is identical for every window, so recomputing it per window
        was pure overhead (it dominated the tensor build).
        """
        return np.concatenate([
            self.clinical_vector(patient_id),
            self.genomic_vector(patient_id),
            self.derived_vector(patient_id),
        ]).astype(np.float32)

    def static_context_vector(self, ctx_row: pd.Series) -> np.ndarray:
        return ctx_row[list(WINDOW_CONTEXT_FEATURES)].to_numpy(dtype=np.float32)

    @property
    def static_dim(self) -> int:
        return (len(STATIC_CLINICAL) + 5 + len(SNP_PANEL) + 1
                + len(DERIVED_CLINICAL) + len(WINDOW_CONTEXT_FEATURES))


# --------------------------------------------------------------------------- #
# Dataset assembly for training
# --------------------------------------------------------------------------- #
@dataclass
class CohortTensors:
    X_time: np.ndarray          # (N, C, T)
    X_static: np.ndarray        # (N, D)
    y_event: np.ndarray         # (N, 3)
    y_risk: np.ndarray          # (N,)
    y_ttx: np.ndarray           # (N,)
    ttx_mask: np.ndarray        # (N,)
    patient_index: np.ndarray   # (N,)
    meta: list[dict[str, Any]]

    def __len__(self) -> int:
        return len(self.X_time)

    def subset(self, idx: np.ndarray) -> "CohortTensors":
        return CohortTensors(
            X_time=self.X_time[idx], X_static=self.X_static[idx],
            y_event=self.y_event[idx], y_risk=self.y_risk[idx],
            y_ttx=self.y_ttx[idx], ttx_mask=self.ttx_mask[idx],
            patient_index=self.patient_index[idx],
            meta=[self.meta[i] for i in idx],
        )


EVENT_INDEX: Final[dict[str, int]] = {
    "silent_ischemia": 0,
    "glycemic_crash": 1,
    "autonomic_arrhythmia": 2,
}


def build_tensors(bundle: CohortBundle, patients: Sequence[str] | None = None,
                  stride_bins: int = 6, max_windows_per_patient: int = 400,
                  seed: int = 20260214, cache: bool = True) -> CohortTensors:
    """Materialise the (window tensor, static vector, label) arrays."""
    cache_path = FEATURE_DIR / f"tensors_{bundle.profile.name}_s{stride_bins}.npz"
    if cache and cache_path.exists():
        try:
            z = np.load(cache_path, allow_pickle=True)
            meta = list(z["meta"])
            return CohortTensors(
                X_time=z["X_time"], X_static=z["X_static"], y_event=z["y_event"],
                y_risk=z["y_risk"], y_ttx=z["y_ttx"], ttx_mask=z["ttx_mask"],
                patient_index=z["patient_index"], meta=meta,
            )
        except Exception:  # pragma: no cover - corrupt cache -> rebuild
            pass

    rng = np.random.default_rng(seed)
    ids = list(patients) if patients is not None else bundle.patient_ids()

    Xt, Xs, Ye, Yr, Yt, Ym, PI, Meta = [], [], [], [], [], [], [], []

    for pid in ids:
        feats = patient_features(pid, bundle.profile)
        wins = build_windows(feats.grid, pid, bundle.events_for(pid),
                             bundle.profile, stride_bins=stride_bins)
        if not wins:
            continue
        if len(wins) > max_windows_per_patient:
            pos = [i for i, w in enumerate(wins) if w.label == 1]
            neg = [i for i, w in enumerate(wins) if w.label == 0]
            keep_pos = rng.choice(pos, size=min(len(pos), max_windows_per_patient // 3),
                                  replace=False) if pos else []
            keep_neg = rng.choice(neg, size=max_windows_per_patient - len(keep_pos),
                                  replace=False) if neg else []
            sel = sorted(list(keep_pos) + list(keep_neg))
            wins = [wins[i] for i in sel]

        tensor = window_tensor(feats.grid, wins)
        ctxs = context_matrix(feats.window_context, wins)
        base_static = bundle.static_base_vector(pid)

        for k, w in enumerate(wins):
            Xt.append(tensor[k])
            Xs.append(np.concatenate([base_static, ctxs[k]]).astype(np.float32))
            y = np.zeros(3, dtype=np.float32)
            if w.event_type in EVENT_INDEX:
                y[EVENT_INDEX[w.event_type]] = 1.0
            Ye.append(y)
            Yr.append(1.0 if w.label else 0.0)
            if w.label and np.isfinite(w.ttx_hours):
                Yt.append(float(np.log1p(max(w.ttx_hours, 0.0))))
                Ym.append(1.0)
            else:
                Yt.append(0.0)
                Ym.append(0.0)
            PI.append(ids.index(pid))
            Meta.append({
                "patient_id": pid, "end_idx": w.end_idx, "label": w.label,
                "event_type": w.event_type, "ttx_hours": w.ttx_hours,
            })

    out = CohortTensors(
        X_time=np.asarray(Xt, dtype=np.float32),
        X_static=np.asarray(Xs, dtype=np.float32),
        y_event=np.asarray(Ye, dtype=np.float32),
        y_risk=np.asarray(Yr, dtype=np.float32),
        y_ttx=np.asarray(Yt, dtype=np.float32),
        ttx_mask=np.asarray(Ym, dtype=np.float32),
        patient_index=np.asarray(PI, dtype=np.int64),
        meta=Meta,
    )
    if cache:
        try:
            np.savez_compressed(
                cache_path,
                X_time=out.X_time, X_static=out.X_static, y_event=out.y_event,
                y_risk=out.y_risk, y_ttx=out.y_ttx, ttx_mask=out.ttx_mask,
                patient_index=out.patient_index,
                meta=np.asarray(Meta, dtype=object),
            )
        except Exception:  # pragma: no cover
            pass
    return out


def patient_split(bundle: CohortBundle, val_frac: float = 0.25,
                  seed: int = 20260214) -> tuple[list[str], list[str]]:
    """Patient-level split (never window-level) to prevent subject leakage."""
    rng = np.random.default_rng(seed)
    ids = list(bundle.patient_ids())
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac)))
    return sorted(ids[n_val:]), sorted(ids[:n_val])
