"""
BioTwin.X — Central Configuration
==================================
Single source of truth for paths, dataset profiles, model hyper-parameters and
the genomic panel definition used across the synthetic generator, the feature
extractor, the CMCL training pipeline and the clinical inference engine.

Everything here is deliberately declarative and frozen-dataclass based so that
runs are reproducible and diffable in version control.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
COHORT_DIR: Final[Path] = DATA_DIR / "cohorts"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
CACHE_DIR: Final[Path] = DATA_DIR / "cache"
FEATURE_DIR: Final[Path] = DATA_DIR / "features"
MODEL_DIR: Final[Path] = DATA_DIR / "models"
ARTIFACT_DIR: Final[Path] = DATA_DIR / "artifacts"

for _p in (DATA_DIR, COHORT_DIR, RAW_DIR, CACHE_DIR, FEATURE_DIR, MODEL_DIR,
           ARTIFACT_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Clinical ontology
# --------------------------------------------------------------------------- #
EVENT_LABELS: Final[tuple[str, ...]] = (
    "silent_ischemia",
    "glycemic_crash",
    "autonomic_arrhythmia",
)

TIERS: Final[tuple[str, ...]] = ("LOW", "ELEVATED", "HIGH", "CRITICAL")

#: Hard clinical SLOs advertised by the platform (see README §6.4).
SLO_ROC_AUC: Final[float] = 0.94
SLO_F1: Final[float] = 0.91
SLO_LATENCY_MS: Final[float] = 120.0


# --------------------------------------------------------------------------- #
# Dataset profiles
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DatasetProfile:
    """Controls the size, cadence and fidelity of the synthetic telemetry."""

    name: str
    n_patients: int
    days: int

    # Per-modality sampling cadence in seconds. Deliberately *different* and
    # mutually non-commensurate: this is the "asynchronous" property the whole
    # architecture is built around.
    rr_period_s: int = 1          # ECG patch — beat-to-beat R-R intervals (1 Hz)
    ppg_period_s: int = 60        # Photoplethysmography ring/watch — 1 min
    cgm_period_s: int = 300       # Continuous Glucose Monitor — 5 min
    spo2_period_s: int = 60       # SpO2 — 1 min
    temp_period_s: int = 300      # Skin temperature — 5 min
    sleep_epoch_s: int = 30       # Sleep staging epochs — 30 s (night only)

    #: Common resampling grid for the temporal encoder (5-minute bins).
    grid_minutes: int = 5

    #: Prediction horizon expressed in hours.
    horizon_hours: int = 12

    seed: int = 20260214

    @property
    def seconds(self) -> int:
        return self.days * 86_400

    @property
    def n_grid_bins(self) -> int:
        return int(self.seconds / (self.grid_minutes * 60))

    def describe(self) -> str:
        return (
            f"{self.name}: {self.n_patients} patients x {self.days} days "
            f"| grid={self.grid_minutes}min | horizon={self.horizon_hours}h"
        )


PROFILES: Final[dict[str, DatasetProfile]] = {
    # `demo` is the zero-friction quick start (a notebook-friendly cohort).
    "demo": DatasetProfile(name="demo", n_patients=20, days=7),
    # `standard` is the headline evaluation cohort: enough *patients* for a
    # patient-level cross-validation to mean something.
    "standard": DatasetProfile(name="standard", n_patients=100, days=14),
    # `research` is the full-scale cohort for a funded deployment.
    "research": DatasetProfile(name="research", n_patients=240, days=30),
    # `smoke` exists purely so the test-suite can run in seconds.
    "smoke": DatasetProfile(name="smoke", n_patients=6, days=2),
}

DEFAULT_PROFILE: Final[str] = os.getenv("BIOTWIN_PROFILE", "demo")


# --------------------------------------------------------------------------- #
# Model hyper-parameters (Cross-Modal Contrastive Learning)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelConfig:
    """Hyper-parameters for the dual-encoder CMCL architecture."""

    # --- encoders ---
    static_input_dim: int = 0          # inferred at build time
    temporal_channels: int = 12        # 10 physiological + 2 asynchrony (staleness) channels
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 3
    ff_mult: int = 2
    dropout: float = 0.20

    #: Probability of hiding the genomic/clinical archetype from a training
    #: sample. This is the single most effective guard against the static
    #: branch being used as a patient-fingerprint memorisation channel.
    archetype_dropout: float = 0.15

    # --- patch embedding (PatchTST-lite) ---
    patch_len: int = 12
    patch_stride: int = 6

    # --- projection heads ---
    proj_dim: int = 128

    # --- optimisation ---
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    epochs: int = 40
    warmup_steps: int = 120
    grad_clip: float = 1.0

    # --- contrastive learning ---
    temperature: float = 0.07
    contrastive_weight: float = 0.35
    label_smoothing: float = 0.03

    # --- loss weights ---
    w_event: float = 1.0
    w_ttx: float = 0.4
    w_risk: float = 0.8

    seed: int = 20260214

    def describe(self) -> str:
        return (
            f"CMCL[d_model={self.d_model}, L={self.n_layers}, heads={self.n_heads}, "
            f"patch={self.patch_len}/{self.patch_stride}, tau={self.temperature}]"
        )


# --------------------------------------------------------------------------- #
# Modality registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModalitySpec:
    key: str
    unit: str
    cadence_s: int
    description: str


MODALITIES: Final[tuple[ModalitySpec, ...]] = (
    ModalitySpec("ecg_rr", "ms", 1, "ECG patch beat-to-beat R-R interval series"),
    ModalitySpec("ppg_hr", "bpm", 60, "Photoplethysmography-derived instantaneous heart rate"),
    ModalitySpec("ppg_rmssd", "ms", 60, "Wrist-device reported RMSSD (HRV)"),
    ModalitySpec("cgm_glucose", "mg/dL", 300, "Continuous Glucose Monitor interstitial glucose"),
    ModalitySpec("spo2", "%", 60, "Peripheral oxygen saturation"),
    ModalitySpec("skin_temp", "degC", 300, "Skin temperature (circadian proxy)"),
    ModalitySpec("sleep_stage", "stage", 30, "Sleep stage epoch (0-4)"),
)

MODALITY_KEYS: Final[tuple[str, ...]] = tuple(m.key for m in MODALITIES)
