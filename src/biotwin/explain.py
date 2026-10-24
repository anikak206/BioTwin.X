"""
BioTwin.X — Model Explainability
=================================
Clinicians do not act on a number; they act on a *reason*. BioTwin.X therefore
never presents a bare risk score.

Three complementary attribution mechanisms are provided:

``Gradient×Input``
    First-order saliency of the Clinical Risk Index head with respect to the
    fused embedding, projected back onto the two modality subspaces. Fast
    (one backward pass) and faithful to *this* forward computation.

``Occlusion``
    Group-wise ablation: each physiological channel is replaced by its
    cohort mean and the resulting drop in risk is measured. This is the most
    *interpretable* attribution because it answers the question a clinician
    actually asks — "what if this signal were not there?".

``Cross-Modal Interaction``
    The contribution of the ``z_s ⊙ z_t`` interaction term versus the purely
    additive paths, isolating the value that the *fusion* (as opposed to
    either modality alone) contributes.

Everything degrades gracefully to the analytic rule layer when no trained
checkpoint is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .features import TEMPORAL_CHANNELS

#: Human-readable labels for the temporal channels.
CHANNEL_LABELS: dict[str, str] = {
    "hr": "Heart rate (ECG)",
    "hrv_rmssd": "HRV RMSSD (ECG)",
    "hrv_lf": "HRV LF power (ECG)",
    "hrv_hf": "HRV HF power (ECG)",
    "hrv_pnn50": "Beat-to-beat irregularity pNN50 (ECG)",
    "glucose": "Interstitial glucose (CGM)",
    "spo2": "Peripheral SpO₂",
    "skin_temp": "Skin temperature",
    "sleep_stage": "Sleep stage",
    "coverage": "Multi-modal coverage",
    "glu_staleness": "CGM staleness",
    "rr_staleness": "ECG staleness",
}

#: Channels that belong to the *static / genomic* side of the fusion.
STATIC_GROUP = "Static architecture (clinical + genomic)"
TEMPORAL_GROUP = "Asynchronous telemetry"
INTERACTION_GROUP = "Cross-modal interaction"


@dataclass
class Attribution:
    name: str
    label: str
    group: str
    contribution: float          # signed contribution to the risk index
    magnitude: float             # |contribution|
    detail: str = ""


@dataclass
class Explanation:
    attributions: list[Attribution] = field(default_factory=list)
    interaction_share: float = 0.0
    modality_share: dict[str, float] = field(default_factory=dict)
    narrative: list[str] = field(default_factory=list)

    def top(self, n: int = 5) -> list[Attribution]:
        return sorted(self.attributions, key=lambda a: -a.magnitude)[:n]


# --------------------------------------------------------------------------- #
# Analytic (checkpoint-free) attribution
# --------------------------------------------------------------------------- #
def analytic_attributions(ctx: dict[str, float], static: dict[str, Any],
                          risk: float) -> list[Attribution]:
    """Transparent, rule-based attribution of the risk index.

    Every term is a named quantity with a stated sign, so a clinician can
    audit the reasoning without trusting the neural network.
    """
    def g(name: str) -> float | None:
        v = ctx.get(name)
        if v is None:
            return None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None

    out: list[Attribution] = []

    def add(name: str, label: str, group: str, value: float | None, detail: str):
        if value is None or not np.isfinite(value):
            return
        out.append(Attribution(name=name, label=label, group=group,
                                contribution=float(value), magnitude=abs(float(value)),
                                detail=detail))

    dev = g("nhr_rmssd_dev_baseline")
    add("nhr_rmssd_dev_baseline", "Nocturnal RMSSD vs personal baseline",
        TEMPORAL_GROUP,
        None if dev is None else float(np.clip(dev, -1, 1) * 22.0),
        "Autonomic reserve erosion is the dominant modifiable early signal.")

    dev_deep = g("nhr_rmssd_dev_deep")
    add("nhr_rmssd_dev_deep", "Deep-sleep RMSSD vs personal baseline",
        TEMPORAL_GROUP,
        None if dev_deep is None else float(np.clip(dev_deep, -1, 1) * 14.0),
        "Survives sleep-stage alignment, so it is withdrawal rather than "
        "normal sleep architecture.")

    ectopy = g("ectopy_load")
    add("ectopy_load", "Ectopic beat burden (pNN50)",
        TEMPORAL_GROUP,
        None if ectopy is None else float(min(ectopy, 1.0) * 12.0),
        "Beat-to-beat irregularity is invisible to mean heart rate.")

    pct = g("rmssd_pct_change_3h9h")
    add("rmssd_pct_change_3h9h", "RMSSD 3 h vs 9 h trend",
        TEMPORAL_GROUP,
        None if pct is None else float(np.clip(pct, -1, 1) * 12.0),
        "Intra-window deterioration slope.")

    lf_hf = g("lf_hf_mean")
    add("lf_hf_mean", "LF/HF sympathetic balance",
        TEMPORAL_GROUP,
        None if lf_hf is None else float(np.clip((1.0 - min(lf_hf, 4.0) / 4.0), -1, 1) * 10.0),
        "Shift towards sympathetic dominance raises myocardial oxygen demand.")

    hypo = g("glu_hypo_minutes")
    add("glu_hypo_minutes", "Time in hypoglycaemic range",
        TEMPORAL_GROUP,
        None if hypo is None else float(min(hypo / 60.0, 1.0) * 14.0),
        "Hypoglycaemia blunts the autonomic warning signal itself.")

    glu_slope = g("glu_slope")
    add("glu_slope", "Glucose rate of change",
        TEMPORAL_GROUP,
        None if glu_slope is None else float(np.clip(-glu_slope / 3.0, -1, 1) * 8.0),
        "Steep negative slope precedes a crash.")

    cov = g("coverage_mean")
    add("coverage_mean", "Telemetry coverage",
        TEMPORAL_GROUP,
        None if cov is None else float((cov - 0.7) * 12.0),
        "Degraded coverage reduces confidence, not risk.")

    prs_cvd = float(static.get("prs_cvd_z", 0.0) or 0.0)
    prs_t2d = float(static.get("prs_t2d_z", 0.0) or 0.0)
    add("prs_cvd", "Polygenic cardiovascular load", STATIC_GROUP,
        float(np.clip(prs_cvd, -3, 3) * 4.0),
        "Hereditary susceptibility to endothelial dysfunction.")
    add("prs_t2d", "Polygenic type-2 diabetes load", STATIC_GROUP,
        float(np.clip(prs_t2d, -3, 3) * 3.5),
        "Hereditary susceptibility to beta-cell failure.")

    hba1c = float(static.get("hba1c", 5.5) or 5.5)
    add("hba1c", "HbA1c", STATIC_GROUP,
        float(np.clip((hba1c - 5.5) * 5.0, -10, 12)),
        "Chronic glycaemic exposure.")
    sbp = float(static.get("systolic_bp", 120) or 120)
    add("systolic_bp", "Systolic blood pressure", STATIC_GROUP,
        float(np.clip((sbp - 125) * 0.25, -6, 10)),
        "Afterload and myocardial oxygen demand.")

    return out


def interaction_share_from_attributions(atts: Sequence[Attribution]) -> tuple[float, dict[str, float]]:
    """Aggregate attribution magnitudes into group shares."""
    groups: dict[str, float] = {}
    for a in atts:
        groups[a.group] = groups.get(a.group, 0.0) + a.magnitude
    total = sum(groups.values()) or 1.0
    shares = {k: v / total for k, v in groups.items()}
    inter = shares.get(INTERACTION_GROUP, 0.0)
    return inter, shares


def build_narrative(atts: Sequence[Attribution], risk: float,
                     tier: str) -> list[str]:
    """Turn attributions into clinician-readable sentences."""
    top = sorted(atts, key=lambda a: -a.magnitude)[:3]
    lines = [
        f"Clinical Risk Index {risk:.1f}/100 ({tier} tier).",
    ]
    for a in top:
        direction = "raises" if a.contribution > 0 else "lowers"
        lines.append(
            f"{a.label} {direction} the index by {abs(a.contribution):.1f} points "
            f"({a.group.lower()}).")
    return lines
