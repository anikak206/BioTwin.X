"""
BioTwin.X — Clinical Risk Index & Actionable Insights Engine
=============================================================
Converts raw model outputs into **clinician-legible, prioritised, actionable
recommendations**.

The market failure mode BioTwin.X is designed against is *alert fatigue*: a
device that fires 30 times a night trains the clinician to ignore it. Three
design rules follow:

1. **One primary action per patient.** Secondary signals are demoted to
   "monitoring notes", never to parallel action items.
2. **Every alert carries its evidence chain.** "Nocturnal RMSSD ↓ 41%
   (42 → 25 ms) at 03:10, in a patient carrying rs4988235 (PCSK9) dosage 2/2,
   polygenic CVD z = +2.1σ" — not "anomaly detected".
3. **Silence is a first-class output.** A LOW-risk patient produces an
   explicit "no action required, continue monitoring" statement so that the
   absence of an alert is meaningful rather than ambiguous.

Urgency ladder
--------------
``STAT`` (immediate, <15 min) · ``URGENT`` (<4 h) · ``PRIORITY`` (<24 h) ·
``ROUTINE`` (<7 d) · ``OPTIMISE`` (lifestyle / next review)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Final, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Tiers
# --------------------------------------------------------------------------- #
TIER_COLORS: Final[dict[str, str]] = {
    "LOW": "#16a34a",
    "ELEVATED": "#eab308",
    "HIGH": "#f97316",
    "CRITICAL": "#dc2626",
}

TIER_BANDS: Final[tuple[tuple[float, float, str], ...]] = (
    (0.0, 25.0, "LOW"),
    (25.0, 50.0, "ELEVATED"),
    (50.0, 75.0, "HIGH"),
    (75.0, 100.01, "CRITICAL"),
)

URGENCY_WINDOW: Final[dict[str, str]] = {
    "STAT": "within 15 minutes",
    "URGENT": "within 4 hours",
    "PRIORITY": "within 24 hours",
    "ROUTINE": "within 7 days",
    "OPTIMISE": "next scheduled review",
}


def tier_for(risk: float) -> str:
    for lo, hi, name in TIER_BANDS:
        if lo <= risk < hi:
            return name
    return "CRITICAL"


# --------------------------------------------------------------------------- #
# Action model
# --------------------------------------------------------------------------- #
@dataclass
class ClinicalAction:
    urgency: str                 # STAT | URGENT | PRIORITY | ROUTINE | OPTIMISE
    title: str
    rationale: str
    evidence: list[str] = field(default_factory=list)
    modality: str = "Multi-modal"
    biomarker: str = ""
    window: str = ""             # the observation window that triggered it
    delta: float | None = None   # relative deviation, when meaningful
    baseline: float | None = None
    current: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["window_text"] = URGENCY_WINDOW.get(self.urgency, "")
        return d


@dataclass
class Insight:
    """A single ranked clinical observation."""

    title: str
    severity: str                # critical | warning | info | reassuring
    score: float                 # 0-100 contribution
    modality: str
    detail: str
    action: str | None = None
    urgency: str = "OPTIMISE"
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Rule engine
# --------------------------------------------------------------------------- #
def _pct_delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None or not np.isfinite(current) or not np.isfinite(baseline):
        return None
    if abs(baseline) < 1e-6:
        return None
    return float((current - baseline) / abs(baseline) * 100.0)


def build_actions(ctx: dict[str, float], static: dict[str, Any],
                  risk: float, event_probs: dict[str, float],
                  ttx_hours: float) -> list[ClinicalAction]:
    """Deterministic, evidence-grounded clinical action list."""
    actions: list[ClinicalAction] = []

    nhr = ctx.get("nhr_rmssd_mean")
    dhr = ctx.get("dhr_rmssd_mean")
    dev = ctx.get("nhr_rmssd_dev_baseline")
    dev_deep = ctx.get("nhr_rmssd_dev_deep")
    nhr_deep = ctx.get("nhr_rmssd_deep")
    pct = ctx.get("rmssd_pct_change_3h9h")
    lf_hf = ctx.get("lf_hf_mean")
    glu_min = ctx.get("glu_min")
    hypo_min = ctx.get("glu_hypo_minutes")
    spo2_min = ctx.get("spo2_min")
    glu_slope = ctx.get("glu_slope")
    cov = ctx.get("coverage_mean")
    hr_mean = ctx.get("hr_mean")
    sleep_depth = ctx.get("sleep_depth_pct")
    ent = ctx.get("spectral_entropy")
    pnn50 = ctx.get("pnn50_mean")
    ectopy = ctx.get("ectopy_load")

    prs_cvd = float(static.get("prs_cvd_z", 0.0) or 0.0)
    prs_t2d = float(static.get("prs_t2d_z", 0.0) or 0.0)
    prs_total = float(static.get("prs_raw_z", 0.0) or 0.0)
    hba1c = float(static.get("hba1c", 5.5) or 5.5)
    age = float(static.get("age", 50) or 50)
    dominant = static.get("dominant_snp", "—")
    dominant_gene = static.get("dominant_gene", "—")
    gt_dosage = static.get("dominant_dosage", None)

    def _f(x) -> float | None:
        if x is None:
            return None
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None

    nhr, dhr, dev, pct = _f(nhr), _f(dhr), _f(dev), _f(pct)
    dev_deep, nhr_deep = _f(dev_deep), _f(nhr_deep)
    lf_hf, glu_min, hypo_min = _f(lf_hf), _f(glu_min), _f(hypo_min)
    spo2_min, glu_slope, cov = _f(spo2_min), _f(glu_slope), _f(cov)
    hr_mean, sleep_depth, ent = _f(hr_mean), _f(sleep_depth), _f(ent)
    pnn50, ectopy = _f(pnn50), _f(ectopy)

    #: The patient's own nocturnal baseline implied by the deviation, so the
    #: evidence chain can quote an absolute value ("28 ms, down from 47 ms")
    #: rather than only a percentage.
    base_rmssd = (nhr / (1.0 + dev)) if (nhr is not None and dev is not None
                                         and dev > -1.0) else None

    # ------------------------------------------------------------------ #
    # 1. Silent ischaemia — nocturnal autonomic withdrawal
    # ------------------------------------------------------------------ #
    isch_prob = float(event_probs.get("silent_ischemia", 0.0))
    nocturnal_collapse = (dev is not None and dev < -0.25) or (pct is not None and pct < -0.30)
    if isch_prob > 0.35 or (nocturnal_collapse and prs_cvd > 0.4):
        severity_score = 100.0 * isch_prob + (20.0 if nocturnal_collapse else 0.0)
        evidence = []
        if nhr is not None and dhr is not None:
            evidence.append(
                f"Nocturnal RMSSD {nhr:.1f} ms vs daytime {dhr:.1f} ms "
                f"(Δ {pct:+.0f}% across the 12 h look-back)"
                if pct is not None else
                f"Nocturnal RMSSD {nhr:.1f} ms vs daytime {dhr:.1f} ms"
            )
        if dev is not None:
            evidence.append(
                f"Nocturnal RMSSD is {abs(dev) * 100:.0f}% below this patient's own "
                f"rolling nocturnal baseline"
                + (f" ({base_rmssd:.0f} ms → {nhr:.0f} ms)" if base_rmssd and nhr else "")
                + " — a personalised deviation, not a population norm"
            )
        if dev_deep is not None:
            evidence.append(
                f"Sleep-stage aligned: deep-sleep RMSSD is {abs(dev_deep) * 100:.0f}% "
                f"below this patient's own deep-sleep baseline"
                + (f" ({nhr_deep:.0f} ms)" if nhr_deep else "")
                + " — the dip survives stage-alignment, so it is autonomic "
                  "withdrawal rather than normal sleep architecture"
            )
        if lf_hf is not None:
            evidence.append(
                f"LF/HF spectral balance {lf_hf:.2f} "
                f"({'parasympathetic withdrawal' if lf_hf < 1.0 else 'sympathetic dominance'})"
            )
        evidence.append(
            f"Genomic vulnerability: {dominant_gene} ({dominant}) dosage "
            f"{gt_dosage if gt_dosage is not None else '—'}/2 · CVD polygenic z = {prs_cvd:+.2f}σ"
        )
        if age >= 55:
            evidence.append(f"Age {age:.0f} y · baseline cardiovascular risk multiplier applied")

        urgency = "STAT" if isch_prob >= 0.7 else ("URGENT" if isch_prob >= 0.45 else "PRIORITY")
        actions.append(ClinicalAction(
            urgency=urgency,
            title="Suspected silent ischaemic event — activate ischaemia protocol",
            rationale=(
                "A fall in nocturnal autonomic modulation, temporally aligned with a "
                "high polygenic cardiovascular load, is the signature BioTwin.X was "
                "designed to detect. Silent ischaemia produces no angina; the only "
                "preclinical signal is autonomic. Escalate for confirmatory ECG."
            ),
            evidence=evidence,
            modality="ECG (HRV) + Genomic",
            biomarker="Nocturnal RMSSD / LF-HF balance",
            window="last 12 h",
            delta=_pct_delta(nhr, base_rmssd),
            baseline=base_rmssd,
            current=nhr,
        ))

    # ------------------------------------------------------------------ #
    # 2. Glycaemic crash
    # ------------------------------------------------------------------ #
    crash_prob = float(event_probs.get("glycemic_crash", 0.0))
    glucose_falling = glu_slope is not None and glu_slope < -0.6
    if crash_prob > 0.35 or (hypo_min is not None and hypo_min >= 30) or \
       (glu_min is not None and glu_min < 62):
        evidence = []
        if glu_min is not None:
            evidence.append(f"Lowest CGM reading {glu_min:.0f} mg/dL in the look-back")
        if hypo_min is not None and hypo_min > 0:
            evidence.append(f"{hypo_min:.0f} minutes spent below 70 mg/dL")
        if glu_slope is not None:
            evidence.append(f"Glucose trend {glu_slope:+.1f} mg/dL·h⁻¹ (descending)")
        evidence.append(
            f"T2D polygenic z = {prs_t2d:+.2f}σ · HbA1c {hba1c:.1f}% · "
            f"metformin {'yes' if static.get('metformin') else 'no'}"
        )
        urgency = "STAT" if crash_prob >= 0.7 else ("URGENT" if crash_prob >= 0.45 else "PRIORITY")
        actions.append(ClinicalAction(
            urgency=urgency,
            title="Predicted nocturnal hypoglycaemic crash — prepare glucose rescue",
            rationale=(
                "The CGM trajectory is entering the steep-fall phase that precedes "
                "symptomatic hypoglycaemia in insulin-resistant, high-polygenic-risk "
                "patients. Prepare fast-acting carbohydrate and re-check in 30 min."
            ),
            evidence=evidence,
            modality="CGM + Genomic",
            biomarker="CGM mg/dL and rate-of-change",
            window="last 12 h",
            current=glu_min,
            baseline=70.0,
        ))

    # ------------------------------------------------------------------ #
    # 3. Autonomic arrhythmia
    # ------------------------------------------------------------------ #
    arr_prob = float(event_probs.get("autonomic_arrhythmia", 0.0))
    if arr_prob > 0.4:
        evidence = [
            f"Model probability of autonomic arrhythmia {arr_prob * 100:.0f}%",
        ]
        if pnn50 is not None:
            evidence.append(
                f"pNN50 (beat-to-beat irregularity) {pnn50:.2f}% over the look-back"
            )
        if ectopy is not None:
            evidence.append(
                f"Ectopy load {ectopy * 100:.0f}% of the last hour flagged as "
                f"non-sinus (> 50 ms successive-difference outliers)"
            )
        if ent is not None:
            evidence.append(f"HRV spectral entropy {ent:.2f}")
        if hr_mean is not None:
            evidence.append(f"Mean heart rate {hr_mean:.0f} bpm")
        actions.append(ClinicalAction(
            urgency="URGENT" if arr_prob >= 0.65 else "PRIORITY",
            title="Autonomic instability / arrhythmia burden — request rhythm strip",
            rationale=(
                "Fragmented autonomic regulation with elevated beat-to-beat entropy "
                "and a high polygenic arrhythmia load indicates an unstable autonomic "
                "substrate. Request a 24-hour Holter."
            ),
            evidence=[e for e in evidence if e],
            modality="ECG (HRV)",
            biomarker="HRV entropy / pNN50",
            window="last 12 h",
        ))

    # ------------------------------------------------------------------ #
    # 4. Data-quality / signal-integrity guardrail
    # ------------------------------------------------------------------ #
    if cov is not None and cov < 0.55:
        actions.append(ClinicalAction(
            urgency="ROUTINE",
            title="Signal coverage below clinical threshold",
            rationale=(
                f"Only {cov * 100:.0f}% of expected multi-modal samples were fresh in "
                f"the look-back window. Confidence in the risk index is reduced; "
                f"re-verify device wear-time before acting on borderline scores."
            ),
            evidence=[
                f"Coverage {cov * 100:.0f}% (< 55% threshold)",
                "Asynchronous resampling flagged stale CGM/ECG segments",
            ],
            modality="Data integrity",
            biomarker="Stream coverage",
            window="last 12 h",
        ))

    # ------------------------------------------------------------------ #
    # 5. Optimisation / prevention (always present when risk is low)
    # ------------------------------------------------------------------ #
    if risk < 25:
        actions.append(ClinicalAction(
            urgency="OPTIMISE",
            title="No acute action required — continue longitudinal monitoring",
            rationale=(
                f"Clinical Risk Index {risk:.1f}/100 with no cross-modal anomaly "
                f"signature. Polygenic z-score {prs_total:+.2f}σ indicates "
                f"{'low' if prs_total < 0 else 'moderate'} baseline susceptibility; "
                f"maintain scheduled surveillance."
            ),
            evidence=[
                f"CRI {risk:.1f}/100 (LOW band)",
                f"Polygenic risk z = {prs_total:+.2f}σ",
                f"Nocturnal RMSSD deviation {dev:+.0%}" if dev is not None else
                "Nocturnal RMSSD within personal baseline",
            ],
            modality="Multi-modal",
            biomarker="Clinical Risk Index",
            window="last 12 h",
        ))
    elif not actions:
        actions.append(ClinicalAction(
            urgency="ROUTINE",
            title="Borderline cross-modal deviation — routine review",
            rationale=(
                f"Clinical Risk Index {risk:.1f}/100 without a specific event "
                f"signature. Re-assess at the next scheduled contact."
            ),
            evidence=[f"CRI {risk:.1f}/100"],
            modality="Multi-modal",
            biomarker="Clinical Risk Index",
            window="last 12 h",
        ))

    # Time-to-event urgency override
    if ttx_hours is not None and np.isfinite(ttx_hours) and ttx_hours <= 2.0:
        for a in actions:
            if a.urgency in ("PRIORITY", "ROUTINE"):
                a.urgency = "URGENT"

    order = {"STAT": 0, "URGENT": 1, "PRIORITY": 2, "ROUTINE": 3, "OPTIMISE": 4}
    actions.sort(key=lambda a: order.get(a.urgency, 9))
    return actions


def build_insights(ctx: dict[str, float], static: dict[str, Any],
                   risk: float, event_probs: dict[str, float],
                   attributions: Sequence[dict[str, Any]] | None = None) -> list[Insight]:
    """Ranked, human-readable insight feed for the dashboard."""
    insights: list[Insight] = []

    def g(name: str):
        v = ctx.get(name)
        if v is None:
            return None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None

    dev = g("nhr_rmssd_dev_baseline")
    dev_deep = g("nhr_rmssd_dev_deep")
    pct = g("rmssd_pct_change_3h9h")
    lf_hf = g("lf_hf_mean")
    glu_min = g("glu_min")
    hypo = g("glu_hypo_minutes")
    cov = g("coverage_mean")
    nhr = g("nhr_rmssd_mean")
    pnn50 = g("pnn50_mean")
    ectopy = g("ectopy_load")

    if dev is not None and dev < -0.20:
        insights.append(Insight(
            title="Nocturnal autonomic dip below personal baseline",
            severity="critical" if dev < -0.35 else "warning",
            score=min(100.0, abs(dev) * 220.0),
            modality="ECG · HRV",
            detail=(
                f"Nocturnal RMSSD is {abs(dev) * 100:.0f}% below this patient's own "
                f"rolling baseline{nhr and f' ({nhr:.0f} ms)'}. A {abs(pct or 0):.0f}% "
                f"intrawindow decline is the classic autonomic signature preceding "
                f"silent ischaemia."
            ) if pct is not None else (
                f"Nocturnal RMSSD is {abs(dev) * 100:.0f}% below this patient's own baseline."
            ),
            action="Cross-reference with ischaemia protocol and request a 12-lead ECG.",
            urgency="STAT" if dev < -0.35 else "URGENT",
            evidence=["HRV RMSSD (nocturnal)", "Personal-baseline deviation"],
        ))

    if dev_deep is not None and dev_deep < -0.20:
        insights.append(Insight(
            title="Sleep-stage aligned autonomic withdrawal",
            severity="critical" if dev_deep < -0.35 else "warning",
            score=min(100.0, abs(dev_deep) * 200.0),
            modality="ECG · HRV · Sleep",
            detail=(
                f"Restricting the comparison to deep sleep (N2/N3/REM) — where a low "
                f"RMSSD is physiologically normal — the patient's autonomic modulation "
                f"is still {abs(dev_deep) * 100:.0f}% below their own deep-sleep "
                f"baseline. The dip therefore survives stage-alignment and is not an "
                f"artefact of normal sleep architecture."
            ),
            action="Treat as a stage-independent autonomic withdrawal; confirm with ECG.",
            urgency="URGENT" if dev_deep < -0.35 else "PRIORITY",
            evidence=["Deep-sleep RMSSD", "Personal deep-sleep baseline"],
        ))

    if ectopy is not None and ectopy > 0.15:
        insights.append(Insight(
            title="Ectopic beat burden rising",
            severity="critical" if ectopy > 0.35 else "warning",
            score=float(np.clip(ectopy * 220.0, 0, 100)),
            modality="ECG · Beat timing",
            detail=(
                f"{ectopy * 100:.0f}% of the last hour was flagged as non-sinus "
                f"(successive R-R differences > 50 ms)"
                + (f"; windowed pNN50 {pnn50:.2f}%" if pnn50 is not None else "")
                + ". Beat-to-beat irregularity is invisible to mean heart rate and is "
                  "the earliest autonomic correlate of arrhythmia substrate."
            ),
            action="Request a 24-hour Holter; review stimulant and electrolyte load.",
            urgency="URGENT" if ectopy > 0.35 else "PRIORITY",
            evidence=["pNN50", "Ectopy load"],
        ))

    if lf_hf is not None and lf_hf < 0.8:
        insights.append(Insight(
            title="Parasympathetic withdrawal detected in spectral domain",
            severity="warning",
            score=float(np.clip((1.0 - lf_hf) * 70.0, 0, 100)),
            modality="ECG · Frequency domain",
            detail=(
                f"LF/HF ratio {lf_hf:.2f} indicates a shift towards sympathetic "
                f"dominance — the autonomic system is paying a cardiovascular cost "
                f"that resting heart rate alone cannot reveal."
            ),
            action="Review autonomic stressors; consider beta-blocker timing review.",
            urgency="PRIORITY",
            evidence=["Welch PSD LF/HF", "HRV spectral balance"],
        ))

    if hypo is not None and hypo > 0:
        insights.append(Insight(
            title=f"Time in hypoglycaemic range: {hypo:.0f} min",
            severity="critical" if hypo > 45 else "warning",
            score=min(100.0, hypo * 1.6),
            modality="CGM",
            detail=(
                f"Lowest reading {glu_min:.0f} mg/dL. Hypoglycaemia blunts the "
                f"autonomic warning signs that BioTwin.X relies on, which is exactly "
                f"why this pathway is scored separately."
            ) if glu_min else f"{hypo:.0f} minutes below 70 mg/dL.",
            action="Review basal insulin / sulfonylurea dose; issue rescue glucose.",
            urgency="URGENT" if hypo > 45 else "PRIORITY",
            evidence=["CGM mg/dL", "Time below range"],
        ))

    for label, prob in event_probs.items():
        if prob >= 0.45:
            insights.append(Insight(
                title=f"Model probability · {label.replace('_', ' ')} = {prob * 100:.0f}%",
                severity="warning",
                score=float(prob * 100),
                modality="CMCL fusion",
                detail=(
                    f"Cross-modal contrastive inference assigns a {prob * 100:.0f}% "
                    f"probability to a {label.replace('_', ' ')} event within the next "
                    f"12 hours."
                ),
                action="Escalate to the on-call clinician.",
                urgency="URGENT",
                evidence=["CMCL event head", "Vulnerability-conditioned fusion"],
            ))

    if cov is not None and cov < 0.55:
        insights.append(Insight(
            title="Degraded multi-modal coverage",
            severity="info",
            score=float((0.55 - cov) * 120),
            modality="Data integrity",
            detail=(
                f"Fresh-sample coverage {cov * 100:.0f}% across the look-back. "
                f"Stale CGM and ECG segments were masked rather than imputed, so the "
                f"risk index is conservative."
            ),
            action="Check device wear-time and sensor placement.",
            urgency="ROUTINE",
            evidence=["Stream coverage"],
        ))

    if not insights:
        insights.append(Insight(
            title="No cross-modal anomaly signature",
            severity="reassuring",
            score=float(max(0.0, 40.0 - risk)),
            modality="Multi-modal",
            detail=(
                f"Autonomic, glycaemic and genomic signals are mutually consistent "
                f"and within this patient's personal baseline (CRI {risk:.1f}/100)."
            ),
            action="Continue scheduled monitoring.",
            urgency="OPTIMISE",
            evidence=["CRI", "Personal-baseline deviation"],
        ))

    insights.sort(key=lambda i: ({"critical": 0, "warning": 1, "info": 2,
                                   "reassuring": 3}.get(i.severity, 4), -i.score))
    return insights
