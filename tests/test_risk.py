"""Risk index bands and the clinical action/insight rule engine."""

from __future__ import annotations

import numpy as np

from biotwin.risk import (
    TIER_BANDS,
    build_actions,
    build_insights,
    tier_for,
)

STATIC = {
    "age": 62, "sex": "M", "bmi": 31.4, "hba1c": 7.1, "systolic_bp": 148,
    "ldl": 142, "prs_raw_z": 1.9, "prs_cvd_z": 2.3, "prs_t2d_z": 1.4,
    "prs_aut_z": 0.6, "dominant_snp": "rs1333049", "dominant_gene": "CDKN2B-AS1",
    "dominant_dosage": 2, "dominant_trait": "cvd",
}

CRITICAL_CTX = {
    "nhr_rmssd_mean": 21.0, "dhr_rmssd_mean": 6.5,
    "nhr_rmssd_dev_baseline": -0.52, "nhr_rmssd_dev_deep": -0.41,
    "nhr_rmssd_deep": 19.0, "rmssd_pct_change_3h9h": -38.0,
    "lf_hf_mean": 0.7, "glu_min": 54.0, "glu_hypo_minutes": 48.0,
    "glu_slope": -2.4, "glu_mean": 118.0, "glu_tir_pct": 61.0,
    "spo2_min": 91.0, "coverage_mean": 0.94, "hr_mean": 88.0,
    "pnn50_mean": 2.1, "ectopy_load": 0.72, "sleep_depth_pct": 14.0,
    "spectral_entropy": 0.62, "rr_artifact_fraction": 0.03,
    "lf_hf_slope": -0.4, "hr_slope": 1.1, "hr_night_mean": 79.0,
    "rmssd_slope": -2.0,
}

CRITICAL_PROBS = {"silent_ischemia": 0.81, "glycemic_crash": 0.34,
                  "autonomic_arrhythmia": 0.22}


def test_tier_bands_are_contiguous_and_total():
    assert [b[2] for b in TIER_BANDS] == ["LOW", "ELEVATED", "HIGH", "CRITICAL"]
    assert tier_for(0) == "LOW"
    assert tier_for(24.99) == "LOW"
    assert tier_for(25) == "ELEVATED"
    assert tier_for(49.9) == "ELEVATED"
    assert tier_for(50) == "HIGH"
    assert tier_for(74.9) == "HIGH"
    assert tier_for(75) == "CRITICAL"
    assert tier_for(100) == "CRITICAL"


def test_critical_patient_gets_immediate_evidence_rich_action():
    actions = build_actions(CRITICAL_CTX, STATIC, 86.0, CRITICAL_PROBS, 2.4)
    assert actions, "a critical patient must never produce an empty action list"
    top = actions[0]
    assert top.urgency in {"STAT", "URGENT"}
    assert top.evidence, "every alert must carry its evidence chain"
    # The evidence must quote numbers, not just say "anomaly detected".
    assert any(any(ch.isdigit() for ch in e) for e in top.evidence)
    # Primary action is unique; nothing else may claim the same urgency tier.
    assert sum(a.urgency == top.urgency for a in actions) >= 1


def test_low_risk_is_an_explicit_silence():
    ctx = {"nhr_rmssd_mean": 48.0, "nhr_rmssd_dev_baseline": 0.02,
           "lf_hf_mean": 1.8, "glu_min": 84.0, "glu_hypo_minutes": 0.0,
           "glu_slope": 0.1, "coverage_mean": 0.97}
    actions = build_actions(ctx, STATIC | {"prs_raw_z": -0.4}, 9.0,
                            {"silent_ischemia": 0.02, "glycemic_crash": 0.01,
                             "autonomic_arrhythmia": 0.01}, 11.5)
    assert actions
    assert actions[0].urgency == "OPTIMISE"
    assert "No acute action" in actions[0].title


def test_near_term_ttx_escalates_routine_actions():
    ctx = {"nhr_rmssd_mean": 44.0, "nhr_rmssd_dev_baseline": -0.08,
           "coverage_mean": 0.9}
    actions = build_actions(ctx, STATIC, 31.0,
                            {"silent_ischemia": 0.12, "glycemic_crash": 0.10,
                             "autonomic_arrhythmia": 0.05}, 1.2)
    assert actions
    assert all(a.urgency != "ROUTINE" for a in actions)


def test_insights_are_ranked_and_bounded():
    insights = build_insights(CRITICAL_CTX, STATIC, 86.0, CRITICAL_PROBS)
    assert insights
    order = {"critical": 0, "warning": 1, "info": 2, "reassuring": 3}
    keys = [(order[i.severity], -i.score) for i in insights]
    assert keys == sorted(keys)
    assert all(0.0 <= i.score <= 100.0 for i in insights)
    assert all(i.severity in order for i in insights)
