"""End-to-end smoke test for the clinical inference engine.

This test generates the tiny ``smoke`` cohort (6 patients x 2 days) on first
run, so it is marked ``slow`` and skipped unless ``BIOTWIN_SLOW=1``. The rest
of the suite runs in a few seconds.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("BIOTWIN_SLOW") != "1",
    reason="set BIOTWIN_SLOW=1 to run cohort-level integration tests",
)


@pytest.fixture(scope="module")
def engine():
    from biotwin.inference import BioTwinEngine

    return BioTwinEngine(profile="smoke", allow_surrogate=True, verbose=False)


def test_engine_selects_a_backend(engine):
    assert engine.backend in {"cmcl", "surrogate", "rule-fallback"}
    assert engine.bundle.patient_ids()


def test_score_patient_is_complete_and_deterministic(engine):
    pid = engine.bundle.patient_ids()[0]
    a = engine.score_patient(pid)
    b = engine.score_patient(pid)
    assert 0.0 <= a.risk_index <= 100.0
    assert a.tier in {"LOW", "ELEVATED", "HIGH", "CRITICAL"}
    assert a.actions, "every report must answer 'what next?'"
    assert a.explanation is not None and a.explanation.attributions
    assert a.risk_index == pytest.approx(b.risk_index)
    assert set(a.event_probs) == {"silent_ischemia", "glycemic_crash",
                                  "autonomic_arrhythmia"}
    assert 0.0 <= a.confidence <= 1.0


def test_report_keeps_baseline_and_current_distinct(engine):
    pid = engine.bundle.patient_ids()[0]
    rep = engine.score_patient(pid)
    if rep.baseline_rmssd is not None and rep.nocturnal_rmssd is not None:
        dev = rep.context.get("nhr_rmssd_dev_baseline", 0.0)
        assert rep.baseline_rmssd == pytest.approx(
            rep.nocturnal_rmssd / (1 + dev), rel=1e-4)


def test_score_cohort_is_sorted_and_typed(engine):
    board = engine.score_cohort()
    assert isinstance(board, pd.DataFrame)
    assert len(board) == len(engine.bundle.patient_ids())
    assert board["risk_index"].is_monotonic_decreasing
    assert board["risk_index"].between(0, 100).all()
    assert np.isfinite(board["p_silent_ischemia"]).all()
    assert board["latency_ms"].max() > 0


def test_risk_series_is_ordered_and_bounded(engine):
    pid = engine.bundle.patient_ids()[0]
    series = engine.risk_series(pid, hours=24, step_minutes=60)
    assert len(series) >= 2
    assert series["ts"].is_monotonic_increasing
    assert series["risk_index"].between(0, 100).all()
    probs = series[["p_silent_ischemia", "p_glycemic_crash",
                    "p_autonomic_arrhythmia"]].to_numpy()
    assert np.isfinite(probs).all()
    assert ((probs >= 0) & (probs <= 1)).all()


def test_json_serialisation(engine):
    import json

    pid = engine.bundle.patient_ids()[0]
    payload = engine.score_patient(pid).to_dict()
    text = json.dumps(payload, default=str)
    assert "risk_index" in text and "actions" in text


def test_latency_benchmark_contract(engine):
    stats = engine.benchmark_latency(n=5, warmup=1)
    assert {"p50_ms", "p95_ms", "slo_ms", "slo_met"} <= set(stats)
    assert stats["p50_ms"] > 0
    assert stats["p95_ms"] >= stats["p50_ms"]
