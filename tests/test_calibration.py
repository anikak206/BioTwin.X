"""Calibration: monotonicity, ECE, Brier, and the numerical guard-rails."""

from __future__ import annotations

import numpy as np

from biotwin.calibration import (
    TemperatureScaler,
    brier_score,
    expected_calibration_error,
)


def test_transform_is_monotone_and_in_unit_interval():
    cal = TemperatureScaler(temperature=1.7, bias=-0.3)
    z = np.linspace(-6, 6, 101)
    p = cal.transform(z)
    assert np.all(np.diff(p) > 0)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_fit_reduces_nll_on_overconfident_logits():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, size=2000)
    # 6-sigma logits with heavy label noise: the model is confidently wrong
    # on a sizeable minority, so a temperature > 1 must improve the NLL.
    z = 6.0 * (2.0 * y - 1.0) + rng.normal(0, 6.0, size=2000)
    cal = TemperatureScaler().fit(z, y)
    assert cal.temperature > 1.0, "temperature must inflate to soften overconfidence"

    def nll(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    before = nll(1.0 / (1.0 + np.exp(-z)))
    after = nll(cal.transform(z))
    assert after <= before + 1e-6


def test_fit_is_a_noop_when_targets_are_degenerate():
    cal = TemperatureScaler().fit(np.zeros(10), np.zeros(10))
    assert cal.temperature == 1.0 and cal.bias == 0.0


def test_roundtrip_serialisation():
    cal = TemperatureScaler(temperature=2.5, bias=0.4)
    again = TemperatureScaler.from_dict(cal.to_dict())
    assert again == cal


def test_ece_and_brier_zero_for_perfect_predictions():
    p = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    assert expected_calibration_error(p, y) < 1e-9
    assert brier_score(p, y) < 1e-9


def test_ece_detects_miscalibration():
    # Confident 0.99 predictions that are right half the time.
    p = np.full(200, 0.99)
    y = np.array([1.0, 0.0] * 100)
    assert expected_calibration_error(p, y) > 0.45
    assert brier_score(p, y) > 0.2
