"""Feature engineering: spectral estimation, artefact handling, window assembly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biotwin.config import PROFILES
from biotwin.features import (
    TEMPORAL_CHANNELS,
    WINDOW_CONTEXT_FEATURES,
    build_windows,
    welch_psd,
    window_tensor,
)


def test_feature_schema_is_stable():
    assert len(WINDOW_CONTEXT_FEATURES) == 24
    assert "nhr_rmssd_dev_deep" in WINDOW_CONTEXT_FEATURES


def test_welch_psd_returns_per_epoch_spectra_with_correct_peak():
    fs = 1.0
    t = np.arange(0, 600, 1.0 / fs)
    # 0.25 Hz sits squarely inside the HF band (0.15-0.40 Hz).
    x = np.sin(2 * np.pi * 0.25 * t)
    freqs, pxx = welch_psd(x, fs=fs, nperseg=300)
    # nperseg=300, 50 % overlap, 600 samples -> 3 epochs.
    assert pxx.ndim == 2 and pxx.shape[0] == 3, "three 50%-overlapping epochs"
    assert pxx.shape[1] == len(freqs)
    peak = freqs[np.argmax(pxx, axis=1)]
    assert np.allclose(peak, 0.25, atol=0.01)


def test_welch_psd_handles_short_input():
    freqs, pxx = welch_psd(np.zeros(10), nperseg=300)
    assert len(freqs) == 0
    assert pxx.shape[0] == 0


def test_temporal_schema_is_stable():
    assert len(TEMPORAL_CHANNELS) == 12
    assert len(set(TEMPORAL_CHANNELS)) == 12
    # The asynchrony channels are what make the grid honest about missingness.
    assert "glu_staleness" in TEMPORAL_CHANNELS
    assert "rr_staleness" in TEMPORAL_CHANNELS
    assert "coverage" in TEMPORAL_CHANNELS


def test_welch_psd_one_sided_power_is_physical():
    rng = np.random.default_rng(0)
    x = rng.normal(size=1200)
    freqs, pxx = welch_psd(x, fs=1.0, nperseg=300)
    # White noise has a flat one-sided PSD; integrating 0..Nyquist recovers
    # the signal variance (~1.0 for a unit-variance series).
    assert 0.7 < np.trapezoid(pxx.mean(axis=0), freqs) < 1.3


def test_build_windows_labels_only_the_forward_horizon():
    prof = PROFILES["smoke"]
    n = prof.n_grid_bins
    ts = pd.date_range("2026-01-01", periods=n, freq="5min")
    grid = pd.DataFrame({c: 0.0 for c in TEMPORAL_CHANNELS} | {"ts": ts})
    event_ts = ts[n // 2]
    wins = build_windows(grid, "P1", [("silent_ischemia", event_ts)], prof,
                         stride_bins=12)
    assert wins
    for w in wins:
        # A window ending after the event must not be labelled positive.
        if ts[w.end_idx - 1] > event_ts:
            assert w.label == 0


def test_window_tensor_shape_and_finiteness():
    prof = PROFILES["smoke"]
    n = prof.n_grid_bins
    ts = pd.date_range("2026-01-01", periods=n, freq="5min")
    rng = np.random.default_rng(1)
    grid = pd.DataFrame(
        {c: rng.normal(size=n) for c in TEMPORAL_CHANNELS} | {"ts": ts})
    wins = build_windows(grid, "P1", [], prof, stride_bins=24)
    tensor = window_tensor(grid, wins[:3])
    assert tensor.shape[1] == len(TEMPORAL_CHANNELS)
    assert np.isfinite(tensor).all()
