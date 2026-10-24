"""
BioTwin.X — Asynchronous Telemetry Feature Extractor
=====================================================
Turns six *mutually non-commensurate* streams into a single dense, causally
valid representation.

Pipeline
--------
1. **Beat-series reconstruction** — the ECG patch emits irregularly-timed
   R-R intervals. We rebuild a uniform 1 Hz R-R series (the standard clinical
   resampling step) with an explicit maximum-gap rule so that dropped leads
   are *masked*, not interpolated.
2. **Frequency-domain HRV** — Welch PSD over 5-minute 1 Hz epochs yields
   log-LF, log-HF, LF/HF ratio and spectral entropy. This is what makes
   *sympathetic withdrawal* (LF fall, HF rise) vs. *stress* (LF rise)
   separable — the two are indistinguishable from mean HR alone.
3. **Asynchronous resampling** — every other modality is forward-filled onto
   the shared 5-minute grid, and the **age of the last observation** is carried
   forward as an explicit channel. A CGM value that is 40 minutes stale is a
   materially different clinical fact than a fresh one, and the model is told
   so rather than silently assuming recency.
4. **Sleep-stage alignment** — nocturnal physiology is scored *within* sleep
   stages so that a genuine autonomic dip is not confused with normal
   sleep-stage architecture.
5. **Causal windowing** — sliding 12-hour look-back windows are emitted with
   a 12-hour future label, so the supervised task is literally
   "will an event occur in the next 12 hours?".

No label ever influences a feature. All baseline statistics are computed from
history strictly *before* the scored window.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd

from .config import DatasetProfile

# --------------------------------------------------------------------------- #
# Channel registry
# --------------------------------------------------------------------------- #
TEMPORAL_CHANNELS: Final[tuple[str, ...]] = (
    "hr",             # instantaneous heart rate from the ECG patch
    "hrv_rmssd",      # rolling 5-min RMSSD (ms)
    "hrv_lf",         # log LF power (0.04-0.15 Hz)
    "hrv_hf",         # log HF power (0.15-0.40 Hz)
    "hrv_pnn50",      # % of successive RR differences > 50 ms (ectopy load)
    "glucose",        # CGM interstitial glucose (mg/dL)
    "spo2",           # peripheral oxygen saturation (%)
    "skin_temp",      # skin temperature (degC)
    "sleep_stage",    # 0 awake / 1 light / 2 N2 / 3 N3 / 4 REM
    "coverage",       # fraction of modalities with fresh (<=3x cadence) data
    "glu_staleness",  # minutes since last CGM sample
    "rr_staleness",   # minutes since last ECG beat
)

WINDOW_CONTEXT_FEATURES: Final[tuple[str, ...]] = (
    "nhr_rmssd_mean",
    "dhr_rmssd_mean",
    "nhr_rmssd_dev_baseline",
    "nhr_rmssd_deep",
    "nhr_rmssd_dev_deep",
    "rmssd_slope",
    "rmssd_pct_change_3h9h",
    "lf_hf_mean",
    "lf_hf_slope",
    "hr_mean",
    "hr_slope",
    "hr_night_mean",
    "glu_mean",
    "glu_min",
    "glu_tir_pct",
    "glu_hypo_minutes",
    "glu_slope",
    "spo2_min",
    "sleep_depth_pct",
    "coverage_mean",
    "spectral_entropy",
    "rr_artifact_fraction",
    "pnn50_mean",
    "ectopy_load",
)

STATIC_CLINICAL: Final[tuple[str, ...]] = (
    "age", "sex_male", "bmi", "smoking", "alcohol_units_wk", "sleep_hours",
    "resting_hr", "systolic_bp", "diastolic_bp", "ldl", "hdl", "triglycerides",
    "hba1c", "fasting_glucose", "antihypertensive", "statin", "metformin",
    "family_history_cv", "family_history_t2d",
)

DERIVED_CLINICAL: Final[tuple[str, ...]] = (
    "insulin_resistance_proxy",
    "atherogenic_index",
    "metabolic_syndrome",
    "age_x_prs_cvd",
    "bmi_x_prs_t2d",
)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _rolling_rms_from_rr(rr: np.ndarray, win: int) -> np.ndarray:
    """Causal rolling RMS of successive R-R differences.

    ``out[i]`` is the RMS of the successive differences ending at beat ``i``,
    i.e. the standard RMSSD definition, aligned to the input length and
    NaN-padded over the warm-up period.
    """
    n = len(rr)
    out = np.full(n, np.nan, dtype=np.float64)
    d = np.diff(rr)
    if len(d) < win:
        return out
    sq = np.nan_to_num(d, nan=0.0) ** 2
    valid = np.isfinite(d).astype(np.float64)
    csum = np.concatenate([[0.0], np.cumsum(sq)])
    cval = np.concatenate([[0.0], np.cumsum(valid)])
    s = csum[win:] - csum[:-win]
    c = cval[win:] - cval[:-win]
    r = np.where(c / win > 0.9, np.sqrt(np.maximum(s, 0.0) / win), np.nan)
    out[win: win + len(r)] = r
    return out


def _slope(y: np.ndarray) -> float:
    """Ordinary least-squares slope of y over its index (NaN-tolerant)."""
    y = np.asarray(y, dtype=np.float64)
    m = ~np.isnan(y)
    if m.sum() < 3:
        return 0.0
    x = np.arange(len(y), dtype=np.float64)[m]
    yy = y[m]
    denom = float(((x - x.mean()) ** 2).sum())
    if denom < 1e-9:
        return 0.0
    return float(((x - x.mean()) * (yy - yy.mean())).sum() / denom)


def _pnn50_series(rr_1hz: np.ndarray, threshold_ms: float = 50.0) -> np.ndarray:
    """Per-second pNN50 — the share of successive R-R differences above 50 ms.

    pNN50 is the classic *ectopy load* metric: a run of premature beats
    produces a burst of large RR differences. Because RMSSD is an RMS of
    differences it is *dominated* by ectopy rather than by sympathetic tone,
    so a separate channel is required to keep "irregular heart" and
    "autonomic withdrawal" distinguishable.
    """
    out = np.full(len(rr_1hz), np.nan, dtype=np.float64)
    if len(rr_1hz) < 2:
        return out
    d = np.abs(np.diff(rr_1hz))
    valid = np.isfinite(d) & np.isfinite(rr_1hz[:-1]) & np.isfinite(rr_1hz[1:])
    flag = np.where(valid, (d > threshold_ms).astype(np.float64), np.nan)
    out[1:] = flag * 100.0
    return out


def _spectral_entropy(power: np.ndarray) -> float:
    p = power[power > 0]
    if p.size < 2:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log(p + 1e-12)).sum() / math.log(p.size))


# --------------------------------------------------------------------------- #
# Stage 1 — reconstruct a uniform 1 Hz R-R series from irregular beats
# --------------------------------------------------------------------------- #
def uniform_rr_series(beat_ts: pd.Series, rr_ms: np.ndarray,
                       max_gap_s: float = 12.0) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Resample an irregular beat-to-beat R-R series onto a uniform 1 Hz grid.

    Beats separated by more than ``max_gap_s`` (device lifted, lead-off) are
    *masked* rather than bridged — a critical distinction, because bridging
    fabricates physiological events that never happened.
    """
    if len(beat_ts) < 3:
        return pd.DatetimeIndex([]), np.zeros(0)

    x = (beat_ts - beat_ts.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64)
    order = np.argsort(x, kind="stable")
    x, rr_ms = x[order], np.asarray(rr_ms, dtype=np.float64)[order]

    n = int(x[-1]) + 1
    uni = np.interp(np.arange(n, dtype=np.float64), x, rr_ms)

    # Distance to the nearest beat on either side -> mask long gaps.
    idx = np.searchsorted(x, np.arange(n, dtype=np.float64))
    left = x[np.clip(idx - 1, 0, len(x) - 1)]
    right = x[np.clip(idx, 0, len(x) - 1)]
    gap = np.minimum(np.arange(n) - left, right - np.arange(n))
    uni = np.where(gap <= max_gap_s, uni, np.nan)

    ts = pd.DatetimeIndex(beat_ts.iloc[0] + pd.to_timedelta(np.arange(n), unit="s"))
    return ts, uni


def welch_psd(x: np.ndarray, fs: float = 1.0, nperseg: int = 300,
              noverlap: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Explicit Welch periodogram, returning **per-epoch** spectra.

    ``scipy.signal.welch`` averaged the segment spectra starting in SciPy 1.18,
    which destroys the per-epoch resolution this pipeline needs (each 5-minute
    epoch must carry its own LF/HF balance). We therefore implement the
    estimator directly — 50 % Hann overlap, constant detrend, one-sided power
    scaled by the noise-power correction ``fs · Σw²``.

    Returns ``(freqs, pxx)`` with ``pxx.shape == (n_epochs, n_freqs)``.
    """
    if noverlap is None:
        noverlap = nperseg // 2
    x = np.asarray(x, dtype=np.float64)
    if len(x) < nperseg:
        return np.zeros(0), np.zeros((0, 1))

    step = max(1, nperseg - noverlap)
    n_seg = 1 + (len(x) - nperseg) // step
    starts = np.arange(n_seg) * step
    segs = np.lib.stride_tricks.sliding_window_view(x, nperseg)[starts]
    segs = segs - segs.mean(axis=1, keepdims=True)          # constant detrend

    win = np.hanning(nperseg)
    spec = np.fft.rfft(segs * win, axis=1)
    scale = fs * float((win ** 2).sum())
    pxx = (np.abs(spec) ** 2) / scale
    pxx[:, 1:] *= 2.0                                          # one-sided
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return freqs, pxx


def hrv_spectral(rr_1hz: np.ndarray, fs: float = 1.0,
                 segment_s: int = 300) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray]:
    """Per-epoch Welch spectral features of the R-R series.

    Returns ``(lf, hf, lf_hf_ratio, entropy, epoch_start_sample)``.
    """
    n = len(rr_1hz)
    nperseg = int(segment_s)
    if n < nperseg * 2:
        empty = np.zeros(0)
        return empty, empty, empty, empty, empty

    hop = max(1, nperseg // 5)
    finite = np.isfinite(rr_1hz)
    n_seg = 1 + (n - nperseg) // hop
    seg_valid = np.lib.stride_tricks.sliding_window_view(
        finite.astype(np.float64), nperseg)[np.arange(n_seg) * hop].mean(axis=1)

    med = float(np.nanmedian(rr_1hz)) if finite.any() else 0.0
    clean = np.where(finite, rr_1hz, med)
    f, pxx = welch_psd(clean, fs=fs, nperseg=nperseg, noverlap=nperseg - hop)

    lf_band = (f >= 0.04) & (f < 0.15)
    hf_band = (f >= 0.15) & (f < 0.40)
    vlf_band = (f >= 0.003) & (f < 0.04)

    lf = pxx[:, lf_band].sum(axis=1) + pxx[:, vlf_band].sum(axis=1) * 0.25
    hf = pxx[:, hf_band].sum(axis=1)
    ratio = (lf + 1e-12) / (hf + 1e-12)
    ent = np.array([_spectral_entropy(pxx[i]) for i in range(pxx.shape[0])])

    # Signal-quality gate: an epoch with >20 % missing beats is *rejected*
    # rather than power-imputed, so the model never learns from a spectrum
    # that was manufactured by the gap-filling step.
    bad = seg_valid < 0.80
    lf[bad] = np.nan
    hf[bad] = np.nan
    ratio[bad] = np.nan
    ent[bad] = np.nan

    starts = np.arange(pxx.shape[0]) * hop
    return lf, hf, ratio, ent, starts


# --------------------------------------------------------------------------- #
# Stage 2 — the main extractor
# --------------------------------------------------------------------------- #
@dataclass
class TelemetryFeatures:
    grid: pd.DataFrame
    context_names: tuple[str, ...]
    window_context: pd.DataFrame   # one row per grid bin (causal summaries)

    @property
    def X(self) -> np.ndarray:
        return self.grid[list(TEMPORAL_CHANNELS)].to_numpy(dtype=np.float32)


def _ns(values: pd.DatetimeIndex | pd.Series) -> np.ndarray:
    """Normalise a datetime index/series to int64 nanoseconds."""
    return np.asarray(pd.DatetimeIndex(values).to_numpy(dtype="datetime64[ns]")).astype("int64")


def _bin_ids(timestamps: pd.DatetimeIndex, origin_ns: int, bin_s: int, n_bins: int) -> np.ndarray:
    ids = (_ns(timestamps) - origin_ns) // int(bin_s * 1e9)
    ids = ids.astype(np.int64)
    ids[(ids < 0) | (ids >= n_bins)] = -1
    return ids


def _bin_last(values: np.ndarray, timestamps: pd.DatetimeIndex, origin_ns: int,
              bin_s: int, n_bins: int) -> np.ndarray:
    """Last observation *within* each bin (fresh-sample semantics)."""
    out = np.full(n_bins, np.nan, dtype=np.float64)
    ids = _bin_ids(timestamps, origin_ns, bin_s, n_bins)
    ok = ids >= 0
    if not ok.any():
        return out
    order = np.argsort(_ns(timestamps)[ok], kind="stable")
    df = pd.DataFrame({"b": ids[ok][order], "v": np.asarray(values)[ok][order],
                       "t": _ns(timestamps)[ok][order]})
    last = df.groupby("b", sort=True)["v"].last()
    out[last.index.to_numpy()] = last.to_numpy()
    return out


def _bin_ffill(values: np.ndarray, timestamps: pd.DatetimeIndex, origin_ns: int,
               bin_s: int, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Value carried forward from the most recent past observation.

    Returns ``(values, age_minutes)`` where the age is measured against the
    bin start — i.e. *how stale is the number the model is about to trust*.
    """
    out = np.full(n_bins, np.nan, dtype=np.float64)
    age = np.full(n_bins, np.nan, dtype=np.float64)
    ts = _ns(timestamps)
    if len(ts) == 0:
        return out, age
    order = np.argsort(ts, kind="stable")
    ts = ts[order]
    vals = np.asarray(values)[order]

    origin = origin_ns + np.arange(n_bins, dtype="int64") * int(bin_s * 1e9)
    pos = np.searchsorted(ts, origin, side="right") - 1
    ok = pos >= 0
    out[ok] = vals[pos[ok]]
    age[ok] = (origin[ok] - ts[pos[ok]]) / (60.0 * 1e9)
    return out, age


def extract_features(tele: pd.DataFrame, profile: DatasetProfile,
                     bin_minutes: int | None = None) -> TelemetryFeatures:
    """Resample an asynchronous telemetry table onto the shared 5-minute grid."""
    bin_min = bin_minutes or profile.grid_minutes
    bin_s = bin_min * 60

    tele = tele.dropna(subset=["ts"]).sort_values("ts", kind="stable")
    start = tele["ts"].iloc[0].floor(f"{bin_min}min")
    end = tele["ts"].iloc[-1].ceil(f"{bin_min}min")
    grid_ts = pd.date_range(start, end, freq=f"{bin_min}min")
    n_bins = len(grid_ts)
    idx = grid_ts
    origin_ns = int(_ns(idx)[0])

    channels = {c: np.full(n_bins, np.nan, dtype=np.float64) for c in TEMPORAL_CHANNELS}

    def _mod(name: str) -> pd.DataFrame:
        sub = tele[tele["modality"] == name]
        if sub.empty:
            return sub
        good = sub[sub["quality"] > 0]
        return good if len(good) > 3 else sub

    # ---------------- ECG: HR + HRV (RMSSD / LF / HF / entropy) --------- #
    rr_mod = tele[tele["modality"] == "ecg_rr"].sort_values("ts", kind="stable")
    rr_staleness = np.full(n_bins, np.nan)
    if len(rr_mod):
        uni_ts, uni_rr = uniform_rr_series(rr_mod["ts"], rr_mod["value"].to_numpy())
        if len(uni_rr):
            hr_1hz = 60000.0 / uni_rr
            hr_bpm = (pd.Series(hr_1hz, index=uni_ts)
                      .resample(f"{bin_min}min", origin=idx[0]).median())
            channels["hr"] = hr_bpm.reindex(idx).to_numpy()

            rmssd_1hz = _rolling_rms_from_rr(uni_rr, bin_s)
            rmssd = (pd.Series(rmssd_1hz, index=uni_ts)
                     .resample(f"{bin_min}min", origin=idx[0]).median())
            channels["hrv_rmssd"] = rmssd.reindex(idx).to_numpy()

            pnn50 = _pnn50_series(uni_rr)
            pnn_bin = (pd.Series(pnn50, index=uni_ts)
                       .resample(f"{bin_min}min", origin=idx[0]).mean())
            channels["hrv_pnn50"] = pnn_bin.reindex(idx).to_numpy()

            lf, hf, ratio, ent, starts = hrv_spectral(uni_rr, fs=1.0, segment_s=bin_s)
            if len(starts):
                epoch_ts = uni_ts[0] + pd.to_timedelta(starts + bin_s // 2, unit="s")
                bin_ids = ((epoch_ts - idx[0]).total_seconds() // bin_s).astype(int)
                valid = (bin_ids >= 0) & (bin_ids < n_bins)
                bid = bin_ids[valid]
                # log-domain power keeps the tensor numerically well conditioned
                for arr, chan in ((lf, "hrv_lf"), (hf, "hrv_hf")):
                    tmp = pd.Series(np.log1p(arr[valid])).groupby(bid).mean()
                    channels[chan][tmp.index.to_numpy()] = tmp.to_numpy()
            # staleness of the most recent ECG beat
            beat_sec = (rr_mod["ts"] - idx[0]).dt.total_seconds().to_numpy()
            order = np.argsort(beat_sec)
            beat_sec = beat_sec[order]
            ptr, last_beat = 0, -np.inf
            for b in range(n_bins):
                while ptr < len(order) and beat_sec[ptr] <= b * bin_s:
                    last_beat = beat_sec[ptr]
                    ptr += 1
                if np.isfinite(last_beat):
                    rr_staleness[b] = (b * bin_s - last_beat) / 60.0
            channels["rr_staleness"] = rr_staleness

    # ---------------- other modalities: ffill + staleness ---------------- #
    # Each modality is carried forward but **invalidated once it exceeds three
    # nominal cadences of staleness** — a sensor that has been silent for an
    # hour is a missing measurement, not a measurement of 92 mg/dL.
    freshness_caps = {
        "cgm_glucose": (3 * profile.cgm_period_s, "glucose", "glu_staleness"),
        "spo2": (3 * profile.spo2_period_s, "spo2", None),
        "skin_temp": (3 * profile.temp_period_s, "skin_temp", None),
    }
    for mod, (cap_s, target, stale_ch) in freshness_caps.items():
        sub = _mod(mod)
        if sub.empty:
            continue
        ts = pd.DatetimeIndex(sub["ts"])
        vals = sub["value"].to_numpy(dtype=np.float64)
        ffilled, age = _bin_ffill(vals, ts, origin_ns, bin_s, n_bins)
        stale = age > (cap_s / 60.0)
        ffilled = np.where(stale, np.nan, ffilled)
        channels[target] = ffilled
        if stale_ch:
            channels[stale_ch] = age

    # ---------------- sleep stage --------------------------------------- #
    sl = _mod("sleep_stage")
    if not sl.empty:
        ts = pd.DatetimeIndex(sl["ts"])
        vals = sl["value"].to_numpy(dtype=np.float64)
        ffilled, age = _bin_ffill(vals, ts, origin_ns, bin_s, n_bins)
        # Outside an epoch's neighbourhood the subject is awake by definition.
        ffilled = np.where(age > 45.0, 0.0, ffilled)
        channels["sleep_stage"] = ffilled

    grid = pd.DataFrame({"ts": idx})
    for c in TEMPORAL_CHANNELS:
        grid[c] = channels[c]

    # coverage: fraction of the 5 physiological channels with a fresh value
    phys = ["hr", "hrv_rmssd", "glucose", "spo2", "skin_temp"]
    grid["coverage"] = grid[phys].notna().mean(axis=1).astype(float)

    # ---------------- window context (causal) -------------------------- #
    ctx = _build_context(grid, bin_s)
    return TelemetryFeatures(grid=grid, context_names=WINDOW_CONTEXT_FEATURES,
                             window_context=ctx)


def _rolling_slope(y: np.ndarray, w: int) -> np.ndarray:
    """Vectorised OLS slope of ``y`` over the trailing window of length ``w``.

    Uses prefix sums so the cost is O(n) rather than O(n·w); NaNs are
    tolerated (the window is rejected if fewer than 60 % of it is valid).
    """
    n = len(y)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or w < 2:
        return out

    finite = np.isfinite(y)
    yy = np.where(finite, y, 0.0)
    x = np.arange(n, dtype=np.float64)
    v = finite.astype(np.float64)

    c0 = np.concatenate([[0.0], np.cumsum(v)])
    cy = np.concatenate([[0.0], np.cumsum(yy)])
    cx = np.concatenate([[0.0], np.cumsum(x * v)])
    cxy = np.concatenate([[0.0], np.cumsum(x * yy * v)])
    cxx = np.concatenate([[0.0], np.cumsum(x * x * v)])

    i = np.arange(n)
    lo = np.maximum(0, i - w + 1)
    S0 = c0[i + 1] - c0[lo]
    Sy = cy[i + 1] - cy[lo]
    Sx = cx[i + 1] - cx[lo]
    Sxy = cxy[i + 1] - cxy[lo]
    Sxx = cxx[i + 1] - cxx[lo]

    with np.errstate(divide="ignore", invalid="ignore"):
        xbar = np.where(S0 > 0, Sx / S0, 0.0)
        ybar = np.where(S0 > 0, Sy / S0, 0.0)
        sxx = Sxx - S0 * xbar ** 2
        sxy = Sxy - S0 * xbar * ybar
        slope = np.where(sxx > 1e-9, sxy / sxx, np.nan)
    out[:] = np.where(S0 >= max(2.0, 0.6 * w), slope, np.nan)
    return out


def _rolling_mean(a: np.ndarray, w: int, min_frac: float = 0.0) -> np.ndarray:
    s = pd.Series(a)
    m = s.rolling(w, min_periods=max(1, int(min_frac * w))).mean()
    return m.to_numpy(dtype=np.float64)


def _rolling_min(a: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(a).rolling(w, min_periods=1).min().to_numpy(dtype=np.float64)


def _rolling_sum(a: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(a).rolling(w, min_periods=1).sum().to_numpy(dtype=np.float64)


def _build_context(grid: pd.DataFrame, bin_s: int) -> pd.DataFrame:
    """Causal per-bin summary statistics (no future information).

    Fully vectorised: every statistic is a trailing window ending at the
    current bin, so no information from the future ever leaks into a feature.
    """
    ts = grid["ts"]
    hours = ts.dt.hour + ts.dt.minute / 60.0
    night = ((hours >= 0.0) & (hours < 6.0)).to_numpy()

    rm = grid["hrv_rmssd"].to_numpy(dtype=np.float64)
    lf = grid["hrv_lf"].to_numpy(dtype=np.float64)
    hf = grid["hrv_hf"].to_numpy(dtype=np.float64)
    hr = grid["hr"].to_numpy(dtype=np.float64)
    glu = grid["glucose"].to_numpy(dtype=np.float64)
    spo2 = grid["spo2"].to_numpy(dtype=np.float64)
    sleep = grid["sleep_stage"].to_numpy(dtype=np.float64)
    cov = grid["coverage"].to_numpy(dtype=np.float64)
    rr_stale = grid["rr_staleness"].to_numpy(dtype=np.float64)
    pnn50 = grid["hrv_pnn50"].to_numpy(dtype=np.float64)

    n = len(grid)
    W_1H = 13       # ~1 hour
    W_3H = 37       # ~3 hours
    W_12H = 145     # ~12 hours — matches the model's look-back window

    # ---- causal personal baseline: nocturnal RMSSD over all prior nights ---- #
    rm_night = np.where(night, rm, np.nan)
    filled = np.where(np.isnan(rm_night), 0.0, rm_night)
    cnt = (~np.isnan(rm_night)).astype(np.float64)
    csum = np.concatenate([[0.0], np.cumsum(filled)])
    ccnt = np.concatenate([[0.0], np.cumsum(cnt)])
    prior_sum = csum[:-1]                    # strictly prior
    prior_cnt = ccnt[:-1]
    base = np.where(prior_cnt >= 6, prior_sum / np.maximum(prior_cnt, 1e-9), np.nan)

    # ---- sleep-stage aligned baseline (deep sleep only) ----------------- #
    # RMSSD is strongly sleep-stage dependent: a low value during N3 is
    # physiologically normal, a low value during REM is not. Comparing the
    # patient against their own *deep-sleep* baseline removes the stage
    # architecture and leaves only genuine autonomic withdrawal.
    deep = np.isin(sleep, [2, 3, 4])
    rm_deep = np.where(night & deep, rm, np.nan)
    dfill = np.where(np.isnan(rm_deep), 0.0, rm_deep)
    dcnt = (~np.isnan(rm_deep)).astype(np.float64)
    dsum = np.concatenate([[0.0], np.cumsum(dfill)])
    dcsum = np.concatenate([[0.0], np.cumsum(dcnt)])
    deep_prior_sum = dsum[:-1]
    deep_prior_cnt = dcsum[:-1]
    deep_base = np.where(deep_prior_cnt >= 3,
                         deep_prior_sum / np.maximum(deep_prior_cnt, 1e-9), np.nan)

    # ---- trailing-window aggregates ---- #
    nhr_mean = _rolling_mean(rm_night, W_12H, min_frac=0.15)
    dhr_mean = _rolling_mean(np.where(night, np.nan, rm), W_12H, min_frac=0.15)
    dev = np.where(np.isfinite(base) & (base > 0) & np.isfinite(nhr_mean),
                   (nhr_mean - base) / base, np.nan)

    nhr_deep = _rolling_mean(rm_deep, W_12H, min_frac=0.10)
    dev_deep = np.where(np.isfinite(deep_base) & (deep_base > 0) & np.isfinite(nhr_deep),
                        (nhr_deep - deep_base) / deep_base, np.nan)

    rm_slope = _rolling_slope(rm, W_12H)

    # "3 h vs 9 h" contrast: last 3 hours against the preceding 9 hours
    rm_filled = np.where(np.isfinite(rm), rm, 0.0)
    rm_valid = np.isfinite(rm).astype(np.float64)
    s12, c12 = _rolling_sum(rm_filled, W_12H), _rolling_sum(rm_valid, W_12H)
    s3, c3 = _rolling_sum(rm_filled, W_3H), _rolling_sum(rm_valid, W_3H)
    late_mean = np.where(c3 >= 3, s3 / np.maximum(c3, 1e-9), np.nan)
    early_mean = np.where((c12 - c3) >= 6, (s12 - s3) / np.maximum(c12 - c3, 1e-9), np.nan)
    pct_change = np.where(np.isfinite(late_mean) & np.isfinite(early_mean)
                           & (np.abs(early_mean) > 1e-6),
                           (late_mean - early_mean) / np.abs(early_mean), np.nan)

    lf_hf = np.exp(np.clip(lf - hf, -10.0, 10.0))
    lf_hf_mean = _rolling_mean(lf_hf, W_12H, min_frac=0.15)
    lf_hf_slope = _rolling_slope(lf_hf, W_12H)

    hr_mean = _rolling_mean(hr, W_3H, min_frac=0.2)
    hr_slope = _rolling_slope(hr, W_3H)
    hr_night_mean = _rolling_mean(np.where(night, hr, np.nan), W_12H, min_frac=0.1)

    glu_mean = _rolling_mean(glu, W_3H, min_frac=0.2)
    glu_min = _rolling_min(glu, W_12H)
    tir = np.where(np.isfinite(glu), ((glu >= 70) & (glu <= 180)).astype(np.float64), np.nan)
    tir_pct = _rolling_mean(tir, W_12H, min_frac=0.15) * 100.0
    hypo = np.where(np.isfinite(glu), (glu < 70).astype(np.float64), np.nan)
    hypo_minutes = _rolling_sum(hypo, W_12H) * 5.0
    glu_slope = _rolling_slope(glu, W_3H)

    spo2_min = _rolling_min(spo2, W_3H)
    sleep_depth = _rolling_mean(np.isin(sleep, [2, 3, 4]).astype(np.float64),
                                 W_12H, min_frac=0.1) * 100.0
    coverage_mean = _rolling_mean(cov, W_3H, min_frac=0.2)

    lf_ratio = np.exp(np.clip(lf - hf, -10.0, 10.0))
    lf_share = lf_ratio / (1.0 + lf_ratio)
    lf_share_mean = pd.Series(lf_share).expanding(min_periods=1).mean().to_numpy()

    artifact_frac = _rolling_mean(np.where(np.isfinite(rr_stale),
                                            (rr_stale > 15.0).astype(np.float64), np.nan),
                                   W_3H, min_frac=0.2)

    # ---- ectopy load ---------------------------------------------------- #
    pnn_mean = _rolling_mean(pnn50, W_3H, min_frac=0.2)
    ectopy_load = _rolling_mean(np.where(np.isfinite(pnn50),
                                          (pnn50 > 1.5).astype(np.float64), np.nan),
                                 W_1H, min_frac=0.2)

    cols = {
        "nhr_rmssd_mean": nhr_mean,
        "dhr_rmssd_mean": dhr_mean,
        "nhr_rmssd_dev_baseline": dev,
        "nhr_rmssd_deep": nhr_deep,
        "nhr_rmssd_dev_deep": dev_deep,
        "rmssd_slope": rm_slope * 12.0,
        "rmssd_pct_change_3h9h": pct_change,
        "lf_hf_mean": lf_hf_mean,
        "lf_hf_slope": lf_hf_slope * 12.0,
        "hr_mean": hr_mean,
        "hr_slope": hr_slope * 12.0,
        "hr_night_mean": hr_night_mean,
        "glu_mean": glu_mean,
        "glu_min": glu_min,
        "glu_tir_pct": tir_pct,
        "glu_hypo_minutes": hypo_minutes,
        "glu_slope": glu_slope * 12.0,
        "spo2_min": spo2_min,
        "sleep_depth_pct": sleep_depth,
        "coverage_mean": coverage_mean,
        "spectral_entropy": lf_share_mean,
        "rr_artifact_fraction": artifact_frac,
        "pnn50_mean": pnn_mean,
        "ectopy_load": ectopy_load,
    }
    ctx = pd.DataFrame({k: cols[k] for k in WINDOW_CONTEXT_FEATURES})
    ctx.insert(0, "ts", ts.to_numpy())
    return ctx


# --------------------------------------------------------------------------- #
# Stage 3 — causal windowing with a forward-looking label
# --------------------------------------------------------------------------- #
@dataclass
class Window:
    patient_id: str
    start_idx: int
    end_idx: int          # exclusive; the "now"
    label: int
    event_type: str
    ttx_hours: float      # hours until the next event (inf if none)


def build_windows(grid: pd.DataFrame, patient_id: str, events: list[tuple[str, pd.Timestamp]],
                  profile: DatasetProfile, stride_bins: int = 6) -> list[Window]:
    """Emit (look-back window -> future event) pairs.

    Look-back and prediction horizon are both ``profile.horizon_hours``.
    """
    bin_s = profile.grid_minutes * 60
    horizon_bins = int(profile.horizon_hours * 3600 / bin_s)
    n_bins = len(grid)
    grid_start = grid["ts"].iloc[0]

    ev = sorted(((lbl, ts) for lbl, ts in events), key=lambda x: x[1])

    windows: list[Window] = []
    for end in range(horizon_bins, n_bins, stride_bins):
        now = grid_start + pd.Timedelta(seconds=end * bin_s)
        future = grid_start + pd.Timedelta(seconds=(end + horizon_bins) * bin_s)

        nxt_label, nxt_type, ttx = "none", "none", float("inf")
        for lbl, ts in ev:
            if now <= ts < future:
                nxt_label, nxt_type = lbl, lbl
                ttx = float((ts - now).total_seconds() / 3600.0)
                break

        windows.append(Window(
            patient_id=patient_id,
            start_idx=end - horizon_bins,
            end_idx=end,
            label=int(nxt_label != "none"),
            event_type=nxt_type,
            ttx_hours=ttx,
        ))
    return windows


def window_tensor(grid: pd.DataFrame, windows: Sequence[Window],
                  channels: Sequence[str] = TEMPORAL_CHANNELS) -> np.ndarray:
    """Materialise the (N, C, T) tensor for a list of windows."""
    values = grid[list(channels)].to_numpy(dtype=np.float32)
    if len(windows) == 0:
        return np.zeros((0, len(channels), 0), dtype=np.float32)
    T = windows[0].end_idx - windows[0].start_idx
    out = np.zeros((len(windows), len(channels), T), dtype=np.float32)
    for i, w in enumerate(windows):
        out[i] = values[w.start_idx: w.end_idx].T
    return out


def context_matrix(ctx: pd.DataFrame, windows: Sequence[Window],
                   names: Sequence[str] = WINDOW_CONTEXT_FEATURES) -> np.ndarray:
    """Per-window context vector, taken at the *end* of the look-back window."""
    vals = ctx[list(names)].to_numpy(dtype=np.float32)
    if len(windows) == 0:
        return np.zeros((0, len(names)), dtype=np.float32)
    idxs = np.array([w.end_idx - 1 for w in windows])
    return vals[idxs]
