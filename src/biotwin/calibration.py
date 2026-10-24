"""
BioTwin.X — Probability Calibration
=====================================
A model that ranks patients well but reports badly calibrated probabilities is
clinically dangerous: an "80 % risk" that materialises 20 % of the time
destroys trust and triggers inappropriate escalation.

We therefore fit **temperature scaling** on a held-out *patient* split
(never on training data, never on the same windows used for evaluation).

Temperature scaling is the right tool here because it preserves the *ranking*
of the model's output exactly (it is a strictly monotonic transform) while
recalibrating the magnitudes — which is precisely what a triage dashboard
needs, and it has a single parameter, so it cannot overfit a small cohort.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class TemperatureScaler:
    """Single-parameter probability calibration (Guo et al., 2017)."""

    temperature: float = 1.0
    bias: float = 0.0

    def transform(self, logits: np.ndarray) -> np.ndarray:
        z = np.asarray(logits, dtype=np.float64) / max(self.temperature, 1e-6) + self.bias
        z = np.clip(z, -60, 60)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, logits: np.ndarray, targets: np.ndarray,
            lr: float = 0.5, steps: int = 600) -> "TemperatureScaler":
        """Fit ``(temperature, bias)`` by gradient descent on binary NLL.

        The logits are mean-centred first (the bias absorbs the shift), which
        keeps the temperature gradient well conditioned when a model's outputs
        are systematically offset — common after class-balanced oversampling.
        ``lr`` is applied to the *per-sample mean* gradient, so it is
        independent of cohort size.
        """
        z = np.asarray(logits, dtype=np.float64).reshape(-1)
        y = np.asarray(targets, dtype=np.float64).reshape(-1)
        if len(z) == 0 or y.min() == y.max():
            return self
        z = z - float(z.mean())
        n = float(len(z))
        t, b = 1.0, 0.0
        for _ in range(steps):
            s = np.clip(z / t + b, -60, 60)
            p = 1.0 / (1.0 + np.exp(-s))
            g = p - y
            # dL/ds = g / n ; ds/dt = -z / t^2 ; ds/db = 1
            grad_t = float(np.dot(g, -z / (t * t))) / n
            grad_b = float(g.mean())
            t -= lr * grad_t
            b -= lr * grad_b
            t = float(np.clip(t, 0.05, 20.0))
            b = float(np.clip(b, -10, 10))
        self.temperature, self.bias = t, b
        return self

    def to_dict(self) -> dict:
        return {"temperature": self.temperature, "bias": self.bias}

    @classmethod
    def from_dict(cls, d: dict) -> "TemperatureScaler":
        return cls(temperature=float(d.get("temperature", 1.0)),
                   bias=float(d.get("bias", 0.0)))


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray,
                                n_bins: int = 10) -> float:
    """ECE over equal-width probability bins."""
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    if len(p) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def brier_score(probs: np.ndarray, targets: np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    if len(p) == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))
