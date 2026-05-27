"""Weighted aggregation of several reliability signals.

When multiple confidence sources (or axes) are combined, each is weighted by its
inverse-Brier reliability so that better-calibrated signals dominate. If no
positive reliability is available the weights fall back to **uniform** rather
than collapsing to zero (a sane default, never a fabricated emphasis).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["normalize_weights", "blend"]


def normalize_weights(raw: Sequence[float] | np.ndarray) -> np.ndarray:
    """Normalise non-negative weights to sum to 1; uniform if none are positive."""
    w = np.asarray(raw, dtype=float)
    if w.size == 0:
        return w
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    total = w.sum()
    if total <= 0.0:
        return np.full(w.size, 1.0 / w.size)
    return w / total


def blend(values: Sequence[float], weights: Sequence[float] | None = None) -> float:
    """Weighted mean of ``values``; uniform weights when ``weights`` is None.

    NaN values are dropped (with their weight) so one missing axis does not
    poison the blend; returns NaN only if nothing finite remains.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan")
    raw = np.ones(v.size) if weights is None else np.asarray(weights, dtype=float)
    finite = np.isfinite(v)
    if not finite.any():
        return float("nan")
    w = normalize_weights(np.where(finite, raw, 0.0))
    return float(np.sum(w[finite] * v[finite]) / w[finite].sum())
