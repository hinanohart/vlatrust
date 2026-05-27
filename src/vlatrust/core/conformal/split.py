r"""Split-conformal thresholds with finite-sample, distribution-free coverage.

Given calibration sequence-scores :math:`S_1,\dots,S_n` and miscoverage level
:math:`\alpha`, the split-conformal threshold is the
:math:`\lceil (n+1)(1-\alpha)\rceil`-th smallest score. Under exchangeability of
trajectories this guarantees

.. math:: P\big(S_{\text{new}} \le \hat q\big) \ge 1 - \alpha .

If :math:`\lceil (n+1)(1-\alpha)\rceil > n` there is too little calibration data
for the requested level; the threshold is :math:`+\infty` (accept nothing —
fail closed) rather than a silently invalid finite value.

Two refinements:
  * **Mondrian** — a separate threshold per group (modality), so coverage holds
    within each group, not just marginally.
  * **Weighted** — covariate-shift weighting (Tibshirani et al., 2019): each
    calibration point carries a likelihood-ratio weight; the threshold is the
    weighted quantile including the test point's weight mass.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

__all__ = ["conformal_quantile", "mondrian_quantiles", "weighted_conformal_quantile"]


def conformal_quantile(calib_scores: Sequence[float] | np.ndarray, alpha: float) -> float:
    r"""Standard split-conformal threshold :math:`\hat q` at level ``alpha``."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    s = np.asarray(calib_scores, dtype=float)
    s = s[np.isfinite(s)]
    n = s.size
    if n == 0:
        return float("inf")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return float("inf")  # not enough calibration data for this alpha
    s_sorted = np.sort(s)
    return float(s_sorted[k - 1])  # k-th smallest (1-indexed)


def mondrian_quantiles(
    grouped_scores: Mapping[str, Sequence[float] | np.ndarray], alpha: float
) -> dict[str, float]:
    """One conformal threshold per group (e.g. per modality)."""
    return {g: conformal_quantile(scores, alpha) for g, scores in grouped_scores.items()}


def weighted_conformal_quantile(
    calib_scores: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    alpha: float,
    *,
    test_weight: float | None = None,
) -> float:
    r"""Weighted split-conformal threshold under covariate shift.

    ``weights`` are non-negative likelihood ratios :math:`w(x_i)` for the
    calibration points. The threshold is the smallest score whose cumulative
    normalised weight (calibration mass plus the test point's mass) reaches
    :math:`1-\alpha`. ``test_weight`` defaults to the mean calibration weight.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    s = np.asarray(calib_scores, dtype=float)
    w = np.asarray(weights, dtype=float)
    if s.shape != w.shape:
        raise ValueError(f"scores {s.shape} and weights {w.shape} must align")
    mask = np.isfinite(s) & np.isfinite(w) & (w >= 0.0)
    s, w = s[mask], w[mask]
    if s.size == 0 or w.sum() <= 0.0:
        return float("inf")
    wt = float(np.mean(w)) if test_weight is None else float(test_weight)
    order = np.argsort(s, kind="stable")
    s_sorted, w_sorted = s[order], w[order]
    total = w_sorted.sum() + wt  # test point sits at +inf, holding mass wt
    cum = np.cumsum(w_sorted) / total
    idx = int(np.searchsorted(cum, 1.0 - alpha, side="left"))
    if idx >= s_sorted.size:
        return float("inf")  # required mass only reached by the test point at +inf
    return float(s_sorted[idx])
