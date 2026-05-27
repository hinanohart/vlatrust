r"""Calibration of confidence against correctness.

Three pieces, reused from the foldconsensus calibration core:

* :func:`pava` — isotonic (non-decreasing) regression by pool-adjacent-violators.
* :func:`expected_calibration_error` — binned ECE.
* :func:`inverse_brier_score` — :math:`1 - \mathrm{Brier}`, the weight a source
  earns when several confidence sources are aggregated.

A policy with :data:`ConfidenceSource.NONE` exposes no confidence, so it is
excluded from calibration entirely (the abstention axis returns ``N/A``); these
functions are only ever fed confidence/outcome pairs that actually exist.
"""

from __future__ import annotations

import numpy as np

from ..types import CalibrationResult, ConfidenceSource, TraceSet

__all__ = [
    "pava",
    "expected_calibration_error",
    "inverse_brier_score",
    "confidence_outcome_pairs",
    "calibration_report",
]


def pava(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Isotonic non-decreasing fit of ``y`` (weighted) via pool-adjacent-violators.

    Returns a fitted array the same length as ``y``. ``y`` is assumed already
    ordered by the predictor (e.g. confidence ascending).
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return y.copy()
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    # blocks: parallel arrays of weighted value, total weight, and span length
    vals = y.copy()
    wts = w.copy()
    lens = np.ones(n, dtype=int)
    # active isotonic blocks (value, total weight, span length)
    block_val: list[float] = []
    block_w: list[float] = []
    block_len: list[int] = []
    for i in range(n):
        v, ww, ll = vals[i], wts[i], lens[i]
        # merge while the new block violates monotonicity with the previous
        while block_val and block_val[-1] > v:
            pv, pw, pl = block_val.pop(), block_w.pop(), block_len.pop()
            v = (pv * pw + v * ww) / (pw + ww)
            ww = pw + ww
            ll = pl + ll
        block_val.append(v)
        block_w.append(ww)
        block_len.append(ll)
    out = np.empty(n, dtype=float)
    pos = 0
    for v, ll in zip(block_val, block_len, strict=True):
        out[pos : pos + ll] = v
        pos += ll
    return out


def expected_calibration_error(
    confidences: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> tuple[float, tuple[tuple[float, float, int], ...]]:
    """Binned ECE plus reliability-diagram bins ``(mean_conf, accuracy, count)``."""
    c = np.asarray(confidences, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    n = c.size
    if n == 0:
        return float("nan"), ()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # bin index in [0, n_bins-1]; clip the right edge into the last bin
    idx = np.clip(np.digitize(c, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    bins: list[tuple[float, float, int]] = []
    for b in range(n_bins):
        sel = idx == b
        cnt = int(np.sum(sel))
        if cnt == 0:
            continue
        conf_mean = float(np.mean(c[sel]))
        acc = float(np.mean(o[sel]))
        ece += (cnt / n) * abs(acc - conf_mean)
        bins.append((conf_mean, acc, cnt))
    return float(ece), tuple(bins)


def inverse_brier_score(confidences: np.ndarray, outcomes: np.ndarray) -> float:
    r""":math:`1 - \frac{1}{n}\sum (c_i - o_i)^2 \in [0, 1]` (higher = better)."""
    c = np.asarray(confidences, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    if c.size == 0:
        return float("nan")
    return float(1.0 - np.mean((c - o) ** 2))


def confidence_outcome_pairs(ts: TraceSet) -> tuple[np.ndarray, np.ndarray]:
    """Per-trace ``(mean confidence, success outcome)`` for usable traces.

    A trace contributes only if it carries a success label, a non-NONE
    confidence source, and at least one step with a confidence value. Returns
    two empty arrays if nothing qualifies (caller must then report
    "uncalibrated" rather than invent a number).
    """
    confs: list[float] = []
    outs: list[float] = []
    for t in ts.traces:
        if t.success is None or t.confidence_source == ConfidenceSource.NONE:
            continue
        step_confs = [s.confidence for s in t.steps if s.confidence is not None]
        if not step_confs:
            continue
        confs.append(float(np.mean(step_confs)))
        outs.append(1.0 if t.success else 0.0)
    return np.asarray(confs, dtype=float), np.asarray(outs, dtype=float)


def calibration_report(ts: TraceSet, n_bins: int = 10) -> CalibrationResult | None:
    """Calibration of a TraceSet's confidence vs. success.

    Returns ``None`` when no trace exposes usable confidence+outcome (fail-closed
    "uncalibrated"), never a fabricated score.
    """
    c, o = confidence_outcome_pairs(ts)
    if c.size == 0:
        return None
    ece, bins = expected_calibration_error(c, o, n_bins=n_bins)
    return CalibrationResult(
        ece=ece, inverse_brier=inverse_brier_score(c, o), n=int(c.size), bins=bins
    )
