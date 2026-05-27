"""Calibrate a conformal threshold on one TraceSet and measure coverage on
another, returning a :class:`~vlatrust.core.types.ConformalResult`.

A test trajectory is ACCEPTED iff its sequence nonconformity is ``<= q_hat``,
else it ABSTAINS. Undefined scores are ``+inf`` and therefore always abstain
(fail closed). Three modes: ``"standard"``, ``"mondrian"`` (per-modality
threshold), ``"weighted"`` (covariate-shift weights).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..types import ConformalResult, Trace, TraceSet
from .nonconformity import DEFAULT_BETA, sequence_nonconformity, trace_step_scores
from .split import conformal_quantile, mondrian_quantiles, weighted_conformal_quantile

__all__ = ["traceset_scores", "group_key", "calibrate"]


def group_key(trace: Trace) -> str:
    """Mondrian group for a trace: its perturbation modality, or ``"clean"``."""
    return trace.modality or "clean"


def traceset_scores(ts: TraceSet, beta: float = DEFAULT_BETA) -> np.ndarray:
    """One spike-preserving sequence score per trace (order preserved)."""
    return np.array(
        [sequence_nonconformity(trace_step_scores(t), beta) for t in ts.traces],
        dtype=float,
    )


def _coverage_abstention(test_scores: np.ndarray, q_hat: float) -> tuple[float, float]:
    if test_scores.size == 0:
        return (float("nan"), float("nan"))
    accept = test_scores <= q_hat  # inf > any finite q -> abstain (fail closed)
    cov = float(np.mean(accept))
    return cov, 1.0 - cov


def calibrate(
    calib_ts: TraceSet,
    test_ts: TraceSet,
    *,
    alpha: float = 0.1,
    beta: float = DEFAULT_BETA,
    mode: str = "standard",
    weight_fn: Callable[[Trace], float] | None = None,
) -> ConformalResult:
    """Fit ``q_hat`` on ``calib_ts`` and report coverage on ``test_ts``."""
    calib_scores = traceset_scores(calib_ts, beta)
    test_scores = traceset_scores(test_ts, beta)
    n_calib, n_test = len(calib_ts), len(test_ts)

    if mode == "standard":
        q_hat = conformal_quantile(calib_scores, alpha)
        cov, abst = _coverage_abstention(test_scores, q_hat)
        return ConformalResult(
            alpha=alpha,
            q_hat=q_hat,
            coverage=cov,
            n_calib=n_calib,
            n_test=n_test,
            abstention_rate=abst,
            beta=beta,
        )

    if mode == "weighted":
        if weight_fn is None:
            raise ValueError("mode='weighted' requires weight_fn")
        weights = np.array([weight_fn(t) for t in calib_ts.traces], dtype=float)
        q_hat = weighted_conformal_quantile(calib_scores, weights, alpha)
        cov, abst = _coverage_abstention(test_scores, q_hat)
        return ConformalResult(
            alpha=alpha,
            q_hat=q_hat,
            coverage=cov,
            n_calib=n_calib,
            n_test=n_test,
            abstention_rate=abst,
            beta=beta,
            weighted=True,
        )

    if mode == "mondrian":
        grouped: dict[str, list[float]] = {}
        for t, s in zip(calib_ts.traces, calib_scores, strict=True):
            grouped.setdefault(group_key(t), []).append(s)
        q_by_group = mondrian_quantiles(grouped, alpha)
        accepts = []
        per_group_cov: dict[str, list[float]] = {}
        for t, s in zip(test_ts.traces, test_scores, strict=True):
            q = q_by_group.get(group_key(t), float("inf"))  # unseen group -> abstain
            ok = bool(s <= q)
            accepts.append(ok)
            per_group_cov.setdefault(group_key(t), []).append(1.0 if ok else 0.0)
        cov = float(np.mean(accepts)) if accepts else float("nan")
        abst = (1.0 - cov) if accepts else float("nan")
        per_group = tuple((g, float(np.mean(v))) for g, v in sorted(per_group_cov.items()))
        # representative q_hat: the worst (largest finite, else inf) group threshold
        finite_qs = [q for q in q_by_group.values() if np.isfinite(q)]
        rep_q = max(finite_qs) if finite_qs else float("inf")
        return ConformalResult(
            alpha=alpha,
            q_hat=rep_q,
            coverage=cov,
            n_calib=n_calib,
            n_test=n_test,
            abstention_rate=abst,
            beta=beta,
            per_group=per_group,
        )

    raise ValueError(f"unknown mode {mode!r} (standard|mondrian|weighted)")
