r"""Reliability gaps between a reference and a target distribution.

Two gaps, each with a bootstrap CI; a gap is only *claimable* when its CI
excludes zero.

* :math:`\Delta_{\text{succ}} = SR_{\text{ref}} - SR_{\text{target}}` — the raw
  success-rate gap (e.g. sim minus real).
* :math:`\Delta_{\text{cov}} = \mathrm{Cov}_{\text{target}}(\hat q_{\text{ref}})
  - (1-\alpha)` — the coverage a reference-fitted conformal threshold actually
  delivers on the target, minus its nominal level. A negative value means the
  threshold was calibrated optimistically: "safe in reference, dangerous in
  target". This second gap is the net-new piece.
"""

from __future__ import annotations

import numpy as np

from .._stats import two_sample_diff_ci
from ..conformal.nonconformity import DEFAULT_BETA
from ..conformal.predictor import traceset_scores
from ..conformal.split import conformal_quantile
from ..types import ReliabilityGap, TraceSet

__all__ = ["reliability_gap"]


def _success_array(ts: TraceSet) -> np.ndarray:
    return np.array(
        [1.0 if t.success else 0.0 for t in ts.traces if t.success is not None],
        dtype=float,
    )


def reliability_gap(
    reference_ts: TraceSet,
    target_ts: TraceSet,
    *,
    alpha: float = 0.1,
    beta: float = DEFAULT_BETA,
    n_boot: int = 2000,
    rng: np.random.Generator | int | None = None,
) -> ReliabilityGap:
    """Compute ``(delta_succ, delta_cov)`` with bootstrap CIs."""
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    # --- delta_succ -------------------------------------------------------
    ref_succ = _success_array(reference_ts)
    tgt_succ = _success_array(target_ts)
    if ref_succ.size and tgt_succ.size:
        d_succ, lo, hi = two_sample_diff_ci(ref_succ, tgt_succ, n_boot=n_boot, alpha=alpha, rng=gen)
        delta_succ: float | None = d_succ
        delta_succ_ci: tuple[float, float] | None = (lo, hi)
    else:
        delta_succ, delta_succ_ci = None, None

    # --- delta_cov --------------------------------------------------------
    q_hat = conformal_quantile(traceset_scores(reference_ts, beta), alpha)
    tgt_scores = traceset_scores(target_ts, beta)
    if tgt_scores.size and np.isfinite(q_hat):
        nominal = 1.0 - alpha
        cov = float(np.mean(tgt_scores <= q_hat))
        delta_cov: float | None = cov - nominal
        n = tgt_scores.size
        idx = gen.integers(0, n, size=(n_boot, n))
        boot_cov = (tgt_scores[idx] <= q_hat).mean(axis=1) - nominal
        delta_cov_ci: tuple[float, float] | None = (
            float(np.quantile(boot_cov, alpha / 2.0)),
            float(np.quantile(boot_cov, 1.0 - alpha / 2.0)),
        )
    else:
        delta_cov, delta_cov_ci = None, None

    return ReliabilityGap(
        delta_succ=delta_succ,
        delta_succ_ci=delta_succ_ci,
        delta_cov=delta_cov,
        delta_cov_ci=delta_cov_ci,
    )
