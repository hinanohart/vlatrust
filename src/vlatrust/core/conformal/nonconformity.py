r"""Per-step and sequence-level nonconformity scores.

A trajectory is one exchangeable unit, but a policy can fail at a *single* step
(grab the wrong object once and the episode is lost). Aggregating per-step
nonconformity by the **mean** dilutes such a spike, so vlatrust aggregates with a
high quantile instead:

.. math::

    S(\tau) = \operatorname{quantile}_\beta(\{s_t\}_{t}), \quad \beta \in [0.9, 1.0]

with :math:`\beta = 1.0` recovering the max. This deliberately preserves
single-step spikes that a mean would wash out; see the critique of mean
aggregation for action streams in arXiv:2603.18342.

The per-step score :math:`s_t` is taken from the first available signal:
``neg_log_prob`` (= :math:`-\log p_\theta(a_t\mid o_t)`), else ``action_residual``.
If neither exists the step's nonconformity is undefined (``nan``); the OOD gate
maps that to ``+inf`` so the decision fails closed (ABSTAIN).
"""

from __future__ import annotations

import numpy as np

from ..types import Step, Trace

__all__ = [
    "step_nonconformity",
    "trace_step_scores",
    "sequence_nonconformity",
    "DEFAULT_BETA",
]

DEFAULT_BETA = 0.95


def step_nonconformity(step: Step) -> float:
    """Per-step score; ``nan`` when no usable signal was recorded."""
    if step.neg_log_prob is not None and np.isfinite(step.neg_log_prob):
        return float(step.neg_log_prob)
    if step.action_residual is not None and np.isfinite(step.action_residual):
        return float(step.action_residual)
    return float("nan")


def trace_step_scores(trace: Trace) -> np.ndarray:
    """Vector of per-step nonconformity scores for one trajectory."""
    if trace.horizon == 0:
        return np.empty(0, dtype=float)
    return np.array([step_nonconformity(s) for s in trace.steps], dtype=float)


def sequence_nonconformity(scores: np.ndarray, beta: float = DEFAULT_BETA) -> float:
    r"""Spike-preserving sequence score :math:`S = \mathrm{quantile}_\beta(\{s_t\})`.

    ``nan`` entries (steps with no usable signal) are ignored when finite scores
    exist; if *every* step is undefined the sequence score is ``+inf`` so the
    trajectory fails closed. An empty trajectory is also ``+inf``.
    """
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"beta must be in [0, 1], got {beta!r}")
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        return float("inf")
    finite = s[np.isfinite(s)]
    if finite.size == 0:
        return float("inf")
    return float(np.quantile(finite, beta))
