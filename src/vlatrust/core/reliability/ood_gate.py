"""Fail-closed out-of-distribution action gate.

A trajectory is ACCEPTED only if both hold:
  * every step is physically valid (no joint-limit / dynamics violation), and
  * its sequence nonconformity is ``<= q_hat``.

Anything undefined fails closed: an unparseable / missing nonconformity becomes
``+inf`` (handled upstream in :func:`sequence_nonconformity`) and therefore
abstains; a physics violation forces ``hard_valid=False`` regardless of the
conformal score. The ``enforce`` flag separates *measuring* abstention (default)
from *blocking* on it, so the same computation serves an audit and a guard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..conformal.nonconformity import DEFAULT_BETA, sequence_nonconformity, trace_step_scores
from ..types import Trace, TraceSet

__all__ = ["GateDecision", "hard_valid", "gate_trace", "gate_traceset"]

# decision reason tags
ACCEPT = "accept"
ABSTAIN_OOD = "abstain-ood"
ABSTAIN_PHYSICS = "abstain-physics-invalid"


@dataclass(frozen=True, slots=True)
class GateDecision:
    accept: bool
    reason: str
    sequence_score: float
    hard_valid: bool
    enforced: bool = False

    @property
    def blocked(self) -> bool:
        """True iff a non-accepting decision is being enforced (a hard stop)."""
        return self.enforced and not self.accept


def hard_valid(trace: Trace) -> bool:
    """True iff every step of the trajectory is physically valid."""
    return all(s.physically_valid for s in trace.steps)


def gate_trace(
    trace: Trace, q_hat: float, *, beta: float = DEFAULT_BETA, enforce: bool = False
) -> GateDecision:
    hv = hard_valid(trace)
    score = sequence_nonconformity(trace_step_scores(trace), beta)
    if not hv:
        return GateDecision(False, ABSTAIN_PHYSICS, score, hv, enforce)
    accept = bool(score <= q_hat)  # inf <= finite q -> False -> fail closed
    reason = ACCEPT if accept else ABSTAIN_OOD
    return GateDecision(accept, reason, score, hv, enforce)


def gate_traceset(
    ts: TraceSet, q_hat: float, *, beta: float = DEFAULT_BETA, enforce: bool = False
) -> tuple[tuple[GateDecision, ...], float]:
    """Gate every trace; return ``(decisions, abstention_rate)``."""
    decisions = tuple(gate_trace(t, q_hat, beta=beta, enforce=enforce) for t in ts.traces)
    if not decisions:
        return decisions, float("nan")
    abstention = float(np.mean([0.0 if d.accept else 1.0 for d in decisions]))
    return decisions, abstention
