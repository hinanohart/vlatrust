"""S2: fail-closed OOD + physics gate — accept / abstain-ood / abstain-physics."""

from __future__ import annotations

import numpy as np

from vlatrust.core.reliability.ood_gate import (
    ABSTAIN_OOD,
    ABSTAIN_PHYSICS,
    ACCEPT,
    gate_trace,
    gate_traceset,
    hard_valid,
)
from vlatrust.core.types import ConfidenceSource, Step, Trace, TraceSet

ACTION_DIM = 7


def _trace(nlps, *, valid=True, trace_id="t") -> Trace:
    """A trace whose per-step nonconformity is the given neg_log_prob values.

    ``nlps`` entries may be ``None`` to leave a step's nonconformity undefined.
    """
    steps = tuple(
        Step(
            action=np.zeros(ACTION_DIM),
            neg_log_prob=nlp,
            physically_valid=valid,
        )
        for nlp in nlps
    )
    return Trace(
        steps=steps,
        confidence_source=ConfidenceSource.TOKEN_ENTROPY,
        success=True,
        trace_id=trace_id,
    )


# --- hard_valid ------------------------------------------------------------- #


def test_hard_valid_true_when_all_steps_valid():
    assert hard_valid(_trace([0.1, 0.2, 0.3]))


def test_hard_valid_false_with_one_violation():
    steps = (
        Step(action=np.zeros(ACTION_DIM), physically_valid=True),
        Step(action=np.zeros(ACTION_DIM), physically_valid=False),
    )
    t = Trace(
        steps=steps, confidence_source=ConfidenceSource.TOKEN_ENTROPY, success=True, trace_id="v"
    )
    assert not hard_valid(t)


# --- gate_trace decisions --------------------------------------------------- #


def test_accept_when_score_within_threshold():
    d = gate_trace(_trace([0.1, 0.1, 0.1]), q_hat=1.0)
    assert d.accept and d.reason == ACCEPT
    assert d.hard_valid is True
    assert d.sequence_score <= 1.0


def test_abstain_ood_when_score_exceeds_threshold():
    d = gate_trace(_trace([5.0, 5.0, 5.0]), q_hat=1.0)
    assert not d.accept and d.reason == ABSTAIN_OOD


def test_abstain_physics_overrides_low_score():
    # even a perfectly in-distribution score is rejected if physics is invalid.
    d = gate_trace(_trace([0.0, 0.0], valid=False), q_hat=1e9)
    assert not d.accept and d.reason == ABSTAIN_PHYSICS


def test_fail_closed_on_undefined_score():
    # all-undefined nonconformity -> +inf -> never <= a finite q_hat -> abstain.
    d = gate_trace(_trace([None, None]), q_hat=1e9)
    assert d.sequence_score == float("inf")
    assert not d.accept and d.reason == ABSTAIN_OOD


def test_enforce_flag_marks_blocked():
    measured = gate_trace(_trace([5.0]), q_hat=1.0, enforce=False)
    enforced = gate_trace(_trace([5.0]), q_hat=1.0, enforce=True)
    assert not measured.accept and not measured.blocked  # measured, not blocked
    assert enforced.blocked  # enforced + non-accepting => hard stop
    accepted = gate_trace(_trace([0.1]), q_hat=1.0, enforce=True)
    assert accepted.accept and not accepted.blocked


# --- gate_traceset ---------------------------------------------------------- #


def test_gate_traceset_abstention_rate():
    ts = TraceSet(
        (
            _trace([0.1], trace_id="a"),  # accept
            _trace([9.0], trace_id="b"),  # abstain-ood
            _trace([0.0], valid=False, trace_id="c"),  # abstain-physics
            _trace([0.1], trace_id="d"),  # accept
        )
    )
    decisions, abstention = gate_traceset(ts, q_hat=1.0)
    assert len(decisions) == 4
    assert np.isclose(abstention, 0.5)  # 2 of 4 abstain


def test_gate_empty_traceset_is_nan():
    decisions, abstention = gate_traceset(TraceSet(()), q_hat=1.0)
    assert decisions == () and np.isnan(abstention)
