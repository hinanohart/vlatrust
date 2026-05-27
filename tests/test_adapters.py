"""S3: adapters — mock falsification, recorded replay, normalize, Tier-A math.

A load-bearing invariant lives here too: importing the adapters package (and
hence core) must not import torch. The Tier-A confidence math is pure numpy and
CI-tested without a model present.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

from vlatrust.adapters import (
    ActionSpaceSpec,
    MockPolicy,
    RecordedBackend,
    doctor_report,
    normalize_traceset,
)
from vlatrust.adapters.base import PolicyBackend
from vlatrust.adapters.policy_openvla import (
    OpenVLABackend,
    step_confidence_from_logits,
    token_logprob,
)
from vlatrust.core.score.scorecard import trust_shift
from vlatrust.core.types import ConfidenceSource

# --- import hygiene: no torch dragged in ------------------------------------ #


def test_importing_adapters_does_not_import_torch():
    # vlatrust.adapters is imported at module load above; torch must stay out.
    assert "torch" not in sys.modules


# --- MockPolicy: protocol + determinism + falsification --------------------- #


def test_mock_satisfies_policy_backend_protocol():
    assert isinstance(MockPolicy(), PolicyBackend)


def test_mock_rollout_is_deterministic_and_sized():
    a = MockPolicy().rollout(8, rng=0)
    b = MockPolicy().rollout(8, rng=0)
    assert len(a) == len(b) == 8
    assert [t.trace_id for t in a] == [t.trace_id for t in b]
    assert a.traces[0].steps[0].action.tolist() == b.traces[0].steps[0].action.tolist()


def test_mock_always_available():
    assert MockPolicy().available() is True


def test_mock_falsification_calibrated_beats_overconfident():
    cal = MockPolicy(calibrated=True).collapse_traceset(rng=0)
    over = MockPolicy(calibrated=False).collapse_traceset(rng=0)
    s_cal = trust_shift(cal, rng=0)
    s_over = trust_shift(over, rng=0)
    assert s_cal is not None and s_over is not None
    assert s_cal > s_over


def test_mock_none_source_yields_no_claim():
    ts = MockPolicy(confidence_source=ConfidenceSource.NONE).collapse_traceset(rng=0)
    assert trust_shift(ts, rng=0) is None


# --- RecordedBackend -------------------------------------------------------- #


def test_recorded_replays_and_detects_source():
    src = MockPolicy().rollout(10, rng=1)
    rb = RecordedBackend(src, name="demo")
    assert isinstance(rb, PolicyBackend)
    assert rb.available() is True
    assert rb.confidence_source == ConfidenceSource.TOKEN_ENTROPY
    assert len(rb.rollout(4)) == 4  # slice
    assert len(rb.rollout(999)) == 10  # capped at what exists


def test_recorded_empty_is_unavailable():
    from vlatrust.core.types import TraceSet

    assert RecordedBackend(TraceSet(())).available() is False


# --- normalize -------------------------------------------------------------- #


def test_action_spec_identity_is_noop():
    a = np.array([1.0, -2.0, 3.0])
    assert np.array_equal(ActionSpaceSpec().apply(a), a)


def test_action_spec_affine_and_sign():
    spec = ActionSpaceSpec(scale=(2.0, 2.0), bias=(1.0, 1.0), sign=(1.0, -1.0))
    out = spec.apply(np.array([5.0, 5.0]))
    assert np.allclose(out, [2.0, -2.0])  # ((5-1)/2)=2, second dim sign-flipped


def test_normalize_traceset_applies_per_step():
    ts = MockPolicy().rollout(3, rng=0)
    spec = ActionSpaceSpec(scale=(2.0,) * 7)
    norm = normalize_traceset(ts, spec)
    assert np.allclose(norm.traces[0].steps[0].action, ts.traces[0].steps[0].action / 2.0)


# --- Tier-A token-entropy math (pure numpy, no model) ----------------------- #


def test_token_logprob_matches_manual_softmax():
    logits = np.array([0.0, np.log(3.0)])  # softmax = [0.25, 0.75]
    assert np.isclose(token_logprob(logits, 1), np.log(0.75))


def test_peaked_logits_give_high_confidence_low_nlp():
    # one near-certain token per dim
    logits = np.full((7, 256), -10.0)
    chosen = np.arange(7) % 256
    logits[np.arange(7), chosen] = 10.0
    conf, nlp = step_confidence_from_logits(logits, chosen)
    assert conf > 0.99 and nlp < 0.01


def test_uniform_logits_give_low_confidence_high_nlp():
    logits = np.zeros((7, 256))  # uniform over 256 -> p=1/256
    chosen = np.zeros(7, dtype=int)
    conf, nlp = step_confidence_from_logits(logits, chosen)
    assert np.isclose(conf, 1.0 / 256.0, atol=1e-6)
    assert np.isclose(nlp, np.log(256.0), atol=1e-6)


def test_confidence_and_nlp_move_together():
    # as the chosen token's mass shrinks, confidence falls and nlp rises in step.
    confs, nlps = [], []
    for peak in (10.0, 3.0, 1.0, 0.0):
        logits = np.zeros((4, 10))
        logits[:, 0] = peak
        conf, nlp = step_confidence_from_logits(logits, np.zeros(4, dtype=int))
        confs.append(conf)
        nlps.append(nlp)
    assert confs == sorted(confs, reverse=True)  # monotone down
    assert nlps == sorted(nlps)  # monotone up


def test_step_confidence_rejects_bad_shape():
    with pytest.raises(ValueError):
        step_confidence_from_logits(np.zeros((3, 10)), np.zeros(4, dtype=int))


def test_openvla_unavailable_without_torch():
    # torch is not installed in CI's core job; backend reports honestly and
    # rollout fails closed with guidance rather than pretending to run.
    be = OpenVLABackend()
    assert be.confidence_source == ConfidenceSource.TOKEN_ENTROPY
    if not be.available():
        with pytest.raises(RuntimeError):
            be.rollout(1)


# --- doctor ----------------------------------------------------------------- #


def test_doctor_report_is_honest():
    rep = {r["name"]: r for r in doctor_report()}
    assert rep["mock"]["available"] is True
    assert "NOT a real policy" in rep["mock"]["note"]
    # openvla availability tracks torch presence; here torch is absent.
    assert rep["openvla"]["available"] == (importlib.util.find_spec("torch") is not None)
