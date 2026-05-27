"""S1: data-model invariants."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vlatrust.core.types import (
    ConfidenceSource,
    Perturbation,
    Scorecard,
    Step,
    Trace,
    TraceSet,
)


def _a(*vals: float) -> np.ndarray:
    return np.asarray(vals, dtype=float)


def test_step_action_is_readonly():
    s = Step(action=_a(1.0, 2.0, 3.0))
    assert s.action.shape == (3,)
    with pytest.raises(ValueError):
        s.action[0] = 9.0  # read-only buffer


def test_step_confidence_bounds():
    Step(action=_a(0.0), confidence=0.0)
    Step(action=_a(0.0), confidence=1.0)
    with pytest.raises(ValueError):
        Step(action=_a(0.0), confidence=1.5)
    with pytest.raises(ValueError):
        Step(action=_a(0.0), confidence=-0.1)


def test_perturbation_intensity_bounds_and_identity():
    assert Perturbation("x", "vision", intensity=0.0).is_identity
    assert not Perturbation("x", "vision", intensity=0.5).is_identity
    with pytest.raises(ValueError):
        Perturbation("x", "vision", intensity=1.5)


def test_perturbation_params_sorted_for_deterministic_identity():
    p = Perturbation("x", "vision", params=(("b", 2.0), ("a", 1.0)))
    assert p.params == (("a", 1.0), ("b", 2.0))


def test_trace_clean_semantics():
    clean = Trace(steps=(Step(action=_a(0.0)),))
    assert clean.is_clean and clean.modality is None and clean.intensity == 0.0
    pert = Trace(
        steps=(Step(action=_a(0.0)),),
        perturbation=Perturbation("n", "dynamics", intensity=0.3),
    )
    assert not pert.is_clean and pert.modality == "dynamics" and pert.intensity == 0.3


def test_traceset_filters_and_views(calibrated_ts):
    ts = calibrated_ts
    assert len(ts.clean()) > 0
    assert len(ts.perturbed()) > 0
    assert len(ts.clean()) + len(ts.perturbed()) == len(ts)
    assert "sensor_noise" in ts.modalities
    assert ts.intensities[0] == 0.0
    sub = ts.at_intensity(0.5)
    assert all(abs(t.intensity - 0.5) < 1e-9 for t in sub)


def test_success_rate_none_when_unlabelled():
    ts = TraceSet((Trace(steps=(Step(action=_a(0.0)),), success=None),))
    assert ts.success_rate() is None
    labelled = TraceSet(
        (
            Trace(steps=(Step(action=_a(0.0)),), success=True),
            Trace(steps=(Step(action=_a(0.0)),), success=False),
        )
    )
    assert labelled.success_rate() == 0.5


def test_scorecard_to_dict_is_json_serializable():
    sc = Scorecard(
        confidence_source=ConfidenceSource.TOKEN_ENTROPY,
        trust_shift=None,  # must survive as null, never fabricated
        hard_valid=False,
        notes=("physics gate failed",),
    )
    d = sc.to_dict()
    s = json.dumps(d)  # must not raise
    back = json.loads(s)
    assert back["confidence_source"] == "token_entropy"
    assert back["trust_shift"] is None
    assert back["hard_valid"] is False


def test_traceset_iteration_and_indexing(calibrated_ts):
    ts = calibrated_ts
    assert isinstance(ts[0], Trace)
    assert sum(1 for _ in ts) == len(ts)
    assert ConfidenceSource.TOKEN_ENTROPY in ts.confidence_sources


def test_step_action_dim():
    assert Step(action=np.zeros(7)).action_dim == 7
