"""S2: the headline Trust-Shift score — the falsification suite.

The whole point of vlatrust is one claim: a policy whose confidence stays high
while competence collapses must score *low*. These tests are the falsification —
if a confidently-collapsing policy ever out-scored a gracefully-degrading one,
the metric would be wrong.
"""

from __future__ import annotations

import dataclasses as dc
import json

import numpy as np

from vlatrust.core.score.aggregate import blend, normalize_weights
from vlatrust.core.score.scorecard import score_traceset, trust_shift
from vlatrust.core.types import ConfidenceSource, TraceSet


def _invalidate_fraction(ts: TraceSet, frac: float) -> TraceSet:
    """Mark the first step of the first ``frac`` of traces physically invalid."""
    n_bad = int(len(ts) * frac)
    out = []
    for i, t in enumerate(ts.traces):
        if i < n_bad and t.steps:
            bad_first = dc.replace(t.steps[0], physically_valid=False)
            t = dc.replace(t, steps=(bad_first, *t.steps[1:]))
        out.append(t)
    return TraceSet(tuple(out))


# --- the falsification: graceful degradation must beat overconfidence ------- #


def test_calibrated_outscores_overconfident(calibrated_ts, overconfident_ts):
    cal = trust_shift(calibrated_ts, rng=0)
    over = trust_shift(overconfident_ts, rng=0)
    assert cal is not None and over is not None
    assert cal > over, f"graceful({cal}) must beat overconfident({over})"


def test_overconfidence_penalised_through_tracking(overconfident_ts):
    sc = score_traceset(overconfident_ts, rng=0)
    notes = dict(n.split("=", 1) for n in sc.notes if "=" in n)
    # the confidence/competence tracking term is what catches the 95%->0% pathology.
    assert float(notes["tracking"]) < 0.6


# --- ConfidenceSource.NONE -> no claim, never fabricated -------------------- #


def test_no_confidence_yields_none(no_confidence_ts):
    sc = score_traceset(no_confidence_ts, rng=0)
    assert sc.trust_shift is None
    assert sc.confidence_source == ConfidenceSource.NONE
    assert any("uncalibrated" in n or "N/A" in n for n in sc.notes)


# --- the multiplicative physics gate pulls toward zero ---------------------- #


def test_physics_violation_reduces_trust(calibrated_ts):
    clean = trust_shift(calibrated_ts, rng=0)
    half_bad = trust_shift(_invalidate_fraction(calibrated_ts, 0.5), rng=0)
    all_bad = score_traceset(_invalidate_fraction(calibrated_ts, 1.0), rng=0)
    assert clean is not None and half_bad is not None
    assert half_bad < clean  # the hard gate drags the score down
    assert all_bad.trust_shift == 0.0  # every trajectory invalid => hard zero
    assert all_bad.hard_valid is False


def test_hard_valid_flag_true_when_all_valid(calibrated_ts):
    sc = score_traceset(calibrated_ts, rng=0)
    assert sc.hard_valid is True


# --- abstention gate helps when it engages ---------------------------------- #


def test_abstention_gate_raises_retained_reliability(calibrated_ts):
    on = score_traceset(calibrated_ts, alpha=0.25, enable_gate=True, rng=0).trust_shift
    off = score_traceset(calibrated_ts, alpha=0.25, enable_gate=False, rng=0).trust_shift
    assert on is not None and off is not None
    assert on > off  # abstaining on high-nonconformity rollouts lifts the score


# --- determinism ------------------------------------------------------------ #


def test_trust_shift_is_deterministic_across_100_calls(calibrated_ts):
    vals = {trust_shift(calibrated_ts, rng=0) for _ in range(100)}
    assert len(vals) == 1


def test_scorecard_serializes_to_json(calibrated_ts):
    sc = score_traceset(calibrated_ts, rng=0)
    blob = json.dumps(sc.to_dict())
    round_trip = json.loads(blob)
    assert round_trip["confidence_source"] == ConfidenceSource.TOKEN_ENTROPY.value
    assert round_trip["trust_shift"] is not None


# --- aggregate helpers ------------------------------------------------------ #


def test_normalize_weights_uniform_fallback():
    w = normalize_weights([0.0, 0.0, 0.0])
    assert np.allclose(w, [1 / 3, 1 / 3, 1 / 3])


def test_blend_drops_nan_axis():
    # a missing (NaN) axis must not poison the blend.
    assert np.isclose(blend([0.8, float("nan"), 0.6]), 0.7)
    assert np.isnan(blend([float("nan"), float("nan")]))
