"""S2: calibration core — PAVA monotonicity, ECE, inverse-Brier, None semantics."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from vlatrust.core.reliability.calibrate import (
    calibration_report,
    confidence_outcome_pairs,
    expected_calibration_error,
    inverse_brier_score,
    pava,
)
from vlatrust.core.types import ConfidenceSource, Step, Trace, TraceSet

ACTION_DIM = 7


def _step(conf: float | None) -> Step:
    return Step(action=np.zeros(ACTION_DIM), confidence=conf)


# --- PAVA: result is non-decreasing ----------------------------------------- #


@settings(max_examples=80, deadline=None)
@given(
    y=st.lists(st.floats(-10.0, 10.0, allow_nan=False), min_size=1, max_size=40),
)
def test_pava_is_monotone_nondecreasing(y):
    out = pava(np.asarray(y, dtype=float))
    assert out.shape == (len(y),)
    assert np.all(np.diff(out) >= -1e-9)


def test_pava_preserves_total_mass():
    y = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    out = pava(y)
    # PAVA is an L2 projection onto the monotone cone => preserves the sum.
    assert np.isclose(out.sum(), y.sum())


def test_pava_already_monotone_is_identity():
    y = np.array([0.1, 0.2, 0.2, 0.9])
    assert np.allclose(pava(y), y)


# --- ECE: perfect calibration => ~0 ----------------------------------------- #


def test_perfect_calibration_has_near_zero_ece():
    # bin centers; within each bin exactly `p` fraction succeed and conf == p,
    # so per-bin accuracy equals per-bin confidence => ECE == 0.
    ps = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    k = 100
    conf = np.repeat(ps, k)
    out = np.concatenate([(np.arange(k) < round(p * k)).astype(float) for p in ps])
    ece, bins = expected_calibration_error(conf, out, n_bins=10)
    assert ece < 1e-9
    assert len(bins) == 10


def test_overconfident_has_large_ece():
    # always 95% confident, only 30% succeed.
    conf = np.full(200, 0.95)
    out = np.zeros(200)
    out[: int(0.3 * 200)] = 1.0
    ece, _ = expected_calibration_error(conf, out, n_bins=10)
    assert ece > 0.5


def test_ece_empty_is_nan():
    ece, bins = expected_calibration_error(np.empty(0), np.empty(0))
    assert np.isnan(ece) and bins == ()


# --- inverse Brier: bounds and ordering ------------------------------------- #


def test_inverse_brier_perfect_is_one():
    c = np.array([0.0, 1.0, 1.0, 0.0])
    assert inverse_brier_score(c, c) == 1.0


def test_inverse_brier_worst_is_zero():
    assert inverse_brier_score(np.ones(4), np.zeros(4)) == 0.0


@settings(max_examples=60, deadline=None)
@given(
    c=st.lists(st.floats(0.0, 1.0), min_size=1, max_size=30),
    seed=st.integers(0, 1000),
)
def test_inverse_brier_in_unit_interval(c, seed):
    c = np.asarray(c)
    o = (np.random.default_rng(seed).random(c.size) < 0.5).astype(float)
    s = inverse_brier_score(c, o)
    assert 0.0 <= s <= 1.0


# --- confidence/outcome extraction + None semantics ------------------------- #


def test_pairs_exclude_none_source_and_unlabelled():
    usable = Trace(
        steps=(_step(0.8), _step(0.6)),
        confidence_source=ConfidenceSource.TOKEN_ENTROPY,
        success=True,
        trace_id="usable",
    )
    no_source = Trace(
        steps=(_step(0.9),),
        confidence_source=ConfidenceSource.NONE,  # excluded: no usable confidence
        success=True,
        trace_id="nosrc",
    )
    unlabelled = Trace(
        steps=(_step(0.9),),
        confidence_source=ConfidenceSource.TOKEN_ENTROPY,
        success=None,  # excluded: no outcome to calibrate against
        trace_id="unlab",
    )
    ts = TraceSet((usable, no_source, unlabelled))
    confs, outs = confidence_outcome_pairs(ts)
    assert confs.shape == (1,) and outs.shape == (1,)
    assert np.isclose(confs[0], 0.7)  # mean(0.8, 0.6)
    assert outs[0] == 1.0


def test_calibration_report_none_when_no_confidence(no_confidence_ts):
    assert calibration_report(no_confidence_ts) is None


def test_calibration_report_distinguishes_calibrated(calibrated_ts, overconfident_ts):
    cal = calibration_report(calibrated_ts)
    over = calibration_report(overconfident_ts)
    assert cal is not None and over is not None
    # the calibrated policy is better calibrated on both axes.
    assert cal.ece < over.ece
    assert cal.inverse_brier > over.inverse_brier


def test_calibration_report_rejects_bad_input_gracefully():
    # a TraceSet with confidence but no labels -> None (cannot calibrate).
    t = Trace(
        steps=(_step(0.5),),
        confidence_source=ConfidenceSource.TOKEN_ENTROPY,
        success=None,
        trace_id="x",
    )
    assert calibration_report(TraceSet((t,))) is None
