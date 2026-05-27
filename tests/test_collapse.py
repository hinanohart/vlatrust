"""S2: collapse curves + fragility — mechanism tags, AUC, bootstrap points."""

from __future__ import annotations

import numpy as np

from vlatrust.core.collapse.curve import collapse_curve, collapse_report
from vlatrust.core.collapse.fragility import (
    BASELINE_FLOOR,
    ROBUST_FRAGILITY,
    fragility_from_points,
)
from vlatrust.core.types import CollapsePoint


def _pts(*pairs: tuple[float, float]) -> tuple[CollapsePoint, ...]:
    return tuple(
        CollapsePoint(intensity=tau, success_rate=sr, ci_low=sr, ci_high=sr, n=40)
        for tau, sr in pairs
    )


# --- fragility mechanism tags ----------------------------------------------- #


def test_insufficient_when_fewer_than_two_points():
    f, mech = fragility_from_points(_pts((0.0, 1.0)))
    assert np.isnan(f) and mech == "insufficient"


def test_robust_curve():
    f, mech = fragility_from_points(_pts((0.0, 1.0), (0.5, 0.95), (1.0, 0.92)))
    assert mech == "robust"
    assert f < ROBUST_FRAGILITY


def test_cliff_curve():
    f, mech = fragility_from_points(_pts((0.0, 1.0), (0.5, 1.0), (1.0, 0.0)))
    assert mech == "cliff"
    assert f > ROBUST_FRAGILITY


def test_gradual_curve():
    f, mech = fragility_from_points(
        _pts((0.0, 1.0), (0.25, 0.8), (0.5, 0.6), (0.75, 0.4), (1.0, 0.3))
    )
    assert mech == "gradual"
    assert f > ROBUST_FRAGILITY


def test_brittle_at_zero():
    # baseline at tau=0 already below the floor => ratio meaningless.
    f, mech = fragility_from_points(_pts((0.0, BASELINE_FLOOR - 0.1), (1.0, 0.1)))
    assert mech == "brittle-at-zero"
    assert f == 1.0


def test_total_collapse_fragility_near_one():
    f, _ = fragility_from_points(_pts((0.0, 1.0), (1.0, 0.0)))
    # straight line 1 -> 0 over [0,1] has AUC 0.5 => fragility 0.5.
    assert np.isclose(f, 0.5, atol=1e-6)


def test_points_unsorted_input_is_handled():
    a, _ = fragility_from_points(_pts((1.0, 0.0), (0.0, 1.0), (0.5, 0.5)))
    b, _ = fragility_from_points(_pts((0.0, 1.0), (0.5, 0.5), (1.0, 0.0)))
    assert np.isclose(a, b)


# --- collapse curve over a synthetic TraceSet ------------------------------- #


def test_collapse_curve_has_baseline_and_declines(calibrated_ts):
    curve = collapse_curve(calibrated_ts, "sensor_noise", rng=0)
    assert curve.modality == "sensor_noise"
    taus = [p.intensity for p in curve.points]
    assert taus[0] == 0.0  # clean baseline first
    assert taus == sorted(taus)
    # success rate should be non-increasing-ish: the last point < the baseline.
    assert curve.points[-1].success_rate < curve.points[0].success_rate
    assert np.isfinite(curve.fragility)


def test_collapse_curve_points_carry_ci(calibrated_ts):
    curve = collapse_curve(calibrated_ts, "sensor_noise", rng=0)
    for p in curve.points:
        assert p.ci_low <= p.success_rate <= p.ci_high
        assert p.n > 0


def test_collapse_report_identifies_modality(calibrated_ts):
    rep = collapse_report(calibrated_ts, rng=0)
    assert rep.most_fragile_modality == "sensor_noise"
    assert np.isfinite(rep.mean_fragility)
    assert len(rep.curves) == 1


def test_collapse_curve_is_deterministic(calibrated_ts):
    a = collapse_curve(calibrated_ts, "sensor_noise", rng=0)
    b = collapse_curve(calibrated_ts, "sensor_noise", rng=0)
    assert [p.success_rate for p in a.points] == [p.success_rate for p in b.points]
    assert [p.ci_low for p in a.points] == [p.ci_low for p in b.points]
