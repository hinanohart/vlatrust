"""S2: conformal core — finite-sample coverage, spike preservation, fail-closed."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vlatrust.core.conformal.nonconformity import sequence_nonconformity
from vlatrust.core.conformal.predictor import calibrate
from vlatrust.core.conformal.split import (
    conformal_quantile,
    mondrian_quantiles,
    weighted_conformal_quantile,
)
from vlatrust.core.trace import split_calib_test

# --- conformal_quantile golden + edge cases -------------------------------- #


def test_conformal_quantile_golden():
    # 1..100, alpha=0.1 -> k = ceil(101*0.9)=91 -> 91st smallest = 91
    assert conformal_quantile(np.arange(1, 101, dtype=float), 0.1) == 91.0


def test_conformal_quantile_too_little_data_is_inf():
    # n=2, alpha=0.1 -> k = ceil(3*0.9)=3 > 2 -> +inf (fail closed)
    assert conformal_quantile([1.0, 2.0], 0.1) == float("inf")


def test_conformal_quantile_ignores_naninf():
    finite = conformal_quantile([1.0, 2.0, 3.0, np.nan, np.inf], 0.5)
    assert np.isfinite(finite)


def test_conformal_quantile_rejects_bad_alpha():
    with pytest.raises(ValueError):
        conformal_quantile([1.0, 2.0], 0.0)


# --- finite-sample coverage guarantee -------------------------------------- #


def test_split_conformal_marginal_coverage():
    # Split conformal guarantees *marginal* coverage >= 1-alpha (averaged over
    # the calibration draw), not conditional coverage for a fixed calib set.
    # Asserting per-seed conditional coverage is statistically wrong (it dips
    # below by ~1 std routinely); the faithful check averages many splits.
    rng = np.random.default_rng(0)
    alpha = 0.1
    covs = []
    for _ in range(300):
        s = rng.normal(size=400)
        calib, test = s[:200], s[200:]
        q = conformal_quantile(calib, alpha)
        covs.append(float(np.mean(test <= q)))
    mean_cov = float(np.mean(covs))
    # ~60k test evaluations => mean_cov concentrates tightly around 1-alpha
    # (slightly conservative due to the ceil in conformal_quantile).
    assert mean_cov >= (1 - alpha) - 0.02


# --- spike preservation: the mean-dilution falsification ------------------- #


def test_spike_preserving_beats_mean_dilution():
    # A localized burst of catastrophic steps occupying >(1-beta) of the
    # trajectory: 2 of 30 steps (~6.7%) exceed the beta=0.95 tail. The mean
    # dilutes them below a benign-level threshold (it would WRONGLY accept the
    # dangerous rollout); the quantile preserves the spike and abstains.
    scores = np.array([0.1] * 28 + [5.0] * 2)
    s_mean = float(np.mean(scores))
    s_quantile = sequence_nonconformity(scores, beta=0.95)
    s_max = sequence_nonconformity(scores, beta=1.0)
    assert s_max == 5.0
    assert s_quantile > s_mean  # quantile keeps the spike; mean washes it out
    threshold = 0.5  # set just above the benign level
    assert s_mean < threshold  # mean -> diluted -> would (wrongly) accept
    assert s_quantile > threshold  # quantile -> spike preserved -> abstains


def test_lone_spike_below_tail_needs_max():
    # Honest design boundary: a single catastrophic step in a long trajectory
    # falls *below* the (1-beta)=5% tail, so the 0.95 quantile cannot see it.
    # Catching any single-step catastrophe is exactly what beta=1.0 (max) is for.
    scores = np.array([0.1] * 30 + [9.0])  # 1 spike in 31 (~3.2% < 5% tail)
    assert sequence_nonconformity(scores, beta=0.95) < 1.0  # quantile misses it
    assert sequence_nonconformity(scores, beta=1.0) == 9.0  # max catches it


@settings(max_examples=50, deadline=None)
@given(
    benign=st.floats(0.0, 0.5),
    spike=st.floats(5.0, 50.0),
    n=st.integers(20, 200),
)
def test_max_aggregation_never_below_mean(benign, spike, n):
    scores = np.array([benign] * n + [spike])
    assert sequence_nonconformity(scores, 1.0) >= float(np.mean(scores))


# --- fail-closed -------------------------------------------------------------- #


def test_all_undefined_sequence_is_inf():
    assert sequence_nonconformity(np.array([np.nan, np.nan]), 0.95) == float("inf")


def test_empty_sequence_is_inf():
    assert sequence_nonconformity(np.empty(0), 0.95) == float("inf")


def test_beta_out_of_range_rejected():
    with pytest.raises(ValueError):
        sequence_nonconformity(np.array([1.0]), 1.5)


# --- Mondrian + weighted ----------------------------------------------------- #


def test_mondrian_per_group_thresholds():
    grouped = {"a": np.arange(1, 101.0), "b": np.arange(1, 11.0)}
    qs = mondrian_quantiles(grouped, 0.1)
    assert set(qs) == {"a", "b"}
    assert qs["a"] == 91.0


def test_weighted_quantile_reduces_to_unweighted_when_uniform():
    s = np.arange(1, 101.0)
    w = np.ones_like(s)
    q_uniform = weighted_conformal_quantile(s, w, 0.1, test_weight=1.0)
    # with equal weights and a unit test point this tracks the standard threshold
    assert 88.0 <= q_uniform <= 94.0


def test_calibrate_returns_result(calibrated_ts):
    calib, test = split_calib_test(calibrated_ts, calib_frac=0.5, rng=0)
    res = calibrate(calib, test, alpha=0.1, mode="standard")
    assert 0.0 <= res.coverage <= 1.0
    assert res.n_calib + res.n_test == len(calibrated_ts)


def test_calibrate_mondrian_has_per_group(calibrated_ts):
    calib, test = split_calib_test(calibrated_ts, calib_frac=0.5, rng=0)
    res = calibrate(calib, test, alpha=0.2, mode="mondrian")
    assert len(res.per_group) >= 1


def test_calibrate_weighted_requires_weight_fn(calibrated_ts):
    calib, test = split_calib_test(calibrated_ts, calib_frac=0.5, rng=0)
    with pytest.raises(ValueError):
        calibrate(calib, test, mode="weighted")
    res = calibrate(calib, test, alpha=0.2, mode="weighted", weight_fn=lambda t: 1.0)
    assert res.weighted is True
