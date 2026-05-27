"""S2: bootstrap CI backbone (_stats) — the R5 substrate behind every claim.

Every ``*_claimable`` flag in the harness rests on these two helpers, so they
get golden, property, determinism, and degenerate-input (fail-closed) tests.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from vlatrust.core._stats import bootstrap_ci, two_sample_diff_ci

# --- bootstrap_ci ----------------------------------------------------------- #


def test_bootstrap_ci_constant_is_exact():
    point, lo, hi = bootstrap_ci(np.full(100, 5.0), rng=0)
    assert point == 5.0 and lo == 5.0 and hi == 5.0


def test_bootstrap_ci_empty_is_nan():
    assert all(np.isnan(x) for x in bootstrap_ci([], rng=0))


def test_bootstrap_ci_singleton_is_degenerate_point():
    assert bootstrap_ci([3.0], rng=0) == (3.0, 3.0, 3.0)


@settings(max_examples=60, deadline=None)
@given(
    vals=st.lists(st.floats(-50, 50, allow_nan=False), min_size=2, max_size=80),
    seed=st.integers(0, 5000),
)
def test_bootstrap_ci_brackets_point(vals, seed):
    point, lo, hi = bootstrap_ci(vals, rng=seed)
    assert lo <= point <= hi


def test_bootstrap_ci_wider_alpha_is_narrower_interval():
    rng_vals = np.random.default_rng(0).normal(size=300)
    _, lo01, hi01 = bootstrap_ci(rng_vals, alpha=0.01, rng=1)
    _, lo20, hi20 = bootstrap_ci(rng_vals, alpha=0.20, rng=1)
    assert (hi20 - lo20) < (hi01 - lo01)  # less coverage => tighter interval


def test_bootstrap_ci_is_deterministic():
    v = np.random.default_rng(2).normal(size=50)
    assert bootstrap_ci(v, rng=7) == bootstrap_ci(v, rng=7)


def test_bootstrap_ci_supports_non_mean_statistic():
    point, lo, hi = bootstrap_ci(np.arange(101.0), statistic=np.median, rng=0)
    assert point == 50.0 and lo <= 50.0 <= hi


# --- two_sample_diff_ci ----------------------------------------------------- #


def _excludes_zero(ci: tuple[float, float]) -> bool:
    return ci[0] > 0.0 or ci[1] < 0.0


def test_diff_ci_separated_means_excludes_zero():
    a = np.ones(200)
    b = np.zeros(200)
    point, lo, hi = two_sample_diff_ci(a, b, n_boot=2000, rng=0)
    assert np.isclose(point, 1.0)
    assert _excludes_zero((lo, hi))  # a genuine, claimable difference


def test_diff_ci_identical_samples_straddles_zero():
    v = np.random.default_rng(0).normal(size=200)
    point, lo, hi = two_sample_diff_ci(v, v.copy(), n_boot=2000, rng=0)
    assert np.isclose(point, 0.0)
    assert not _excludes_zero((lo, hi))  # no claim from identical data


def test_diff_ci_empty_is_nan():
    assert all(np.isnan(x) for x in two_sample_diff_ci([], [1.0, 2.0], rng=0))
    assert all(np.isnan(x) for x in two_sample_diff_ci([1.0, 2.0], [], rng=0))


def test_diff_ci_singleton_side_fails_closed():
    # The defect Monitor-C caught: two singletons gave a zero-width CI that
    # excludes zero => a "claimable" difference from one observation per side.
    # Now the CI is NaN (point reported, but never claimable).
    point, lo, hi = two_sample_diff_ci([1.0], [0.0], rng=0)
    assert point == 1.0
    assert np.isnan(lo) and np.isnan(hi)
    assert not _excludes_zero((lo, hi))


def test_diff_ci_one_singleton_side_fails_closed():
    _point, lo, hi = two_sample_diff_ci([1.0], [0.0, 0.0, 0.0, 0.0], rng=0)
    assert np.isnan(lo) and np.isnan(hi)
    assert not _excludes_zero((lo, hi))


def test_diff_ci_is_deterministic():
    a = np.random.default_rng(1).normal(size=40)
    b = np.random.default_rng(2).normal(size=40)
    assert two_sample_diff_ci(a, b, rng=5) == two_sample_diff_ci(a, b, rng=5)
