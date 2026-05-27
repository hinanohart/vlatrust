"""S2: reliability gap — Delta_succ and the net-new Delta_cov (sim->real).

Delta_cov is the harness's named net-new metric: the coverage a reference-fitted
conformal threshold actually delivers on a shifted target, minus its nominal
1-alpha. A negative value is the load-bearing signal "safe in reference,
dangerous in target". A gap is only *claimable* when its bootstrap CI excludes 0.
"""

from __future__ import annotations

import synthetic

from vlatrust.core.reliability.gap import reliability_gap
from vlatrust.core.types import TraceSet


def test_degraded_target_has_positive_succ_gap_and_negative_cov_gap(calibrated_ts):
    ref = calibrated_ts.clean()  # high success, in-distribution
    tgt = calibrated_ts.perturbed()  # success collapsed under shift
    g = reliability_gap(ref, tgt, alpha=0.1, n_boot=2000, rng=0)

    # SR_ref - SR_tgt > 0 (reference is more reliable) and the CI excludes 0.
    assert g.delta_succ is not None and g.delta_succ > 0.0
    assert g.succ_claimable

    # the reference-fitted conformal threshold UNDER-covers the shifted target:
    # "safe in reference, dangerous in target" => delta_cov < 0, claimable.
    assert g.delta_cov is not None and g.delta_cov < 0.0
    assert g.cov_claimable


def test_identical_distributions_make_no_claim():
    # ref == target: both gaps ~0 and neither CI excludes 0 (R5 -> no claim).
    ts = synthetic.make_collapse_traceset(calibrated=True, seed=3).clean()
    g = reliability_gap(ts, ts, alpha=0.1, n_boot=2000, rng=0)

    assert g.delta_succ is not None and abs(g.delta_succ) < 0.05
    assert not g.succ_claimable
    # delta_cov on identical data ~ 0 (only the conformal ceil's small upward
    # bias); the CI must still straddle 0 so no coverage claim is made.
    assert g.delta_cov is not None and abs(g.delta_cov) < 0.1
    assert not g.cov_claimable


def test_reliability_gap_is_deterministic(calibrated_ts):
    a = reliability_gap(calibrated_ts.clean(), calibrated_ts.perturbed(), rng=0)
    b = reliability_gap(calibrated_ts.clean(), calibrated_ts.perturbed(), rng=0)
    assert a.delta_succ == b.delta_succ
    assert a.delta_cov == b.delta_cov
    assert a.delta_succ_ci == b.delta_succ_ci
    assert a.delta_cov_ci == b.delta_cov_ci


def test_empty_reference_fails_closed_to_none(calibrated_ts):
    # no reference traces => no SR gap and no fittable threshold => both None,
    # never a fabricated gap.
    g = reliability_gap(TraceSet(()), calibrated_ts.perturbed(), rng=0)
    assert g.delta_succ is None and g.delta_succ_ci is None
    assert g.delta_cov is None and g.delta_cov_ci is None
    assert not g.succ_claimable and not g.cov_claimable
