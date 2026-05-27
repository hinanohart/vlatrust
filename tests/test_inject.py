"""S4: perturbation injection, ranking, and the deferred sim/Tier-B stubs."""

from __future__ import annotations

import numpy as np
import pytest

from vlatrust.adapters.mock import MockPolicy
from vlatrust.adapters.policy_openpi import OpenPiBackend
from vlatrust.adapters.policy_smolvla import SmolVLABackend
from vlatrust.adapters.sim_maniskill import ManiSkillSim
from vlatrust.adapters.sim_mujoco import MuJoCoSim
from vlatrust.core.collapse.curve import collapse_report
from vlatrust.core.perturb.inject import (
    intensity_sweep_shift,
    perturb_payloads,
    shift_magnitude,
)
from vlatrust.core.perturb.ranking import rank_modalities
from vlatrust.core.types import ConfidenceSource


def _obs(n=6, dim=5, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=dim) for _ in range(n)]


# --- injection -------------------------------------------------------------- #


def test_inject_identity_at_zero_intensity():
    payloads = _obs()
    out = perturb_payloads(payloads, "gaussian_sensor_noise", 0.0, root_seed=7)
    for a, b in zip(payloads, out, strict=True):
        assert np.array_equal(a, b)


def test_inject_is_reproducible():
    payloads = _obs()
    a = perturb_payloads(payloads, "gaussian_sensor_noise", 0.5, root_seed=7)
    b = perturb_payloads(payloads, "gaussian_sensor_noise", 0.5, root_seed=7)
    c = perturb_payloads(payloads, "gaussian_sensor_noise", 0.5, root_seed=8)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c, strict=True))


def test_inject_text_payloads():
    out = perturb_payloads(["pick up the red block"] * 4, "word_dropout", 0.0, root_seed=1)
    assert out == ["pick up the red block"] * 4  # identity at tau=0


def test_inject_renderer_dim_fails_closed():
    with pytest.raises(NotImplementedError):
        perturb_payloads(_obs(), "relighting", 0.5, root_seed=0)


def test_inject_unknown_perturbation_raises():
    with pytest.raises(KeyError):
        perturb_payloads(_obs(), "no_such_dim", 0.5, root_seed=0)


# --- sweep: injection produces graded, zero-at-clean input shift ------------ #


def test_intensity_sweep_shift_is_zero_at_clean_and_monotone():
    payloads = _obs(n=8, dim=6, seed=3)
    sweep = intensity_sweep_shift(
        payloads, "gaussian_sensor_noise", (0.0, 0.25, 0.5, 0.75, 1.0), root_seed=11
    )
    shifts = [s for _, s in sweep]
    assert shifts[0] == 0.0  # identity at tau=0
    assert shifts == sorted(shifts)  # non-decreasing distribution shift
    assert shifts[-1] > shifts[0]  # genuinely moved


def test_shift_magnitude_zero_for_identical():
    p = _obs()
    assert shift_magnitude(p, p) == 0.0


# --- injected -> collapse (output) while clean stays stable ----------------- #


def test_injected_collapse_declines_while_clean_is_stable():
    ts = MockPolicy(calibrated=True, nominal_success=0.95).collapse_traceset(rng=0)
    rep = collapse_report(ts, rng=0)
    curve = rep.curves[0]
    # clean baseline (tau=0) is near nominal; the highest intensity is far below.
    sr0 = curve.points[0].success_rate
    sr_last = curve.points[-1].success_rate
    assert sr0 > 0.8  # clean -> stable, high success
    assert sr_last < sr0  # injected -> collapse
    assert curve.fragility > 0.0


# --- ranking (non-aggregating) ---------------------------------------------- #


def test_rank_modalities_orders_by_fragility_and_drops_nonfinite():
    from vlatrust.core.types import CollapseCurve, FragilityReport

    rep = FragilityReport(
        curves=(
            CollapseCurve("sensor_noise", (), 0.8, "cliff"),
            CollapseCurve("language", (), 0.2, "gradual"),
            CollapseCurve("dynamics", (), float("nan"), "insufficient"),
        ),
        most_fragile_modality="sensor_noise",
        mean_fragility=0.5,
    )
    ranked = rank_modalities(rep)
    assert [r[0] for r in ranked] == ["sensor_noise", "language"]  # nan dropped
    assert ranked[0][2] == 0.8  # weighted == raw when no weights


def test_rank_modalities_respects_explicit_weights():
    from vlatrust.core.types import CollapseCurve, FragilityReport

    rep = FragilityReport(
        curves=(
            CollapseCurve("sensor_noise", (), 0.8, "cliff"),
            CollapseCurve("language", (), 0.2, "gradual"),
        ),
        most_fragile_modality="sensor_noise",
        mean_fragility=0.5,
    )
    ranked = rank_modalities(rep, weights={"language": 10.0})
    assert ranked[0][0] == "language"  # 0.2*10 = 2.0 outranks 0.8*1


# --- deferred stubs report honestly + fail closed --------------------------- #


def test_sim_backends_unavailable_and_not_implemented():
    for sim in (MuJoCoSim(), ManiSkillSim()):
        assert sim.available() is False
        with pytest.raises(NotImplementedError):
            sim.rerollout(None, None)


def test_tier_b_backends_are_non_claim_and_deferred():
    for be in (SmolVLABackend(), OpenPiBackend()):
        assert be.available() is False
        assert be.confidence_source == ConfidenceSource.SAMPLING_VARIANCE
        with pytest.raises(NotImplementedError):
            be.rollout(1)
