"""Deterministic synthetic TraceSet builders for tests.

These are NOT calibration claims about any real policy — they are controllable
ground-truth fixtures used to (a) golden-test the core math and (b) falsify the
Trust-Shift metric (a known confidently-collapsing policy must score low).

Two archetypes:
  * ``calibrated``    — success rate falls with perturbation intensity AND the
                        policy's confidence falls in step (graceful degradation).
  * ``overconfident`` — success rate falls identically, but confidence stays
                        high (the 90%->0% pathology). The metric must penalise it.

Everything is seeded; calling with the same seed yields byte-identical traces.
"""

from __future__ import annotations

import numpy as np

from vlatrust.core.types import ConfidenceSource, Perturbation, Step, Trace, TraceSet

ACTION_DIM = 7


def _make_trace(
    *,
    rng: np.random.Generator,
    success_prob: float,
    calibrated: bool,
    horizon: int,
    perturbation: Perturbation | None,
    trace_id: str,
) -> Trace:
    success = bool(rng.random() < success_prob)
    steps: list[Step] = []
    for _ in range(horizon):
        action = rng.normal(scale=0.1, size=ACTION_DIM)
        if calibrated:
            # confidence tracks the actual chance of success; a touch of noise.
            # Floor/cap keeps -log(conf) bounded so nonconformity is well behaved.
            conf = float(np.clip(success_prob + rng.normal(scale=0.03), 0.02, 0.98))
        else:
            # overconfident: confidence pinned high regardless of competence
            conf = float(np.clip(0.95 + rng.normal(scale=0.02), 0.02, 0.98))
        # nonconformity (residual) grows as competence drops, identically for
        # both archetypes — only the confidence signal differs between them.
        resid = float(abs(rng.normal(loc=(1.0 - success_prob), scale=0.05)))
        nlp = float(-np.log(conf))
        steps.append(
            Step(
                action=action,
                confidence=conf,
                neg_log_prob=nlp,
                action_residual=resid,
                physically_valid=True,
            )
        )
    return Trace(
        steps=tuple(steps),
        confidence_source=ConfidenceSource.TOKEN_ENTROPY,
        success=success,
        perturbation=perturbation,
        task="synthetic-pick-place",
        trace_id=trace_id,
    )


def make_collapse_traceset(
    *,
    calibrated: bool,
    modality: str = "sensor_noise",
    intensities: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    n_per_intensity: int = 40,
    horizon: int = 12,
    nominal_success: float = 0.95,
    seed: int = 0,
) -> TraceSet:
    """Build a TraceSet whose success rate collapses from ``nominal_success`` at
    tau=0 toward ~0 at tau=1, linearly in intensity."""
    rng = np.random.default_rng(seed)
    traces: list[Trace] = []
    for tau in intensities:
        success_prob = float(np.clip(nominal_success * (1.0 - tau), 0.0, 1.0))
        pert = (
            None
            if tau == 0.0
            else Perturbation(name=modality, modality=modality, intensity=tau, seed=seed)
        )
        for j in range(n_per_intensity):
            traces.append(
                _make_trace(
                    rng=rng,
                    success_prob=success_prob,
                    calibrated=calibrated,
                    horizon=horizon,
                    perturbation=pert,
                    trace_id=f"{modality}-t{tau:.2f}-{j}",
                )
            )
    return TraceSet(tuple(traces))


def make_no_confidence_traceset(seed: int = 0, n: int = 20, horizon: int = 8) -> TraceSet:
    """A TraceSet whose policy exposes no confidence (ConfidenceSource.NONE)."""
    rng = np.random.default_rng(seed)
    traces: list[Trace] = []
    for j in range(n):
        steps = tuple(Step(action=rng.normal(scale=0.1, size=ACTION_DIM)) for _ in range(horizon))
        traces.append(
            Trace(
                steps=steps,
                confidence_source=ConfidenceSource.NONE,
                success=bool(rng.random() < 0.8),
                trace_id=f"noconf-{j}",
            )
        )
    return TraceSet(tuple(traces))
