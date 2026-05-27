"""A deterministic, dependency-free mock policy backend.

This is a *real, shippable* backend (not a test fixture): it lets ``vlatrust``
run end-to-end — CLI, report, falsification demo — with no torch, no simulator,
and no weights, so the calibration math can be exercised on every machine and in
CI. It is also the falsification harness: it can synthesise both a
gracefully-degrading ("calibrated") and a confidently-collapsing
("overconfident") policy, and the Trust-Shift metric must score the first higher
than the second.

It is *labelled* mock everywhere (``name="mock"``, ``confidence_source`` chosen
explicitly) so ``vlatrust doctor`` never lets mock output pass for a live run.
Everything is seeded: the same seed yields byte-identical traces.
"""

from __future__ import annotations

import numpy as np

from ..core.types import ConfidenceSource, Perturbation, Step, Trace, TraceSet

__all__ = ["MockPolicy"]

ACTION_DIM = 7


class MockPolicy:
    """Deterministic synthetic policy.

    Parameters
    ----------
    calibrated:
        If True, confidence tracks the true success probability (graceful
        degradation). If False, confidence stays pinned high regardless of
        competence — the 90%->0% overconfidence pathology the metric must catch.
    confidence_source:
        The source tag the produced traces carry. ``NONE`` yields traces with no
        per-step confidence (the abstention axis must then report N/A).
    nominal_success:
        Clean (tau=0) success probability.
    """

    name = "mock"

    def __init__(
        self,
        *,
        calibrated: bool = True,
        confidence_source: ConfidenceSource = ConfidenceSource.TOKEN_ENTROPY,
        nominal_success: float = 0.95,
        horizon: int = 12,
    ) -> None:
        self.calibrated = calibrated
        self.confidence_source = confidence_source
        self.nominal_success = float(nominal_success)
        self.horizon = int(horizon)

    def available(self) -> bool:
        """A mock is always runnable (that is its whole point)."""
        return True

    # -- single trajectory ---------------------------------------------------

    def _trace(
        self,
        rng: np.random.Generator,
        *,
        success_prob: float,
        perturbation: Perturbation | None,
        trace_id: str,
    ) -> Trace:
        success = bool(rng.random() < success_prob)
        steps: list[Step] = []
        has_conf = self.confidence_source != ConfidenceSource.NONE
        for _ in range(self.horizon):
            action = rng.normal(scale=0.1, size=ACTION_DIM)
            conf: float | None = None
            nlp: float | None = None
            if has_conf:
                if self.calibrated:
                    conf = float(np.clip(success_prob + rng.normal(scale=0.03), 0.02, 0.98))
                else:
                    conf = float(np.clip(0.95 + rng.normal(scale=0.02), 0.02, 0.98))
                nlp = float(-np.log(conf))
            # residual grows as competence drops, identically for both archetypes
            # — only the confidence signal distinguishes them.
            resid = float(abs(rng.normal(loc=(1.0 - success_prob), scale=0.05)))
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
            confidence_source=self.confidence_source,
            success=success,
            perturbation=perturbation,
            task="mock-pick-place",
            trace_id=trace_id,
        )

    # -- PolicyBackend protocol ---------------------------------------------

    def rollout(self, n_episodes: int, *, rng: np.random.Generator | int | None = None) -> TraceSet:
        """``n_episodes`` clean (unperturbed) trajectories at nominal success."""
        gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        traces = tuple(
            self._trace(
                gen, success_prob=self.nominal_success, perturbation=None, trace_id=f"mock-{j}"
            )
            for j in range(n_episodes)
        )
        return TraceSet(traces)

    # -- falsification / demo helper (beyond the protocol) -------------------

    def collapse_traceset(
        self,
        *,
        modality: str = "sensor_noise",
        intensities: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
        n_per_intensity: int = 40,
        rng: np.random.Generator | int | None = None,
    ) -> TraceSet:
        """A graded TraceSet whose success rate collapses linearly from
        ``nominal_success`` at tau=0 toward ~0 at tau=1 — the falsification input."""
        gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        traces: list[Trace] = []
        for tau in intensities:
            success_prob = float(np.clip(self.nominal_success * (1.0 - tau), 0.0, 1.0))
            pert = (
                None
                if tau == 0.0
                else Perturbation(name=modality, modality=modality, intensity=tau, seed=0)
            )
            for j in range(n_per_intensity):
                traces.append(
                    self._trace(
                        gen,
                        success_prob=success_prob,
                        perturbation=pert,
                        trace_id=f"mock-{modality}-t{tau:.2f}-{j}",
                    )
                )
        return TraceSet(tuple(traces))
