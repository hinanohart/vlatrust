"""ManiSkill re-rollout backend — placeholder for renderer-dependent dims (v0.1.1).

See :mod:`vlatrust.adapters.sim_mujoco` for the rationale; this is the same
deferred capability against the ManiSkill simulator.
"""

from __future__ import annotations

from ..core.types import Perturbation, TraceSet

__all__ = ["ManiSkillSim"]


class ManiSkillSim:
    name = "maniskill"

    def available(self) -> bool:
        return False

    def rerollout(self, traces: TraceSet, perturbation: Perturbation, *, rng=None) -> TraceSet:  # noqa: ARG002
        raise NotImplementedError(
            "ManiSkill re-rollout (renderer-dependent perturbation dims) is deferred to v0.1.1."
        )
