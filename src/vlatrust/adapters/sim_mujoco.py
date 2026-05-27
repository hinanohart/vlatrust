"""MuJoCo re-rollout backend — placeholder for the renderer-dependent dims.

The three renderer-only perturbation dimensions (external force, distractor
objects, relighting) cannot be applied post-hoc to a recorded trace; they need a
simulator to re-render and re-roll the policy. That is a GPU-shaped, v0.1.1
feature. This stub exists so the :class:`SimBackend` surface is real and
``available()`` answers honestly (False), rather than the capability silently
not existing.
"""

from __future__ import annotations

from ..core.types import Perturbation, TraceSet

__all__ = ["MuJoCoSim"]


class MuJoCoSim:
    name = "mujoco"

    def available(self) -> bool:
        return False

    def rerollout(self, traces: TraceSet, perturbation: Perturbation, *, rng=None) -> TraceSet:  # noqa: ARG002
        raise NotImplementedError(
            "MuJoCo re-rollout (renderer-dependent perturbation dims: external_force, "
            "distractor_objects, relighting) is deferred to v0.1.1."
        )
