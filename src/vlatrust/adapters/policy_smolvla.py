"""Tier-B backend stub: SmolVLA (flow-matching) — sampling-variance, NON-claim.

Flow-matching policies (SmolVLA, pi0, GR00T) have **no native confidence**: they
do not emit a per-action token distribution. The only confidence one can derive
is the *variance across sampled action rollouts*, which is an opt-in, GPU-shaped,
and explicitly NON-claim signal (:data:`ConfidenceSource.SAMPLING_VARIANCE`).
That adapter is deferred to v0.1.1; this stub fixes the contract and reports its
unavailability honestly so a flow-matching policy is never silently treated as a
Tier-A token-confidence claim.
"""

from __future__ import annotations

from ..core.types import ConfidenceSource

__all__ = ["SmolVLABackend"]


class SmolVLABackend:
    name = "smolvla"
    confidence_source = ConfidenceSource.SAMPLING_VARIANCE  # opt-in, NON-claim

    def available(self) -> bool:
        return False

    def rollout(self, n_episodes: int, *, rng=None):  # noqa: ARG002
        raise NotImplementedError(
            "Tier-B sampling-variance adapter is deferred to v0.1.1. Flow-matching "
            "policies expose no native confidence, so this is a NON-claim signal "
            "(ConfidenceSource.SAMPLING_VARIANCE), not the Tier-A abstention claim."
        )
