"""Tier-B backend stub: openpi / pi0 (flow-matching) — sampling-variance, NON-claim.

Same rationale as :mod:`vlatrust.adapters.policy_smolvla`: flow-matching, no
native confidence, sampling-variance only (opt-in, NON-claim), deferred to
v0.1.1. The openpi weights (Gemma-based) are ToU-licensed and are *not* bundled;
the live adapter would fetch them, never redistribute.
"""

from __future__ import annotations

from ..core.types import ConfidenceSource

__all__ = ["OpenPiBackend"]


class OpenPiBackend:
    name = "openpi"
    confidence_source = ConfidenceSource.SAMPLING_VARIANCE  # opt-in, NON-claim

    def available(self) -> bool:
        return False

    def rollout(self, n_episodes: int, *, rng=None):  # noqa: ARG002
        raise NotImplementedError(
            "Tier-B sampling-variance adapter is deferred to v0.1.1. openpi/pi0 is "
            "flow-matching (no native confidence); weights are ToU-licensed and not "
            "bundled (fetched, never redistributed)."
        )
