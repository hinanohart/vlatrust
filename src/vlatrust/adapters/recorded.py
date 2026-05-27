"""The default, GPU-free backend: replay already-recorded trajectories.

A :class:`RecordedBackend` wraps a :class:`TraceSet` that was loaded from disk
(``core.trace.load_traceset``) or ingested from a LeRobot dataset
(:mod:`vlatrust.adapters.lerobot_dataset`). "Rollout" here is replay: it returns
the recorded trajectories unchanged. This is the path that needs no policy
weights, no torch, and no simulator — the one exercised in CI and the default for
post-hoc perturbation analysis.

The backend reports the confidence source of the traces it holds (so an ingested
dataset with no per-step confidence correctly surfaces as
:data:`ConfidenceSource.NONE` → abstention axis N/A).
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..core.trace import load_traceset
from ..core.types import ConfidenceSource, TraceSet

__all__ = ["RecordedBackend"]


class RecordedBackend:
    """Replays a recorded :class:`TraceSet`."""

    name = "recorded"

    def __init__(self, traces: TraceSet, *, name: str = "recorded") -> None:
        self._traces = traces
        self.name = name
        self.confidence_source = _dominant_source(traces)

    @classmethod
    def from_json(cls, path, *, name: str = "recorded") -> RecordedBackend:
        """Load traces from a vlatrust JSON file (never pickle)."""
        return cls(load_traceset(path), name=name)

    def available(self) -> bool:
        """Recorded replay is always runnable; it only reads in-memory traces."""
        return len(self._traces) > 0

    def rollout(self, n_episodes: int, *, rng: np.random.Generator | int | None = None) -> TraceSet:
        """Return up to ``n_episodes`` recorded trajectories (replay, no sampling)."""
        if n_episodes >= len(self._traces):
            return self._traces
        return TraceSet(self._traces.traces[:n_episodes])

    @property
    def traces(self) -> TraceSet:
        return self._traces


def _dominant_source(ts: TraceSet) -> ConfidenceSource:
    sources = [t.confidence_source for t in ts.traces]
    non_none = [s for s in sources if s != ConfidenceSource.NONE]
    if not non_none:
        return ConfidenceSource.NONE
    return Counter(non_none).most_common(1)[0][0]
