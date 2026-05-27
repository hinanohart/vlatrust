"""Adapter contracts: how a TraceSet is *produced*.

The whole point of vlatrust's layering is that :mod:`vlatrust.core` only ever
sees :class:`~vlatrust.core.types.TraceSet`. Adapters are the *only* code that
touches a simulator, a policy's weights, or a recorded dataset, and they convert
those into TraceSets. Core never imports an adapter; adapters import core. Heavy
dependencies (torch, a simulator) are imported lazily *inside* a backend's
methods, so importing this module — and the whole core — stays torch-free.

Two backend kinds:

* :class:`PolicyBackend` — yields trajectories (rollout, recorded replay, or
  live inference). Carries an explicit :class:`ConfidenceSource` so the abstention
  axis knows whether the confidence signal is a real claim (TOKEN_ENTROPY),
  opt-in/non-claim (SAMPLING_VARIANCE), or absent (NONE → fail-closed N/A).
* :class:`SimBackend` — re-rolls trajectories under a perturbation that cannot be
  applied post-hoc to a recorded trace (the renderer-dependent dims). No concrete
  SimBackend ships in v0.1.0a1; the renderer dims raise ``NotImplementedError``.

``available()`` is the honesty hook: ``vlatrust doctor`` reports each backend's
real availability (e.g. a live policy whose weights/torch are absent), so a mock
run is never silently mistaken for a live one.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..core.types import ConfidenceSource, Perturbation, TraceSet

__all__ = ["PolicyBackend", "SimBackend"]


@runtime_checkable
class PolicyBackend(Protocol):
    """Produces a :class:`TraceSet` of trajectories.

    Implementations: :class:`~vlatrust.adapters.mock.MockPolicy` (deterministic
    synthetic, bundled), :class:`~vlatrust.adapters.recorded.RecordedBackend`
    (replays loaded traces, GPU-free, the default), and
    :class:`~vlatrust.adapters.policy_openvla.OpenVLABackend` (Tier-A
    token-entropy, lazy torch).
    """

    name: str
    confidence_source: ConfidenceSource

    def available(self) -> bool:
        """True iff this backend can actually run here (deps + weights present).

        ``vlatrust doctor`` surfaces this so mock output is never mistaken for a
        live policy run.
        """
        ...

    def rollout(self, n_episodes: int, *, rng: np.random.Generator | int | None = None) -> TraceSet:
        """Return up to ``n_episodes`` trajectories as a TraceSet."""
        ...


@runtime_checkable
class SimBackend(Protocol):
    """Re-rolls trajectories under a perturbation that needs a renderer/simulator.

    Used only for the renderer-dependent perturbation dims (external force,
    distractor objects, relighting). No concrete implementation ships in
    v0.1.0a1 — these dims raise ``NotImplementedError`` (deferred to v0.1.1).
    """

    name: str

    def available(self) -> bool: ...

    def rerollout(
        self,
        traces: TraceSet,
        perturbation: Perturbation,
        *,
        rng: np.random.Generator | int | None = None,
    ) -> TraceSet: ...
