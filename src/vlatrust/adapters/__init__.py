"""Backends that *produce* TraceSets (the only code that touches torch/sims/data).

Core never imports this package; heavy deps are imported lazily inside backend
methods, so ``import vlatrust.adapters`` stays torch-free.
"""

from __future__ import annotations

from .base import PolicyBackend, SimBackend
from .mock import MockPolicy
from .normalize import ActionSpaceSpec, normalize_traceset
from .recorded import RecordedBackend

__all__ = [
    "PolicyBackend",
    "SimBackend",
    "MockPolicy",
    "RecordedBackend",
    "ActionSpaceSpec",
    "normalize_traceset",
    "doctor_report",
]


def doctor_report() -> list[dict]:
    """Honest availability of every known backend (live vs mock vs unavailable).

    Importing the Tier-A backend must not drag torch in, so it is imported lazily
    here. Each entry is ``{name, kind, confidence_source, available, note}``; the
    CLI ``vlatrust doctor`` renders it so a mock run is never read as a live one.
    """
    from .policy_openvla import OpenVLABackend  # lazy: keeps torch out of import

    mock = MockPolicy()
    openvla = OpenVLABackend()
    return [
        {
            "name": mock.name,
            "kind": "mock",
            "confidence_source": mock.confidence_source.value,
            "available": mock.available(),
            "note": "deterministic synthetic; NOT a real policy",
        },
        {
            "name": "recorded",
            "kind": "recorded",
            "confidence_source": "depends-on-data",
            "available": True,
            "note": "replays a loaded/ingested TraceSet (GPU-free, default)",
        },
        {
            "name": openvla.name,
            "kind": "policy/tier-a",
            "confidence_source": openvla.confidence_source.value,
            "available": openvla.available(),
            "note": "live rollout deferred to v0.1.1; token-entropy math is CI-tested",
        },
    ]
