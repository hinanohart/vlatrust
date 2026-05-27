"""Action-space normalisation so nonconformity is comparable across backends.

Different VLA policies emit actions in different conventions (absolute vs delta,
per-dimension scale, gripper sign). The action-residual nonconformity score only
makes sense when actions live in a common frame, so a backend declares an
:class:`ActionSpaceSpec` and the recorded actions are mapped through it once.

This is pure numpy and deliberately minimal: a per-dimension affine map
``(a - bias) / scale`` plus optional per-dimension sign flips (e.g. a gripper
axis whose polarity is reversed between datasets).
"""

from __future__ import annotations

import dataclasses as dc

import numpy as np

from ..core.types import Trace, TraceSet

__all__ = ["ActionSpaceSpec", "normalize_action", "normalize_trace", "normalize_traceset"]


@dc.dataclass(frozen=True, slots=True)
class ActionSpaceSpec:
    """Per-dimension affine normalisation of an action vector.

    ``scale`` and ``bias`` broadcast against the action; ``sign`` (``+1``/``-1``
    per dim) flips polarity after the affine map. All default to identity.
    """

    scale: tuple[float, ...] | None = None
    bias: tuple[float, ...] | None = None
    sign: tuple[float, ...] | None = None

    def apply(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=float)
        bias = 0.0 if self.bias is None else np.asarray(self.bias, dtype=float)
        scale = 1.0 if self.scale is None else np.asarray(self.scale, dtype=float)
        sign = 1.0 if self.sign is None else np.asarray(self.sign, dtype=float)
        return sign * ((a - bias) / scale)


def normalize_action(action: np.ndarray, spec: ActionSpaceSpec) -> np.ndarray:
    return spec.apply(action)


def normalize_trace(trace: Trace, spec: ActionSpaceSpec) -> Trace:
    """Return ``trace`` with every step's action mapped through ``spec``."""
    new_steps = tuple(dc.replace(s, action=spec.apply(s.action)) for s in trace.steps)
    return dc.replace(trace, steps=new_steps)


def normalize_traceset(ts: TraceSet, spec: ActionSpaceSpec) -> TraceSet:
    return TraceSet(tuple(normalize_trace(t, spec) for t in ts.traces))
