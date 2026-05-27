r"""Perturbation algebra: identity, seeded application, and composition.

An operation is a map ``P : (payload, seed) -> payload`` parameterised by an
intensity ``tau in [0, 1]``. The algebra guarantees:

* **identity** — :data:`IDENTITY` (and any op at ``tau = 0``) returns its input
  unchanged: ``id . P = P . id = P``.
* **reproducibility** — applying the same op (or pipeline) with the same root
  seed yields byte-identical output, because each stage draws from a child seed
  derived from ``(root_seed, name, intensity, position)``.
* **composition** — :class:`Pipeline` chains ops left to right, each on its own
  derived seed, so order is explicit and reseeding is structural.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..._seed import rng_for

__all__ = ["Op", "IDENTITY", "apply_op", "Pipeline", "compose"]

PayloadFn = Callable[..., object]


@dataclass(frozen=True, slots=True)
class Op:
    """A named perturbation operation with its modality and payload kind."""

    name: str
    modality: str
    fn: PayloadFn
    payload_kind: str  # "text" | "array"

    def __call__(self, payload, intensity, rng):
        return self.fn(payload, intensity, rng)


def _identity_fn(payload, intensity, rng):  # noqa: ARG001 - uniform op signature
    return payload


IDENTITY = Op(name="identity", modality="none", fn=_identity_fn, payload_kind="any")


def apply_op(op: Op, payload, intensity: float, *, root_seed: int, position: int = 0):
    """Apply one op on a seed derived from ``(root_seed, name, intensity, position)``."""
    if not (0.0 <= intensity <= 1.0):
        raise ValueError(f"intensity must be in [0, 1], got {intensity!r}")
    rng = rng_for(root_seed, op.name, intensity, position)
    return op.fn(payload, intensity, rng)


@dataclass(frozen=True, slots=True)
class Pipeline:
    """An ordered composition of ops applied left to right."""

    ops: tuple[Op, ...]

    def apply(
        self,
        payload,
        intensity: float | Sequence[float],
        *,
        root_seed: int,
    ):
        """Run the pipeline. ``intensity`` is a scalar (shared) or one per op."""
        if isinstance(intensity, (int, float)):
            taus = [float(intensity)] * len(self.ops)
        else:
            taus = [float(x) for x in intensity]
            if len(taus) != len(self.ops):
                raise ValueError(f"expected {len(self.ops)} intensities, got {len(taus)}")
        out = payload
        for position, (op, tau) in enumerate(zip(self.ops, taus, strict=True)):
            out = apply_op(op, out, tau, root_seed=root_seed, position=position)
        return out

    def __len__(self) -> int:
        return len(self.ops)


def compose(*ops: Op) -> Pipeline:
    """Compose ops into a :class:`Pipeline` (empty compose is the identity)."""
    return Pipeline(ops=tuple(ops) if ops else (IDENTITY,))
