"""Registry of the 14 trace-applicable perturbation ops, keyed by name.

Also names the 3 renderer-only dimensions that are *not* available in the core
(they need a simulator to re-render and ship in the sim adapters at v0.1.1).
Asking the registry for one of those raises ``NotImplementedError`` with a clear
pointer rather than silently returning nothing.
"""

from __future__ import annotations

from . import ops
from .algebra import Op

__all__ = [
    "REGISTRY",
    "TRACE_APPLICABLE",
    "RENDERER_REQUIRED",
    "get",
    "by_modality",
    "all_ops",
]


def _op(name: str, modality: str, fn, kind: str = "array") -> Op:
    return Op(name=name, modality=modality, fn=fn, payload_kind=kind)


REGISTRY: dict[str, Op] = {
    op.name: op
    for op in (
        _op("word_dropout", "language", ops.word_dropout, "text"),
        _op("word_shuffle", "language", ops.word_shuffle, "text"),
        _op("instruction_truncate", "language", ops.instruction_truncate, "text"),
        _op("gaussian_state_jitter", "init_state", ops.gaussian_state_jitter),
        _op("state_dropout", "init_state", ops.state_dropout),
        _op("gaussian_sensor_noise", "sensor_noise", ops.gaussian_sensor_noise),
        _op("sensor_dropout", "sensor_noise", ops.sensor_dropout),
        _op("brightness_shift", "sensor_noise", ops.brightness_shift),
        _op("salt_pepper", "sensor_noise", ops.salt_pepper),
        _op("action_bias", "actuation", ops.action_bias),
        _op("action_scale", "actuation", ops.action_scale),
        _op("latency_shift", "dynamics", ops.latency_shift),
        _op("step_dropout", "dynamics", ops.step_dropout),
        _op("translate_shift", "camera", ops.translate_shift),
    )
}

#: The 14 trace-applicable dimension names (CPU, no renderer).
TRACE_APPLICABLE: tuple[str, ...] = tuple(REGISTRY.keys())

#: The 3 renderer-only dimensions, deferred to the sim adapters (v0.1.1).
RENDERER_REQUIRED: tuple[str, ...] = (
    "external_force",
    "distractor_objects",
    "relighting",
)


def get(name: str) -> Op:
    if name in RENDERER_REQUIRED:
        raise NotImplementedError(
            f"perturbation {name!r} requires a renderer; it ships in the sim "
            "adapters at v0.1.1, not in the core injector."
        )
    try:
        return REGISTRY[name]
    except KeyError as e:
        raise KeyError(f"unknown perturbation {name!r}; known: {sorted(REGISTRY)}") from e


def by_modality(modality: str) -> tuple[Op, ...]:
    return tuple(op for op in REGISTRY.values() if op.modality == modality)


def all_ops() -> tuple[Op, ...]:
    return tuple(REGISTRY.values())
