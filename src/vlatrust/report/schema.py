"""JSON Schema (draft-07) for the serialized Scorecard, plus a dep-free check.

The scorecard's ``to_dict()`` is the machine-readable contract. We publish a
draft-07 schema for external/cross-language consumers, and ship a small
pure-python structural validator so the contract is enforced in CI without
adding ``jsonschema`` as a dependency. Non-finite floats serialize to ``null``
(see ``core.types._to_jsonable``), so every numeric field is ``["number", "null"]``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["SCORECARD_SCHEMA", "validate_scorecard", "SchemaError"]

_NUM = {"type": ["number", "null"]}
_NUM_PAIR = {"type": ["array", "null"], "items": {"type": "number"}, "minItems": 2, "maxItems": 2}

SCORECARD_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "vlatrust Scorecard",
    "type": "object",
    "required": ["confidence_source", "trust_shift", "hard_valid", "notes"],
    "additionalProperties": True,
    "properties": {
        "confidence_source": {"enum": ["token_entropy", "sampling_variance", "none"]},
        # null is the honest, designed value when no usable confidence exists.
        "trust_shift": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0},
        "hard_valid": {"type": "boolean"},
        "notes": {"type": "array", "items": {"type": "string"}},
        "conformal": {
            "type": ["object", "null"],
            "properties": {
                "alpha": _NUM,
                "q_hat": _NUM,  # may be null = +inf fail-closed sentinel
                "coverage": _NUM,
                "abstention_rate": _NUM,
                "beta": _NUM,
                "weighted": {"type": "boolean"},
            },
        },
        "calibration": {
            "type": ["object", "null"],
            "properties": {"ece": _NUM, "inverse_brier": _NUM, "n": {"type": "integer"}},
        },
        "reliability": {
            "type": ["object", "null"],
            "properties": {
                "delta_succ": _NUM,
                "delta_succ_ci": _NUM_PAIR,
                "delta_cov": _NUM,
                "delta_cov_ci": _NUM_PAIR,
            },
        },
        "fragility": {
            "type": ["object", "null"],
            "properties": {
                "most_fragile_modality": {"type": ["string", "null"]},
                "mean_fragility": _NUM,
                "curves": {"type": "array"},
            },
        },
    },
}


class SchemaError(ValueError):
    """Raised when a scorecard dict violates the published contract."""


def validate_scorecard(d: dict) -> None:
    """Structural check of a scorecard dict against the contract (raises on error).

    Pure python — no ``jsonschema`` dependency. Checks the load-bearing
    invariants: required keys, the confidence-source enum, and that
    ``trust_shift`` is either ``null`` or a number in ``[0, 1]`` (never a
    fabricated out-of-range or non-finite value).
    """
    if not isinstance(d, dict):
        raise SchemaError(f"scorecard must be an object, got {type(d).__name__}")
    for key in ("confidence_source", "trust_shift", "hard_valid", "notes"):
        if key not in d:
            raise SchemaError(f"missing required key {key!r}")
    if d["confidence_source"] not in ("token_entropy", "sampling_variance", "none"):
        raise SchemaError(f"bad confidence_source {d['confidence_source']!r}")
    ts = d["trust_shift"]
    if ts is not None:
        if not isinstance(ts, (int, float)) or isinstance(ts, bool):
            raise SchemaError(f"trust_shift must be number or null, got {ts!r}")
        if not (0.0 <= float(ts) <= 1.0):
            raise SchemaError(f"trust_shift out of [0,1]: {ts!r}")
    if not isinstance(d["hard_valid"], bool):
        raise SchemaError("hard_valid must be boolean")
    if not isinstance(d["notes"], list):
        raise SchemaError("notes must be an array")
