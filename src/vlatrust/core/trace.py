"""TraceSet serialization (JSON, never pickle) and per-trajectory splitting.

The conformal guarantee rests on exchangeability at the *trajectory* level, so
calibration/test splits are always drawn over whole traces — never over
individual steps. All randomness goes through an explicit, seeded
``numpy.random.Generator`` for byte-for-byte reproducibility (the seedloop
discipline).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .types import ConfidenceSource, Perturbation, Step, Trace, TraceSet

__all__ = [
    "traceset_to_dict",
    "traceset_from_dict",
    "save_traceset",
    "load_traceset",
    "split_calib_test",
]

SCHEMA_VERSION = 1


def _perturbation_to_dict(p: Perturbation | None) -> dict | None:
    if p is None:
        return None
    return {
        "name": p.name,
        "modality": p.modality,
        "intensity": p.intensity,
        "seed": p.seed,
        "reversible": p.reversible,
        "params": [[k, v] for k, v in p.params],
    }


def _perturbation_from_dict(d: dict | None) -> Perturbation | None:
    if d is None:
        return None
    return Perturbation(
        name=d["name"],
        modality=d["modality"],
        intensity=float(d.get("intensity", 0.0)),
        seed=int(d.get("seed", 0)),
        reversible=bool(d.get("reversible", True)),
        params=tuple((k, float(v)) for k, v in d.get("params", [])),
    )


def _step_to_dict(s: Step) -> dict:
    return {
        "action": s.action.tolist(),
        "confidence": s.confidence,
        "neg_log_prob": s.neg_log_prob,
        "action_residual": s.action_residual,
        "physically_valid": s.physically_valid,
    }


def _step_from_dict(d: dict) -> Step:
    return Step(
        action=np.asarray(d["action"], dtype=float),
        confidence=d.get("confidence"),
        neg_log_prob=d.get("neg_log_prob"),
        action_residual=d.get("action_residual"),
        physically_valid=bool(d.get("physically_valid", True)),
    )


def _trace_to_dict(t: Trace) -> dict:
    return {
        "trace_id": t.trace_id,
        "task": t.task,
        "success": t.success,
        "confidence_source": t.confidence_source.value,
        "perturbation": _perturbation_to_dict(t.perturbation),
        "steps": [_step_to_dict(s) for s in t.steps],
    }


def _trace_from_dict(d: dict) -> Trace:
    return Trace(
        steps=tuple(_step_from_dict(s) for s in d["steps"]),
        confidence_source=ConfidenceSource(d.get("confidence_source", "none")),
        success=d.get("success"),
        perturbation=_perturbation_from_dict(d.get("perturbation")),
        task=d.get("task"),
        trace_id=d.get("trace_id"),
    )


def traceset_to_dict(ts: TraceSet) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "traces": [_trace_to_dict(t) for t in ts.traces],
    }


def traceset_from_dict(d: dict) -> TraceSet:
    version = d.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported TraceSet schema_version {version!r}")
    return TraceSet(tuple(_trace_from_dict(t) for t in d["traces"]))


def save_traceset(ts: TraceSet, path: str | Path) -> None:
    Path(path).write_text(json.dumps(traceset_to_dict(ts), indent=2), encoding="utf-8")


def load_traceset(path: str | Path) -> TraceSet:
    return traceset_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def split_calib_test(
    ts: TraceSet,
    *,
    calib_frac: float = 0.5,
    rng: np.random.Generator | int | None = None,
) -> tuple[TraceSet, TraceSet]:
    """Partition traces (whole trajectories) into calibration and test sets.

    Exchangeability for split-conformal holds at the trajectory level, so the
    split never cuts within a trace. ``rng`` may be a ``Generator``, an int seed,
    or ``None`` (non-deterministic); pass a seed for reproducible splits.
    """
    if not (0.0 < calib_frac < 1.0):
        raise ValueError(f"calib_frac must be in (0, 1), got {calib_frac!r}")
    n = len(ts)
    if n < 2:
        raise ValueError(f"need >=2 traces to split, got {n}")
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    perm = generator.permutation(n)
    n_calib = max(1, min(n - 1, int(round(calib_frac * n))))
    calib_idx = sorted(perm[:n_calib].tolist())
    test_idx = sorted(perm[n_calib:].tolist())
    calib = TraceSet(tuple(ts.traces[i] for i in calib_idx))
    test = TraceSet(tuple(ts.traces[i] for i in test_idx))
    return calib, test
