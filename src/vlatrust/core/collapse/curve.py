"""Collapse curves: success rate vs. perturbation intensity, per modality.

The clean (tau=0) success rate is the shared baseline for every modality. Each
intensity's success rate carries a bootstrap CI over trajectories, so a decline
is only read as real when the intervals separate (R5).
"""

from __future__ import annotations

import numpy as np

from .._stats import bootstrap_ci
from ..types import CollapseCurve, CollapsePoint, FragilityReport, TraceSet
from .fragility import fragility_from_points

__all__ = ["collapse_curve", "collapse_report"]


def _success_array(ts: TraceSet) -> np.ndarray:
    return np.array(
        [1.0 if t.success else 0.0 for t in ts.traces if t.success is not None],
        dtype=float,
    )


def _point(intensity: float, succ: np.ndarray, *, alpha, n_boot, rng) -> CollapsePoint | None:
    if succ.size == 0:
        return None
    sr, lo, hi = bootstrap_ci(succ, n_boot=n_boot, alpha=alpha, rng=rng)
    return CollapsePoint(
        intensity=intensity, success_rate=sr, ci_low=lo, ci_high=hi, n=int(succ.size)
    )


def collapse_curve(
    ts: TraceSet,
    modality: str,
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    rng: np.random.Generator | int | None = None,
) -> CollapseCurve:
    """Collapse curve for one modality (clean baseline + that modality's taus)."""
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    points: list[CollapsePoint] = []

    clean_pt = _point(0.0, _success_array(ts.clean()), alpha=alpha, n_boot=n_boot, rng=gen)
    if clean_pt is not None:
        points.append(clean_pt)

    mod_ts = ts.by_modality(modality)
    for tau in mod_ts.intensities:
        if tau == 0.0:
            continue  # baseline already taken from clean()
        succ = _success_array(mod_ts.at_intensity(tau))
        pt = _point(tau, succ, alpha=alpha, n_boot=n_boot, rng=gen)
        if pt is not None:
            points.append(pt)

    pts = tuple(sorted(points, key=lambda p: p.intensity))
    fragility, mechanism = fragility_from_points(pts)
    return CollapseCurve(modality=modality, points=pts, fragility=fragility, mechanism=mechanism)


def collapse_report(
    ts: TraceSet,
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    rng: np.random.Generator | int | None = None,
) -> FragilityReport:
    """Collapse curves for every observed modality, plus the most fragile one."""
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    curves = tuple(
        collapse_curve(ts, m, alpha=alpha, n_boot=n_boot, rng=gen) for m in ts.modalities
    )
    measurable = [c.fragility for c in curves if np.isfinite(c.fragility)]
    mean_fragility = float(np.mean(measurable)) if measurable else float("nan")
    most_fragile = None
    finite_curves = [c for c in curves if np.isfinite(c.fragility)]
    if finite_curves:
        most_fragile = max(finite_curves, key=lambda c: c.fragility).modality
    return FragilityReport(
        curves=curves, most_fragile_modality=most_fragile, mean_fragility=mean_fragility
    )
