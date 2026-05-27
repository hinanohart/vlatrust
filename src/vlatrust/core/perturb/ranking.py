"""Rank perturbation modalities by how fragile the policy is to each.

Deliberately *non-aggregating*: it returns the per-modality fragilities in order,
never a single hidden "robustness number". Weights, if a caller wants to
emphasise some modalities (e.g. weight a safety-critical sensor failure above a
cosmetic one), are supplied explicitly and shown alongside the raw value — the
caller owns that value judgement, the harness does not bury it in a default.
"""

from __future__ import annotations

import math

from ..types import FragilityReport

__all__ = ["rank_modalities"]


def rank_modalities(
    report: FragilityReport, *, weights: dict[str, float] | None = None
) -> tuple[tuple[str, float, float], ...]:
    """Return ``(modality, fragility, weighted_fragility)`` most-fragile first.

    Only modalities with a finite (measurable) fragility are included. With
    ``weights=None`` the weighted column equals the raw fragility (no emphasis).
    """
    w = weights or {}
    rows = [
        (c.modality, float(c.fragility), float(c.fragility) * float(w.get(c.modality, 1.0)))
        for c in report.curves
        if math.isfinite(c.fragility)
    ]
    return tuple(sorted(rows, key=lambda r: r[2], reverse=True))
