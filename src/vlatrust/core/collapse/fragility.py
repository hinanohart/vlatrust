r"""Fragility of a collapse curve.

Given success rate vs. intensity points, fragility summarises how fast the
policy collapses:

.. math:: F_m = 1 - \mathrm{AUC}_\tau\!\left(\frac{SR_m(\tau)}{SR_m(0)}\right) \in [0, 1]

computed as ``1 - (mean normalised success rate over the observed tau range)``
via the trapezoid rule. ``0`` = perfectly robust, ``1`` = total collapse.

A coarse mechanism tag is also returned:
  * ``"brittle-at-zero"`` — baseline success at tau=0 is already low (the policy
    barely works even unperturbed), so a fragility ratio is not meaningful;
  * ``"cliff"``           — a single large drop between adjacent intensities;
  * ``"gradual"``         — a steady decline with no cliff;
  * ``"robust"``          — little degradation across the range;
  * ``"insufficient"``    — fewer than two intensities to measure a slope.
"""

from __future__ import annotations

import numpy as np

from ..types import CollapsePoint

__all__ = ["fragility_from_points", "BASELINE_FLOOR", "CLIFF_DROP", "ROBUST_FRAGILITY"]

BASELINE_FLOOR = 0.5  # SR(0) below this => brittle-at-zero
CLIFF_DROP = 0.5  # normalised drop between adjacent taus that counts as a cliff
ROBUST_FRAGILITY = 0.15  # fragility below this => robust


def fragility_from_points(points: tuple[CollapsePoint, ...]) -> tuple[float, str]:
    """Return ``(fragility, mechanism)`` for collapse-curve points."""
    pts = sorted(points, key=lambda p: p.intensity)
    if len(pts) < 2:
        return float("nan"), "insufficient"

    tau = np.array([p.intensity for p in pts], dtype=float)
    sr = np.array([p.success_rate for p in pts], dtype=float)
    baseline = sr[0]

    if baseline <= BASELINE_FLOOR:
        return 1.0, "brittle-at-zero"

    norm = np.clip(sr / baseline, 0.0, 1.0)
    span = tau[-1] - tau[0]
    if span <= 0.0:
        return float("nan"), "insufficient"
    # trapezoid rule, computed directly for numpy 1.24..2.x portability
    area = float(np.sum(0.5 * (norm[:-1] + norm[1:]) * (tau[1:] - tau[:-1])))
    auc = area / span  # mean normalised SR over the observed range
    fragility = float(np.clip(1.0 - auc, 0.0, 1.0))

    drops = norm[:-1] - norm[1:]  # positive => decline between adjacent taus
    max_drop = float(np.max(drops)) if drops.size else 0.0

    if fragility < ROBUST_FRAGILITY:
        mechanism = "robust"
    elif max_drop >= CLIFF_DROP:
        mechanism = "cliff"
    else:
        mechanism = "gradual"
    return fragility, mechanism
