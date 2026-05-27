"""Inject a perturbation into a sequence of observation payloads.

This is the trace-applicable injection path: given the per-step observation
payloads of a trajectory (text instruction, sensor array, state vector, …) it
applies a registry perturbation at a chosen intensity, seeded so the result is
byte-reproducible and exactly the identity at ``intensity = 0``. Asking for a
renderer-only dimension fails closed (``NotImplementedError`` from the registry).

What injection does *not* do is invent an outcome: perturbing observations
changes the policy's *inputs*: the success label must come from re-rolling the
policy (a live/sim backend) or from a recorded re-evaluation. The collapse curve
that results is measured by :mod:`vlatrust.core.collapse`, never fabricated here.
:func:`intensity_sweep_shift` quantifies the input-space shift the injection
actually produces (monotone in intensity, zero when clean) so a sweep can be
validated without a policy.
"""

from __future__ import annotations

import numpy as np

from .algebra import apply_op
from .registry import get

__all__ = ["perturb_payloads", "shift_magnitude", "intensity_sweep_shift"]


def perturb_payloads(payloads, name: str, intensity: float, *, root_seed: int) -> list:
    """Apply perturbation ``name`` to every payload (seeded, identity at tau=0).

    Each payload gets a child seed derived from its position, so the whole
    sequence is reproducible yet not identically perturbed step-to-step.
    """
    op = get(name)  # NotImplementedError for renderer dims; KeyError if unknown
    return [
        apply_op(op, p, intensity, root_seed=root_seed, position=i) for i, p in enumerate(payloads)
    ]


def shift_magnitude(clean, perturbed) -> float:
    """Mean L2 deviation between clean and perturbed array payloads.

    A scalar summary of how far injection moved the observations; 0 means the
    perturbation was the identity. Only defined for array payloads.
    """
    diffs = []
    for c, p in zip(clean, perturbed, strict=True):
        c_arr = np.asarray(c, dtype=float)
        p_arr = np.asarray(p, dtype=float)
        diffs.append(float(np.linalg.norm((p_arr - c_arr).ravel())))
    return float(np.mean(diffs)) if diffs else 0.0


def intensity_sweep_shift(
    payloads, name: str, intensities, *, root_seed: int
) -> list[tuple[float, float]]:
    """``[(intensity, mean shift)]`` for an injection sweep.

    The shift is 0 at ``intensity = 0`` (identity) and rises with intensity — the
    input-space evidence that injection produces graded distribution shift,
    measurable without running any policy.
    """
    out: list[tuple[float, float]] = []
    for tau in intensities:
        perturbed = perturb_payloads(payloads, name, tau, root_seed=root_seed)
        out.append((float(tau), shift_magnitude(payloads, perturbed)))
    return out
