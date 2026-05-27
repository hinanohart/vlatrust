"""Small statistical helpers shared by the reliability and collapse modules.

Percentile bootstrap confidence intervals only — no parametric assumptions. All
randomness is seeded so intervals are reproducible (R5: never claim a difference
whose CI straddles zero).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = ["bootstrap_ci", "two_sample_diff_ci"]


def _as_gen(rng: np.random.Generator | int | None) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def bootstrap_ci(
    values,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | int | None = None,
) -> tuple[float, float, float]:
    """Return ``(point, ci_low, ci_high)`` for ``statistic`` over ``values``.

    Returns NaNs for an empty input. The interval is the percentile bootstrap at
    level ``alpha`` (two-sided).
    """
    v = np.asarray(values, dtype=float)
    n = v.size
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(statistic(v))
    if n == 1:
        return (point, point, point)
    gen = _as_gen(rng)
    idx = gen.integers(0, n, size=(n_boot, n))
    resamples = v[idx]
    if statistic is np.mean:
        boot = resamples.mean(axis=1)
    else:
        boot = np.array([statistic(row) for row in resamples], dtype=float)
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return (point, lo, hi)


def two_sample_diff_ci(
    a,
    b,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | int | None = None,
) -> tuple[float, float, float]:
    """Bootstrap CI for ``statistic(a) - statistic(b)`` (independent resamples).

    Returns ``(point, ci_low, ci_high)``; NaNs if either sample is empty.
    """
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    if av.size == 0 or bv.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(statistic(av)) - float(statistic(bv))
    gen = _as_gen(rng)
    ia = gen.integers(0, av.size, size=(n_boot, av.size))
    ib = gen.integers(0, bv.size, size=(n_boot, bv.size))
    ra, rb = av[ia], bv[ib]
    if statistic is np.mean:
        boot = ra.mean(axis=1) - rb.mean(axis=1)
    else:
        boot = np.array([statistic(ra[i]) - statistic(rb[i]) for i in range(n_boot)], dtype=float)
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return (point, lo, hi)
