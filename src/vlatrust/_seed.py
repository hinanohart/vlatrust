"""Deterministic seed derivation (the seedloop discipline, reused here).

A single root seed deterministically derives a tree of child seeds keyed by an
arbitrary path. This lets every perturbation, bootstrap resample, and rollout
draw from an independent, reproducible stream while only one integer is ever
configured by the user. Derivation uses BLAKE2b (stable across processes and
platforms) rather than Python's randomised ``hash()``.
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["derive_seed", "rng_for"]

_MASK64 = (1 << 64) - 1


def derive_seed(root: int, *path: object) -> int:
    """Return a stable 64-bit child seed for ``(root, *path)``.

    The same arguments always yield the same seed, in any process, on any OS.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(repr((int(root), tuple(map(repr, path)))).encode("utf-8"))
    return int.from_bytes(h.digest(), "little") & _MASK64


def rng_for(root: int, *path: object) -> np.random.Generator:
    """A numpy Generator seeded deterministically from ``(root, *path)``."""
    return np.random.default_rng(derive_seed(root, *path))
