"""The 14 post-hoc perturbation operations (the trace-applicable injector).

Each op has the signature ``op(payload, intensity, rng) -> payload`` where
``intensity`` (tau) is in ``[0, 1]`` and ``rng`` is a seeded
``numpy.random.Generator``. Two invariants every op obeys:

* **identity at zero** — ``op(x, 0.0, rng) == x`` exactly (the algebra's ``id``);
* **purity** — the input is never mutated (recorded traces are immutable).

Payloads are primitive observation pieces: ``str`` for language, ``np.ndarray``
for state / sensor / action / sequence data. These are exactly the dimensions
applicable to recorded data with no renderer; the three renderer-only dimensions
(external force, distractor objects, relighting) live in the sim adapters and
raise ``NotImplementedError`` until v0.1.1.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "word_dropout",
    "word_shuffle",
    "instruction_truncate",
    "gaussian_state_jitter",
    "state_dropout",
    "gaussian_sensor_noise",
    "sensor_dropout",
    "brightness_shift",
    "salt_pepper",
    "action_bias",
    "action_scale",
    "latency_shift",
    "step_dropout",
    "translate_shift",
]


def _arr(x) -> np.ndarray:
    return np.array(x, dtype=float, copy=True)


# --- language (str) --------------------------------------------------------- #


def word_dropout(text: str, intensity: float, rng: np.random.Generator) -> str:
    if intensity <= 0.0:
        return text
    words = text.split()
    if not words:
        return text
    keep = rng.random(len(words)) >= intensity
    kept = [w for w, k in zip(words, keep, strict=True) if k]
    return " ".join(kept) if kept else words[0]


def word_shuffle(text: str, intensity: float, rng: np.random.Generator) -> str:
    if intensity <= 0.0:
        return text
    words = text.split()
    n = len(words)
    k = int(round(intensity * n))
    if n < 2 or k < 2:
        return text
    idx = rng.choice(n, size=k, replace=False)
    perm = rng.permutation(idx)
    out = list(words)
    for dst, src in zip(idx, perm, strict=True):
        out[dst] = words[src]
    return " ".join(out)


def instruction_truncate(text: str, intensity: float, rng: np.random.Generator) -> str:
    if intensity <= 0.0:
        return text
    words = text.split()
    if not words:
        return text
    keep = max(1, int(round((1.0 - intensity) * len(words))))
    return " ".join(words[:keep])


# --- state / init (np.ndarray) ---------------------------------------------- #


def gaussian_state_jitter(
    x, intensity: float, rng: np.random.Generator, *, scale: float = 1.0
) -> np.ndarray:
    a = _arr(x)
    if intensity <= 0.0:
        return a
    return a + rng.normal(0.0, intensity * scale, size=a.shape)


def state_dropout(x, intensity: float, rng: np.random.Generator) -> np.ndarray:
    a = _arr(x)
    if intensity <= 0.0:
        return a
    mask = rng.random(a.shape) < intensity
    a[mask] = 0.0
    return a


# --- sensor / vision array (np.ndarray) ------------------------------------- #


def gaussian_sensor_noise(
    x, intensity: float, rng: np.random.Generator, *, scale: float = 1.0
) -> np.ndarray:
    a = _arr(x)
    if intensity <= 0.0:
        return a
    return a + rng.normal(0.0, intensity * scale, size=a.shape)


def sensor_dropout(x, intensity: float, rng: np.random.Generator) -> np.ndarray:
    a = _arr(x)
    if intensity <= 0.0:
        return a
    mask = rng.random(a.shape) < intensity
    a[mask] = 0.0
    return a


def brightness_shift(
    x, intensity: float, rng: np.random.Generator, *, scale: float = 1.0
) -> np.ndarray:
    a = _arr(x)
    if intensity <= 0.0:
        return a
    sign = 1.0 if rng.random() < 0.5 else -1.0
    return a + sign * intensity * scale


def salt_pepper(x, intensity: float, rng: np.random.Generator) -> np.ndarray:
    a = _arr(x)
    if intensity <= 0.0 or a.size == 0:
        return a
    mask = rng.random(a.shape) < intensity
    lo, hi = float(np.min(a)), float(np.max(a))
    extremes = np.where(rng.random(a.shape) < 0.5, lo, hi)
    a[mask] = extremes[mask]
    return a


# --- actuation (np.ndarray) ------------------------------------------------- #


def action_bias(x, intensity: float, rng: np.random.Generator, *, scale: float = 0.5) -> np.ndarray:
    a = _arr(x)
    if intensity <= 0.0:
        return a
    bias = rng.normal(0.0, intensity * scale, size=a.shape[-1:])
    return a + bias


def action_scale(x, intensity: float, rng: np.random.Generator, *, k: float = 0.5) -> np.ndarray:
    a = _arr(x)
    if intensity <= 0.0:
        return a
    return a * (1.0 + intensity * k)


# --- dynamics (sequence, axis 0 = time) ------------------------------------- #


def latency_shift(
    x, intensity: float, rng: np.random.Generator, *, max_shift: int = 3
) -> np.ndarray:
    a = _arr(x)
    if intensity <= 0.0 or a.shape[0] < 2:
        return a
    shift = int(round(intensity * max_shift))
    if shift <= 0:
        return a
    out = np.roll(a, shift, axis=0)
    out[:shift] = a[0]  # hold the first frame rather than wrapping
    return out


def step_dropout(x, intensity: float, rng: np.random.Generator) -> np.ndarray:
    a = _arr(x)
    if intensity <= 0.0 or a.shape[0] < 2:
        return a
    drop = rng.random(a.shape[0]) < intensity
    out = a.copy()
    last = a[0]
    for t in range(a.shape[0]):
        if drop[t]:
            out[t] = last
        else:
            last = a[t]
    return out


# --- camera (spatial roll on the last axis) --------------------------------- #


def translate_shift(
    x, intensity: float, rng: np.random.Generator, *, max_shift: int = 4
) -> np.ndarray:
    a = _arr(x)
    if intensity <= 0.0:
        return a
    shift = int(round(intensity * max_shift))
    if shift <= 0:
        return a
    return np.roll(a, shift, axis=-1)
