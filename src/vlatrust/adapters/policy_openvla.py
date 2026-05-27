r"""Tier-A backend: OpenVLA token-entropy confidence (the CLAIM-bearing adapter).

OpenVLA is autoregressive — it discretises each action dimension into 256 bins
and predicts one token per dimension — so it exposes a *native* confidence: the
probability the model assigned to the action token it actually emitted. That is
the only confidence source vlatrust treats as a full claim
(:data:`ConfidenceSource.TOKEN_ENTROPY`); flow-matching policies (pi0, SmolVLA,
GR00T) have no such native signal and are handled, opt-in and non-claim, by a
separate sampling-variance adapter (deferred to v0.1.1).

Two layers, split on purpose:

* :func:`step_confidence_from_logits` / :func:`token_logprob` — **pure numpy**,
  no torch. Given the per-dimension action-token logits and the chosen token
  ids, they compute the step confidence and ``neg_log_prob``. This is the actual
  Tier-A math and is unit-tested in CI with no model present.
* :class:`OpenVLABackend` — loads the 7B model via ``transformers`` (**lazy
  import**) and drives it. Heavy and GPU-shaped; ``available()`` reports honestly
  whether it can run here. The live env-stepping rollout is deferred to v0.1.1;
  the confidence extractor above is what users with logits can call today.
"""

from __future__ import annotations

import importlib.util

import numpy as np

from ..core.types import ConfidenceSource

__all__ = ["token_logprob", "step_confidence_from_logits", "OpenVLABackend"]


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=-1, keepdims=True)
    return z - np.log(np.sum(np.exp(z), axis=-1, keepdims=True))


def token_logprob(logits: np.ndarray, chosen: int) -> float:
    r"""``log p(chosen)`` under a softmax over a single token's ``logits``."""
    lp = _log_softmax(np.asarray(logits, dtype=float))
    return float(lp[int(chosen)])


def step_confidence_from_logits(
    logits_per_dim: np.ndarray, chosen_per_dim: np.ndarray
) -> tuple[float, float]:
    r"""Step confidence and ``neg_log_prob`` from OpenVLA action-token logits.

    ``logits_per_dim`` has shape ``(action_dim, vocab)``; ``chosen_per_dim`` the
    emitted token id per dimension. Returns ``(confidence, neg_log_prob)`` where

    * ``confidence`` is the mean probability the model put on the tokens it chose
      (mean over action dims), in ``[0, 1]`` — high when the policy is decisive;
    * ``neg_log_prob`` is ``-mean_d log p(chosen_d)`` — the per-step nonconformity
      score (0 when perfectly confident, growing as the policy hesitates).

    Both are derived from the same softmax, so they move together by construction
    — confidence falling exactly as neg_log_prob rises.
    """
    logits = np.asarray(logits_per_dim, dtype=float)
    chosen = np.asarray(chosen_per_dim, dtype=int)
    if logits.ndim != 2 or chosen.ndim != 1 or logits.shape[0] != chosen.shape[0]:
        raise ValueError("logits_per_dim must be (action_dim, vocab) matching chosen_per_dim")
    lp = _log_softmax(logits)  # (action_dim, vocab)
    chosen_lp = lp[np.arange(chosen.shape[0]), chosen]  # (action_dim,)
    confidence = float(np.mean(np.exp(chosen_lp)))
    neg_log_prob = float(-np.mean(chosen_lp))
    return confidence, neg_log_prob


class OpenVLABackend:
    """Live OpenVLA policy (Tier-A). Heavy: lazy ``transformers``/torch import."""

    name = "openvla"
    confidence_source = ConfidenceSource.TOKEN_ENTROPY

    def __init__(self, model_id: str = "openvla/openvla-7b", *, device: str = "cuda") -> None:
        self.model_id = model_id
        self.device = device

    def available(self) -> bool:
        """True only if torch + transformers are importable (weights still need
        downloading on first use). Reported honestly by ``vlatrust doctor`` so a
        mock run is never mistaken for a live OpenVLA run."""
        return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))

    def rollout(self, n_episodes: int, *, rng=None):  # noqa: ARG002
        """Live env-stepping rollout is deferred to v0.1.1.

        v0.1.0a1 ships the Tier-A *confidence extractor*
        (:func:`step_confidence_from_logits`), which is the CLAIM-bearing math and
        is CI-tested without a model. Driving the 7B policy through an environment
        needs a simulator/real robot and GPU, which are out of the a1 CPU scope.
        """
        if not self.available():
            raise RuntimeError(
                "OpenVLA backend unavailable: install `vlatrust[openvla]` (torch + "
                "transformers) and a CUDA device. The Tier-A confidence math is in "
                "step_confidence_from_logits() and needs no model."
            )
        raise NotImplementedError(
            "Live OpenVLA env rollout is deferred to v0.1.1; use a recorded TraceSet "
            "via RecordedBackend, or call step_confidence_from_logits() on your own logits."
        )
