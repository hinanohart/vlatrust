r"""The headline Trust-Shift score.

Trust-Shift composes three axes through a multiplicative physics gate:

.. math::

   \text{Trust-Shift} = h \cdot \operatorname{blend}(\,T,\; C,\; R\,)

* :math:`h` — **hard physics gate**, the fraction of trajectories that are
  physically valid. A policy that commands joint-limit violations is pulled
  toward 0 no matter how confident it is (the foldconsensus multiplicative
  ``hard_valid -> Q=0`` pattern).
* :math:`T` — **confidence/competence tracking**: ``1 - mean_tau |C(tau) - SR(tau)|``.
  This is the claim made flesh — confidence must fall *in step with* success as
  shift rises. A policy that stays 95% confident while success goes 90% -> 0%
  earns a large, growing gap and a low :math:`T`.
* :math:`C` — **calibration** (inverse-Brier of confidence vs. success).
* :math:`R` — **retained reliability**: success rate among trajectories the
  abstention gate accepts. With the gate enabled a useful gate abstains on
  high-nonconformity rollouts, so :math:`R` (and thus the score) rises above the
  accept-everything baseline.

If the policy exposes no usable confidence (:data:`ConfidenceSource.NONE`) the
score is ``None`` — never fabricated — and the abstention axis reports N/A.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..collapse.curve import collapse_report
from ..conformal.nonconformity import DEFAULT_BETA
from ..conformal.predictor import calibrate
from ..reliability.calibrate import calibration_report
from ..reliability.gap import reliability_gap
from ..reliability.ood_gate import gate_traceset, hard_valid
from ..trace import split_calib_test
from ..types import ConfidenceSource, Scorecard, TraceSet
from .aggregate import blend

__all__ = ["score_traceset", "trust_shift"]


def _dominant_source(ts: TraceSet) -> ConfidenceSource:
    sources = [t.confidence_source for t in ts.traces]
    non_none = [s for s in sources if s != ConfidenceSource.NONE]
    if not non_none:
        return ConfidenceSource.NONE
    return Counter(non_none).most_common(1)[0][0]


def _hard_valid_factor(ts: TraceSet) -> float:
    if len(ts) == 0:
        return 0.0
    return float(np.mean([1.0 if hard_valid(t) else 0.0 for t in ts.traces]))


def _mean_confidence(trace) -> float | None:
    cs = [s.confidence for s in trace.steps if s.confidence is not None]
    return float(np.mean(cs)) if cs else None


def _tracking(ts: TraceSet) -> float:
    """``1 - mean_tau |mean confidence(tau) - success rate(tau)|`` in ``[0, 1]``."""
    gaps: list[float] = []
    for tau in ts.intensities:
        sub = ts.clean() if tau == 0.0 else ts.perturbed().at_intensity(tau)
        sr = sub.success_rate()
        confs = [c for c in (_mean_confidence(t) for t in sub) if c is not None]
        if sr is None or not confs:
            continue
        gaps.append(abs(float(np.mean(confs)) - sr))
    if not gaps:
        return float("nan")
    return float(np.clip(1.0 - float(np.mean(gaps)), 0.0, 1.0))


def _retained_reliability(test: TraceSet, decisions, *, enable_gate: bool) -> float:
    if enable_gate:
        labelled = [
            t
            for t, d in zip(test.traces, decisions, strict=True)
            if d.accept and t.success is not None
        ]
    else:
        labelled = [t for t in test.traces if t.success is not None]
    if not labelled:
        return float("nan")
    return float(np.mean([1.0 if t.success else 0.0 for t in labelled]))


def score_traceset(
    ts: TraceSet,
    *,
    alpha: float = 0.1,
    beta: float = DEFAULT_BETA,
    calib_ts: TraceSet | None = None,
    enable_gate: bool = True,
    n_boot: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Scorecard:
    """Full scorecard for a TraceSet. See module docstring for the composition."""
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    source = _dominant_source(ts)
    hv_factor = _hard_valid_factor(ts)

    fragility = collapse_report(ts, rng=gen)  # P1 wing

    # P2 wing: conformal calibration + abstention gate.
    if calib_ts is None and len(ts) >= 2:
        calib, test = split_calib_test(ts, calib_frac=0.5, rng=gen)
    else:
        calib, test = (calib_ts if calib_ts is not None else ts), ts
    conformal = calibrate(calib, test, alpha=alpha, beta=beta, mode="standard")
    decisions, _abst = gate_traceset(test, conformal.q_hat, beta=beta, enforce=enable_gate)
    retained = _retained_reliability(test, decisions, enable_gate=enable_gate)

    calibration = calibration_report(ts)

    # sim-vs-real style gap between clean (reference) and perturbed (target).
    clean, perturbed = ts.clean(), ts.perturbed()
    reliability = None
    if len(clean) and len(perturbed):
        reliability = reliability_gap(
            clean, perturbed, alpha=alpha, beta=beta, n_boot=n_boot, rng=gen
        )

    if source == ConfidenceSource.NONE or calibration is None:
        return Scorecard(
            confidence_source=ConfidenceSource.NONE,
            trust_shift=None,  # fail-closed: no confidence -> no claim
            hard_valid=(hv_factor >= 1.0),
            fragility=fragility,
            reliability=reliability,
            conformal=conformal,
            calibration=calibration,
            notes=("uncalibrated: no usable confidence; abstention axis = N/A",),
        )

    tracking = _tracking(ts)
    core = blend([tracking, calibration.inverse_brier, retained])
    score = float(np.clip(hv_factor * core, 0.0, 1.0)) if np.isfinite(core) else None

    notes = (
        f"tracking={tracking:.3f}",
        f"inverse_brier={calibration.inverse_brier:.3f}",
        f"retained_reliability={retained:.3f}",
        f"hard_valid_factor={hv_factor:.3f}",
        f"abstention_gate={'on' if enable_gate else 'off'}",
    )
    return Scorecard(
        confidence_source=source,
        trust_shift=score,
        hard_valid=(hv_factor >= 1.0),
        fragility=fragility,
        reliability=reliability,
        conformal=conformal,
        calibration=calibration,
        notes=notes,
    )


def trust_shift(ts: TraceSet, **kwargs) -> float | None:
    """Convenience wrapper returning just the Trust-Shift scalar (or None)."""
    return score_traceset(ts, **kwargs).trust_shift
