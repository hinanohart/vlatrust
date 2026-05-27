#!/usr/bin/env python3
"""S6 measurement: run the falsification suite and emit an env-stamped record.

This produces ``bench_results/v0.1.0a1_falsification.json``, the single source of
every number that appears in the README (no hand-writing).

HONESTY BOUNDARY (read this before citing any number here):
  * ``mode`` is ``"synthetic"``. The inputs are the deterministic ``MockPolicy``
    archetypes, NOT recorded real-robot traces. These numbers validate that the
    *metric* behaves correctly (a confidently-collapsing policy scores below a
    gracefully-degrading one; abstention helps; a physics violation zeroes the
    score) — they are NOT an empirical claim about any real VLA policy.
  * ``gate_real`` is ``"deferred-v0.1.1"``. Reproducing high-nominal-success then
    collapse-under-perturbation on a *recorded real trace* requires either live
    OpenVLA inference (GPU; out of the a1 CPU scope) or a permissively-licensed
    graded-perturbation benchmark (LIBERO-plus has no license → G-LIB blocks it).
    No real-collapse number is fabricated to fill that gap.

The deterministic Tier-A confidence extractor and the conformal coverage are
exact-math checks (no policy needed) and are recorded as ``real-math`` evidence.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vlatrust.adapters.mock import MockPolicy  # noqa: E402
from vlatrust.adapters.policy_openvla import step_confidence_from_logits  # noqa: E402
from vlatrust.core._stats import bootstrap_ci  # noqa: E402
from vlatrust.core.conformal.split import conformal_quantile  # noqa: E402
from vlatrust.core.score.scorecard import score_traceset  # noqa: E402

SEED = 0
OUT = ROOT / "bench_results" / "v0.1.0a1_falsification.json"


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _env_stamp() -> dict:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "seed": SEED,
    }


def _falsification() -> dict:
    """Synthetic falsification: calibrated must out-score overconfident, with CIs."""
    results = {}
    for alpha in (0.1, 0.25):
        cal = MockPolicy(calibrated=True).collapse_traceset(rng=SEED)
        over = MockPolicy(calibrated=False).collapse_traceset(rng=SEED)
        s_cal = score_traceset(cal, alpha=alpha, rng=SEED)
        s_over = score_traceset(over, alpha=alpha, rng=SEED)
        # CI over the per-trajectory success contributions is not the metric's CI;
        # we report the headline scalars plus the gate on/off contrast.
        on = score_traceset(cal, alpha=alpha, enable_gate=True, rng=SEED).trust_shift
        off = score_traceset(cal, alpha=alpha, enable_gate=False, rng=SEED).trust_shift
        results[f"alpha_{alpha}"] = {
            "calibrated_trust_shift": s_cal.trust_shift,
            "overconfident_trust_shift": s_over.trust_shift,
            "calibrated_beats_overconfident": s_cal.trust_shift > s_over.trust_shift,
            "abstention_on": on,
            "abstention_off": off,
            "abstention_helps": on > off,
        }
    # physics gate: every trajectory invalid => hard zero.
    import dataclasses as dc

    cal = MockPolicy(calibrated=True).collapse_traceset(rng=SEED)
    bad = []
    for t in cal.traces:
        if t.steps:
            s0 = dc.replace(t.steps[0], physically_valid=False)
            t = dc.replace(t, steps=(s0, *t.steps[1:]))
        bad.append(t)
    from vlatrust.core.types import TraceSet

    sc_bad = score_traceset(TraceSet(tuple(bad)), rng=SEED)
    results["physics_gate"] = {
        "all_invalid_trust_shift": sc_bad.trust_shift,
        "hard_valid_flag": sc_bad.hard_valid,
        "zeroed": sc_bad.trust_shift == 0.0,
    }
    return results


def _real_math() -> dict:
    """Exact-math checks that need no policy and no synthetic behaviour."""
    # Tier-A token-entropy extractor on exact logits (peaked vs uniform).
    peaked = np.full((7, 256), -10.0)
    chosen = np.arange(7) % 256
    peaked[np.arange(7), chosen] = 10.0
    conf_peak, nlp_peak = step_confidence_from_logits(peaked, chosen)
    conf_unif, nlp_unif = step_confidence_from_logits(np.zeros((7, 256)), np.zeros(7, dtype=int))

    # conformal marginal coverage on iid samples (averaged over many splits).
    rng = np.random.default_rng(SEED)
    covs = []
    for _ in range(300):
        s = rng.normal(size=400)
        q = conformal_quantile(s[:200], 0.1)
        covs.append(float(np.mean(s[200:] <= q)))
    point, lo, hi = bootstrap_ci(np.asarray(covs), n_boot=2000, alpha=0.05, rng=SEED)
    return {
        "token_entropy": {
            "peaked_confidence": conf_peak,
            "peaked_neg_log_prob": nlp_peak,
            "uniform_confidence": conf_unif,
            "uniform_neg_log_prob": nlp_unif,
            "uniform_matches_1_over_256": bool(abs(conf_unif - 1 / 256) < 1e-6),
        },
        "conformal_marginal_coverage": {
            "nominal": 0.9,
            "mean": point,
            "ci95": [lo, hi],
            "meets_guarantee": bool(point >= 0.9 - 0.02),
        },
    }


def main() -> int:
    record = {
        "project": "vlatrust",
        "version": "0.1.0a1",
        "mode": "synthetic",
        "disclaimer": (
            "Inputs are deterministic MockPolicy archetypes, NOT recorded real-robot "
            "traces. These numbers validate the METRIC's behaviour, not any real VLA "
            "policy. Empirical real-trace collapse validation is deferred to v0.1.1."
        ),
        "gate_real": {
            "status": "deferred-v0.1.1",
            "reason": (
                "Reproducing high-success->collapse on a recorded real trace needs live "
                "OpenVLA inference (GPU, out of a1 CPU scope) or a permissively-licensed "
                "graded-perturbation benchmark; LIBERO-plus has no license (G-LIB blocks it)."
            ),
        },
        "env": _env_stamp(),
        "falsification_synthetic": _falsification(),
        "real_math": _real_math(),
    }
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, allow_nan=False)
    print(f"wrote {OUT.relative_to(ROOT)}")
    fc = record["falsification_synthetic"]
    print(
        "falsification (synthetic): "
        f"alpha0.1 cal={fc['alpha_0.1']['calibrated_trust_shift']:.3f} > "
        f"over={fc['alpha_0.1']['overconfident_trust_shift']:.3f}; "
        f"physics-zeroed={fc['physics_gate']['zeroed']}; "
        f"gate-real={record['gate_real']['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
