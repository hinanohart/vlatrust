"""``vlatrust`` command-line interface (stdlib argparse, zero extra deps).

Three commands:

* ``vlatrust doctor`` — honest backend availability (live vs mock vs unavailable).
* ``vlatrust score`` — the Trust-Shift scorecard for a TraceSet, optionally
  written as JSON and/or a self-contained HTML report.
* ``vlatrust calibrate`` — calibration of confidence vs success; prints
  "uncalibrated" (never a fabricated number) when no usable confidence exists.

Honest eval mode: a TraceSet with no success labels cannot yield a reliability
gap, and one with no usable confidence yields ``trust_shift = null`` — both are
reported plainly rather than papered over. ``--mock`` input is always labelled as
illustrative (not a measurement) in stdout and in the HTML disclaimer.
"""

from __future__ import annotations

import argparse
import json
import sys

from ..adapters import doctor_report
from ..adapters.mock import MockPolicy
from ..core.reliability.calibrate import calibration_report
from ..core.score.scorecard import score_traceset
from ..core.trace import load_traceset
from ..core.types import TraceSet
from .html import scorecard_to_html
from .schema import validate_scorecard

__all__ = ["main", "build_parser"]

_MOCK_DISCLAIMER = (
    "Input is a deterministic MOCK TraceSet — illustrative, not an empirical measurement."
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vlatrust", description="Calibration-under-shift trust harness for VLA policies."
    )
    sub = p.add_subparsers(dest="command", required=True)

    p.add_argument("--version", action="store_true", help=argparse.SUPPRESS)

    sub.add_parser("doctor", help="report backend availability (live vs mock)")

    def _add_input(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("traces", nargs="?", help="path to a vlatrust traces JSON file")
        sp.add_argument(
            "--mock",
            action="store_true",
            help="use a deterministic mock collapse TraceSet (illustrative)",
        )
        sp.add_argument(
            "--overconfident",
            action="store_true",
            help="with --mock, generate the overconfident archetype",
        )
        sp.add_argument("--seed", type=int, default=0)

    ps = sub.add_parser("score", help="compute the Trust-Shift scorecard")
    _add_input(ps)
    ps.add_argument("--alpha", type=float, default=0.1)
    ps.add_argument(
        "--no-gate", action="store_true", help="measure abstention without enforcing it"
    )
    ps.add_argument("--out", help="write the scorecard JSON to this path")
    ps.add_argument("--html", help="write a self-contained HTML report to this path")

    pc = sub.add_parser("calibrate", help="report calibration of confidence vs success")
    _add_input(pc)
    return p


def _load(args) -> tuple[TraceSet, str | None]:
    """Return (traceset, disclaimer). Disclaimer is set for mock input."""
    if getattr(args, "mock", False):
        ts = MockPolicy(calibrated=not args.overconfident).collapse_traceset(rng=args.seed)
        return ts, _MOCK_DISCLAIMER
    if not args.traces:
        raise SystemExit("error: provide a traces JSON path or --mock")
    return load_traceset(args.traces), None


def _cmd_doctor() -> int:
    print("vlatrust backends:")
    for r in doctor_report():
        flag = "available" if r["available"] else "unavailable"
        print(f"  - {r['name']:<10} [{r['kind']}] {flag}: {r['note']}")
    return 0


def _cmd_score(args) -> int:
    ts, disclaimer = _load(args)
    if ts.success_rate() is None:
        print("note: no success labels present — reliability gap and success-based axes are N/A")
    sc = score_traceset(ts, alpha=args.alpha, enable_gate=not args.no_gate, rng=args.seed)
    d = sc.to_dict()
    validate_scorecard(d)  # enforce the published JSON contract before emitting

    if disclaimer:
        print(f"** {disclaimer}")
    ts_val = "N/A (no usable confidence)" if sc.trust_shift is None else f"{sc.trust_shift:.3f}"
    print(
        f"Trust-Shift: {ts_val}  (source={sc.confidence_source.value}, physically_valid={sc.hard_valid})"
    )
    for n in sc.notes:
        print(f"  {n}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=2, allow_nan=False)
        print(f"wrote scorecard JSON -> {args.out}")
    if args.html:
        with open(args.html, "w", encoding="utf-8") as fh:
            fh.write(scorecard_to_html(sc, disclaimer=disclaimer))
        print(f"wrote HTML report -> {args.html}")
    return 0


def _cmd_calibrate(args) -> int:
    ts, disclaimer = _load(args)
    if disclaimer:
        print(f"** {disclaimer}")
    rep = calibration_report(ts)
    if rep is None:
        src = "none" if not ts.traces else ts.traces[0].confidence_source.value
        print(f"uncalibrated: no usable confidence (source={src}); abstention axis = N/A")
        return 0
    print(f"ECE={rep.ece:.4f}  inverse_brier={rep.inverse_brier:.4f}  n={rep.n}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    # allow `vlatrust --version` without a subcommand
    if argv is None:
        argv = sys.argv[1:]
    if "--version" in argv and len(argv) == 1:
        from .. import __version__

        print(__version__)
        return 0
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _cmd_doctor()
    if args.command == "score":
        return _cmd_score(args)
    if args.command == "calibrate":
        return _cmd_calibrate(args)
    return 2  # unreachable: subparser is required


if __name__ == "__main__":
    raise SystemExit(main())
