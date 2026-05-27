#!/usr/bin/env python3
"""Per-step completion verifier for the vlatrust autonomous build.

Usage:
    python scripts/verify_step.py <STEP> [--dry-run]

Each step maps to a list of concrete, non-empty assertions (file existence,
import smoke, symbol presence, optional pytest selection). Exit 0 == the step's
artifacts are present and importable; non-zero == not complete (resume there).

`--dry-run` performs the same read-only checks but never runs pytest (used by
the /compact resume procedure to decide where to restart without side effects).

This script is itself audited at S8 (it must contain real assertions, not pass).
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _file(rel: str) -> tuple[bool, str]:
    p = ROOT / rel
    return p.exists() and p.stat().st_size > 0, f"file non-empty: {rel}"


def _safe_import(mod: str):
    # Only first-party modules are ever verified here; reject anything else so
    # the dynamic import cannot be steered to arbitrary code.
    if not (mod == "vlatrust" or mod.startswith("vlatrust.")):
        raise ValueError(f"refusing non-vlatrust import target: {mod!r}")
    return importlib.import_module(mod)  # nosemgrep


def _imports(mod: str) -> tuple[bool, str]:
    try:
        _safe_import(mod)
        return True, f"import {mod}"
    except Exception as e:  # noqa: BLE001
        return False, f"import {mod} -> {type(e).__name__}: {e}"


def _has_symbol(mod: str, name: str) -> tuple[bool, str]:
    try:
        m = _safe_import(mod)
        ok = hasattr(m, name)
        return ok, f"{mod}.{name} present" + ("" if ok else " (MISSING)")
    except Exception as e:  # noqa: BLE001
        return False, f"{mod}.{name} -> {type(e).__name__}: {e}"


def _pytest(expr: str, dry: bool) -> tuple[bool, str]:
    if dry:
        return True, f"pytest -k {expr!r} (skipped: --dry-run)"
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-k", expr],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]
    return r.returncode == 0, f"pytest -k {expr!r} -> {tail[0]}"


def checks(step: str, dry: bool) -> list[tuple[bool, str]]:
    s = step.upper()
    out: list[tuple[bool, str]] = []
    if s in ("S0", "S0.5"):
        out += [
            _file("pyproject.toml"),
            _file("LICENSE"),
            _file("NOTICE"),
            _file("README.md"),
            _file(".gitignore"),
            _file(".github/workflows/ci.yml"),
            _file("scripts/verify_step.py"),
            _imports("vlatrust"),
            _has_symbol("vlatrust", "__version__"),
        ]
    if s == "S1":
        out += [
            _file("src/vlatrust/core/types.py"),
            _imports("vlatrust.core.types"),
            _has_symbol("vlatrust.core.types", "TraceSet"),
            _has_symbol("vlatrust.core.types", "ConfidenceSource"),
            _has_symbol("vlatrust.core.types", "Perturbation"),
            _has_symbol("vlatrust.core.types", "Scorecard"),
        ]
    if s == "S2":
        out += [
            _imports("vlatrust.core.perturb.ops"),
            _imports("vlatrust.core.conformal.predictor"),
            _imports("vlatrust.core.reliability.gap"),
            _imports("vlatrust.core.reliability.ood_gate"),
            _imports("vlatrust.core.collapse.curve"),
            _imports("vlatrust.core.score.scorecard"),
            _pytest("core or conformal or perturb or reliability or collapse or score", dry),
        ]
    if s == "S3":
        out += [
            _imports("vlatrust.adapters.base"),
            _imports("vlatrust.adapters.mock"),
            _has_symbol("vlatrust.adapters.base", "PolicyBackend"),
        ]
    if s == "S4":
        out += [
            _imports("vlatrust.core.perturb.inject"),
            _imports("vlatrust.core.perturb.ranking"),
            _imports("vlatrust.adapters.sim_mujoco"),
            _has_symbol("vlatrust.core.perturb.inject", "perturb_payloads"),
            _has_symbol("vlatrust.core.perturb.ranking", "rank_modalities"),
            _pytest("inject or perturb or collapse", dry),
        ]
    if s == "S5":
        out += [
            _imports("vlatrust.report.cli"),
            _has_symbol("vlatrust.report.cli", "main"),
        ]
    if s == "S6":
        out += [
            _file("scripts/measure_falsification.py"),
            _file("bench_results/v0.1.0a1_falsification.json"),
            _pytest("measurement or ingest", dry),
        ]
    if not out:
        out.append((False, f"unknown step {step!r}"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    results = checks(args.step, args.dry_run)
    all_ok = True
    for ok, desc in results:
        print(f"  [{'OK ' if ok else 'FAIL'}] {desc}")
        all_ok = all_ok and ok
    print(f"{args.step}: {'COMPLETE' if all_ok else 'INCOMPLETE'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
