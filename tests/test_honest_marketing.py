"""S7: the honest-marketing firewall actually runs and actually trips.

These tests guard against the firewall silently degrading into a no-op (the
ship-and-yank failure mode). They run the real shell script: the live tree must
pass, and the planted-bad-fixture self-test must confirm the detector trips.
Skipped where bash is unavailable (the CI honest-marketing job runs on ubuntu).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "honest_marketing_check.sh"

# The shell firewall is authoritatively run by the ubuntu honest-marketing CI job.
# Invoking it from pytest is a convenience; skip where a POSIX bash is unreliable
# (Windows git-bash mktemp/trap differ). The pure-python README test runs anywhere.
_needs_bash = pytest.mark.skipif(
    shutil.which("bash") is None or sys.platform == "win32",
    reason="POSIX bash required",
)


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=ROOT, capture_output=True, text=True)


@_needs_bash
def test_live_tree_passes_firewall():
    r = _run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "honest-marketing: OK" in r.stdout


@_needs_bash
def test_selftest_proves_detector_trips():
    # --selftest plants a synthetic-without-disclaimer fixture and asserts the
    # detector catches it (and does not false-positive on a disclaimer'd one).
    r = _run("--selftest")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "selftest: OK" in r.stdout


def test_readme_numbers_come_from_bench_results():
    import json

    rec = json.loads((ROOT / "bench_results" / "v0.1.0a1_falsification.json").read_text())
    readme = (ROOT / "README.md").read_text()
    a01 = rec["falsification_synthetic"]["alpha_0.1"]
    a025 = rec["falsification_synthetic"]["alpha_0.25"]
    cov = rec["real_math"]["conformal_marginal_coverage"]["mean"]
    # every headline number printed in the README must exist in the bench record.
    for val in (
        a01["calibrated_trust_shift"],
        a01["overconfident_trust_shift"],
        a025["calibrated_trust_shift"],
        a025["overconfident_trust_shift"],
    ):
        assert f"{val:.3f}" in readme, f"README missing measured value {val:.3f}"
    assert f"{cov:.4f}" in readme, "README missing measured conformal coverage"
