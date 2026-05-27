"""S5: CLI — doctor / score / calibrate, honest eval mode, file outputs."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vlatrust.core.trace import save_traceset
from vlatrust.core.types import ConfidenceSource, Step, Trace, TraceSet
from vlatrust.report.cli import main


def test_doctor_returns_zero(capsys):
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "mock" in out and "openvla" in out


def test_version():
    assert main(["--version"]) == 0


def test_score_mock_writes_json_and_html(tmp_path, capsys):
    out = tmp_path / "sc.json"
    html = tmp_path / "sc.html"
    rc = main(["score", "--mock", "--out", str(out), "--html", str(html)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "MOCK" in text and "Trust-Shift" in text  # disclaimer + headline
    d = json.loads(out.read_text())
    assert d["trust_shift"] is not None
    assert html.read_text().lstrip().startswith("<!doctype html>")


def test_score_mock_overconfident_lower(capsys):
    main(["score", "--mock"])
    cal_line = [ln for ln in capsys.readouterr().out.splitlines() if "Trust-Shift" in ln][0]
    main(["score", "--mock", "--overconfident"])
    over_line = [ln for ln in capsys.readouterr().out.splitlines() if "Trust-Shift" in ln][0]
    cal = float(cal_line.split(":")[1].split("(")[0])
    over = float(over_line.split(":")[1].split("(")[0])
    assert cal > over  # falsification holds through the CLI surface too


def test_calibrate_mock(capsys):
    assert main(["calibrate", "--mock"]) == 0
    assert "ECE" in capsys.readouterr().out


def test_calibrate_uncalibrated_is_honest(tmp_path, capsys):
    # a TraceSet with no usable confidence -> "uncalibrated", never a number.
    ts = TraceSet(
        (
            Trace(
                steps=(Step(action=np.zeros(7)),),
                confidence_source=ConfidenceSource.NONE,
                success=True,
                trace_id="x",
            ),
        )
    )
    path = tmp_path / "noconf.json"
    save_traceset(ts, path)
    assert main(["calibrate", str(path)]) == 0
    assert "uncalibrated" in capsys.readouterr().out


def test_score_no_labels_reports_na(tmp_path, capsys):
    ts = TraceSet(
        (
            Trace(
                steps=(Step(action=np.zeros(7), confidence=0.8, neg_log_prob=0.2),),
                confidence_source=ConfidenceSource.TOKEN_ENTROPY,
                success=None,  # unlabelled
                trace_id="u",
            ),
        )
    )
    path = tmp_path / "nolabels.json"
    save_traceset(ts, path)
    assert main(["score", str(path)]) == 0
    assert "no success labels" in capsys.readouterr().out


def test_score_requires_input():
    with pytest.raises(SystemExit):
        main(["score"])  # neither path nor --mock
