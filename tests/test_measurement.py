"""S6: real-data ingest (physics on joint states) + bench-record honesty.

The bench record's honesty is machine-checked here: synthetic mode MUST carry a
disclaimer and gate_real MUST be explicitly deferred (never silently claimed).
This is the firewall against shipping synthetic numbers as if they were real.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vlatrust.adapters.lerobot_dataset import traceset_from_rows
from vlatrust.core.reliability.ood_gate import hard_valid
from vlatrust.core.types import ConfidenceSource

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench_results" / "v0.1.0a1_falsification.json"


# --- ingest: physics gate on real-style joint states ------------------------ #


def test_ingest_groups_episodes_and_tags_none_source():
    actions = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
    episode_index = np.array([0, 0, 1])
    ts = traceset_from_rows(actions=actions, episode_index=episode_index)
    assert len(ts) == 2  # two episodes
    assert [len(t.steps) for t in ts.traces] == [2, 1]
    assert all(
        t.confidence_source == ConfidenceSource.NONE for t in ts.traces
    )  # datasets have no confidence


def test_ingest_physics_gate_flags_out_of_limit_joint_state():
    actions = np.zeros((3, 1))
    episode_index = np.array([0, 0, 1])
    states = np.array([[0.5], [5.0], [0.5]])  # ep0 step1 is outside [0,1]
    ts = traceset_from_rows(
        actions=actions,
        episode_index=episode_index,
        states=states,
        joint_limits=(np.array([0.0]), np.array([1.0])),
    )
    ep0, ep1 = ts.traces
    assert hard_valid(ep0) is False  # contains an out-of-limit state
    assert hard_valid(ep1) is True  # within limits


def test_ingest_without_limits_is_all_valid():
    ts = traceset_from_rows(
        actions=np.zeros((2, 1)),
        episode_index=np.array([0, 1]),
        states=np.array([[99.0], [-99.0]]),  # ignored when no limits given
    )
    assert all(hard_valid(t) for t in ts.traces)


def test_ingest_success_labels_from_last_row():
    ts = traceset_from_rows(
        actions=np.zeros((3, 1)),
        episode_index=np.array([0, 0, 1]),
        success_by_episode={0: True, 1: False},
    )
    assert ts.traces[0].success is True
    assert ts.traces[1].success is False


# --- bench-record honesty (the firewall) ------------------------------------ #


def test_bench_record_exists_and_is_strict_json():
    assert BENCH.exists(), "run scripts/measure_falsification.py to produce the bench record"
    json.loads(BENCH.read_text())  # parses; written with allow_nan=False


def test_synthetic_mode_requires_disclaimer():
    rec = json.loads(BENCH.read_text())
    if rec["mode"] == "synthetic":
        assert rec.get("disclaimer"), "synthetic mode MUST carry a disclaimer (firewall)"
        assert "NOT" in rec["disclaimer"]  # must say what it is not


def test_gate_real_is_explicitly_resolved_not_silently_claimed():
    rec = json.loads(BENCH.read_text())
    assert "gate_real" in rec
    assert rec["gate_real"]["status"] in ("met", "deferred-v0.1.1")
    if rec["gate_real"]["status"].startswith("deferred"):
        assert rec["gate_real"]["reason"]  # a deferral must explain itself


def test_recorded_falsification_holds():
    rec = json.loads(BENCH.read_text())
    a = rec["falsification_synthetic"]["alpha_0.1"]
    assert a["calibrated_beats_overconfident"] is True
    assert rec["falsification_synthetic"]["physics_gate"]["zeroed"] is True
