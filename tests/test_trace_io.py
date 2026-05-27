"""S1: TraceSet JSON round-trip and per-trajectory split."""

from __future__ import annotations

import numpy as np
import pytest
import synthetic

from vlatrust.core.trace import (
    load_traceset,
    save_traceset,
    split_calib_test,
    traceset_from_dict,
    traceset_to_dict,
)


def _equal_traceset(a, b) -> bool:
    da, db = traceset_to_dict(a), traceset_to_dict(b)
    return da == db


def test_roundtrip_dict_preserves_everything():
    ts = synthetic.make_collapse_traceset(calibrated=True, seed=3)
    back = traceset_from_dict(traceset_to_dict(ts))
    assert len(back) == len(ts)
    assert _equal_traceset(ts, back)


def test_roundtrip_file(tmp_path):
    ts = synthetic.make_collapse_traceset(calibrated=False, seed=4, n_per_intensity=5)
    p = tmp_path / "ts.json"
    save_traceset(ts, p)
    loaded = load_traceset(p)
    assert _equal_traceset(ts, loaded)
    # no pickle: the file must be valid UTF-8 JSON text
    assert p.read_text(encoding="utf-8").lstrip().startswith("{")


def test_split_is_deterministic_and_disjoint():
    ts = synthetic.make_collapse_traceset(calibrated=True, seed=5, n_per_intensity=10)
    c1, t1 = split_calib_test(ts, calib_frac=0.5, rng=42)
    c2, t2 = split_calib_test(ts, calib_frac=0.5, rng=42)
    assert _equal_traceset(c1, c2) and _equal_traceset(t1, t2)
    ids_c = {tr.trace_id for tr in c1}
    ids_t = {tr.trace_id for tr in t1}
    assert ids_c.isdisjoint(ids_t)
    assert len(ids_c) + len(ids_t) == len(ts)


def test_split_respects_frac():
    ts = synthetic.make_collapse_traceset(calibrated=True, seed=6, n_per_intensity=10)
    calib, test = split_calib_test(ts, calib_frac=0.3, rng=0)
    assert abs(len(calib) / len(ts) - 0.3) < 0.05


def test_split_requires_two_traces():
    one = synthetic.make_collapse_traceset(
        calibrated=True, intensities=(0.0,), n_per_intensity=1, seed=7
    )
    with pytest.raises(ValueError):
        split_calib_test(one)


def test_split_accepts_generator_instance():
    ts = synthetic.make_collapse_traceset(calibrated=True, seed=8, n_per_intensity=6)
    gen = np.random.default_rng(11)
    calib, test = split_calib_test(ts, rng=gen)
    assert len(calib) + len(test) == len(ts)
