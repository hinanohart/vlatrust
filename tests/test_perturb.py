"""S2: perturbation algebra — identity law, reproducibility, purity, registry."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from vlatrust.core.perturb.algebra import IDENTITY, Pipeline, apply_op, compose
from vlatrust.core.perturb.registry import (
    REGISTRY,
    RENDERER_REQUIRED,
    TRACE_APPLICABLE,
    by_modality,
    get,
)

ARRAY_OPS = [op for op in REGISTRY.values() if op.payload_kind == "array"]
TEXT_OPS = [op for op in REGISTRY.values() if op.payload_kind == "text"]

_finite = arrays(
    dtype=np.float64,
    shape=array_shapes(min_dims=1, max_dims=2, min_side=2, max_side=6),
    elements=st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False),
)


def test_registry_has_14_trace_applicable_ops():
    assert len(TRACE_APPLICABLE) == 14
    assert len(REGISTRY) == 14


def test_renderer_required_ops_raise_not_implemented():
    assert len(RENDERER_REQUIRED) == 3
    for name in RENDERER_REQUIRED:
        with pytest.raises(NotImplementedError):
            get(name)


def test_unknown_op_raises_keyerror():
    with pytest.raises(KeyError):
        get("no_such_perturbation")


@settings(max_examples=60, deadline=None)
@given(x=_finite)
def test_array_ops_identity_at_zero(x):
    rng = np.random.default_rng(0)
    for op in ARRAY_OPS:
        out = op.fn(x, 0.0, rng)
        assert np.array_equal(np.asarray(out, dtype=float), x.astype(float)), op.name


@given(words=st.lists(st.text(min_size=1, max_size=5), min_size=0, max_size=8))
def test_text_ops_identity_at_zero(words):
    text = " ".join(words)
    rng = np.random.default_rng(0)
    for op in TEXT_OPS:
        assert op.fn(text, 0.0, rng) == text, op.name


@settings(max_examples=40, deadline=None)
@given(x=_finite, tau=st.floats(0.0, 1.0))
def test_array_ops_are_pure(x, tau):
    original = x.copy()
    rng = np.random.default_rng(1)
    for op in ARRAY_OPS:
        op.fn(x, tau, rng)
    assert np.array_equal(x, original)  # input untouched


def test_apply_op_is_reproducible():
    x = np.arange(12, dtype=float).reshape(3, 4)
    op = REGISTRY["gaussian_sensor_noise"]
    a = apply_op(op, x, 0.5, root_seed=123, position=0)
    b = apply_op(op, x, 0.5, root_seed=123, position=0)
    c = apply_op(op, x, 0.5, root_seed=999, position=0)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)  # different root seed -> different draw


def test_pipeline_composition_reproducible_and_ordered():
    x = np.linspace(0, 1, 20)
    pipe = compose(
        REGISTRY["gaussian_sensor_noise"],
        REGISTRY["action_scale"],
        REGISTRY["translate_shift"],
    )
    assert isinstance(pipe, Pipeline) and len(pipe) == 3
    out1 = pipe.apply(x, 0.4, root_seed=7)
    out2 = pipe.apply(x, 0.4, root_seed=7)
    assert np.array_equal(out1, out2)


def test_pipeline_identity_at_zero():
    x = np.random.default_rng(0).normal(size=(4, 5))
    pipe = compose(*ARRAY_OPS)
    out = pipe.apply(x, 0.0, root_seed=42)
    assert np.array_equal(np.asarray(out, dtype=float), x)


def test_empty_compose_is_identity():
    pipe = compose()
    x = np.array([1.0, 2.0, 3.0])
    assert pipe.ops == (IDENTITY,)
    assert np.array_equal(np.asarray(pipe.apply(x, 0.9, root_seed=1)), x)


def test_intensity_out_of_range_rejected():
    with pytest.raises(ValueError):
        apply_op(REGISTRY["action_scale"], np.zeros(3), 1.5, root_seed=0)


def test_by_modality_groups():
    assert {op.name for op in by_modality("language")} == {
        "word_dropout",
        "word_shuffle",
        "instruction_truncate",
    }
