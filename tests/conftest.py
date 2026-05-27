"""Shared pytest fixtures. ``tests/`` has no ``__init__.py`` so ``synthetic`` is
importable directly under pytest's default prepend import mode."""

from __future__ import annotations

import pytest
import synthetic


@pytest.fixture
def calibrated_ts():
    return synthetic.make_collapse_traceset(calibrated=True, seed=1)


@pytest.fixture
def overconfident_ts():
    return synthetic.make_collapse_traceset(calibrated=False, seed=1)


@pytest.fixture
def no_confidence_ts():
    return synthetic.make_no_confidence_traceset(seed=1)
