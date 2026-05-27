"""S5: report layer — JSON schema contract + self-contained HTML."""

from __future__ import annotations

import json

import pytest

from vlatrust.adapters.mock import MockPolicy
from vlatrust.core.score.scorecard import score_traceset
from vlatrust.core.types import ConfidenceSource
from vlatrust.report.html import scorecard_to_html
from vlatrust.report.schema import SchemaError, validate_scorecard


@pytest.fixture
def mock_scorecard():
    ts = MockPolicy(calibrated=True).collapse_traceset(rng=0)
    return score_traceset(ts, rng=0)


# --- schema ----------------------------------------------------------------- #


def test_valid_scorecard_passes(mock_scorecard):
    d = mock_scorecard.to_dict()
    validate_scorecard(d)  # must not raise
    json.dumps(d, allow_nan=False)  # and must be strict JSON


def test_fail_closed_scorecard_passes_schema():
    # NONE-source scorecard has trust_shift=null; the contract must accept it.
    ts = MockPolicy(confidence_source=ConfidenceSource.NONE).collapse_traceset(rng=0)
    d = score_traceset(ts, rng=0).to_dict()
    validate_scorecard(d)
    assert d["trust_shift"] is None


def test_schema_rejects_missing_key():
    with pytest.raises(SchemaError):
        validate_scorecard({"trust_shift": 0.5, "hard_valid": True, "notes": []})


def test_schema_rejects_out_of_range_trust():
    bad = {
        "confidence_source": "token_entropy",
        "trust_shift": 1.5,
        "hard_valid": True,
        "notes": [],
    }
    with pytest.raises(SchemaError):
        validate_scorecard(bad)


def test_schema_rejects_bad_enum():
    bad = {"confidence_source": "made_up", "trust_shift": 0.5, "hard_valid": True, "notes": []}
    with pytest.raises(SchemaError):
        validate_scorecard(bad)


def test_schema_rejects_bool_trust():
    bad = {"confidence_source": "none", "trust_shift": True, "hard_valid": True, "notes": []}
    with pytest.raises(SchemaError):
        validate_scorecard(bad)


# --- html ------------------------------------------------------------------- #


def test_html_is_self_contained(mock_scorecard):
    html = scorecard_to_html(mock_scorecard, title="Test")
    assert html.lstrip().startswith("<!doctype html>")
    # no external resources of any kind.
    for needle in ("http://", "https://", "src=", "cdn", "<script"):
        assert needle not in html, needle


def test_html_shows_score_and_disclaimer(mock_scorecard):
    html = scorecard_to_html(mock_scorecard, disclaimer="MOCK input")
    assert f"{mock_scorecard.trust_shift:.3f}" in html
    assert "MOCK input" in html
    assert "disclaimer" in html


def test_html_renders_none_score():

    ts = MockPolicy(confidence_source=ConfidenceSource.NONE).collapse_traceset(rng=0)
    html = scorecard_to_html(score_traceset(ts, rng=0))
    assert "N/A" in html
