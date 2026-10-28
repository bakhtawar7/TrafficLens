"""Tests for the test set checker logic."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

_MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "check_testset.py"
_spec = importlib.util.spec_from_file_location("check_testset", _MODULE_PATH)
check_testset = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(check_testset)


def _sources(rows):
    return pd.DataFrame(rows, columns=["item_id", "filename", "permission", "condition_tags"])


def test_reported_clean_when_files_and_rows_match():
    sources = _sources([["tl001", "tl001_a.jpg", "own", "day"]])
    labels = pd.DataFrame(columns=["item_id"])
    errors, counts = check_testset.check(sources, labels, {"tl001_a.jpg"})
    assert errors == []
    assert counts[sorted(check_testset.VALID_CONDITION_TAGS).index("day")] == 1


def test_flags_file_missing_from_sources():
    sources = _sources([])
    labels = pd.DataFrame(columns=["item_id"])
    errors, _ = check_testset.check(sources, labels, {"tl002_a.jpg"})
    assert any("missing from SOURCES.csv" in e for e in errors)


def test_flags_row_with_no_file():
    sources = _sources([["tl003", "tl003_a.jpg", "own", "day"]])
    labels = pd.DataFrame(columns=["item_id"])
    errors, _ = check_testset.check(sources, labels, set())
    assert any("no such file" in e for e in errors)


def test_flags_missing_permission():
    sources = _sources([["tl004", "tl004_a.jpg", "", "day"]])
    labels = pd.DataFrame(columns=["item_id"])
    errors, _ = check_testset.check(sources, labels, {"tl004_a.jpg"})
    assert any("no permission" in e for e in errors)


def test_flags_invalid_condition_tag():
    sources = _sources([["tl005", "tl005_a.jpg", "own", "rainy"]])
    labels = pd.DataFrame(columns=["item_id"])
    errors, _ = check_testset.check(sources, labels, {"tl005_a.jpg"})
    assert any("invalid condition tag" in e for e in errors)


def test_empty_scaffold_has_no_errors():
    sources = pd.DataFrame(columns=["item_id", "filename", "permission", "condition_tags"])
    labels = pd.DataFrame(columns=["item_id"])
    errors, counts = check_testset.check(sources, labels, set())
    assert errors == []
    assert sum(counts) == 0
