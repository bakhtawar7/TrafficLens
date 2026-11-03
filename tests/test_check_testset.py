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


_EXTENDED_COLUMNS = [
    "item_id", "filename", "type", "dataset", "split", "scene",
    "source", "licence", "url", "condition_tags", "notes",
]


def _extended_sources(rows):
    """A SOURCES frame in the new build_testset format with the provenance columns."""
    return pd.DataFrame(rows, columns=_EXTENDED_COLUMNS)


_LABEL_COLUMNS = ["item_id", *check_testset.dataset.LABEL_CATEGORIES, "labeller_pass", "notes"]


def _labels(rows):
    """A LABELS frame with the seven category count columns plus the pass marker."""
    return pd.DataFrame(rows, columns=_LABEL_COLUMNS)


def _clean_row(**overrides):
    """One well-formed extended SOURCES row; override a single field to force an error."""
    row = {
        "item_id": "tl0001", "filename": "tl0001_a.jpg", "type": "image",
        "dataset": "dhaka_ai", "split": "train", "scene": "train:blk0000",
        "source": "archive", "licence": "UNVERIFIED", "url": "",
        "condition_tags": "day", "notes": "",
    }
    row.update(overrides)
    return [row[c] for c in _EXTENDED_COLUMNS]


def test_reported_clean_when_files_and_rows_match():
    sources = _sources([["tl0001", "tl0001_a.jpg", "own", "day"]])
    labels = pd.DataFrame(columns=["item_id"])
    errors, counts = check_testset.check(sources, labels, {"tl0001_a.jpg"})
    assert errors == []
    assert counts[sorted(check_testset.VALID_CONDITION_TAGS).index("day")] == 1


def test_flags_file_missing_from_sources():
    sources = _sources([])
    labels = pd.DataFrame(columns=["item_id"])
    errors, _ = check_testset.check(sources, labels, {"tl0002_a.jpg"})
    assert any("missing from SOURCES.csv" in e for e in errors)


def test_flags_row_with_no_file():
    sources = _sources([["tl0003", "tl0003_a.jpg", "own", "day"]])
    labels = pd.DataFrame(columns=["item_id"])
    errors, _ = check_testset.check(sources, labels, set())
    assert any("no such file" in e for e in errors)


def test_flags_missing_permission():
    sources = _sources([["tl0004", "tl0004_a.jpg", "", "day"]])
    labels = pd.DataFrame(columns=["item_id"])
    errors, _ = check_testset.check(sources, labels, {"tl0004_a.jpg"})
    assert any("no permission" in e for e in errors)


def test_flags_invalid_condition_tag():
    sources = _sources([["tl0005", "tl0005_a.jpg", "own", "rainy"]])
    labels = pd.DataFrame(columns=["item_id"])
    errors, _ = check_testset.check(sources, labels, {"tl0005_a.jpg"})
    assert any("invalid condition tag" in e for e in errors)


def test_flags_source_row_with_no_condition_tag():
    """A tagless item belongs to no condition stratum, so a blank cell is an error."""
    for blank in ("", "  ", None):
        sources = _sources([["tl0012", "tl0012_a.jpg", "own", blank]])
        labels = pd.DataFrame(columns=["item_id"])
        errors, counts = check_testset.check(sources, labels, {"tl0012_a.jpg"})
        assert any("no condition tags recorded" in e and "tl0012" in e for e in errors), blank
        assert sum(counts) == 0


def test_flags_extended_source_row_with_no_condition_tag():
    sources = _extended_sources([_clean_row(condition_tags="")])
    errors, _ = check_testset.check(sources, pd.DataFrame(columns=["item_id"]), {"tl0001_a.jpg"})
    assert any("no condition tags recorded" in e for e in errors)


def test_empty_scaffold_has_no_errors():
    sources = pd.DataFrame(columns=["item_id", "filename", "permission", "condition_tags"])
    labels = pd.DataFrame(columns=["item_id"])
    errors, counts = check_testset.check(sources, labels, set())
    assert errors == []
    assert sum(counts) == 0


def test_empty_scaffold_with_new_columns_has_no_errors():
    sources = _extended_sources([])
    labels = _labels([])
    errors, counts = check_testset.check(sources, labels, set())
    assert errors == []
    assert sum(counts) == 0


# ---------------------------------------------------------------------------
# New SOURCES provenance columns
# ---------------------------------------------------------------------------

def test_accepts_new_sources_columns():
    sources = _extended_sources([_clean_row()])
    errors, _ = check_testset.check(sources, pd.DataFrame(columns=["item_id"]), {"tl0001_a.jpg"})
    assert errors == []


def test_allows_empty_url():
    sources = _extended_sources([_clean_row(url="")])
    errors, _ = check_testset.check(sources, pd.DataFrame(columns=["item_id"]), {"tl0001_a.jpg"})
    assert errors == []


def test_flags_blank_dataset():
    sources = _extended_sources([_clean_row(dataset="")])
    errors, _ = check_testset.check(sources, pd.DataFrame(columns=["item_id"]), {"tl0001_a.jpg"})
    assert any("no dataset recorded" in e for e in errors)


def test_flags_blank_split():
    sources = _extended_sources([_clean_row(split="")])
    errors, _ = check_testset.check(sources, pd.DataFrame(columns=["item_id"]), {"tl0001_a.jpg"})
    assert any("no split recorded" in e for e in errors)


def test_flags_blank_licence():
    sources = _extended_sources([_clean_row(licence="")])
    errors, _ = check_testset.check(sources, pd.DataFrame(columns=["item_id"]), {"tl0001_a.jpg"})
    assert any("no licence recorded" in e for e in errors)


def test_flags_invalid_url():
    sources = _extended_sources([_clean_row(url="ftp://example.com/a.jpg")])
    errors, _ = check_testset.check(sources, pd.DataFrame(columns=["item_id"]), {"tl0001_a.jpg"})
    assert any("invalid url" in e for e in errors)


# ---------------------------------------------------------------------------
# LABELS value validation
# ---------------------------------------------------------------------------

def _label_row(item_id="tl0001", **overrides):
    row = {
        "item_id": item_id, "car": 0, "bus": 0, "truck": 0, "motorcycle": 0,
        "rickshaw": 0, "loader_rickshaw": 0, "other": 0, "labeller_pass": 1, "notes": "",
    }
    row.update(overrides)
    return [row[column] for column in _LABEL_COLUMNS]


def _labels_only(**overrides):
    """Check a single LABELS row against a matching SOURCES row, isolating value errors."""
    sources = _extended_sources([_clean_row()])
    labels = _labels([_label_row(**overrides)])
    errors, _ = check_testset.check(sources, labels, {"tl0001_a.jpg"})
    return errors


def test_accepts_valid_label_values():
    assert _labels_only(car=3, bus=0, labeller_pass=2) == []


def test_allows_blank_label_values():
    assert _labels_only(
        car="", bus="", truck="", motorcycle="", rickshaw="", loader_rickshaw="",
        other="", labeller_pass="",
    ) == []


def test_flags_negative_label_count():
    assert any("non-negative integer" in e and "car" in e for e in _labels_only(car=-1))


def test_flags_non_integer_label_count():
    assert any("non-negative integer" in e and "bus" in e for e in _labels_only(bus=2.5))


def test_flags_non_numeric_label_count():
    assert any("non-negative integer" in e and "truck" in e for e in _labels_only(truck="abc"))


def test_flags_invalid_labeller_pass():
    assert any("labeller_pass must be 1 or 2" in e for e in _labels_only(labeller_pass=3))


# ---------------------------------------------------------------------------
# Item id format and the LABELS to SOURCES join
# ---------------------------------------------------------------------------

def test_flags_legacy_three_digit_id_and_names_the_canonical_form():
    """A ``tl001`` id is reported, with the four digit spelling it should carry."""
    sources = _extended_sources([_clean_row(item_id="tl001", filename="tl001_a.jpg")])
    errors, _ = check_testset.check(sources, pd.DataFrame(columns=["item_id"]), {"tl001_a.jpg"})
    assert any("four digit form" in e and "tl0001" in e for e in errors)


def test_flags_item_id_that_is_not_a_traffic_lens_id():
    sources = _extended_sources([_clean_row(item_id="img-17", filename="tl0001_a.jpg")])
    errors, _ = check_testset.check(sources, pd.DataFrame(columns=["item_id"]), {"tl0001_a.jpg"})
    assert any("not a TrafficLens item id" in e for e in errors)


def test_labels_row_joins_sources_by_number_not_by_spelling():
    """``tl001`` in LABELS is item 1, so it matches SOURCES ``tl0001`` on the number."""
    sources = _extended_sources([_clean_row()])
    labels = _labels([_label_row(item_id="tl001")])
    errors, _ = check_testset.check(sources, labels, {"tl0001_a.jpg"})
    assert not any("no matching row" in e for e in errors)


def test_flags_label_row_with_no_source_row_at_all():
    """An orphan label number is reported even when the spelling is canonical."""
    sources = _extended_sources([_clean_row()])
    labels = _labels([_label_row(item_id="tl0046")])
    errors, _ = check_testset.check(sources, labels, {"tl0001_a.jpg"})
    assert any("LABELS.csv item 'tl0046' has no matching row in SOURCES.csv" in e for e in errors)
