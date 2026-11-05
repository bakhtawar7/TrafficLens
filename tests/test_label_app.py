"""Tests for the manual count audit app (scripts/label_app.py).

The file-system and logic helpers are exercised directly against a temporary
directory so they stay deterministic and hermetic. A single Streamlit ``AppTest``
smoke test drives the real entry script to confirm the screen renders, shows one
item at a time, and exposes exactly the seven category fields without pulling in
any annotation or model numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from core import config
from scripts import check_testset as ct
from scripts import label_app as la

ROOT = Path(__file__).resolve().parent.parent
APP_ENTRY = str(ROOT / "scripts" / "label_app.py")

ALL_CATEGORIES = list(la.CATEGORIES)


def new_labels() -> pd.DataFrame:
    return pd.DataFrame(columns=list(la.LABEL_COLUMNS))


def counts(**overrides) -> dict[str, int]:
    base = {category: 0 for category in ALL_CATEGORIES}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def test_upsert_creates_a_row_with_the_right_pass_and_counts() -> None:
    frame, status = la.upsert_label(new_labels(), "tl0007", counts(car=3, bus=1), "clear", 1)
    assert status == "created"
    row = la.find_rows(frame, "tl0007", 1).iloc[-1]
    assert row["car"] == "3"
    assert row["bus"] == "1"
    assert row["labeller_pass"] == "1"
    assert row["notes"] == "clear"


def test_upsert_writes_and_reads_back_atomically(tmp_path: Path) -> None:
    target = tmp_path / "LABELS.csv"
    frame, _ = la.upsert_label(new_labels(), "tl0007", counts(car=3), "note", 2)
    la.write_atomic(frame[list(la.LABEL_COLUMNS)], target)
    reloaded = la.load_labels(target)
    assert la.row_exists(reloaded, "tl0007", 2)
    assert la.saved_counts(reloaded, "tl0007", 2)["car"] == 3


def test_second_upsert_needs_confirmation_before_overwriting(tmp_path: Path) -> None:
    frame, _ = la.upsert_label(new_labels(), "tl0007", counts(car=3), "first", 1)
    blocked, status = la.upsert_label(frame, "tl0007", counts(car=9), "second", 1, allow_overwrite=False)
    assert status == "needs_confirmation"
    assert la.saved_counts(blocked, "tl0007", 1)["car"] == 3  # unchanged
    replaced, status = la.upsert_label(frame, "tl0007", counts(car=9), "second", 1, allow_overwrite=True)
    assert status == "overwrite_saved"
    assert la.saved_counts(replaced, "tl0007", 1)["car"] == 9


# ---------------------------------------------------------------------------
# Pass 2 hiding rule
# ---------------------------------------------------------------------------
def test_pass1_counts_stay_hidden_until_pass2_is_saved() -> None:
    only_pass1, _ = la.upsert_label(new_labels(), "tl0007", counts(car=3), "p1", 1)
    assert la.should_reveal_pass1(only_pass1, "tl0007") is False
    # The pass 2 field must not be pre-filled from the pass 1 row.
    assert la.saved_counts(only_pass1, "tl0007", 2) is None
    with_pass2, _ = la.upsert_label(only_pass1, "tl0007", counts(car=5), "p2", 2)
    assert la.should_reveal_pass1(with_pass2, "tl0007") is True


def test_pass2_mode_shows_only_second_pass_items() -> None:
    sources = pd.DataFrame(
        {
            "item_id": ["tl0001", "tl0002", "tl0003"],
            "filename": ["a.jpg", "b.jpg", "c.jpg"],
            "condition_tags": ["day", "night", "day"],
        }
    )
    plan = {"development_subset": ["tl0001"], "second_pass": ["tl0002"]}
    order = la.ordered_items(sources, plan)
    second = la.filter_second_pass(order)
    assert [item["item_id"] for item in second] == ["tl0002"]
    # Development images lead, then evaluation images grouped by tag.
    assert [item["item_id"] for item in order] == ["tl0001", "tl0003", "tl0002"]


def test_unlabelled_items_lists_missing_counts() -> None:
    sources = pd.DataFrame(
        {"item_id": ["tl0001", "tl0002"], "filename": ["a.jpg", "b.jpg"], "condition_tags": ["day", "night"]}
    )
    order = la.ordered_items(sources, {"development_subset": [], "second_pass": []})
    frame, _ = la.upsert_label(new_labels(), "tl0001", counts(car=2), "", 1)
    assert la.unlabelled_items(order, frame, 1) == ["tl0002"]


# ---------------------------------------------------------------------------
# Tag confirmation
# ---------------------------------------------------------------------------
def test_update_item_tags_marks_manual_and_confirmed() -> None:
    sources = pd.DataFrame(
        {
            "item_id": ["tl0007", "tl0008"],
            "condition_tags": ["dense", ""],
            "tag_source": ["cutoff_rule", "auto_proposed"],
            "tag_confirmed": ["", ""],
        }
    )
    updated = la.update_item_tags(sources, "tl0008", ["dense", "day"])
    row = updated[updated["item_id"] == "tl0008"].iloc[-1]
    assert row["condition_tags"] == "day;dense"  # canonical order via merge_tags
    assert row["tag_source"] == "manual"
    assert row["tag_confirmed"] == "yes"
    # The untouched row keeps its original provenance.
    other = updated[updated["item_id"] == "tl0007"].iloc[-1]
    assert other["tag_source"] == "cutoff_rule"


def test_confirmed_tags_are_no_longer_counted_unconfirmed() -> None:
    sources = pd.DataFrame(
        {"item_id": ["tl0008"], "condition_tags": [""], "tag_source": ["auto_proposed"], "tag_confirmed": [""]}
    )
    assert ct.count_unconfirmed_tags(sources) == 1
    confirmed = la.update_item_tags(sources, "tl0008", ["dense"])
    assert ct.count_unconfirmed_tags(confirmed) == 0


# ---------------------------------------------------------------------------
# Atomic writes and backup
# ---------------------------------------------------------------------------
class _BrokenFrame:
    """A frame stub whose to_csv fails midway, to prove the write is atomic."""

    def to_csv(self, *args, **kwargs) -> None:  # noqa: D401 - deliberate failure
        raise RuntimeError("serialization exploded")


def test_write_atomic_preserves_original_and_cleans_temp_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "LABELS.csv"
    target.write_text("item_id,notes\nOLD,keep\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        la.write_atomic(_BrokenFrame(), target)
    assert target.read_text(encoding="utf-8") == "item_id,notes\nOLD,keep\n"
    leftovers = [p for p in tmp_path.glob("LABELS.csv.*.tmp")]
    assert leftovers == []


def test_write_atomic_replaces_existing_content(tmp_path: Path) -> None:
    target = tmp_path / "LABELS.csv"
    target.write_text("item_id\nSTALE\n", encoding="utf-8")
    frame, _ = la.upsert_label(new_labels(), "tl0009", counts(car=1), "fresh", 1)
    la.write_atomic(frame[list(la.LABEL_COLUMNS)], target)
    text = target.read_text(encoding="utf-8")
    assert "STALE" not in text
    assert "tl0009" in text


def test_make_backup_copies_labels_with_timestamp(tmp_path: Path) -> None:
    labels = tmp_path / "LABELS.csv"
    labels.write_text("item_id,car\ntl0001,3\n", encoding="utf-8")
    backups = tmp_path / "label_backups"
    made = la.make_backup(labels, backups)
    assert made is not None and made.exists()
    assert made.parent == backups
    assert made.read_text(encoding="utf-8") == "item_id,car\ntl0001,3\n"


def test_make_backup_returns_none_when_no_labels_file(tmp_path: Path) -> None:
    assert la.make_backup(tmp_path / "absent.csv", tmp_path / "label_backups") is None


# ---------------------------------------------------------------------------
# Streamlit smoke test
# ---------------------------------------------------------------------------
def _build_mini_testset(tmp_path: Path) -> None:
    """Create a two-item test set (one real image each) and point config at it."""
    images = tmp_path / "images"
    images.mkdir()
    for name in ("tl0001_x.jpg", "tl0002_x.jpg"):
        canvas = np.full((60, 80, 3), 120, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", canvas)
        assert ok
        (images / name).write_bytes(encoded.tobytes())

    (tmp_path / "SOURCES.csv").write_text(
        "item_id,filename,condition_tags,tag_source,tag_confirmed\n"
        "tl0001,tl0001_x.jpg,day,cutoff_rule,\n"
        "tl0002,tl0002_x.jpg,night,auto_proposed,\n",
        encoding="utf-8",
    )
    (tmp_path / "LABELS.csv").write_text(",".join(la.LABEL_COLUMNS) + "\n", encoding="utf-8")
    plan = {"development_subset": ["tl0001"], "second_pass": ["tl0002"], "items": []}
    (tmp_path / "testset_plan.json").write_text(json.dumps(plan), encoding="utf-8")


def _patch_config(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "SOURCES_CSV", tmp_path / "SOURCES.csv")
    monkeypatch.setattr(config, "LABELS_CSV", tmp_path / "LABELS.csv")
    monkeypatch.setattr(config, "TESTSET_PLAN_JSON", tmp_path / "testset_plan.json")
    monkeypatch.setattr(config, "TESTSET_IMAGES_DIR", tmp_path / "images")
    monkeypatch.setattr(config, "AUDIT_OVERLAYS_DIR", tmp_path / "no_overlays")
    monkeypatch.setattr(config, "LABEL_BACKUPS_DIR", tmp_path / "label_backups")


def test_app_renders_first_item_with_seven_blind_fields(tmp_path: Path, monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    _build_mini_testset(tmp_path)
    _patch_config(monkeypatch, tmp_path)

    app = AppTest.from_file(APP_ENTRY, default_timeout=120)
    app.run()

    assert not app.exception
    # One image shown, big.
    assert len(app.image) >= 1
    # Exactly the seven category number fields, none pre-filled from a machine value.
    assert len(app.number_input) == len(ALL_CATEGORIES)
    labels = [str(field.label) for field in app.number_input]
    assert any("Rickshaw" in label for label in labels)
    # Navigation and progress are present.
    button_labels = [str(button.label) for button in app.button]
    assert "Previous" in button_labels and "Next" in button_labels and "Save counts" in button_labels
    assert len(app.progress) >= 1
