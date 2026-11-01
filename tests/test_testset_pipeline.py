"""End to end tests for the test set construction scripts.

A miniature archive is built inside ``tmp_path``, every path in :mod:`core.config` is
redirected there, and the four scripts are run in order. The suite proves that the chain
holds together and that the generated ``SOURCES.csv`` and ``LABELS.csv`` still satisfy
``scripts/check_testset.py``.
"""
from __future__ import annotations

import importlib.util

import cv2
import numpy as np
import pandas as pd
import pytest

from core import config, dataset
from scripts import apply_review, index_dataset, make_review_sheet, select_testset

_MODULE_PATH = config.PROJECT_ROOT / "scripts" / "check_testset.py"
_spec = importlib.util.spec_from_file_location("check_testset_gate", _MODULE_PATH)
check_testset = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(check_testset)

BOXES = [("car", (20, 20, 180, 140)), ("bus", (200, 60, 420, 260)), ("rickshaw", (60, 300, 200, 420))]


def write_image(path, brightness: int = 170, seed: int = 0) -> None:
    """Write a sharp, textured image that passes the screening rules."""
    rng = np.random.RandomState(seed)
    image = np.clip(rng.randint(0, 255, (500, 700, 3)) + brightness // 4, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), image)


def write_xml(path, filename: str, boxes=BOXES) -> None:
    """Write a PASCAL VOC annotation with the given boxes."""
    entries = "".join(
        f"<object><name>{name}</name><truncated>0</truncated><difficult>0</difficult>"
        f"<bndbox><xmin>{b[0]}</xmin><ymin>{b[1]}</ymin><xmax>{b[2]}</xmax><ymax>{b[3]}</ymax></bndbox></object>"
        for name, b in boxes
    )
    path.write_text(
        f"<annotation><filename>{filename}</filename>"
        f"<size><width>700</width><height>500</height><depth>3</depth></size>{entries}</annotation>",
        encoding="utf-8",
    )


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point every test set path at a scratch directory and build a small archive."""
    testset = tmp_path / "testset"
    monkeypatch.setattr(config, "TESTSET_DIR", testset)
    monkeypatch.setattr(config, "TESTSET_IMAGES_DIR", testset / "images")
    monkeypatch.setattr(config, "TESTSET_CLIPS_DIR", testset / "clips")
    monkeypatch.setattr(config, "SOURCES_CSV", testset / "SOURCES.csv")
    monkeypatch.setattr(config, "LABELS_CSV", testset / "LABELS.csv")
    monkeypatch.setattr(config, "TESTSET_INVENTORY_CSV", testset / "inventory.csv")
    monkeypatch.setattr(config, "TESTSET_SELECTION_CSV", testset / "selection_log.csv")
    monkeypatch.setattr(config, "TESTSET_REVIEW_DIR", testset / "review")
    monkeypatch.setattr(config, "TESTSET_REVIEW_CHECKLIST_CSV", testset / "review" / "REVIEW.csv")

    (config.TESTSET_IMAGES_DIR).mkdir(parents=True)
    (config.TESTSET_CLIPS_DIR).mkdir(parents=True)

    archive = tmp_path / "archive"
    folder = archive / "Final Train Dataset"
    folder.mkdir(parents=True)
    for index in range(4):
        name = f"Ma_{index + 100}.jpg"
        write_image(folder / name, seed=index)
        write_xml(folder / name.replace(".jpg", ".xml"), name)
    write_image(folder / "no_annotation.jpg", seed=9)
    (folder / "999.xml").write_text("0.1 0.2 0.3", encoding="utf-8")
    write_image(folder / "999.jpg", seed=8)
    return archive


def test_index_reports_every_image_and_screens_them(sandbox):
    rows = index_dataset.main(["--source", str(sandbox), "--out", str(config.TESTSET_INVENTORY_CSV)])
    frame = pd.read_csv(config.TESTSET_INVENTORY_CSV)
    assert rows == 0
    assert len(frame) == 6
    assert (frame["reject_reasons"].fillna("").astype(str) == "").sum() == 4
    assert any("no XML annotation" in str(reason) for reason in frame["reject_reasons"])
    assert any("not a PASCAL VOC annotation" in str(reason) for reason in frame["reject_reasons"])


def test_full_chain_produces_a_consistent_test_set(sandbox):
    assert index_dataset.main(["--source", str(sandbox), "--out", str(config.TESTSET_INVENTORY_CSV)]) == 0
    assert select_testset.main(["--inventory", str(config.TESTSET_INVENTORY_CSV), "--apply", "--target", "3"]) == 0

    sources = pd.read_csv(config.SOURCES_CSV)
    labels = pd.read_csv(config.LABELS_CSV)
    assert len(sources) == 3
    assert set(sources["filename"]) == {path.name for path in config.TESTSET_IMAGES_DIR.glob("*.jpg")}
    for tags in sources["condition_tags"]:
        valid, invalid = dataset.split_tags(tags)
        assert valid and not invalid
    assert labels["labeller_pass"].unique().tolist() == [1]
    assert int(labels["car"].sum()) == 3

    assert make_review_sheet.main([]) == 0
    review = pd.read_csv(config.TESTSET_REVIEW_CHECKLIST_CSV)
    assert len(review) == 3
    assert list(config.TESTSET_REVIEW_DIR.glob("sheets/sheet_*.jpg"))

    review["tags_are_correct"] = "yes"
    review["counts_are_correct"] = "yes"
    review["privacy_ok"] = "yes"
    review["reviewer_initials"] = "ab"
    review["corrected_condition_tags"] = "night;dense"
    review.to_csv(config.TESTSET_REVIEW_CHECKLIST_CSV, index=False)

    assert apply_review.main([]) == 0
    final_labels = pd.read_csv(config.LABELS_CSV)
    assert sorted(final_labels["labeller_pass"].unique().tolist()) == [1, 2]
    assert len(final_labels) == 6
    final_sources = pd.read_csv(config.SOURCES_CSV)
    assert set(final_sources["condition_tags"]) == {"night;dense"}
    assert all((config.TESTSET_IMAGES_DIR / name).exists() for name in final_sources["filename"])

    errors, _counts = check_testset.check(
        final_sources,
        final_labels,
        {path.name for path in config.TESTSET_IMAGES_DIR.iterdir()},
    )
    assert errors == []


def test_select_testset_rejects_an_unknown_quota_tag(sandbox):
    index_dataset.main(["--source", str(sandbox), "--out", str(config.TESTSET_INVENTORY_CSV)])
    assert select_testset.main(["--min-tag", "rain=5"]) == 1


def test_apply_review_refuses_an_unanswered_sheet(sandbox, capsys):
    index_dataset.main(["--source", str(sandbox), "--out", str(config.TESTSET_INVENTORY_CSV)])
    select_testset.main(["--inventory", str(config.TESTSET_INVENTORY_CSV), "--apply", "--target", "2"])
    make_review_sheet.main([])
    assert apply_review.main([]) == 1
    assert "is not answered" in capsys.readouterr().out


def test_second_pass_draw_is_deterministic_and_rounded_up():
    ids = [f"tl{index:03d}" for index in range(1, 13)]
    first = make_review_sheet.second_pass_draw(ids)
    assert len(first) == 3
    assert first == make_review_sheet.second_pass_draw(list(reversed(ids)))
    assert make_review_sheet.second_pass_draw(["tl001"]) == {"tl001"}


def test_review_frame_carries_draft_counts_and_tags():
    rows = [
        {
            "item_id": "tl001",
            "filename": "tl001_mock_day.jpg",
            "original_filename": "mock.jpg",
            "condition_tags": "day",
            "location_area": "mock",
            "counts": {category: 0 for category in dataset.LABEL_CATEGORIES} | {"car": 2},
            "objects": (),
        }
    ]
    frame = make_review_sheet.review_frame(rows, {"tl001"})
    assert frame.loc[0, "car"] == 2
    assert frame.loc[0, "second_pass_due"] == "yes"
    assert frame.loc[0, "corrected_condition_tags"] == "day"
