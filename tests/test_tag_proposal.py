"""Tests for the auto-proposed condition tag pipeline and the checker's new tolerance.

The proposal helpers are exercised with synthetic features and tiny real files in
``tmp_path``: no archive download, no model, and no writes outside the temporary folder.
The checker tests drive ``check_testset.check`` and ``count_unconfirmed_tags`` directly to
prove that an auto proposed row is accepted (even when blank), that a cut-off-rule row is
still required to carry a tag, and that unconfirmed proposals are counted as a warning.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from core import curation, dataset
from scripts import check_testset as ct
from scripts import tag_proposal as tp

_CUTOFFS = {
    "brightness": {"q1": 90.0, "median": 110.0, "q3": 130.0},
    "saturation": {"q1": 0.02, "median": 0.05, "q3": 0.15},
    "vehicle_count": {"q1": 3.0, "median": 6.0, "q3": 9.0},
    "median_box_fraction": {"q1": 0.0028, "median": 0.008, "q3": 0.024},
}


def _feature(**overrides) -> curation.ImageFeature:
    base = dict(
        image_path="img.jpg", brightness=120.0, saturation=0.05,
        vehicle_count=7, median_box_fraction=0.005, rickshaw_or_cng=False,
    )
    base.update(overrides)
    return curation.ImageFeature(**base)


def _sources(rows, columns=None):
    columns = columns or ["item_id", "filename", "type", "dataset", "split", "scene",
                          "source", "licence", "url", "condition_tags", "notes"]
    return pd.DataFrame(rows, columns=columns)


def _tagged_sources(rows):
    """A SOURCES frame in the post-proposal shape (with the two new provenance columns)."""
    columns = ["item_id", "filename", "type", "dataset", "split", "scene",
               "source", "licence", "url", "condition_tags", "notes", "tag_source", "tag_confirmed"]
    return pd.DataFrame(rows, columns=columns)


def _clean_row(**overrides):
    row = {
        "item_id": "tl0001", "filename": "tl0001_a.jpg", "type": "image",
        "dataset": "dhaka_ai", "split": "train", "scene": "train:blk0000",
        "source": "archive", "licence": "UNVERIFIED", "url": "",
        "condition_tags": "day", "notes": "", "tag_source": "cutoff_rule", "tag_confirmed": "",
    }
    row.update(overrides)
    return list(row.values())


# --- decide_tag_source -----------------------------------------------------
def test_decide_tag_source_seeks_cutoff_rule_for_existing_tags() -> None:
    assert tp.decide_tag_source("day;dense") == tp.CUTOFF_RULE_SOURCE
    assert tp.decide_tag_source("") == tp.AUTO_PROPOSED_SOURCE
    assert tp.decide_tag_source(None) == tp.AUTO_PROPOSED_SOURCE


# --- mid-band images get no proposal, never a fabricated tag ---------------
def test_assign_tags_leaves_a_mid_band_image_empty() -> None:
    assert curation.assign_tags(_feature(), _CUTOFFS) == []


def test_assign_tags_proposes_a_tag_for_an_extreme_image() -> None:
    assert curation.assign_tags(_feature(brightness=140.0), _CUTOFFS) == ["day"]


# --- review rendering is deterministic and non destructive -----------------
def test_compose_review_is_deterministic_and_leaves_original_intact() -> None:
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    before = image.copy()
    lines = tp.review_lines("tl0012", _feature(), _CUTOFFS, [])
    first = tp.compose_review(image, lines)
    second = tp.compose_review(image, lines)
    assert np.array_equal(first, second)
    assert np.array_equal(image, before)  # input array untouched
    assert first.shape == image.shape


def test_review_lines_report_features_and_empty_verdict() -> None:
    lines = tp.review_lines("tl0012", _feature(), _CUTOFFS, [])
    joined = " | ".join(lines)
    assert "id tl0012" in joined
    assert "brightness" in joined and "vehicle_count" in joined
    assert "NONE (unconfirmed)" in joined
    assert "day>=" in joined and "night<=" in joined


# --- archive matching by content -------------------------------------------
def test_match_archive_sources_resolves_exact_content(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    npImage = np.full((20, 20, 3), 123, dtype=np.uint8)
    cv2.imwrite(str(archive / "one.jpg"), npImage)
    copied = tmp_path / "copied"
    copied.mkdir()
    shutil.copy2(archive / "one.jpg", copied / "tl0001.jpg")
    other = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.imwrite(str(copied / "tl0002.jpg"), other)  # no archive twin

    resolved = tp.match_archive_sources(archive, {
        "tl0001": copied / "tl0001.jpg", "tl0002": copied / "tl0002.jpg",
    })
    assert set(resolved) == {"tl0001"}
    assert resolved["tl0001"].name == "one.jpg"


def test_match_archive_sources_is_empty_without_archive(tmp_path: Path) -> None:
    assert tp.match_archive_sources(tmp_path / "missing", {"tl0001": tmp_path / "x.jpg"}) == {}


# --- apply_updates preserves every existing row ----------------------------
def test_apply_updates_adds_provenance_and_preserves_existing_tags() -> None:
    sources = _sources([
        ["tl0001", "tl0001_a.jpg", "image", "dhaka_ai", "train", "s", "src", "lic", "", "day;dense", ""],
        ["tl0012", "tl0012_a.jpg", "image", "dhaka_ai", "train", "s", "src", "lic", "", "", ""],
    ])
    updates = {
        "tl0001": {"condition_tags": "day;dense", "tag_source": "cutoff_rule", "tag_confirmed": ""},
        "tl0012": {"condition_tags": "", "tag_source": "auto_proposed", "tag_confirmed": ""},
    }
    result = tp.apply_updates(sources, updates)
    assert list(result["tag_source"]) == ["cutoff_rule", "auto_proposed"]
    assert list(result["condition_tags"]) == ["day;dense", ""]
    assert all(result["tag_confirmed"] == "")


# --- checker accepts auto proposed rows ------------------------------------
def test_checker_accepts_blank_auto_proposed_row() -> None:
    sources = _tagged_sources([_clean_row(item_id="tl0012", filename="tl0012_a.jpg",
                                          condition_tags="", tag_source="auto_proposed")])
    errors, counts = ct.check(sources, pd.DataFrame(columns=["item_id"]), {"tl0012_a.jpg"})
    assert not any("no condition tags recorded" in e for e in errors)
    assert sum(counts) == 0


def test_checker_still_flags_blank_cutoff_rule_row() -> None:
    sources = _tagged_sources([_clean_row(item_id="tl0009", filename="tl0009_a.jpg",
                                          condition_tags="", tag_source="cutoff_rule")])
    errors, _ = ct.check(sources, pd.DataFrame(columns=["item_id"]), {"tl0009_a.jpg"})
    assert any("no condition tags recorded" in e and "tl0009" in e for e in errors)


def test_checker_accepts_auto_proposed_row_with_a_tag() -> None:
    sources = _tagged_sources([_clean_row(item_id="tl0012", filename="tl0012_a.jpg",
                                          condition_tags="night", tag_source="auto_proposed")])
    errors, counts = ct.check(sources, pd.DataFrame(columns=["item_id"]), {"tl0012_a.jpg"})
    assert errors == []
    assert counts[sorted(ct.VALID_CONDITION_TAGS).index("night")] == 1


def test_checker_legacy_frame_without_tag_source_is_strict() -> None:
    """A SOURCES file with no tag_source column keeps the old hard error on a blank tag."""
    sources = _sources([["tl0012", "tl0012_a.jpg", "image", "dhaka_ai", "train", "s",
                         "src", "lic", "", "", ""]])
    errors, _ = ct.check(sources, pd.DataFrame(columns=["item_id"]), {"tl0012_a.jpg"})
    assert any("no condition tags recorded" in e for e in errors)


def test_count_unconfirmed_tags_counts_only_blank_auto_proposed() -> None:
    sources = _tagged_sources([
        _clean_row(item_id="tl0001", tag_source="cutoff_rule", tag_confirmed=""),
        _clean_row(item_id="tl0007", filename="tl0007_a.jpg", tag_source="auto_proposed", tag_confirmed=""),
        _clean_row(item_id="tl0012", filename="tl0012_a.jpg", tag_source="auto_proposed", tag_confirmed="yes"),
    ])
    assert ct.count_unconfirmed_tags(sources) == 1
