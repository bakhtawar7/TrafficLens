"""Tests for the curation cleaning pass using tiny synthetic images.

Everything is written into a pytest ``tmp_path`` so no real dataset is read or changed.
The synthetic scene exercises each drop reason: unreadable file, missing annotation,
annotation mapped entirely to ignore, and a near duplicate frame.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from core import curation


def _write_image(path: Path, seed: int) -> None:
    """Write a small random but deterministic image so hashes differ across content."""
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    cv2.imwrite(str(path), array)


def _write_voc(xml_path: Path, label: str = "car") -> None:
    """Write a minimal PASCAL VOC annotation with one box."""
    xml_path.write_text(
        "<annotation>"
        "<filename>x.jpg</filename>"
        "<size><width>24</width><height>24</height><depth>3</depth></size>"
        "<object><name>" + label + "</name>"
        "<bndbox><xmin>2</xmin><ymin>2</ymin><xmax>20</xmax><ymax>20</ymax></bndbox>"
        "</object></annotation>",
        encoding="utf-8",
    )


def _build_dataset(root: Path) -> None:
    """Create a mixed dataset: keepable, unreadable, unannotated, and a duplicate."""
    root.mkdir(parents=True, exist_ok=True)
    _write_image(root / "keep_a.jpg", seed=1)
    _write_voc(root / "keep_a.xml")
    _write_image(root / "keep_b.jpg", seed=2)
    _write_voc(root / "keep_b.xml")
    _write_image(root / "dup_a.jpg", seed=3)
    _write_voc(root / "dup_a.xml")
    # Exact byte copy of dup_a, so its perceptual hash is identical.
    (root / "dup_b.jpg").write_bytes((root / "dup_a.jpg").read_bytes())
    _write_voc(root / "dup_b.xml")
    # A readable image with no annotation file.
    _write_image(root / "noxml.jpg", seed=4)
    # A file that cannot be decoded, with an annotation present.
    (root / "broken.jpg").write_bytes(b"this is not a real image")
    _write_voc(root / "broken.xml")


def test_clean_dataset_drops_by_reason(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "ds"
    _build_dataset(dataset_dir)
    log_path = tmp_path / "log.csv"

    report = curation.clean_dataset(
        dataset_dir, mapping_path=tmp_path / "missing_mapping.json", log_path=log_path
    )

    reasons = report.dropped_by_reason
    assert reasons.get(curation.DROP_NO_ANNOTATION, 0) >= 1  # noxml.jpg
    assert reasons.get(curation.DROP_UNREADABLE, 0) >= 1  # broken.jpg
    assert reasons.get(curation.DROP_DUPLICATE, 0) >= 1  # dup_b.jpg
    assert report.loaded == 6
    assert report.remaining == 3  # keep_a, keep_b, dup_a
    assert report.duplicates_removed >= 1


def test_log_written_with_rows(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "ds"
    _build_dataset(dataset_dir)
    log_path = tmp_path / "log.csv"

    curation.clean_dataset(
        dataset_dir, mapping_path=tmp_path / "missing.json", log_path=log_path
    )

    with log_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "expected at least one dropped image"
    assert {"image_path", "reason", "detail"} == set(rows[0].keys())


def test_ignore_mapping_drops_all_ignored(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "ds"
    _build_dataset(dataset_dir)
    # Reviewed mapping sends car to ignore, leaving no kept annotations.
    mapping_path = tmp_path / "class_mapping.json"
    mapping_path.write_text(
        json.dumps({"classes": {"car": {"proposed": "ignore"}}}), encoding="utf-8"
    )
    log_path = tmp_path / "log.csv"

    report = curation.clean_dataset(
        dataset_dir, mapping_path=mapping_path, log_path=log_path
    )

    assert report.dropped_by_reason.get(curation.DROP_ALL_IGNORED, 0) >= 1
    # Only the unreadable file survives no further; nothing with a car annotation is kept.
    assert all("broken" not in p for p in report.kept_paths)


def test_hamming_threshold_is_configurable(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    _write_image(dataset_dir / "a.jpg", seed=10)
    _write_voc(dataset_dir / "a.xml")
    # Exact byte copy, so the distance is 0 under any positive threshold.
    (dataset_dir / "b.jpg").write_bytes((dataset_dir / "a.jpg").read_bytes())
    _write_voc(dataset_dir / "b.xml")
    log_path = tmp_path / "log.csv"

    strict = curation.clean_dataset(
        dataset_dir, hamming_threshold=1, mapping_path=tmp_path / "m.json", log_path=log_path
    )
    assert strict.duplicates_removed == 1

    # Threshold 0 never treats anything as a duplicate because distance < 0 is impossible.
    off = curation.clean_dataset(
        dataset_dir, hamming_threshold=0, mapping_path=tmp_path / "m.json", log_path=log_path
    )
    assert off.duplicates_removed == 0


def test_original_folder_not_modified(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "ds"
    _build_dataset(dataset_dir)
    before = {p.name for p in dataset_dir.iterdir()}

    curation.clean_dataset(
        dataset_dir, mapping_path=tmp_path / "m.json", log_path=tmp_path / "log.csv"
    )

    after = {p.name for p in dataset_dir.iterdir()}
    assert before == after  # no files added or removed inside the dataset
