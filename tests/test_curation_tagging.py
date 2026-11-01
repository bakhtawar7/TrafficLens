"""Tests for quartile based feature measurement and tagging.

Everything is synthetic and written to ``tmp_path``: flat colour images give exact,
predictable brightness and saturation, and hand written annotations give exact counts and
box fractions. No test reads a real dataset and none uses a model detector.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from core import curation


def _flat_image(path: Path, bgr: tuple[int, int, int]) -> None:
    """Write a constant colour image so its luma and saturation are predictable."""
    array = np.zeros((40, 40, 3), dtype=np.uint8)
    array[:, :] = bgr
    cv2.imwrite(str(path), array)


def _voc(xml_path: Path, boxes: list[tuple[str, tuple[int, int, int, int]]]) -> None:
    """Write a VOC file with the given (label, xmin/ymin/xmax/ymax) objects."""
    parts = ["<annotation><filename>x.jpg</filename>",
             "<size><width>40</width><height>40</height><depth>3</depth></size>"]
    for label, (x0, y0, x1, y1) in boxes:
        parts.append(
            f"<object><name>{label}</name><bndbox>"
            f"<xmin>{x0}</xmin><ymin>{y0}</ymin><xmax>{x1}</xmax><ymax>{y1}</ymax>"
            f"</bndbox></object>"
        )
    parts.append("</annotation>")
    xml_path.write_text("".join(parts), encoding="utf-8")


def test_saturation_fraction(tmp_path: Path) -> None:
    red = np.zeros((10, 10, 3), dtype=np.uint8)
    red[:, :] = (0, 0, 255)  # BGR pure red, fully saturated
    grey = np.full((10, 10, 3), 128, dtype=np.uint8)
    assert curation.saturation_fraction(red) == 1.0
    assert curation.saturation_fraction(grey) == 0.0


def test_compute_feature_values(tmp_path: Path) -> None:
    image = tmp_path / "a.jpg"
    _flat_image(image, (0, 0, 0))  # black -> brightness near zero
    _voc(image.with_suffix(".xml"), [("car", (10, 10, 30, 30))])  # 20x20 box on 40x40

    feature = curation.compute_feature(image)
    assert feature.brightness < 1.0
    assert feature.saturation == 0.0
    assert feature.vehicle_count == 1
    assert feature.median_box_fraction == 0.25  # 400 / 1600
    assert feature.rickshaw_or_cng is False


def test_compute_feature_detects_rickshaw_and_cng(tmp_path: Path) -> None:
    image = tmp_path / "b.jpg"
    _flat_image(image, (255, 255, 255))
    _voc(image.with_suffix(".xml"), [("rickshaw", (5, 5, 25, 25))])
    assert curation.compute_feature(image).rickshaw_or_cng is True

    cng = tmp_path / "c.jpg"
    _flat_image(cng, (255, 255, 255))
    _voc(cng.with_suffix(".xml"), [("Three Wheelers (CNG)", (5, 5, 25, 25))])
    assert curation.compute_feature(cng).rickshaw_or_cng is True


def _synthetic_features() -> list[curation.ImageFeature]:
    return [
        curation.ImageFeature("1", brightness=10.0, saturation=0.05, vehicle_count=2,
                              median_box_fraction=0.01, rickshaw_or_cng=False),
        curation.ImageFeature("2", brightness=120.0, saturation=0.40, vehicle_count=20,
                              median_box_fraction=0.02, rickshaw_or_cng=True),
        curation.ImageFeature("3", brightness=200.0, saturation=0.60, vehicle_count=40,
                              median_box_fraction=0.30, rickshaw_or_cng=False),
        curation.ImageFeature("4", brightness=180.0, saturation=0.80, vehicle_count=30,
                              median_box_fraction=0.50, rickshaw_or_cng=True),
    ]


def test_compute_cutoffs_match_numpy_quartiles() -> None:
    features = _synthetic_features()
    cutoffs = curation.compute_cutoffs(features)
    brightness = np.asarray([f.brightness for f in features])
    assert cutoffs["brightness"]["q1"] == float(np.percentile(brightness, 25))
    assert cutoffs["brightness"]["q3"] == float(np.percentile(brightness, 75))
    assert set(cutoffs) == set(curation.FEATURE_NAMES)


def test_assign_tags_and_counts() -> None:
    features = _synthetic_features()
    cutoffs = curation.compute_cutoffs(features)
    counts = curation.tag_counts(features, cutoffs)
    # The brightest two clear the q3 brightness line, the darkest two the q1 line.
    assert counts["day"] == sum(
        1 for f in features if f.brightness >= cutoffs["brightness"]["q3"]
    )
    assert counts["dense"] == sum(
        1 for f in features if f.vehicle_count >= cutoffs["vehicle_count"]["q3"]
    )
    # Every counted image is within the pool size.
    assert all(counts[tag] <= len(features) for tag in counts)


def test_write_tag_cutoffs_before_sampling(tmp_path: Path) -> None:
    cutoffs = curation.compute_cutoffs(_synthetic_features())
    out = tmp_path / "configs" / "tag_cutoffs.json"
    curation.write_tag_cutoffs(cutoffs, source="x", images_used=4, path=out)

    assert out.exists()  # written up front, before any sampling step
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["images_used"] == 4
    assert set(payload["cutoffs"]) == set(curation.FEATURE_NAMES)
    assert set(payload["tag_rules"]) == set(curation.TAG_RULES)
    assert "before any" in payload["_note"]
