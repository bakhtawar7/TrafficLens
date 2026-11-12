"""Synthetic box matching and unverified H3 workflow tests."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core import confusion


def _annotation(box=(0.0, 0.0, 10.0, 10.0), category="rickshaw"):
    return confusion.AnnotationBox(
        box_id="box-a",
        item_id="tl0001",
        image_path=Path("synthetic.jpg"),
        relative_image="synthetic.jpg",
        dataset_class="rickshaw",
        category=category,
        xyxy=box,
        scene="scene-a",
    )


def _detection(class_name="car", confidence=0.9, box=(0.0, 0.0, 10.0, 10.0)):
    return confusion.DetectionBox(class_name, confidence, box)


def test_exact_box_match_reports_detector_class():
    result = confusion.match_annotations([_annotation()], [_detection("car")])
    assert result == [{
        "box_id": "box-a",
        "outcome": "car",
        "matched_detection_class": "car",
    }]


def test_partial_overlap_at_iou_threshold_matches_class_agnostically():
    # 6x10 intersection over a 10x10 union is exactly 0.6.
    result = confusion.match_annotations(
        [_annotation()], [_detection("bus", box=(0.0, 0.0, 6.0, 10.0))]
    )
    assert result[0]["outcome"] == "bus"


def test_partial_overlap_below_threshold_is_not_detected():
    result = confusion.match_annotations(
        [_annotation()], [_detection("car", box=(0.0, 0.0, 4.0, 10.0))]
    )
    assert result[0]["outcome"] == "not seen"


def test_absent_detection_reports_not_seen():
    result = confusion.match_annotations([_annotation()], [])
    assert result[0]["outcome"] == "not seen"
    assert result[0]["matched_detection_class"] == ""


def test_class_swap_is_reported_as_detector_class_not_as_a_miss():
    result = confusion.match_annotations([_annotation()], [_detection("truck")])
    assert result[0]["outcome"] == "truck"


def test_low_confidence_is_separate_from_not_seen():
    result = confusion.match_annotations(
        [_annotation()],
        [_detection("car", confidence=0.2)],
    )
    assert result[0]["outcome"] == "low confidence"


def test_greedy_matching_never_reuses_a_detection():
    annotations = [
        _annotation(box=(0, 0, 10, 10)),
        _annotation(box=(1, 0, 11, 10)),
    ]
    result = confusion.match_annotations(annotations, [_detection()])
    assert sum(row["outcome"] == "car" for row in result) == 1
    assert sum(row["outcome"] == "not seen" for row in result) == 1


def test_summary_uses_scene_level_denominators_and_marks_small_categories():
    rows = [
        {
            "category": "rickshaw", "model": "yolov8n", "scene": f"scene-{i % 2}",
            "outcome": "low confidence" if i == 0 else "car",
        }
        for i in range(3)
    ]
    summary = confusion.summarize_outcomes(rows, categories=("rickshaw",),
                                           models=("yolov8n",), samples=40, seed=7)
    assert summary[0]["annotated_boxes"] == 3
    assert summary[0]["scene_count"] == 2
    assert summary[0]["miss_count_at_default"] == 1
    assert summary[0]["low confidence_count"] == 1
    assert summary[0]["category_exploratory_below_50_boxes"] is True
    assert summary[0]["annotation_status"] == "annotation referenced, unverified"
    assert summary[0]["scope_status"] == "exploratory"


def test_verified_box_class_replaces_dataset_mapping():
    box = _annotation(category="rickshaw")
    revised = confusion.apply_verified_classes(
        [box], {"box-a": "car"}, {"car": "car"}
    )
    assert revised[0].category == "car"
    assert box.category == "rickshaw"


def test_audit_crops_show_dataset_class_without_detection_overlay(tmp_path: Path):
    source = tmp_path / "source.jpg"
    image = np.full((80, 100, 3), 120, dtype=np.uint8)
    cv2.rectangle(image, (35, 25), (65, 55), (0, 255, 0), -1)
    assert cv2.imwrite(str(source), image)
    annotations = [
        confusion.AnnotationBox(
            box_id=f"box-{index}",
            item_id="",
            image_path=source,
            relative_image="source.jpg",
            dataset_class="three wheelers (CNG)",
            category="rickshaw",
            xyxy=(35.0, 25.0, 65.0, 55.0),
            scene="source",
        )
        for index in range(30)
    ]
    result = confusion.create_box_audit(annotations, tmp_path / "audit", count=30, seed=1)
    assert result["selected_boxes"] == 30
    sheet = tmp_path / "audit" / "verification_sheet.csv"
    contents = sheet.read_text(encoding="utf-8")
    assert "verified_class" in contents
    assert "annotation referenced, unverified" in contents
    assert "exploratory" in contents
    assert len(list((tmp_path / "audit").glob("box_*.png"))) == 30
    crop = cv2.imread(str(tmp_path / "audit" / "box_01.png"))
    assert crop is not None
    assert crop.shape[0] > 30 and crop.shape[1] > 30
