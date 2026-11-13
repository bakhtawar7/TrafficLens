"""Tests for JSON and CSV export field completeness and file naming."""
from __future__ import annotations

import csv
import io
import json

from core import export
from core.counting import build_count_result
from core.detector import Detection

_DETECTIONS = [
    Detection(2, "car", 0.88, (0.0, 0.0, 50.0, 50.0)),
    Detection(7, "truck", 0.51, (10.0, 10.0, 110.0, 90.0)),
]


def _result():
    return build_count_result(
        detections=_DETECTIONS,
        selected_classes=["car", "motorcycle", "bus", "truck"],
        confidence_threshold=0.25,
        model_name="yolov8n",
        image_width=640,
        image_height=480,
        inference_ms=12.5,
    )


def test_payload_has_all_required_fields():
    payload = export.build_payload(_result(), "street.jpg", timestamp_utc="2026-01-01T00:00:00Z")
    required = {
        "input_filename",
        "timestamp_utc",
        "model",
        "weights_sha256",
        "confidence_threshold",
        "selected_classes",
        "image_width",
        "image_height",
        "inference_ms",
        "total_count",
        "counts_by_class",
        "detections",
    }
    assert required.issubset(payload.keys())
    assert payload["total_count"] == 2
    assert payload["counts_by_class"] == {"car": 1, "motorcycle": 0, "bus": 0, "truck": 1}


def test_detection_entries_have_expected_shape():
    payload = export.build_payload(_result(), "street.jpg", timestamp_utc="2026-01-01T00:00:00Z")
    for detection in payload["detections"]:
        assert set(detection) == {"class", "confidence", "bbox_xyxy"}
        assert len(detection["bbox_xyxy"]) == 4


def test_json_bytes_round_trip():
    payload = export.build_payload(
        _result(), "street.jpg", timestamp_utc="2026-01-01T00:00:00Z",
        weights_sha256="deadbeef",
    )
    decoded = json.loads(export.to_json_bytes(payload).decode("utf-8"))
    assert decoded["model"] == "yolov8n"
    assert decoded["weights_sha256"] == "deadbeef"
    assert decoded["total_count"] == 2


def test_csv_has_scalar_and_per_class_columns():
    payload = export.build_payload(_result(), "street.jpg", timestamp_utc="2026-01-01T00:00:00Z")
    text = export.to_csv_bytes(payload).decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    row = next(reader)
    for field in export.SCALAR_FIELDS:
        assert field in row
    for class_name in payload["counts_by_class"]:
        assert class_name in row
    assert row["car"] == "1" and row["truck"] == "1" and row["bus"] == "0"


def test_finetuned_model_name_and_weights_hash_are_in_both_exports():
    model_name = "YOLOv8n Rickshaw Fine-tuned"
    weights_hash = "abc123"
    result = build_count_result(
        detections=[Detection(0, "rickshaw", 0.9, (0.0, 0.0, 20.0, 20.0))],
        selected_classes=["rickshaw"],
        confidence_threshold=0.25,
        model_name=model_name,
        image_width=100,
        image_height=100,
        inference_ms=1.0,
    )
    payload = export.build_payload(
        result,
        "street.jpg",
        timestamp_utc="2026-01-01T00:00:00Z",
        weights_sha256=weights_hash,
    )
    json_payload = json.loads(export.to_json_bytes(payload).decode("utf-8"))
    csv_row = next(csv.DictReader(io.StringIO(export.to_csv_bytes(payload).decode("utf-8"))))
    assert json_payload["model"] == model_name
    assert json_payload["weights_sha256"] == weights_hash
    assert csv_row["model"] == model_name
    assert csv_row["weights_sha256"] == weights_hash


def test_export_filename_pattern():
    name = export.make_json_filename("street.jpg", "2026-01-01T12:30:00Z")
    assert name.startswith("trafficlens_street_")
    assert name.endswith(".json")
    assert "20260101123000Z" in name
