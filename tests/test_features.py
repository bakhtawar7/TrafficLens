"""Tests for core.features and scripts/build_features.py.

Everything is synthetic and offline: hand built Detection lists, constant and
patterned numpy images, and temporary cache/settings files. No label file or
annotation is read anywhere in this module, mirroring the feature contract.
Each feature group is checked on empty input and a single detection as well as
the ordinary case.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from core import config, features
from core.detector import Detection
from scripts import build_features as bf


def _det(conf: float = 0.9, box: tuple = (0.0, 0.0, 10.0, 10.0),
         cls: tuple = (2, "car")) -> Detection:
    return Detection(cls[0], cls[1], conf, box)


def _flat_image(value: int = 100, size: int = 32) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    encoded.tofile(str(path))


# ---------------------------------------------------------------------------
# Confidence features
# ---------------------------------------------------------------------------
def test_confidence_features_empty_is_missing():
    assert features.confidence_features([]) == {
        "mean_confidence": None, "min_confidence": None, "mid_band_share": None}


def test_confidence_features_single_detection():
    inside = features.confidence_features([_det(conf=0.3)])
    assert inside == {"mean_confidence": 0.3, "min_confidence": 0.3, "mid_band_share": 1.0}
    outside = features.confidence_features([_det(conf=0.9)])
    assert outside["mid_band_share"] == 0.0
    assert outside["mean_confidence"] == 0.9


def test_confidence_features_band_bounds_are_inclusive():
    detections = [_det(conf=0.25), _det(conf=0.50), _det(conf=0.249), _det(conf=0.501)]
    result = features.confidence_features(detections)
    assert result["mid_band_share"] == pytest.approx(0.5)
    assert result["min_confidence"] == pytest.approx(0.249)
    assert result["mean_confidence"] == pytest.approx((0.25 + 0.50 + 0.249 + 0.501) / 4)


# ---------------------------------------------------------------------------
# Threshold sensitivity
# ---------------------------------------------------------------------------
def test_threshold_sensitivity_spans_the_sweep():
    detections = [_det(conf=0.15), _det(conf=0.25), _det(conf=0.35), _det(conf=0.50)]
    # totals at (0.15, 0.25, 0.35, 0.50) are 4, 3, 2, 1.
    assert features.threshold_sensitivity(detections) == 3.0


def test_threshold_sensitivity_empty_and_single():
    assert features.threshold_sensitivity([]) == 0.0
    assert features.threshold_sensitivity([_det(conf=0.9)]) == 0.0
    assert features.threshold_sensitivity([_det(conf=0.15)]) == 1.0


def test_threshold_sensitivity_ignores_non_vehicle_classes():
    detections = [_det(conf=0.15, cls=(0, "person")), _det(conf=0.9)]
    assert features.threshold_sensitivity(detections) == 0.0


# ---------------------------------------------------------------------------
# Scene features
# ---------------------------------------------------------------------------
def test_scene_features_missing_image_is_missing():
    assert features.scene_features(None) == {
        "mean_brightness": None, "contrast": None, "saturated_share": None, "blur": None}


def test_scene_features_flat_image():
    result = features.scene_features(_flat_image(100))
    assert result["mean_brightness"] == pytest.approx(100.0)
    assert result["contrast"] == pytest.approx(0.0)
    assert result["saturated_share"] == 0.0
    assert result["blur"] == pytest.approx(0.0)


def test_scene_features_saturated_image():
    result = features.scene_features(_flat_image(255))
    assert result["saturated_share"] == 1.0
    assert result["mean_brightness"] == pytest.approx(255.0)


def test_scene_features_split_image_contrast_and_saturation():
    image = _flat_image(0)
    image[:, 16:, :] = 255
    result = features.scene_features(image)
    assert result["mean_brightness"] == pytest.approx(127.5)
    assert result["contrast"] == pytest.approx(127.5)
    assert result["saturated_share"] == pytest.approx(0.5)
    assert result["blur"] > 0.0


def test_scene_features_saturation_is_any_channel_clipped():
    image = _flat_image(100)
    image[0, 0] = (100, 100, 255)  # one channel clipped, luma far below 245
    result = features.scene_features(image)
    assert result["saturated_share"] == pytest.approx(1 / (32 * 32))


# ---------------------------------------------------------------------------
# Geometry features
# ---------------------------------------------------------------------------
def test_geometry_features_empty():
    assert features.geometry_features([], 1000, 1000) == {
        "median_box_area_fraction": None, "small_box_share": None, "detection_density": 0.0}


def test_geometry_features_single_box():
    result = features.geometry_features([_det(box=(0.0, 0.0, 100.0, 100.0))], 1000, 1000)
    assert result["median_box_area_fraction"] == pytest.approx(0.01)
    assert result["small_box_share"] == 0.0
    assert result["detection_density"] == pytest.approx(1.0)  # one box per megapixel


def test_geometry_features_small_threshold_is_strict():
    # 10x10 on 1000x1000 is 0.0001 (small); 100x10 is exactly 0.001 (not small).
    detections = [_det(box=(0.0, 0.0, 10.0, 10.0)), _det(box=(0.0, 0.0, 100.0, 10.0))]
    result = features.geometry_features(detections, 1000, 1000)
    assert result["small_box_share"] == pytest.approx(0.5)
    assert result["median_box_area_fraction"] == pytest.approx((0.0001 + 0.001) / 2)
    assert result["detection_density"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Whole-image features and agreement
# ---------------------------------------------------------------------------
def _all_models(detections: list[Detection]) -> dict[str, list[Detection]]:
    return {key: list(detections) for key in config.MODEL_REGISTRY}


def test_image_features_identical_models_agree_perfectly():
    row = features.image_features(_all_models([_det()]), _flat_image())
    for first, second in (("YOLOv8n", "YOLOv8s"), ("YOLOv8n", "RT-DETR-L"), ("YOLOv8s", "RT-DETR-L")):
        assert row[f"{first} vs {second} Box Agreement"] == 1.0
        assert row[f"{first} vs {second} Count Disagreement"] == 0.0
    assert row["within_family_box_agreement"] == 1.0
    assert row["within_family_count_disagreement"] == 0.0
    # Cross-family aggregate uses best YOLO (YOLOv8s) versus RT-DETR-L only.
    assert row["cross_family_box_agreement"] == 1.0
    assert row["cross_family_count_disagreement"] == 0.0
    assert set(row) == set(features.TABLE_COLUMNS[2:])


def test_image_features_disjoint_boxes_break_agreement():
    detections = {
        "yolov8n": [_det(box=(0.0, 0.0, 10.0, 10.0))],
        "yolov8s": [_det(box=(500.0, 500.0, 510.0, 510.0))],
        "rtdetr-l": [_det(box=(900.0, 900.0, 910.0, 910.0))],
    }
    row = features.image_features(detections, _flat_image())
    assert row["YOLOv8n vs YOLOv8s Box Agreement"] == 0.0
    assert row["within_family_box_agreement"] == 0.0
    assert row["cross_family_box_agreement"] == 0.0


def test_image_features_empty_detection_sets():
    row = features.image_features(_all_models([]), _flat_image())
    assert row["yolov8n_mean_confidence"] is None
    assert row["yolov8n_threshold_sensitivity"] == 0.0
    # Two empty sets agree by the H2 definition.
    assert row["YOLOv8n vs YOLOv8s Box Agreement"] == 1.0
    assert row["YOLOv8n vs YOLOv8s Count Disagreement"] == 0.0
    assert row["median_box_area_fraction"] is None
    assert row["detection_density"] == 0.0


def test_image_features_missing_model_leaves_columns_missing():
    row = features.image_features({"yolov8n": [_det()]}, _flat_image())
    assert row["yolov8s_mean_confidence"] is None
    assert row["rtdetr-l_threshold_sensitivity"] is None
    assert row["YOLOv8n vs YOLOv8s Box Agreement"] is None
    assert row["within_family_box_agreement"] is None
    assert row["cross_family_box_agreement"] is None
    assert row["yolov8n_mean_confidence"] == pytest.approx(0.9)


def test_image_features_missing_image_leaves_pixel_columns_missing():
    row = features.image_features(_all_models([_det()]), None)
    assert row["mean_brightness"] is None
    assert row["blur"] is None
    assert row["median_box_area_fraction"] is None
    assert row["detection_density"] is None
    assert row["yolov8n_mean_confidence"] == pytest.approx(0.9)


def test_image_features_filters_at_default_confidence():
    # The 0.2 box survives the 0.15 cache load but not the default 0.25 filter.
    row = features.image_features(_all_models([_det(conf=0.9), _det(conf=0.2)]), _flat_image())
    assert row["yolov8n_mean_confidence"] == pytest.approx(0.9)
    assert row["yolov8n_threshold_sensitivity"] == 1.0


# ---------------------------------------------------------------------------
# Feature table, summaries, and the recompute decision
# ---------------------------------------------------------------------------
@pytest.fixture
def table_workspace(tmp_path: Path) -> dict:
    images = tmp_path / "images"
    images.mkdir()
    _write_image(images / "tl0001_dhaka_ai.png", _flat_image(100))
    cache = {
        (item, key, "standard"): [_det()]
        for item in (1, 2) for key in config.MODEL_REGISTRY
    }
    return {"cache": cache, "image_paths": {1: images / "tl0001_dhaka_ai.png"},
            "output": tmp_path / "features.csv"}


def test_build_feature_table_rows_and_missing_image(table_workspace):
    ws = table_workspace
    rows = features.build_feature_table(ws["cache"], ws["image_paths"], "hash123")
    assert [row["item_id"] for row in rows] == ["tl0001", "tl0002"]
    for row in rows:
        assert list(row) == list(features.TABLE_COLUMNS)
        assert row["settings_sha256"] == "hash123"
    assert rows[0]["mean_brightness"] == pytest.approx(100.0)
    # Item 2 has detections but no image file: pixel and geometry columns missing.
    assert rows[1]["mean_brightness"] is None
    assert rows[1]["detection_density"] is None
    assert rows[1]["yolov8n_mean_confidence"] == pytest.approx(0.9)


def test_summarise_features_reports_means_ranges_and_missing(table_workspace):
    rows = features.build_feature_table(table_workspace["cache"], table_workspace["image_paths"], "h")
    summary = features.summarise_features(rows)
    brightness = summary["mean_brightness"]
    assert brightness["mean"] == pytest.approx(100.0)
    assert brightness["min"] == pytest.approx(100.0)
    assert brightness["max"] == pytest.approx(100.0)
    assert brightness["missing"] == 1
    assert summary["yolov8n_mean_confidence"]["missing"] == 0


def _write_table(path: Path, hashes: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(features.TABLE_COLUMNS))
        writer.writeheader()
        for value in hashes:
            writer.writerow({"item_id": "tl0001", "settings_sha256": value})


def test_needs_recompute_decision(tmp_path):
    path = tmp_path / "features.csv"
    assert features.needs_recompute(path, "current") is True   # missing file
    _write_table(path, ["current", "current"])
    assert features.needs_recompute(path, "current") is False  # same hash everywhere
    assert features.needs_recompute(path, "changed") is True   # hash moved on
    _write_table(path, ["current", "older"])
    assert features.needs_recompute(path, "current") is True   # mixed stamps
    _write_table(path, [])
    assert features.needs_recompute(path, "current") is True   # header only


# ---------------------------------------------------------------------------
# Runner: write, reuse, recompute, and aggregate-only printing
# ---------------------------------------------------------------------------
@pytest.fixture
def runner_workspace(tmp_path: Path) -> dict:
    images = tmp_path / "images"
    images.mkdir()
    _write_image(images / "tl0001_dhaka_ai.png", _flat_image(100))
    cache_path = tmp_path / "raw_detections.jsonl"
    entries = [
        {"image": "tl0001_dhaka_ai.png", "model": key, "mode": "standard",
         "detections": [{"class_id": 2, "class_name": "car", "confidence": 0.9,
                         "xyxy": [0.0, 0.0, 10.0, 10.0]}]}
        for key in config.MODEL_REGISTRY
    ]
    cache_path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n",
                          encoding="utf-8")
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"models": {}}), encoding="utf-8")
    return {"images": images, "cache": cache_path, "settings": settings_path,
            "output": tmp_path / "features.csv"}


def _runner_args(ws: dict) -> list[str]:
    return ["--cache", str(ws["cache"]), "--images-dir", str(ws["images"]),
            "--settings", str(ws["settings"]), "--output", str(ws["output"])]


def test_runner_writes_table_and_prints_aggregates_only(runner_workspace, capsys):
    ws = runner_workspace
    assert bf.main(_runner_args(ws)) == 0
    with ws["output"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["item_id"] == "tl0001"
    expected = hashlib.sha256(ws["settings"].read_bytes()).hexdigest()
    assert rows[0]["settings_sha256"] == expected
    printed = capsys.readouterr().out
    assert "mean=" in printed and "missing=" in printed
    assert "tl0001" not in printed          # no per-image values in the console
    assert "detections" not in printed.replace("cached standard-mode detections", "")


def test_runner_reuses_table_when_hash_unchanged(runner_workspace, monkeypatch):
    ws = runner_workspace
    assert bf.main(_runner_args(ws)) == 0

    def _never_called(*args, **kwargs):
        raise AssertionError("feature table recomputed despite an unchanged settings hash")

    monkeypatch.setattr(features, "build_feature_table", _never_called)
    assert bf.main(_runner_args(ws)) == 0


def test_runner_recomputes_when_hash_changes(runner_workspace, monkeypatch):
    ws = runner_workspace
    assert bf.main(_runner_args(ws)) == 0
    ws["settings"].write_text(json.dumps({"models": {"yolov8n": {"input_size": 1280}}}),
                              encoding="utf-8")
    calls = []
    original = features.build_feature_table

    def _spy(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(features, "build_feature_table", _spy)
    assert bf.main(_runner_args(ws)) == 0
    assert calls == [True]
    with ws["output"].open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["settings_sha256"] == hashlib.sha256(ws["settings"].read_bytes()).hexdigest()


def test_runner_missing_settings_and_empty_cache_fail(runner_workspace, tmp_path):
    ws = runner_workspace
    args = _runner_args(ws)
    args[args.index("--settings") + 1] = str(tmp_path / "absent.json")
    assert bf.main(args) == 1
    empty_cache = tmp_path / "empty.jsonl"
    empty_cache.write_text("", encoding="utf-8")
    args = _runner_args(ws)
    args[args.index("--cache") + 1] = str(empty_cache)
    assert bf.main(args) == 1