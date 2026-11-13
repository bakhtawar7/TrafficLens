"""Regression coverage for upload, model-fallback, and history failure paths."""
from __future__ import annotations

import numpy as np
import pytest

from app.ui.image_tab import _count_with_pretrained_fallback
from app.ui.reliability_tab import _run_models
from core import config, history
from core.detector import Detection


class FailingDetector:
    family = "yolo"

    def __init__(self, name: str, message: str = "offline or incomplete download"):
        self.name = name
        self.message = message

    def detect(self, image, confidence_threshold, **kwargs):
        raise OSError(self.message)


class WorkingDetector:
    family = "yolo"
    name = "YOLOv8s"

    def detect(self, image, confidence_threshold, **kwargs):
        return [Detection(2, "car", 0.9, (0.0, 0.0, 10.0, 10.0))]


def test_image_count_uses_pretrained_fallback_and_surfaces_failure_reason():
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    result, failures = _count_with_pretrained_fallback(
        image,
        [FailingDetector("YOLOv8n", "connection dropped"), WorkingDetector()],
        list(config.TARGET_CLASS_NAMES),
        0.25,
    )
    assert result.model_name == "YOLOv8s"
    assert result.counts_by_class["car"] == 1
    assert failures == [("YOLOv8n", "OSError: connection dropped")]


def test_image_count_reports_all_pretrained_failures():
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="No pretrained detector completed inference"):
        _count_with_pretrained_fallback(
            image,
            [FailingDetector("YOLOv8n"), FailingDetector("YOLOv8s")],
            list(config.TARGET_CLASS_NAMES),
            0.25,
        )


def test_reliability_omits_failed_model_and_returns_reason(monkeypatch):
    class SettingsStub:
        @staticmethod
        def load_settings(path):
            return {
                "models": {
                    "yolov8n": {"input_size": 640, "uses_nms": True, "nms_iou": 0.7},
                    "yolov8s": {"input_size": 640, "uses_nms": True, "nms_iou": 0.7},
                },
                "sliced_tiles": {},
            }

    monkeypatch.setattr("app.ui.reliability_tab.settings", SettingsStub)
    good = WorkingDetector()
    good.name = "YOLOv8s"
    rows, detections, families, raw, timings, failures = _run_models(
        np.zeros((24, 32, 3), dtype=np.uint8),
        [FailingDetector("YOLOv8n"), good],
        ["car"],
        0.25,
        False,
    )
    assert [row["Model"] for row in rows] == ["YOLOv8s"]
    assert list(detections) == ["YOLOv8s"]
    assert list(raw) == ["yolov8s"]
    assert timings["YOLOv8s"] >= 0
    assert failures == [
        ("YOLOv8n", "OSError: offline or incomplete download")
    ]


def test_locked_history_database_returns_clear_error_without_blocking_counting(tmp_path):
    database = tmp_path / "history.db"
    history.migrate(database)
    blocker = __import__("sqlite3").connect(database, timeout=0.1)
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(history.HistoryStoreError, match="unavailable or corrupt"):
            history.record_run(
                {
                    "input_filename": "road.jpg",
                    "image_sha256": "sha",
                    "width": 10,
                    "height": 10,
                    "settings_sha256": "settings",
                    "mode": "standard",
                    "selected_classes": ["car"],
                    "threshold": 0.25,
                    "rickshaw_mode": False,
                },
                [],
                path=database,
            )
    finally:
        blocker.rollback()
        blocker.close()
