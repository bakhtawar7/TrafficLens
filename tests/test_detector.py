"""Tests for the Detector protocol, the model registry, and the ultralytics wrappers.

The protocol conformance tests run offline. The warm-up and load-timing tests inject a
fake ultralytics module into ``sys.modules`` so no weights are touched. The real
inference test needs downloaded model weights and network on first use, so it is marked
``requires_weights`` and skipped unless the environment opt in flag is set.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

from core import config
from core.detector import Detection, Detector, RtdetrDetector, YoloDetector, build_detector


def test_yolo_detector_satisfies_protocol():
    detector = YoloDetector()
    assert isinstance(detector, Detector)
    assert hasattr(detector, "detect")


def test_detection_is_frozen_and_typed():
    detection = Detection(2, "car", 0.5, (1.0, 2.0, 3.0, 4.0))
    assert detection.class_id == 2
    assert detection.xyxy == (1.0, 2.0, 3.0, 4.0)
    with pytest.raises(Exception):
        detection.class_id = 3  # frozen dataclass rejects mutation


@pytest.mark.requires_weights
@pytest.mark.skipif(
    os.environ.get("TRAFFICLENS_RUN_WEIGHTS") != "1",
    reason="requires YOLO model weights; set TRAFFICLENS_RUN_WEIGHTS=1 to run",
)
def test_yolo_detector_real_inference():
    detector = YoloDetector()
    image = np.zeros((320, 320, 3), dtype=np.uint8)
    detections = detector.detect(image, 0.25)
    assert isinstance(detections, list)


# ---------------------------------------------------------------------------
# Model registry and factory, offline
# ---------------------------------------------------------------------------

def test_model_registry_lists_three_pretrained_models() -> None:
    assert set(config.MODEL_REGISTRY) == {"yolov8n", "yolov8s", "rtdetr-l"}
    for spec in config.MODEL_REGISTRY.values():
        assert spec.display_name
        assert spec.family in config.MODEL_FAMILIES
        assert spec.weights_file.endswith(".pt")


def test_build_detector_follows_the_registry() -> None:
    fast = build_detector("yolov8s")
    assert isinstance(fast, YoloDetector)
    assert isinstance(fast, Detector)
    assert (fast.name, fast.family, fast.weights) == ("YOLOv8s", "yolo", "yolov8s.pt")

    detr = build_detector("rtdetr-l")
    assert isinstance(detr, RtdetrDetector)
    assert isinstance(detr, Detector)
    assert (detr.name, detr.family, detr.weights) == ("RT-DETR-L", "detr", "rtdetr-l.pt")


def test_default_yolo_detector_keeps_its_week_one_identity() -> None:
    # Regression guard: the bare constructor must still carry the legacy name and weights.
    assert YoloDetector().name == config.DEFAULT_MODEL_NAME
    assert YoloDetector().weights == config.DEFAULT_WEIGHTS_FILE


def test_every_registered_detector_offers_warmup() -> None:
    for key in config.MODEL_REGISTRY:
        assert callable(getattr(build_detector(key), "warmup", None))


# ---------------------------------------------------------------------------
# Warmup and load timing with a fake ultralytics module (no weights, no network)
# ---------------------------------------------------------------------------

class _FakeWrapper:
    """Stand-in for the ultralytics YOLO/RTDETR classes with a per subclass counter."""

    names = {0: "person"}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.calls = 0

    def __init__(self, weights: str) -> None:
        type(self).calls += 1
        self.weights = weights

    def predict(self, image, **kwargs):  # noqa: ANN001, ANN201 - fake model surface
        return []


class FakeYOLO(_FakeWrapper):
    pass


class FakeRTDETR(_FakeWrapper):
    pass


def _install_fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.SimpleNamespace(YOLO=FakeYOLO, RTDETR=FakeRTDETR)
    monkeypatch.setitem(sys.modules, "ultralytics", module)


def test_warmup_records_load_once_and_reuses_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeYOLO.calls = 0
    _install_fake_ultralytics(monkeypatch)

    detector = build_detector("yolov8n")
    assert detector.load_seconds is None  # lazy: construction loads nothing

    detector.warmup()
    assert detector.load_seconds is not None
    assert detector.load_seconds >= 0.0
    assert FakeYOLO.calls == 1

    detections = detector.detect(np.zeros((32, 32, 3), dtype=np.uint8), 0.25)
    assert detections == []
    assert FakeYOLO.calls == 1  # inference did not trigger a second load


def test_rtdetr_warmup_uses_the_detr_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRTDETR.calls = 0
    FakeYOLO.calls = 0
    _install_fake_ultralytics(monkeypatch)

    detector = build_detector("rtdetr-l")
    detector.warmup()
    assert FakeRTDETR.calls == 1
    assert FakeYOLO.calls == 0
    assert detector.load_seconds is not None
