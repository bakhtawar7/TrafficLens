"""Tests for the Detector protocol, the model registry, and the ultralytics wrappers.

The protocol conformance tests run offline. The warm-up and load-timing tests inject a
fake ultralytics module into ``sys.modules`` so no weights are touched. The real
inference test needs downloaded model weights and network on first use, so it is marked
``requires_weights`` and skipped unless the environment opt in flag is set. The
SlicedDetector tests use a synthetic blob detector, so
they never need weights either.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

from core import config
from core.detector import (
    Detection,
    Detector,
    RtdetrDetector,
    SlicedDetector,
    YoloDetector,
    build_detector,
)


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

def test_model_registry_keeps_pretrained_models_and_adds_finetuned_model() -> None:
    assert set(config.PRETRAINED_MODEL_KEYS) == {"yolov8n", "yolov8s", "rtdetr-l"}
    assert "yolov8n-rickshaw-finetuned" in config.MODEL_REGISTRY
    assert config.MODEL_REGISTRY["yolov8n-rickshaw-finetuned"].finetuned
    assert config.MODEL_REGISTRY["yolov8n-rickshaw-finetuned"].weights_sha256
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
        self.predict_calls = []

    def predict(self, image, **kwargs):  # noqa: ANN001, ANN201 - fake model surface
        self.predict_calls.append(kwargs)
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

    finetuned = build_detector(config.FINETUNED_MODEL_KEY)
    assert finetuned.name == config.FINETUNED_MODEL_DISPLAY_NAME
    assert finetuned.weights == str(config.FINETUNED_MODEL_WEIGHTS)
    finetuned.warmup()
    assert finetuned.load_seconds is not None


def test_rtdetr_warmup_uses_the_detr_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRTDETR.calls = 0
    FakeYOLO.calls = 0
    _install_fake_ultralytics(monkeypatch)

    detector = build_detector("rtdetr-l")
    detector.warmup()
    assert FakeRTDETR.calls == 1
    assert FakeYOLO.calls == 0
    assert detector.load_seconds is not None


def test_detector_passes_input_size_and_yolo_nms_iou(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_ultralytics(monkeypatch)
    detector = build_detector("yolov8n")
    detector.detect(
        np.zeros((32, 32, 3), dtype=np.uint8),
        0.35,
        input_size=1280,
        nms_iou=0.6,
    )
    kwargs = detector._model.predict_calls[-1]
    assert kwargs["conf"] == 0.35
    assert kwargs["imgsz"] == 1280
    assert kwargs["iou"] == 0.6


def test_rtdetr_rejects_an_nms_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_ultralytics(monkeypatch)
    detector = build_detector("rtdetr-l")
    with pytest.raises(ValueError, match="does not use NMS"):
        detector.detect(
            np.zeros((32, 32, 3), dtype=np.uint8),
            0.25,
            nms_iou=0.6,
        )


# ---------------------------------------------------------------------------
# SlicedDetector: tiled inference with class-aware merging (offline, synthetic)
# ---------------------------------------------------------------------------

class BlobDetector:
    """Toy detector satisfying the Detector protocol without any model.

    Pure red pixels form a car, pure green pixels form a bus. Boxes are
    tile-local, exactly like a real model's output, and confidence grows with
    the visible blob area, so a tile that sees a whole vehicle outscores a tile
    that sees only a fragment of it.
    """

    name = "blob"

    def __init__(self) -> None:
        self.warmed = False
        self.tile_calls = 0

    def warmup(self) -> None:
        self.warmed = True

    def detect(self, image: np.ndarray, confidence_threshold: float) -> list[Detection]:
        self.tile_calls += 1
        found: list[Detection] = []
        candidates = (
            ((image[..., 2] >= 250) & (image[..., 0] <= 5) & (image[..., 1] <= 5), 2, "car"),
            ((image[..., 1] >= 250) & (image[..., 0] <= 5) & (image[..., 2] <= 5), 5, "bus"),
        )
        for mask, class_id, class_name in candidates:
            if not mask.any():
                continue
            cols = np.where(mask.any(axis=0))[0]
            rows = np.where(mask.any(axis=1))[0]
            confidence = min(0.95, 0.3 + float(mask.sum()) / 3000.0)
            found.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    xyxy=(float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1)),
                )
            )
        return found


class RecordingDetector:
    """Detects nothing but records every threshold and tile shape it was given."""

    name = "rec"

    def __init__(self) -> None:
        self.warmed = False
        self.thresholds: list[float] = []
        self.tile_shapes: list[tuple[int, ...]] = []

    def warmup(self) -> None:
        self.warmed = True

    def detect(self, image: np.ndarray, confidence_threshold: float) -> list[Detection]:
        self.thresholds.append(confidence_threshold)
        self.tile_shapes.append(tuple(image.shape))
        return []


def _blank(height: int = 200, width: int = 300) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _draw(image: np.ndarray, color: tuple[int, int, int], x0: int, y0: int, x1: int, y1: int) -> None:
    image[y0:y1, x0:x1] = color


CAR_RED = (0, 0, 255)
BUS_GREEN = (0, 255, 0)


def test_sliced_detector_satisfies_protocol_and_delegates_warmup() -> None:
    inner = BlobDetector()
    sliced = SlicedDetector(inner)
    assert isinstance(sliced, Detector)
    assert "blob" in sliced.name
    assert sliced.load_seconds is None
    sliced.warmup()
    assert inner.warmed


def test_sliced_detector_rejects_bad_geometry() -> None:
    with pytest.raises(ValueError):
        SlicedDetector(BlobDetector(), tile_size=0)
    with pytest.raises(ValueError):
        SlicedDetector(BlobDetector(), overlap_ratio=1.0)
    with pytest.raises(ValueError):
        SlicedDetector(BlobDetector(), overlap_ratio=-0.1)


def test_straddling_vehicle_is_offset_and_merged_once() -> None:
    # Car drawn across the tile border at x=100 (tiles of 100 px, 50 px overlap):
    # two tiles see fragments, one tile sees the whole car. The fragment boxes
    # overlap the whole box at IoU 0.5, so a 0.3 threshold merges all of them.
    image = _blank()
    _draw(image, CAR_RED, 80, 60, 120, 90)

    sliced = SlicedDetector(BlobDetector(), tile_size=100, overlap_ratio=0.5, iou_threshold=0.3)
    detections = sliced.detect(image, 0.25)

    assert len(detections) == 1
    kept = detections[0]
    assert (kept.class_id, kept.class_name) == (2, "car")
    assert kept.xyxy == (80.0, 60.0, 120.0, 90.0)  # full-image coordinates restored


def test_even_split_at_high_threshold_double_counts_border_vehicle() -> None:
    # The honest downside: with IoU threshold 0.95 the fragment boxes (IoU 0.5
    # with the whole box) are never suppressed, so one physical car is counted
    # three times. This test pins the double-counting risk SlicedDetector
    # mitigates but cannot eliminate.
    image = _blank()
    _draw(image, CAR_RED, 80, 60, 120, 90)

    sliced = SlicedDetector(BlobDetector(), tile_size=100, overlap_ratio=0.5, iou_threshold=0.95)
    detections = sliced.detect(image, 0.25)

    cars = [d for d in detections if d.class_name == "car"]
    assert len(cars) == 3  # whole box plus one fragment per side of the border


def test_separate_cars_counted_once_each_in_full_image_coordinates() -> None:
    image = _blank()
    _draw(image, CAR_RED, 5, 10, 45, 40)      # wholly inside the first tile column
    _draw(image, CAR_RED, 210, 150, 250, 180)  # wholly inside the last tile column

    sliced = SlicedDetector(BlobDetector(), tile_size=100, overlap_ratio=0.5)
    detections = sliced.detect(image, 0.25)

    assert len(detections) == 2
    assert {d.xyxy for d in detections} == {(5.0, 10.0, 45.0, 40.0), (210.0, 150.0, 250.0, 180.0)}


def test_merge_is_class_aware_for_overlapping_boxes() -> None:
    # A car and a bus drawn on top of each other overlap well past the IoU
    # threshold, but NMS is class aware, so both survive while same-class
    # duplicates across tiles collapse.
    image = _blank()
    _draw(image, CAR_RED, 100, 60, 150, 90)
    _draw(image, BUS_GREEN, 105, 62, 145, 88)

    sliced = SlicedDetector(BlobDetector(), tile_size=100, overlap_ratio=0.5)
    detections = sliced.detect(image, 0.25)

    names = sorted(d.class_name for d in detections)
    assert names == ["bus", "car"]


def test_confidence_threshold_is_passed_to_every_tile() -> None:
    inner = RecordingDetector()
    sliced = SlicedDetector(inner, tile_size=100, overlap_ratio=0.5)
    detections = sliced.detect(_blank(), 0.42)

    assert detections == []
    assert inner.thresholds  # the image was sliced, not skipped
    assert set(inner.thresholds) == {0.42}
    for height, width, channels in inner.tile_shapes:
        assert height <= 100 and width <= 100 and channels == 3


def test_image_smaller_than_a_tile_is_passed_through_untiled() -> None:
    image = _blank(height=60, width=80)
    _draw(image, CAR_RED, 10, 15, 40, 35)

    inner = BlobDetector()
    sliced = SlicedDetector(inner, tile_size=100)
    detections = sliced.detect(image, 0.25)

    assert inner.tile_calls == 1
    assert len(detections) == 1
    assert detections[0].xyxy == (10.0, 15.0, 40.0, 35.0)
