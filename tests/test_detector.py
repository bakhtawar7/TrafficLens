"""Tests for the Detector protocol and the YOLO implementation.

The protocol conformance test runs offline. The real inference test needs downloaded
model weights and network on first use, so it is marked ``requires_weights`` and skipped
unless the environment opt in flag is set.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from core.detector import Detection, Detector, YoloDetector


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
