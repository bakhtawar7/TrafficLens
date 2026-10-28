"""Detector abstraction for TrafficLens.

A detector turns an image into a list of :class:`Detection` objects above a confidence
threshold. Week 2 adds YOLOv8s and RT-DETR-L behind the same :class:`Detector` protocol,
so all downstream code depends only on the protocol and the ``Detection`` shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from . import config


@dataclass(frozen=True)
class Detection:
    """A single detected object.

    Attributes:
        class_id: Dataset class id (COCO id for the pretrained models).
        class_name: Human readable class name reported by the model.
        confidence: Detection score in ``[0, 1]``.
        xyxy: Bounding box as ``(x_min, y_min, x_max, y_max)`` in pixel coordinates.
    """

    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


@runtime_checkable
class Detector(Protocol):
    """Minimal interface every TrafficLens detector must satisfy."""

    name: str

    def detect(self, image: np.ndarray, confidence_threshold: float) -> list[Detection]:
        """Detect objects in a BGR image above the given confidence threshold.

        Args:
            image: ``(H, W, 3)`` uint8 BGR array.
            confidence_threshold: Minimum score for a detection to be returned.

        Returns:
            A list of :class:`Detection`, order unspecified.
        """
        ...  # pragma: no cover - protocol definition


class YoloDetector:
    """Ultralytics YOLO wrapper implementing the :class:`Detector` protocol.

    The default ``yolov8n`` weights download on first use and are cached on disk by
    ultralytics. The model instance is loaded lazily so import and cache wiring stay
    cheap until the first real inference.
    """

    def __init__(self, weights: str = config.DEFAULT_WEIGHTS_FILE) -> None:
        self.weights = weights
        self.name = config.DEFAULT_MODEL_NAME
        self._model = None

    def _ensure_model(self):  # noqa: ANN202 - ultralytics returns a YOLO instance
        """Load and cache the underlying ultralytics model on first use."""
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.weights)
        return self._model

    def detect(self, image: np.ndarray, confidence_threshold: float) -> list[Detection]:
        """Run inference and convert ultralytics results into ``Detection`` objects."""
        model = self._ensure_model()
        results = model.predict(image, conf=confidence_threshold, verbose=False)

        detections: list[Detection] = []
        names = model.names
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            for (x0, y0, x1, y1), cls_id, conf in zip(xyxy, class_ids, confs):
                detections.append(
                    Detection(
                        class_id=int(cls_id),
                        class_name=str(names.get(int(cls_id), str(int(cls_id)))),
                        confidence=float(conf),
                        xyxy=(float(x0), float(y0), float(x1), float(y1)),
                    )
                )
        return detections
