"""Detector abstraction for TrafficLens.

A detector turns an image into a list of :class:`Detection` objects above a confidence
threshold. Every model listed in ``config.MODEL_REGISTRY`` (YOLOv8n, YOLOv8s, and
RT-DETR-L) sits behind the same :class:`Detector` protocol, so all downstream code
depends only on the protocol and the ``Detection`` shape. Weights load lazily and the
load is timed into ``load_seconds``; :meth:`Detector.warmup` pays for that load and the
first-run overhead up front so measured inference excludes them.
"""
from __future__ import annotations

import time
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

    def warmup(self) -> None:
        """Load weights and prime the model so timed inference excludes the load.

        The model load is recorded separately (``load_seconds``) and a tiny dummy
        frame pays for first-call setup; after warmup, timing a ``detect`` call
        measures prediction work only.
        """
        ...  # pragma: no cover - protocol definition


#: Edge length in pixels of the dummy frame used by :meth:`Detector.warmup`.
WARMUP_EDGE: int = 64


class _UltralyticsDetector:
    """Shared ultralytics plumbing: lazy timed load, warmup, and detection shape.

    Subclasses fix which ultralytics wrapper class loads the pretrained weights
    (``YOLO`` for the yolo family, ``RTDETR`` for the detr family).
    """

    def __init__(self, weights: str, name: str, family: str, class_name: str) -> None:
        self.weights = weights
        self.name = name
        self.family = family
        self._class_name = class_name
        self._model = None
        self.load_seconds: float | None = None

    def _ensure_model(self):  # noqa: ANN202 - ultralytics returns the wrapper instance
        """Load and cache the underlying model once, timing only the load."""
        if self._model is None:
            import ultralytics

            started = time.perf_counter()
            self._model = getattr(ultralytics, self._class_name)(self.weights)
            self.load_seconds = time.perf_counter() - started
        return self._model

    def warmup(self) -> None:
        """Load the weights (recorded in ``load_seconds``) and run one dummy frame."""
        model = self._ensure_model()
        dummy = np.zeros((WARMUP_EDGE, WARMUP_EDGE, 3), dtype=np.uint8)
        model.predict(dummy, verbose=False)

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


class YoloDetector(_UltralyticsDetector):
    """Ultralytics YOLO wrapper implementing the :class:`Detector` protocol.

    The default ``yolov8n`` weights download on first use and are cached on disk by
    ultralytics. The model instance is loaded lazily so import and cache wiring stay
    cheap until the first real inference.
    """

    def __init__(
        self,
        weights: str = config.DEFAULT_WEIGHTS_FILE,
        name: str = config.DEFAULT_MODEL_NAME,
    ) -> None:
        super().__init__(weights=weights, name=name, family="yolo", class_name="YOLO")


class RtdetrDetector(_UltralyticsDetector):
    """Ultralytics RT-DETR wrapper implementing the :class:`Detector` protocol."""

    def __init__(
        self,
        weights: str = config.MODEL_REGISTRY["rtdetr-l"].weights_file,
        name: str = config.MODEL_REGISTRY["rtdetr-l"].display_name,
    ) -> None:
        super().__init__(weights=weights, name=name, family="detr", class_name="RTDETR")


_FAMILY_TO_CLASS: dict[str, type[_UltralyticsDetector]] = {
    "yolo": YoloDetector,
    "detr": RtdetrDetector,
}


def build_detector(model_key: str) -> Detector:
    """Construct the detector registered under ``model_key`` in ``config.MODEL_REGISTRY``.

    Raises:
        KeyError: when the key is not present in the registry.
    """
    spec = config.MODEL_REGISTRY[model_key]
    detector_class = _FAMILY_TO_CLASS[spec.family]
    return detector_class(weights=spec.weights_file, name=spec.display_name)
