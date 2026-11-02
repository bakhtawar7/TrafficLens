"""Detector abstraction for TrafficLens.

A detector turns an image into a list of :class:`Detection` objects above a confidence
threshold. Every model listed in ``config.MODEL_REGISTRY`` (YOLOv8n, YOLOv8s, and
RT-DETR-L) sits behind the same :class:`Detector` protocol, so all downstream code
depends only on the protocol and the ``Detection`` shape. Weights load lazily and the
load is timed into ``load_seconds``; :meth:`Detector.warmup` pays for that load and the
first-run overhead up front so measured inference excludes them. ``SlicedDetector``
wraps any detector and runs it on overlapping tiles, merging the duplicates.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import supervision as sv

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


def _detections_to_sv(detections: list[Detection]) -> sv.Detections:
    """Convert ``Detection`` objects into the supervision container the slicer wants."""
    count = len(detections)
    xyxy = np.array([d.xyxy for d in detections], dtype=np.float32).reshape(count, 4)
    confidence = np.array([d.confidence for d in detections], dtype=np.float32).reshape(count)
    class_id = np.array([d.class_id for d in detections], dtype=np.int32).reshape(count)
    return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)


def _sv_to_detections(detections: sv.Detections, class_names: dict[int, str]) -> list[Detection]:
    """Convert merged slicer output back into ``Detection`` objects."""
    if len(detections) == 0:
        return []
    class_ids = detections.class_id if detections.class_id is not None else np.zeros(len(detections), dtype=int)
    confidences = detections.confidence if detections.confidence is not None else np.ones(len(detections))
    return [
        Detection(
            class_id=int(cls_id),
            class_name=class_names.get(int(cls_id), str(int(cls_id))),
            confidence=float(conf),
            xyxy=(float(x0), float(y0), float(x1), float(y1)),
        )
        for (x0, y0, x1, y1), cls_id, conf in zip(detections.xyxy, class_ids, confidences)
    ]


class SlicedDetector:
    """Tiled inference wrapper around any :class:`Detector`.

    The image is cut into overlapping tiles of ``tile_size`` pixels with an
    overlap of ``overlap_ratio`` of one tile edge; the inner detector runs on
    every tile, its tile-local boxes are shifted back to full-image
    coordinates, and duplicates across tiles are merged by class-aware
    non-max suppression at ``iou_threshold``. The tiling, offsetting, and
    suppression are supervision's ``InferenceSlicer`` (0.31 API), whose
    ``with_nms`` merge is class aware by default; this class only converts to
    and from the ``Detection`` shape.

    Double counting risk: a vehicle that straddles a tile border is seen by
    several tiles, each reporting only the part inside its own crop. Class-
    aware NMS merges those fragments only when their boxes overlap enough; if
    a vehicle is split so evenly that neither half overlaps the other past
    ``iou_threshold``, both halves survive as separate detections and are
    counted twice. Overlap margin and a suitably low ``iou_threshold`` are the
    mitigations, and small distant vehicles near borders are the cases to
    audit.
    """

    def __init__(
        self,
        detector: Detector,
        tile_size: int = config.SLICE_TILE_SIZE,
        overlap_ratio: float = config.SLICE_OVERLAP_RATIO,
        iou_threshold: float = config.SLICE_IOU_THRESHOLD,
        name: str | None = None,
    ) -> None:
        """Wrap ``detector``.

        Args:
            detector: Any detector to run per tile.
            tile_size: Edge length in pixels of each square tile.
            overlap_ratio: Fraction of a tile edge shared with neighbours, in ``[0, 1)``.
            iou_threshold: IoU above which same-class boxes merge.
            name: Display name; defaults to the inner name marked as sliced.

        Raises:
            ValueError: when ``tile_size`` is not positive or ``overlap_ratio``
                is outside ``[0, 1)``.
        """
        if tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {tile_size}")
        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError(f"overlap_ratio must be in [0, 1), got {overlap_ratio}")
        self._detector = detector
        self.tile_size = int(tile_size)
        self.overlap_px = min(max(0, round(self.tile_size * overlap_ratio)), self.tile_size - 1)
        self.iou_threshold = float(iou_threshold)
        self.name = name if name is not None else f"{detector.name} (sliced)"

    @property
    def load_seconds(self) -> float | None:
        """Load time of the wrapped detector, kept separate from inference."""
        return getattr(self._detector, "load_seconds", None)

    def warmup(self) -> None:
        """Warm the wrapped detector; the tiling itself holds no model state."""
        self._detector.warmup()

    def detect(self, image: np.ndarray, confidence_threshold: float) -> list[Detection]:
        """Run the inner detector on every tile and merge the results.

        Args:
            image: ``(H, W, 3)`` uint8 BGR array.
            confidence_threshold: Passed unchanged to the inner detector per tile.

        Returns:
            Merged detections in full-image coordinates, one per surviving box.
        """
        class_names: dict[int, str] = {}

        def callback(tile: np.ndarray) -> sv.Detections:
            tile_detections = self._detector.detect(tile, confidence_threshold)
            for detection in tile_detections:
                class_names[detection.class_id] = detection.class_name
            return _detections_to_sv(tile_detections)

        slicer = sv.InferenceSlicer(
            callback=callback,
            slice_wh=(self.tile_size, self.tile_size),
            overlap_wh=(self.overlap_px, self.overlap_px),
            iou_threshold=self.iou_threshold,
        )
        return _sv_to_detections(slicer(image), class_names)
