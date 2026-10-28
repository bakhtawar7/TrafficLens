"""Class filtering, per class counting, and the result container.

The detector returns every class it sees above the confidence threshold. This module
narrows those detections to the selected vehicle classes, aggregates per class counts,
and packages everything into an immutable :class:`CountResult` that the UI, exports, and
tests share.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import config
from .detector import Detection, Detector


@dataclass(frozen=True)
class CountResult:
    """The complete outcome of counting vehicles in one image.

    Attributes:
        detections: Detections that survived class filtering and thresholding.
        total_count: Number of retained detections.
        counts_by_class: Per class count. Every selected class is present, zero included.
        confidence_threshold: Threshold applied during detection.
        selected_classes: Class names the user asked to count, in display order.
        model_name: Name of the detector that produced the detections.
        image_width: Source image width in pixels.
        image_height: Source image height in pixels.
        inference_ms: Detector wall clock time in milliseconds.
    """

    detections: list[Detection]
    total_count: int
    counts_by_class: dict[str, int]
    confidence_threshold: float
    selected_classes: list[str]
    model_name: str
    image_width: int
    image_height: int
    inference_ms: float


def filter_by_classes(
    detections: list[Detection], selected_classes: list[str]
) -> list[Detection]:
    """Keep only detections whose class name is in ``selected_classes``.

    Args:
        detections: Raw detections from a detector.
        selected_classes: Class names to retain.

    Returns:
        A new list containing only the matching detections.
    """
    allowed = set(selected_classes)
    return [d for d in detections if d.class_name in allowed]


def count_by_class(
    detections: list[Detection], selected_classes: list[str]
) -> dict[str, int]:
    """Count detections per class, guaranteeing every selected class appears.

    Args:
        detections: Already filtered detections.
        selected_classes: Classes that must appear in the output, zero included.

    Returns:
        Mapping of class name to count, ordered like ``selected_classes``.
    """
    counts = {name: 0 for name in selected_classes}
    for detection in detections:
        if detection.class_name in counts:
            counts[detection.class_name] += 1
    return counts


def build_count_result(
    detections: list[Detection],
    selected_classes: list[str],
    confidence_threshold: float,
    model_name: str,
    image_width: int,
    image_height: int,
    inference_ms: float,
) -> CountResult:
    """Filter detections and assemble a :class:`CountResult`.

    Class filtering happens here so callers can pass raw detector output directly.
    """
    filtered = filter_by_classes(detections, selected_classes)
    counts = count_by_class(filtered, selected_classes)
    return CountResult(
        detections=filtered,
        total_count=len(filtered),
        counts_by_class=counts,
        confidence_threshold=confidence_threshold,
        selected_classes=list(selected_classes),
        model_name=model_name,
        image_width=image_width,
        image_height=image_height,
        inference_ms=inference_ms,
    )


def count_vehicles(
    image: np.ndarray,
    detector: Detector,
    selected_classes: list[str] = list(config.TARGET_CLASS_NAMES),
    confidence_threshold: float = config.DEFAULT_CONFIDENCE,
) -> CountResult:
    """Run a detector on an image and return a fully populated :class:`CountResult`.

    Args:
        image: ``(H, W, 3)`` uint8 BGR array.
        detector: Any object satisfying the :class:`Detector` protocol.
        selected_classes: Vehicle classes to count.
        confidence_threshold: Minimum detection score, clamped to the allowed range.

    Returns:
        The counting outcome, including timing measured around the detector call.
    """
    threshold = max(config.MIN_CONFIDENCE, min(config.MAX_CONFIDENCE, confidence_threshold))
    height, width = image.shape[:2]
    start = time.perf_counter()
    detections = detector.detect(image, threshold)
    inference_ms = (time.perf_counter() - start) * 1000.0
    return build_count_result(
        detections=detections,
        selected_classes=selected_classes,
        confidence_threshold=threshold,
        model_name=detector.name,
        image_width=width,
        image_height=height,
        inference_ms=inference_ms,
    )
