"""Threshold-sweep reliability measurements for TrafficLens detectors.

Running a model once per sweep threshold is expensive; running it once at the
lowest threshold and deriving the higher thresholds by confidence filtering
costs a single inference. The derivation is exact for the ultralytics stack
because ``conf`` acts as a lower bound (detections with ``confidence >= conf``
are kept) and greedy NMS processes boxes in descending score order, so the
survivors above any higher threshold are precisely the low-run survivors whose
confidence passes it.

Documented deviations from running each threshold directly:

* ``max_det`` truncation. Ultralytics returns at most ``max_det`` boxes per
  image (300 by default), keeping the highest scores. If a low-threshold run
  produces more survivors than that cap, the excess low-confidence boxes are
  dropped before this module ever sees them, and a derived higher threshold can
  undercount relative to a direct run at that threshold. Only pathologically
  dense frames are affected; when the two disagree, the direct run is the
  reference.
* Threshold equality. Both the derivation and ultralytics keep ``>=``, so a
  detection whose confidence equals the threshold appears at that threshold in
  either method.

Two ranges summarise the sweep: a model's ``threshold_range`` is the
``(min, max)`` of its total vehicle counts across the sweep thresholds, and a
report's ``full_range`` is the ``(min, max)`` across every model and every
threshold together. Both bound how much the headline count moves with the
threshold choice, which is the reliability question this module answers;
counts stay defensible only inside those bounds until manual labels settle
which model is accurate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config, counting
from .detector import Detection, Detector


@dataclass(frozen=True)
class ThresholdCounts:
    """Vehicle counts derived at one sweep threshold.

    Attributes:
        threshold: The confidence threshold this row was derived at.
        counts_by_class: Per class count; every selected class appears, zero included.
        total: Number of counted vehicle detections at this threshold.
    """

    threshold: float
    counts_by_class: dict[str, int]
    total: int


@dataclass(frozen=True)
class ModelSweep:
    """One detector's counts across the sweep.

    Attributes:
        model_name: Name reported by the detector.
        base_threshold: The lowest threshold, the one the detector actually ran at.
        per_threshold: :class:`ThresholdCounts` rows in ascending threshold order.
        threshold_range: ``(min, max)`` of the per-threshold totals for this model.
    """

    model_name: str
    base_threshold: float
    per_threshold: tuple[ThresholdCounts, ...]
    threshold_range: tuple[int, int]


@dataclass(frozen=True)
class SweepReport:
    """All swept models plus the cross-model range.

    Attributes:
        thresholds: The ordered thresholds every model was swept at.
        sweeps: One :class:`ModelSweep` per detector, in input order.
        full_range: ``(min, max)`` of totals across all models and thresholds.
        image_width: Source image width in pixels.
        image_height: Source image height in pixels.
    """

    thresholds: tuple[float, ...]
    sweeps: tuple[ModelSweep, ...]
    full_range: tuple[int, int]
    image_width: int
    image_height: int


def _ordered_thresholds(thresholds: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    """Validate and sort sweep thresholds, dropping duplicates.

    Raises:
        ValueError: when no threshold is given or one lies outside the counting
            range configured in ``config.MIN_CONFIDENCE`` / ``config.MAX_CONFIDENCE``.
    """
    ordered = sorted({float(threshold) for threshold in thresholds})
    if not ordered:
        raise ValueError("thresholds must not be empty")
    for threshold in ordered:
        if not config.MIN_CONFIDENCE <= threshold <= config.MAX_CONFIDENCE:
            raise ValueError(
                f"threshold {threshold} outside [{config.MIN_CONFIDENCE}, {config.MAX_CONFIDENCE}]"
            )
    return tuple(ordered)


def _counts_at(
    detections: list[Detection], threshold: float, selected_classes: list[str]
) -> ThresholdCounts:
    """Derive the counts for one threshold from a lower-threshold detection run."""
    kept = [detection for detection in detections if detection.confidence >= threshold]
    filtered = counting.filter_by_classes(kept, selected_classes)
    return ThresholdCounts(
        threshold=threshold,
        counts_by_class=counting.count_by_class(filtered, selected_classes),
        total=len(filtered),
    )


def sweep_detector(
    detector: Detector,
    image: np.ndarray,
    thresholds: tuple[float, ...] | list[float] = config.THRESHOLD_SWEEP,
    selected_classes: list[str] = list(config.TARGET_CLASS_NAMES),
) -> ModelSweep:
    """Run ``detector`` once at the lowest threshold and derive the whole sweep.

    Args:
        detector: Any object satisfying the :class:`Detector` protocol.
        image: ``(H, W, 3)`` uint8 BGR array.
        thresholds: Sweep values; duplicates are dropped and the order need not
            be sorted. The detector runs at ``min(thresholds)`` only.
        selected_classes: Vehicle classes to count, as in :func:`counting.count_vehicles`.

    Returns:
        A :class:`ModelSweep` with one row per threshold and the model's range.

    Raises:
        ValueError: from threshold validation (empty or outside the counting range).
    """
    ordered = _ordered_thresholds(thresholds)
    base = ordered[0]
    detections = detector.detect(image, base)
    per_threshold = tuple(_counts_at(detections, threshold, selected_classes) for threshold in ordered)
    totals = [row.total for row in per_threshold]
    return ModelSweep(
        model_name=detector.name,
        base_threshold=base,
        per_threshold=per_threshold,
        threshold_range=(min(totals), max(totals)),
    )


def sweep_all(
    detectors: list[Detector],
    image: np.ndarray,
    thresholds: tuple[float, ...] | list[float] = config.THRESHOLD_SWEEP,
    selected_classes: list[str] = list(config.TARGET_CLASS_NAMES),
) -> SweepReport:
    """Sweep every detector once and compute the cross-model range.

    Args:
        detectors: Detectors to sweep; each runs exactly once on ``image``.
        image: ``(H, W, 3)`` uint8 BGR array.
        thresholds: Shared sweep values for every model.
        selected_classes: Vehicle classes to count.

    Returns:
        A :class:`SweepReport` whose ``full_range`` spans all models and thresholds.

    Raises:
        ValueError: when no detector is given or the thresholds are invalid.
    """
    if not detectors:
        raise ValueError("at least one detector is required")
    ordered = _ordered_thresholds(thresholds)
    sweeps = tuple(
        sweep_detector(detector, image, ordered, selected_classes) for detector in detectors
    )
    totals = [row.total for sweep in sweeps for row in sweep.per_threshold]
    height, width = image.shape[:2]
    return SweepReport(
        thresholds=ordered,
        sweeps=sweeps,
        full_range=(min(totals), max(totals)),
        image_width=width,
        image_height=height,
    )


def format_report(report: SweepReport) -> str:
    """Render a sweep report as human readable lines, one block per model."""
    lines = [
        f"Threshold sweep {tuple(report.thresholds)} on a {report.image_width}x{report.image_height} image",
        "(each detector ran once at the lowest threshold; higher rows are filtered)",
    ]
    for sweep in report.sweeps:
        lines.append("")
        lines.append(f"{sweep.model_name} (ran at {sweep.base_threshold:.2f})")
        for row in sweep.per_threshold:
            classes = " ".join(f"{name}={count}" for name, count in row.counts_by_class.items())
            lines.append(f"  {row.threshold:.2f}: total={row.total:>3} | {classes}")
        lines.append(f"  threshold range: {sweep.threshold_range}")
    lines.append("")
    lines.append(f"full range across all models and thresholds: {report.full_range}")
    return "\n".join(lines)
