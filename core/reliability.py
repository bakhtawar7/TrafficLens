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

The module also computes the two agreement measures behind hypothesis H2 for
every model pair and aggregates them within family and across families:

* Count disagreement, per class: ``d_c = |a_c - b_c| / (a_c + b_c)``, defined
  as ``0`` when both counts are ``0``, and reported overall as the mean of
  ``d_c`` across the selected classes. It lies in ``[0, 1]``: ``0`` means the
  two models counted identically, ``1`` means every counted vehicle of that
  class was claimed by one model only. ``docs/hypotheses.md`` H2 fixes the
  measure as "the normalised difference in per class counts" but does not
  spell out the closed form; this is the normalisation implemented, and the
  choice is part of the definition of the metric.
* Box agreement: greedy class-aware one-to-one IoU matching at
  ``config.BOX_IOU_THRESHOLD`` (0.5): same-class box pairs are considered in
  descending IoU and matched when their IoU is at or above the threshold and
  neither box is taken yet; the score is ``2 * matches / (n_a + n_b)``, the
  symmetric F1 form. Identical sets score ``1``, disjoint sets ``0``, two
  empty sets score ``1`` by definition.
"""
from __future__ import annotations

import itertools
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


# ---------------------------------------------------------------------------
# Agreement measures for hypothesis H2
# ---------------------------------------------------------------------------

#: Ultralytics capacity suffixes in ascending model size, used to pick the
#: best YOLO variant for the cross-family aggregate.
_CAPACITY_RANK: dict[str, int] = {"n": 0, "s": 1, "m": 2, "l": 3, "x": 4}


@dataclass(frozen=True)
class Agreement:
    """The two H2 measures between one pair of detection sets.

    Attributes:
        count_disagreement_by_class: Normalised per class difference in [0, 1].
        count_disagreement: Mean of the per class values over the selected classes.
        box_agreement: Greedy class-aware IoU match score ``2m / (n_a + n_b)``.
    """

    count_disagreement_by_class: dict[str, float]
    count_disagreement: float
    box_agreement: float


@dataclass(frozen=True)
class AgreementPair:
    """Agreement for one named model pair.

    Attributes:
        models: The two model display names, in pair order.
        family_relation: ``"within"`` for same-family pairs, ``"cross"`` otherwise.
        agreement: The measures between the two models' detections.
    """

    models: tuple[str, str]
    family_relation: str
    agreement: Agreement


@dataclass(frozen=True)
class ReliabilityReport:
    """Pairwise agreement for every model plus the two H2 aggregates.

    Attributes:
        threshold: Confidence threshold every detector ran at.
        iou_threshold: IoU at which same-class boxes count as one vehicle.
        pairs: One :class:`AgreementPair` per unordered model pair.
        within_family: Mean agreement over all same-family pairs, ``None`` if none.
        cross_family: Mean agreement over best-YOLO-versus-each-other-family
            pairs, ``None`` if no such pair exists.
        best_yolo: Name of the YOLO variant chosen for the cross-family
            aggregate by ultralytics capacity suffix (n < s < m < l < x),
            ``None`` when the registry has no yolo model.
    """

    threshold: float
    iou_threshold: float
    pairs: tuple[AgreementPair, ...]
    within_family: Agreement | None
    cross_family: Agreement | None
    best_yolo: str | None


def _iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Intersection over union of two axis-aligned xyxy boxes."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def count_disagreement_by_class(
    counts_a: dict[str, int],
    counts_b: dict[str, int],
    selected_classes: list[str] = list(config.TARGET_CLASS_NAMES),
) -> dict[str, float]:
    """Per class normalised difference ``|a - b| / (a + b)``, 0 when both are 0."""
    disagreement: dict[str, float] = {}
    for name in selected_classes:
        a = counts_a.get(name, 0)
        b = counts_b.get(name, 0)
        total = a + b
        disagreement[name] = abs(a - b) / total if total else 0.0
    return disagreement


def box_agreement(
    detections_a: list[Detection],
    detections_b: list[Detection],
    iou_threshold: float = config.BOX_IOU_THRESHOLD,
) -> float:
    """Greedy class-aware IoU matching score ``2 * matches / (n_a + n_b)``.

    Same-class box pairs are considered in descending IoU (ties resolved by
    input order, so the result is deterministic) and matched when their IoU is
    at or above ``iou_threshold`` and neither box is already taken. Two empty
    detection sets agree by definition and score 1.0.
    """
    if not detections_a and not detections_b:
        return 1.0
    if not detections_a or not detections_b:
        return 0.0

    candidates: list[tuple[float, int, int]] = []
    for i, first in enumerate(detections_a):
        for j, second in enumerate(detections_b):
            if first.class_name != second.class_name:
                continue
            overlap = _iou(first.xyxy, second.xyxy)
            if overlap >= iou_threshold:
                candidates.append((overlap, i, j))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    taken_a: set[int] = set()
    taken_b: set[int] = set()
    matches = 0
    for _overlap, i, j in candidates:
        if i in taken_a or j in taken_b:
            continue
        taken_a.add(i)
        taken_b.add(j)
        matches += 1
    return 2.0 * matches / (len(detections_a) + len(detections_b))


def agreement_between(
    detections_a: list[Detection],
    detections_b: list[Detection],
    selected_classes: list[str] = list(config.TARGET_CLASS_NAMES),
    iou_threshold: float = config.BOX_IOU_THRESHOLD,
) -> Agreement:
    """Compute both H2 measures between two models' detections on one image.

    Detections of classes outside ``selected_classes`` are excluded from the
    counts and from box matching, so the measures only concern vehicles.
    """
    kept_a = counting.filter_by_classes(detections_a, selected_classes)
    kept_b = counting.filter_by_classes(detections_b, selected_classes)
    per_class = count_disagreement_by_class(
        counting.count_by_class(kept_a, selected_classes),
        counting.count_by_class(kept_b, selected_classes),
        selected_classes,
    )
    mean = sum(per_class.values()) / len(per_class) if per_class else 0.0
    return Agreement(
        count_disagreement_by_class=per_class,
        count_disagreement=mean,
        box_agreement=box_agreement(kept_a, kept_b, iou_threshold),
    )


def _aggregate(agreements: list[Agreement]) -> Agreement:
    """Mean of several agreements, per class and per measure."""
    if not agreements:
        raise ValueError("at least one agreement is required")
    classes = list(agreements[0].count_disagreement_by_class)
    per_class = {
        name: sum(a.count_disagreement_by_class[name] for a in agreements) / len(agreements)
        for name in classes
    }
    return Agreement(
        count_disagreement_by_class=per_class,
        count_disagreement=sum(a.count_disagreement for a in agreements) / len(agreements),
        box_agreement=sum(a.box_agreement for a in agreements) / len(agreements),
    )


def _capacity_rank(name: str) -> int:
    """Ultralytics size suffix rank of a model name; unknown suffixes rank last."""
    return _CAPACITY_RANK.get(name[-1].lower(), -1) if name else -1


def build_reliability_report(
    detections_by_model: dict[str, list[Detection]],
    families_by_model: dict[str, str],
    threshold: float,
    iou_threshold: float = config.BOX_IOU_THRESHOLD,
    selected_classes: list[str] = list(config.TARGET_CLASS_NAMES),
) -> ReliabilityReport:
    """Assemble the pairwise and aggregated agreement for a model set.

    Args:
        detections_by_model: Each model's already-detected vehicle-relevant
            detections for one image, keyed by display name.
        families_by_model: Registry family (``yolo`` or ``detr``) per name.
        threshold: The detection threshold the lists were produced at.
        iou_threshold: IoU used for box matching.
        selected_classes: Vehicle classes that take part in both measures.

    Returns:
        A :class:`ReliabilityReport` with every unordered pair plus the
        within-family and best-YOLO-versus-other-family aggregates.
    """
    names = list(detections_by_model)
    pairs: list[AgreementPair] = []
    for first, second in itertools.combinations(names, 2):
        relation = "within" if families_by_model[first] == families_by_model[second] else "cross"
        pairs.append(
            AgreementPair(
                models=(first, second),
                family_relation=relation,
                agreement=agreement_between(
                    detections_by_model[first], detections_by_model[second], selected_classes, iou_threshold
                ),
            )
        )

    within = [pair.agreement for pair in pairs if pair.family_relation == "within"]

    yolo_names = [name for name in names if families_by_model.get(name) == "yolo"]
    best_yolo = max(yolo_names, key=_capacity_rank) if yolo_names else None
    cross: list[Agreement] = []
    if best_yolo is not None:
        for pair in pairs:
            if pair.family_relation != "cross":
                continue
            if best_yolo in pair.models:
                cross.append(pair.agreement)

    return ReliabilityReport(
        threshold=threshold,
        iou_threshold=iou_threshold,
        pairs=tuple(pairs),
        within_family=_aggregate(within) if within else None,
        cross_family=_aggregate(cross) if cross else None,
        best_yolo=best_yolo,
    )


def agreement_report(
    detectors: list[Detector],
    image: np.ndarray,
    threshold: float = config.DEFAULT_CONFIDENCE,
    iou_threshold: float = config.BOX_IOU_THRESHOLD,
    selected_classes: list[str] = list(config.TARGET_CLASS_NAMES),
) -> ReliabilityReport:
    """Run every detector once at ``threshold`` and compute the agreement matrix.

    Args:
        detectors: Detectors to compare; each runs exactly once on ``image``.
        image: ``(H, W, 3)`` uint8 BGR array.
        threshold: Single confidence threshold all models are compared at.
        iou_threshold: IoU used for box matching.
        selected_classes: Vehicle classes that take part in both measures.

    Returns:
        The same report structure as :func:`build_reliability_report`.

    Raises:
        ValueError: when no detector is given or two share a name.
    """
    if not detectors:
        raise ValueError("at least one detector is required")
    detections_by_model: dict[str, list[Detection]] = {}
    families_by_model: dict[str, str] = {}
    for detector in detectors:
        if detector.name in detections_by_model:
            raise ValueError(f"duplicate detector name {detector.name}")
        detections_by_model[detector.name] = detector.detect(image, threshold)
        families_by_model[detector.name] = getattr(detector, "family", "other")
    return build_reliability_report(
        detections_by_model, families_by_model, threshold, iou_threshold, selected_classes
    )


def format_agreement(report: ReliabilityReport) -> str:
    """Render an agreement report as human readable lines."""
    lines = [
        f"Model agreement at confidence {report.threshold:.2f}, IoU threshold {report.iou_threshold:.2f}",
    ]
    for pair in report.pairs:
        agreement = pair.agreement
        classes = " ".join(f"{name}={value:.2f}" for name, value in agreement.count_disagreement_by_class.items())
        lines.append("")
        lines.append(f"{pair.models[0]} vs {pair.models[1]} ({pair.family_relation}-family)")
        lines.append(
            f"  count disagreement: {agreement.count_disagreement:.3f} | per class: {classes}"
        )
        lines.append(f"  box agreement : {agreement.box_agreement:.3f}")
    lines.append("")
    if report.best_yolo is not None:
        lines.append(f"best yolo for the cross-family aggregate: {report.best_yolo}")
    for label, agreement in (("within family", report.within_family), ("cross family", report.cross_family)):
        if agreement is None:
            lines.append(f"{label} aggregate: not applicable (no such pair)")
        else:
            lines.append(
                f"{label} aggregate: count disagreement {agreement.count_disagreement:.3f}, "
                f"box agreement {agreement.box_agreement:.3f}"
            )
    return "\n".join(lines)
