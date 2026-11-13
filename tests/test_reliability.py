"""Tests for the threshold sweep in core.reliability.

The offline tests use a fake detector that applies the ultralytics semantics
(keep detections with ``confidence >= threshold``) so the derivation plumbing,
class filtering, and range maths are proven without weights or network. The
real-model equivalence test needs cached weights and is skipped unless
``TRAFFICLENS_RUN_WEIGHTS=1``; it is the only test that can check the sweep
against genuine ultralytics NMS behaviour. The agreement tests for the H2
measures (count disagreement and class-aware box agreement) are fully
synthetic and run offline.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from core import config, counting, reliability
from core.detector import Detection


def _detection(class_id: int, class_name: str, confidence: float) -> Detection:
    return Detection(class_id, class_name, confidence, (1.0, 1.0, 5.0, 5.0))


class ThresholdFake:
    """Detector-shaped fake that filters its candidate list like ultralytics does."""

    name = "fake"

    def __init__(self, candidates: list[Detection]) -> None:
        self.candidates = candidates
        self.calls = 0
        self.thresholds: list[float] = []

    def warmup(self) -> None:
        pass

    def detect(self, image: np.ndarray, confidence_threshold: float) -> list[Detection]:
        self.calls += 1
        self.thresholds.append(confidence_threshold)
        return [d for d in self.candidates if d.confidence >= confidence_threshold]


CANDIDATES = [
    _detection(2, "car", 0.62),
    _detection(2, "car", 0.30),
    _detection(2, "car", 0.25),  # exactly on a sweep threshold
    _detection(2, "car", 0.18),
    _detection(3, "motorcycle", 0.55),
    _detection(3, "motorcycle", 0.10),  # below the lowest sweep value
    _detection(5, "bus", 0.44),
    _detection(7, "truck", 0.21),
    _detection(0, "person", 0.90),  # never a counted class
]

IMAGE = np.zeros((50, 50, 3), dtype=np.uint8)


def _fake() -> ThresholdFake:
    return ThresholdFake(list(CANDIDATES))


# ---------------------------------------------------------------------------
# Equivalence of the derived sweep with direct per-threshold runs (offline)
# ---------------------------------------------------------------------------

def test_derived_sweep_matches_direct_runs() -> None:
    # The fake proves the derivation plumbing: filtering one low-threshold run
    # must reproduce what running the same detector at each threshold gives.
    thresholds = (0.15, 0.25, 0.35, 0.50)
    sweep = reliability.sweep_detector(_fake(), IMAGE, thresholds)

    for row in sweep.per_threshold:
        direct = counting.count_vehicles(IMAGE, _fake(), confidence_threshold=row.threshold)
        assert row.total == direct.total_count
        assert row.counts_by_class == direct.counts_by_class


def test_detector_runs_once_at_the_lowest_threshold() -> None:
    first, second = _fake(), ThresholdFake([_detection(2, "car", 0.20)])
    report = reliability.sweep_all([first, second], IMAGE, (0.15, 0.25, 0.35, 0.50))

    assert report.sweeps[0].base_threshold == 0.15
    assert first.calls == 1 and first.thresholds == [0.15]
    assert second.calls == 1 and second.thresholds == [0.15]


def test_confidence_exactly_at_threshold_is_kept() -> None:
    sweep = reliability.sweep_detector(_fake(), IMAGE, (0.15, 0.25, 0.35, 0.50))
    by_threshold = {row.threshold: row for row in sweep.per_threshold}

    cars_at_025 = by_threshold[0.25].counts_by_class["car"]
    assert cars_at_025 == 3  # 0.62, 0.30, and the boundary 0.25
    assert by_threshold[0.35].counts_by_class["car"] == 1  # only 0.62 clears 0.35


def test_non_vehicle_classes_are_excluded() -> None:
    sweep = reliability.sweep_detector(_fake(), IMAGE, (0.15, 0.25, 0.35, 0.50))
    lowest = sweep.per_threshold[0]
    assert set(lowest.counts_by_class) == set(config.TARGET_CLASS_NAMES)
    assert lowest.total == 7  # the 0.90 person detection is never counted


def test_threshold_range_is_min_and_max_of_totals() -> None:
    sweep = reliability.sweep_detector(_fake(), IMAGE, (0.15, 0.25, 0.35, 0.50))
    totals = [row.total for row in sweep.per_threshold]
    assert sweep.threshold_range == (min(totals), max(totals)) == (2, 7)


def test_full_range_spans_all_models_and_thresholds() -> None:
    sparse = ThresholdFake([_detection(2, "car", 0.20), _detection(2, "car", 0.16)])
    report = reliability.sweep_all([_fake(), sparse], IMAGE, (0.15, 0.25, 0.35, 0.50))

    assert report.sweeps[1].threshold_range == (0, 2)
    assert report.full_range == (0, 7)


def test_unsorted_and_duplicate_thresholds_are_normalised() -> None:
    sweep = reliability.sweep_detector(_fake(), IMAGE, [0.5, 0.25, 0.25, 0.15])
    assert [row.threshold for row in sweep.per_threshold] == [0.15, 0.25, 0.5]
    assert sweep.base_threshold == 0.15


# ---------------------------------------------------------------------------
# Validation and reporting
# ---------------------------------------------------------------------------

def test_invalid_thresholds_and_detectors_are_rejected() -> None:
    with pytest.raises(ValueError):
        reliability.sweep_detector(_fake(), IMAGE, [])
    with pytest.raises(ValueError):
        reliability.sweep_detector(_fake(), IMAGE, [0.97])  # above MAX_CONFIDENCE
    with pytest.raises(ValueError):
        reliability.sweep_detector(_fake(), IMAGE, [0.0])
    with pytest.raises(ValueError):
        reliability.sweep_all([], IMAGE)


def test_format_report_lists_every_model_and_threshold() -> None:
    report = reliability.sweep_all([_fake()], IMAGE, (0.15, 0.25, 0.35, 0.50))
    text = reliability.format_report(report)

    assert "fake" in text
    for threshold in ("0.15", "0.25", "0.35", "0.50"):
        assert threshold in text
    assert "threshold range: (2, 7)" in text
    assert "full range across all models and thresholds: (2, 7)" in text


# ---------------------------------------------------------------------------
# Real-model equivalence, gated behind cached weights
# ---------------------------------------------------------------------------

@pytest.mark.requires_weights
@pytest.mark.skipif(
    os.environ.get("TRAFFICLENS_RUN_WEIGHTS") != "1",
    reason="requires cached model weights; set TRAFFICLENS_RUN_WEIGHTS=1 to run",
)
@pytest.mark.parametrize("model_key", sorted(config.PRETRAINED_MODEL_KEYS))
def test_real_sweep_matches_direct_runs(model_key: str) -> None:
    images = sorted(
        path
        for path in config.TESTSET_IMAGES_DIR.iterdir()
        if path.suffix.lower() in config.ALLOWED_EXTENSIONS
    )
    if not images:
        pytest.skip("no test set image available")

    import cv2

    from core.detector import build_detector

    image = cv2.imread(str(images[0]))
    detector = build_detector(model_key)
    detector.warmup()

    thresholds = tuple(config.THRESHOLD_SWEEP)
    sweep = reliability.sweep_detector(detector, image, thresholds)
    for row in sweep.per_threshold:
        direct = counting.count_vehicles(image, detector, confidence_threshold=row.threshold)
        assert row.total == direct.total_count, (model_key, row.threshold)
        assert row.counts_by_class == direct.counts_by_class, (model_key, row.threshold)


# ---------------------------------------------------------------------------
# H2 agreement measures: synthetic detections, offline
# ---------------------------------------------------------------------------

_CLASS_IDS = {name: class_id for class_id, name in config.TARGET_CLASS_MAP.items()}


def _box_det(class_name: str, xyxy: tuple[float, float, float, float], confidence: float = 0.9) -> Detection:
    return Detection(_CLASS_IDS[class_name], class_name, confidence, xyxy)


class FixedDetector:
    """Returns a canned detection list and counts how often it was called."""

    def __init__(self, name: str, family: str, detections: list[Detection]) -> None:
        self.name = name
        self.family = family
        self.detections = detections
        self.calls = 0
        self.thresholds: list[float] = []

    def warmup(self) -> None:
        pass

    def detect(self, image: np.ndarray, confidence_threshold: float) -> list[Detection]:
        self.calls += 1
        self.thresholds.append(confidence_threshold)
        return list(self.detections)


def test_identical_detections_agree_perfectly() -> None:
    shared = [_box_det("car", (0, 0, 10, 10)), _box_det("bus", (50, 50, 60, 60))]
    agreement = reliability.agreement_between(shared, list(shared))

    assert agreement.count_disagreement == 0.0
    assert all(value == 0.0 for value in agreement.count_disagreement_by_class.values())
    assert agreement.box_agreement == 1.0


def test_same_counts_in_different_places_disagree_on_boxes_only() -> None:
    left = [_box_det("car", (0, 0, 10, 10)), _box_det("car", (20, 20, 30, 30))]
    right = [_box_det("car", (200, 200, 210, 210)), _box_det("car", (220, 220, 230, 230))]
    agreement = reliability.agreement_between(left, right)

    assert agreement.count_disagreement == 0.0  # both counted two cars
    assert agreement.box_agreement == 0.0  # but no box is within IoU 0.5 of its twin


def test_one_sided_counts_are_maximal_disagreement() -> None:
    left = [_box_det("car", (0, 0, 10, 10)), _box_det("car", (20, 20, 30, 30))]
    agreement = reliability.agreement_between(left, [])

    assert agreement.count_disagreement_by_class["car"] == 1.0
    assert agreement.count_disagreement == 0.25  # one of four classes in disagreement
    assert agreement.box_agreement == 0.0
    assert reliability.box_agreement([], []) == 1.0  # two empty sets agree by definition


def test_partial_box_overlap_scores_the_f1_form() -> None:
    left = [_box_det("car", (0, 0, 10, 10)), _box_det("car", (100, 100, 110, 110))]
    right = [_box_det("car", (0, 0, 10, 10)), _box_det("car", (200, 200, 210, 210))]
    agreement = reliability.agreement_between(left, right)

    assert agreement.box_agreement == 0.5  # 2 * 1 match / (2 + 2)


def test_matching_is_class_aware() -> None:
    car = [_box_det("car", (0, 0, 10, 10))]
    bus = [_box_det("bus", (0, 0, 10, 10))]  # same geometry, different class
    agreement = reliability.agreement_between(car, bus)

    assert agreement.box_agreement == 0.0
    assert agreement.count_disagreement_by_class["car"] == 1.0
    assert agreement.count_disagreement_by_class["bus"] == 1.0
    assert agreement.count_disagreement == 0.5


def test_iou_exactly_at_threshold_counts_as_a_match() -> None:
    # Boxes (0,0,6,1) and (2,0,8,1): intersection 4, union 8, IoU exactly 0.5.
    agreement = reliability.box_agreement(
        [_box_det("car", (0.0, 0.0, 6.0, 1.0))],
        [_box_det("car", (2.0, 0.0, 8.0, 1.0))],
    )
    assert agreement == 1.0


def test_greedy_matching_is_one_to_one() -> None:
    left = [_box_det("car", (0, 0, 10, 10))]
    right = [_box_det("car", (0, 0, 10, 10)), _box_det("car", (1, 0, 11, 10))]
    # Both right boxes clear IoU 0.5 against the single left box, but the
    # greedy pass may match it only once: 2 * 1 / (1 + 2).
    assert reliability.box_agreement(left, right) == pytest.approx(2.0 / 3.0)


def test_non_vehicle_classes_join_neither_measure() -> None:
    person = Detection(0, "person", 0.95, (0, 0, 10, 10))
    car = _box_det("car", (0, 0, 10, 10))
    agreement = reliability.agreement_between([car, person], [car])

    assert agreement.count_disagreement == 0.0
    assert agreement.box_agreement == 1.0


def test_report_pairs_every_model_and_aggregates_within_and_cross() -> None:
    shared_car = _box_det("car", (0, 0, 10, 10))
    far_truck = _box_det("truck", (100, 100, 120, 120))
    detections = {
        "YOLOv8n": [shared_car],
        "YOLOv8s": [shared_car],
        "RT-DETR-L": [shared_car, far_truck],
    }
    families = {"YOLOv8n": "yolo", "YOLOv8s": "yolo", "RT-DETR-L": "detr"}
    report = reliability.build_reliability_report(detections, families, threshold=0.25)

    assert len(report.pairs) == 3
    assert report.best_yolo == "YOLOv8s"
    relations = {pair.models: pair.family_relation for pair in report.pairs}
    assert relations[("YOLOv8n", "YOLOv8s")] == "within"
    assert relations[("YOLOv8n", "RT-DETR-L")] == "cross"
    assert relations[("YOLOv8s", "RT-DETR-L")] == "cross"

    assert report.within_family.count_disagreement == 0.0
    assert report.within_family.box_agreement == 1.0
    # Cross aggregate is best YOLO versus the detr model only.
    assert report.cross_family.count_disagreement_by_class == {
        "car": 0.0, "motorcycle": 0.0, "bus": 0.0, "truck": 1.0,
    }
    assert report.cross_family.count_disagreement == 0.25
    assert report.cross_family.box_agreement == pytest.approx(2.0 / 3.0)


def test_cross_aggregate_averages_over_several_other_family_models() -> None:
    car = _box_det("car", (0, 0, 10, 10))
    truck = _box_det("truck", (100, 100, 120, 120))
    detections = {"n": [car], "s": [car], "d1": [truck], "d2": []}
    families = {"n": "yolo", "s": "yolo", "d1": "detr", "d2": "detr"}
    report = reliability.build_reliability_report(detections, families, threshold=0.25)

    # Cross pairs are (s, d1) with mean 0.5 and (s, d2) with mean 0.25; the
    # best YOLO (suffix s outranks n) never pairs with (n, *) within-family rows.
    assert report.best_yolo == "s"
    assert report.cross_family.count_disagreement == pytest.approx((0.5 + 0.25) / 2)
    assert report.cross_family.box_agreement == 0.0


def test_agreement_report_runs_each_detector_once() -> None:
    shared_car = _box_det("car", (0, 0, 10, 10))
    detectors = [
        FixedDetector("YOLOv8n", "yolo", [shared_car]),
        FixedDetector("YOLOv8s", "yolo", [shared_car]),
        FixedDetector("RT-DETR-L", "detr", [shared_car]),
    ]
    report = reliability.agreement_report(detectors, IMAGE, threshold=0.25)

    assert all(detector.calls == 1 for detector in detectors)
    assert all(detector.thresholds == [0.25] for detector in detectors)
    assert len(report.pairs) == 3
    assert report.within_family.box_agreement == 1.0
    assert report.cross_family.box_agreement == 1.0


def test_agreement_report_rejects_bad_detector_sets() -> None:
    with pytest.raises(ValueError):
        reliability.agreement_report([], IMAGE)
    duplicate = FixedDetector("x", "yolo", [])
    with pytest.raises(ValueError):
        reliability.agreement_report([duplicate, duplicate], IMAGE)


def test_format_agreement_lists_pairs_and_aggregates() -> None:
    car = _box_det("car", (0, 0, 10, 10))
    detectors = [
        FixedDetector("YOLOv8n", "yolo", [car]),
        FixedDetector("RT-DETR-L", "detr", [car]),
    ]
    text = reliability.format_agreement(reliability.agreement_report(detectors, IMAGE, threshold=0.25))

    assert "YOLOv8n vs RT-DETR-L (cross-family)" in text
    assert "count disagreement" in text
    assert "box agreement" in text
    assert "best yolo for the cross-family aggregate: YOLOv8n" in text
    assert "within family aggregate: not applicable" in text
