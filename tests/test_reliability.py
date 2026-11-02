"""Tests for the threshold sweep in core.reliability.

The offline tests use a fake detector that applies the ultralytics semantics
(keep detections with ``confidence >= threshold``) so the derivation plumbing,
class filtering, and range maths are proven without weights or network. The
real-model equivalence test needs cached weights and is skipped unless
``TRAFFICLENS_RUN_WEIGHTS=1``; it is the only test that can check the sweep
against genuine ultralytics NMS behaviour.
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
@pytest.mark.parametrize("model_key", sorted(config.MODEL_REGISTRY))
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
