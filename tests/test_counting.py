"""Tests for class filtering, threshold clamping, aggregation, and empty scenes."""
from __future__ import annotations

import numpy as np

from core import config, counting
from core.counting import count_by_class, filter_by_classes


def test_filter_by_classes_keeps_only_selected(sample_detections):
    filtered = filter_by_classes(sample_detections, ["car", "bus"])
    assert {d.class_name for d in filtered} == {"car", "bus"}
    assert len(filtered) == 3  # two cars plus one bus


def test_count_by_class_includes_zero_for_selected(sample_detections):
    filtered = filter_by_classes(sample_detections, list(config.TARGET_CLASS_NAMES))
    counts = count_by_class(filtered, list(config.TARGET_CLASS_NAMES))
    assert counts == {"car": 2, "motorcycle": 1, "bus": 1, "truck": 1}


def test_count_by_class_selected_class_with_no_detections_is_zero(sample_detections):
    filtered = filter_by_classes(sample_detections, ["car"])
    counts = count_by_class(filtered, ["car", "truck"])
    assert counts["truck"] == 0
    assert set(counts) == {"car", "truck"}


def test_count_vehicles_total_matches_filtered(stub_factory, sample_detections):
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    result = counting.count_vehicles(image, stub_factory(sample_detections))
    assert result.total_count == 5
    assert result.image_width == 200 and result.image_height == 100
    assert result.inference_ms >= 0.0


def test_count_vehicles_clamps_threshold_to_range(stub_factory, sample_detections):
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    detector = stub_factory(sample_detections)
    low = counting.count_vehicles(image, detector, confidence_threshold=0.0)
    assert low.confidence_threshold == config.MIN_CONFIDENCE
    high = counting.count_vehicles(image, stub_factory(sample_detections), confidence_threshold=1.0)
    assert high.confidence_threshold == config.MAX_CONFIDENCE


def test_empty_scene_returns_zero_without_error(stub_factory):
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    result = counting.count_vehicles(image, stub_factory([]))
    assert result.total_count == 0
    assert result.detections == []
    assert all(value == 0 for value in result.counts_by_class.values())
