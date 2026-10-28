"""Tests for annotation output shape and empty scene handling."""
from __future__ import annotations

import numpy as np

from core import annotate
from core.detector import Detection


def test_annotate_empty_scene_returns_unchanged_copy():
    image = np.zeros((30, 40, 3), dtype=np.uint8)
    out = annotate.annotate(image, [])
    assert out.shape == image.shape
    assert out is not image
    assert np.array_equal(out, image)


def test_annotate_with_detections_keeps_shape_and_modifies():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    detections = [
        Detection(2, "car", 0.9, (10.0, 10.0, 60.0, 50.0)),
        Detection(7, "truck", 0.6, (20.0, 20.0, 110.0, 90.0)),
    ]
    out = annotate.annotate(image, detections)
    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, image)


def test_annotate_does_not_mutate_input():
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    snapshot = image.copy()
    annotate.annotate(image, [Detection(2, "car", 0.9, (1.0, 1.0, 20.0, 20.0))])
    assert np.array_equal(image, snapshot)
