"""Shared pytest fixtures.

Tests use a stub detector and synthetic images so the suite runs offline with no model
weights. Anything that genuinely needs the YOLO weights is marked ``requires_weights``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.detector import Detection  # noqa: E402


def pytest_configure(config) -> None:
    """Register custom markers so strict marker checking passes."""
    config.addinivalue_line(
        "markers",
        "requires_weights: test needs YOLO model weights and network on first use.",
    )


class StubDetector:
    """A detector that returns a fixed detection list, for deterministic tests."""

    name = "stub"

    def __init__(self, detections: list[Detection]) -> None:
        self._detections = detections
        self.last_threshold: float | None = None

    def detect(self, image, confidence_threshold: float) -> list[Detection]:
        self.last_threshold = confidence_threshold
        return list(self._detections)


@pytest.fixture
def stub_factory():
    """Return a factory that builds a StubDetector over given detections."""
    return StubDetector


@pytest.fixture
def sample_detections() -> list[Detection]:
    """A small mixed detection set used across tests."""
    return [
        Detection(2, "car", 0.9, (0.0, 0.0, 50.0, 50.0)),
        Detection(2, "car", 0.4, (10.0, 10.0, 60.0, 60.0)),
        Detection(3, "motorcycle", 0.7, (5.0, 5.0, 40.0, 70.0)),
        Detection(5, "bus", 0.6, (20.0, 20.0, 120.0, 90.0)),
        Detection(7, "truck", 0.3, (30.0, 30.0, 130.0, 110.0)),
        Detection(0, "person", 0.8, (1.0, 1.0, 20.0, 40.0)),
    ]
