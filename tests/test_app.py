"""Dashboard shell tests driven by Streamlit's own AppTest harness.

The Week 1 scope includes a working multi tab dashboard, so the shell is tested the same
way the counting logic is. Streamlit executes ``app/main.py`` in process, which keeps the
real sidebar, tabs, and Image tab wiring under test while a fake detector stands in for
the YOLO weights. Nothing here touches the network.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent

from core import config  # noqa: E402
from core.detector import Detection  # noqa: E402

APP_ENTRY = str(ROOT / "app" / "main.py")

TAB_LABELS = ["Image", "Video", "Reliability", "Evaluation", "History"]

BASE_DETECTIONS = [
    Detection(2, "car", 0.90, (0.0, 0.0, 40.0, 30.0)),
    Detection(2, "car", 0.60, (50.0, 20.0, 90.0, 55.0)),
    Detection(3, "motorcycle", 0.70, (10.0, 40.0, 45.0, 70.0)),
    Detection(5, "bus", 0.80, (60.0, 10.0, 150.0, 80.0)),
    Detection(7, "truck", 0.30, (30.0, 30.0, 130.0, 110.0)),
    Detection(0, "person", 0.95, (5.0, 5.0, 20.0, 40.0)),
]


class FakeYolo:
    """A detector with the shape of ``YoloDetector`` but no model behind it."""

    name = "fake"
    thresholds: list[float] = []

    def __init__(self, weights: str = "") -> None:
        self.weights = weights

    def detect(self, image, confidence_threshold: float) -> list[Detection]:
        """Return the fixed detections that clear the requested threshold."""
        FakeYolo.thresholds.append(confidence_threshold)
        return [d for d in BASE_DETECTIONS if d.confidence >= confidence_threshold]


def png_bytes(width: int = 320, height: int = 240) -> bytes:
    """Encode a plain grey image as PNG so the uploader has something to read."""
    ok, encoded = cv2.imencode(".png", np.full((height, width, 3), 120, dtype=np.uint8))
    assert ok
    return encoded.tobytes()


@pytest.fixture
def app(monkeypatch):
    """Return an AppTest bound to the real entry point with the detector replaced."""
    from streamlit.testing.v1 import AppTest

    FakeYolo.thresholds = []
    monkeypatch.setattr("core.detector.YoloDetector", FakeYolo)
    return AppTest.from_file(APP_ENTRY, default_timeout=120)


def uploaded(app) -> None:
    """Run the shell, put a valid image in the uploader, and run again."""
    app.run()
    app.sidebar.file_uploader[0].upload("street.png", png_bytes(), "image/png")
    app.run()


def metrics_of(app) -> dict[str, str]:
    """Map every metric tile label to its displayed value."""
    return {metric.label: str(metric.value) for metric in app.metric}


def test_shell_renders_the_five_week_one_tabs(app) -> None:
    app.run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == TAB_LABELS
    assert app.title[0].value == "TrafficLens"


def test_sidebar_exposes_the_counting_controls(app) -> None:
    app.run()
    assert [uploader.label for uploader in app.sidebar.file_uploader] == ["Upload an image"]

    classes = app.sidebar.multiselect[0]
    assert classes.options == list(config.TARGET_CLASS_NAMES)
    assert classes.value == list(config.TARGET_CLASS_NAMES)

    slider = app.sidebar.slider[0]
    assert slider.value == pytest.approx(config.DEFAULT_CONFIDENCE)
    assert slider.proto.min == pytest.approx(config.MIN_CONFIDENCE)
    assert slider.proto.max == pytest.approx(config.MAX_CONFIDENCE)
    assert slider.proto.step == pytest.approx(0.05)

    assert any("YOLOv8n" in item.value for item in app.sidebar.markdown)
    assert [expander.label for expander in app.sidebar.expander] == ["Limitations"]


def test_uploader_gates_file_type_and_size(app) -> None:
    app.run()
    proto = app.sidebar.file_uploader[0].proto
    assert sorted(proto.type) == sorted(config.ALLOWED_EXTENSIONS)
    assert proto.max_upload_size_mb == config.MAX_UPLOAD_BYTES // (1024 * 1024)


def test_placeholder_tabs_declare_their_week_and_show_no_results(app) -> None:
    app.run()
    messages = [warning.value for warning in app.warning]
    assert sum("Planned for Week 3" in message for message in messages) == 2
    assert sum("Planned for Week 4" in message for message in messages) == 2
    assert not app.metric
    assert not app.download_button


def test_image_tab_asks_for_an_upload_before_doing_anything(app) -> None:
    app.run()
    assert any("Upload a JPG" in info.value for info in app.info)
    assert not app.error
    assert FakeYolo.thresholds == []


def test_image_tab_counts_an_uploaded_image(app) -> None:
    uploaded(app)
    assert not app.error
    assert FakeYolo.thresholds == [config.DEFAULT_CONFIDENCE]

    counts = metrics_of(app)
    assert counts["Total vehicles"] == "5"
    assert (counts["Car"], counts["Motorcycle"], counts["Bus"], counts["Truck"]) == ("2", "1", "1", "1")

    assert len(app.image) == 2, "original and annotated image should both render"
    assert sorted(button.label for button in app.download_button) == ["Download CSV", "Download JSON"]


def test_class_selection_narrows_the_count(app) -> None:
    uploaded(app)
    app.sidebar.multiselect[0].set_value(["car"]).run()
    assert not app.error

    counts = metrics_of(app)
    assert counts["Total vehicles"] == "2"
    assert counts["Car"] == "2"
    assert "Bus" not in counts, "unselected classes should not be offered as tiles"


def test_confidence_slider_is_passed_through_to_the_detector(app) -> None:
    uploaded(app)
    app.sidebar.slider[0].set_value(0.85).run()
    assert FakeYolo.thresholds[-1] == pytest.approx(0.85)
    assert metrics_of(app)["Total vehicles"] == "1"


def test_image_tab_reports_an_empty_file_without_detecting(app) -> None:
    app.run()
    app.sidebar.file_uploader[0].upload("blank.png", b"", "image/png")
    app.run()
    assert any("empty" in error.value for error in app.error)
    assert FakeYolo.thresholds == []
    assert not app.download_button
