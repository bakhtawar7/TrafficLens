"""Dashboard shell tests driven by Streamlit's own AppTest harness.

The Week 1 scope includes a working multi tab dashboard, so the shell is tested the same
way the counting logic is. Streamlit executes ``app/main.py`` in process, which keeps the
real sidebar, tabs, and Image tab wiring under test while a fake detector stands in for
the YOLO weights. Nothing here touches the network.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import json
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


# Per model detection sets for the Reliability tab, keyed by registry display
# name. Counts are deliberately different (5, 4, 3) so the count range and the
# agreement measures are non-trivial offline. Person detections are dropped by
# the tab's class filter, so YOLOv8n counts five vehicles.
RELIABILITY_DETECTIONS: dict[str, list[Detection]] = {
    "YOLOv8n": BASE_DETECTIONS,
    "YOLOv8s": BASE_DETECTIONS[:4],
    "RT-DETR-L": BASE_DETECTIONS[:3],
}


class FakeModel:
    """A registered-model stand-in exposing the attributes the Reliability tab reads."""

    def __init__(self, name: str, family: str, detections: list[Detection]) -> None:
        self.name = name
        self.family = family
        self._detections = detections
        self.calls = 0

    def warmup(self) -> None:
        pass

    def detect(self, image, confidence_threshold: float, **kwargs) -> list[Detection]:
        self.calls += 1
        return [d for d in self._detections if d.confidence >= confidence_threshold]


def fake_build_detector(model_key: str) -> FakeModel:
    """Build a ``FakeModel`` mirroring what ``core.detector.build_detector`` returns."""
    spec = config.MODEL_REGISTRY[model_key]
    return FakeModel(spec.display_name, spec.family, RELIABILITY_DETECTIONS[spec.display_name])


def png_bytes(width: int = 320, height: int = 240) -> bytes:
    """Encode a plain grey image as PNG so the uploader has something to read."""
    ok, encoded = cv2.imencode(".png", np.full((height, width, 3), 120, dtype=np.uint8))
    assert ok
    return encoded.tobytes()


@pytest.fixture
def app(monkeypatch, tmp_path):
    """Return an AppTest bound to the real entry point with the detector replaced."""
    from streamlit.testing.v1 import AppTest

    FakeYolo.thresholds = []
    monkeypatch.setattr("core.detector.YoloDetector", FakeYolo)
    monkeypatch.setattr("core.detector.build_detector", fake_build_detector)
    # Keep the dashboard hermetic. The Evaluation tab goes live when
    # results/evaluation_table.csv exists, but that file is an untracked benchmark
    # output, so whether it happens to be on disk must not change these tests. Point
    # the paths at a non-existent temp file; tests that want the populated branch
    # override them explicitly.
    monkeypatch.setattr("core.config.EVALUATION_TABLE_CSV", tmp_path / "missing_table.csv")
    monkeypatch.setattr("core.config.EVALUATION_SUMMARY_JSON", tmp_path / "missing_summary.json")
    monkeypatch.setattr(
        "app.ui.reliability_tab.PREDICTOR_REPORT_PATH",
        tmp_path / "missing_predictor_report.json",
    )
    monkeypatch.setattr(
        "app.ui.reliability_tab.WEEK3_SUMMARY_PATH",
        tmp_path / "missing_week3_summary.md",
    )
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


def test_limitations_panel_covers_required_caveats(app) -> None:
    app.run()
    text = "\n".join(element.value for element in app.sidebar.markdown)
    required_phrases = (
        "Occlusion and overlap",
        "Night scenes and glare",
        "Small and distant vehicles",
        "Rickshaws and other local vehicle types",
        "provisional and unverified",
        "pretrained models applied to public Dhaka road imagery",
        "Reliability signals are measured",
        "not shown to predict error",
    )
    for phrase in required_phrases:
        assert phrase in text


def test_uploader_gates_file_type_and_size(app) -> None:
    app.run()
    proto = app.sidebar.file_uploader[0].proto
    assert sorted(proto.type) == sorted(config.ALLOWED_EXTENSIONS)
    assert proto.max_upload_size_mb == config.MAX_UPLOAD_BYTES // (1024 * 1024)


def test_video_tab_disclaims_accuracy_and_is_no_longer_a_placeholder(app) -> None:
    app.run()
    messages = [warning.value for warning in app.warning]
    assert not any("Planned for Week 3" in message for message in messages)
    assert sum("Planned for Week 4" in message for message in messages) == 2
    assert not app.metric
    assert not app.download_button
    # Reliability is no longer a placeholder: it prompts for an upload instead.
    assert any("compare the models" in info.value for info in app.info)
    assert any("Upload a video" in info.value for info in app.info)
    assert any("carry no accuracy claim" in value.value for value in app.markdown)
    assert not any("reliability scoring" in message.lower() for message in messages)


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


def test_image_tab_optional_zone_reports_standard_model_counts_and_occupancy(app) -> None:
    uploaded(app)
    next(widget for widget in app.checkbox if widget.label == "Define an analysis zone").set_value(True)
    app.run()
    next(widget for widget in app.text_area if widget.label == "Zone polygon coordinates (x,y per line)").set_value(
        "0,0\n100,0\n100,100\n0,100"
    )
    app.run()

    assert not app.exception
    table = next(element for element in app.dataframe if "Vehicles in zone" in element.value.columns)
    assert set(table.value["Model"]) == {"YOLOv8n", "YOLOv8s", "RT-DETR-L"}
    occupancy = [float(value.rstrip("%")) for value in table.value["Occupancy"]]
    assert all(0 < value < 100 for value in occupancy)
    assert any("PCU needs a cited source" in element.value for element in app.info)


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


# ---------------------------------------------------------------------------
# Reliability tab
# ---------------------------------------------------------------------------
def test_reliability_tab_asks_for_an_upload_before_doing_anything(app) -> None:
    app.run()
    assert any("compare the models" in info.value for info in app.info)
    assert not app.metric


def test_reliability_tab_reports_counts_range_and_agreement(app) -> None:
    uploaded(app)
    assert not app.exception
    counts = metrics_of(app)
    # Model totals differ (5, 4, 3) so the range is the spread between them.
    assert counts["Lowest model total"] == "3"
    assert counts["Highest model total"] == "5"
    assert counts["Count range"] == "3-5"
    # Both aggregate agreement values are shown; YOLOv8n vs YOLOv8s is within
    # family and YOLOv8s vs RT-DETR-L is cross family.
    assert "Within family: count disagreement" in counts
    assert "Within family: box agreement" in counts
    assert "Cross family: count disagreement" in counts
    assert "Cross family: box agreement" in counts


def test_reliability_sliced_toggle_runs_without_error(app) -> None:
    uploaded(app)
    app.sidebar.toggle[0].set_value(True).run()
    assert not app.exception
    counts = metrics_of(app)
    # A 320x240 frame fits in one 640 tile, so tiled counts match the standard run.
    assert counts["Count range"] == "3-5"


def _stored_predictor_report(verdict: str, exploratory: bool = False) -> dict:
    columns = ["yolov8n_mean_confidence", "yolov8s_mean_confidence", "rtdetr-l_mean_confidence"]
    model = {
        "feature_columns": columns,
        "imputation_medians": [0.5, 0.5, 0.5],
        "means": [0.5, 0.5, 0.5],
        "scales": [0.2, 0.2, 0.2],
        "coefficients": [1.0, 0.5, -0.25],
        "intercept": 1.0,
        "flag_threshold": 0.5,
    }
    validation_metrics = {
        "primary_rule": {"precision": 0.6, "recall": 0.75},
        "scene_bootstrap_95ci": {
            "precision": {"low": 0.4, "high": 0.8},
            "recall": {"low": 0.5, "high": 0.9},
        },
    }
    return {
        "sample_size": 40,
        "exploratory": exploratory,
        "audited_label_readiness": {"evaluation_audited": 40},
        "hypotheses": {"H1": {"verdict": verdict}},
        "flag_policy": {"fallback_baseline": "mean_confidence"},
        "deployment": {
            "flag_fraction": 0.25,
            "models": {"ridge": model, "mean_confidence": model},
        },
        "models": {
            "ridge": {"needs_review": validation_metrics},
            "mean_confidence": {"needs_review": validation_metrics},
        },
    }


def test_reliability_without_predictor_shows_only_unvalidated_message(app) -> None:
    from app.ui.reliability_tab import reliability_prediction

    uploaded(app)
    assert any(reliability_prediction.UNVALIDATED_MESSAGE == item.value for item in app.info)
    assert "Predicted counting error" not in metrics_of(app)
    assert "Needs review flag" not in metrics_of(app)
    all_text = "\n".join(
        [item.value for item in app.markdown]
        + [item.value for item in app.caption]
        + [item.value for item in app.info]
    )
    assert "Top contributing features" not in all_text


@pytest.mark.parametrize(
    ("verdict", "expected_source"),
    [
        ("supported", "learned ridge predictor"),
        ("not supported", "simple `mean_confidence` baseline"),
        ("inconclusive", "simple `mean_confidence` baseline"),
    ],
)
def test_reliability_flag_source_follows_stored_h1_verdict(
    app, tmp_path, monkeypatch, verdict, expected_source
) -> None:
    from app.ui import reliability_tab

    report_path = tmp_path / "predictor_report.json"
    report_path.write_text(json.dumps(_stored_predictor_report(verdict)), encoding="utf-8")
    monkeypatch.setattr(reliability_tab, "PREDICTOR_REPORT_PATH", report_path)

    uploaded(app)

    assert not app.exception
    labels = metrics_of(app)
    assert labels["Predicted counting error"].endswith("vehicles")
    assert labels["Needs review flag"] == "Yes"
    captions = "\n".join(item.value for item in app.caption)
    assert expected_source in captions
    assert f"H1: {verdict}" in captions
    assert "Audited images: 40" in captions
    assert "Precision: 0.600 (95% scene interval 0.400 to 0.800)" in captions
    assert "Recall: 0.750 (95% scene interval 0.500 to 0.900)" in captions
    assert "estimated from 40 audited images on Dhaka street imagery; it may not transfer to other scenes." in captions
    assert "guarantee" not in captions.lower()
    assert "accuracy" not in captions.lower()
    assert any("Top contributing features" in item.value for item in app.markdown)


def test_reliability_flag_status_reports_exploratory_result(app, tmp_path, monkeypatch) -> None:
    from app.ui import reliability_tab

    report_path = tmp_path / "predictor_report.json"
    report_path.write_text(
        json.dumps(_stored_predictor_report("supported", exploratory=True)),
        encoding="utf-8",
    )
    monkeypatch.setattr(reliability_tab, "PREDICTOR_REPORT_PATH", report_path)

    uploaded(app)

    assert not app.exception
    assert any("Exploratory: yes" in item.value for item in app.caption)


def test_reliability_does_not_show_flag_when_h1_verdict_is_missing(app, tmp_path, monkeypatch) -> None:
    from app.ui import reliability_tab

    report = _stored_predictor_report("supported")
    report["hypotheses"] = {}
    report_path = tmp_path / "predictor_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(reliability_tab, "PREDICTOR_REPORT_PATH", report_path)

    uploaded(app)

    from app.ui.reliability_tab import reliability_prediction

    assert any(reliability_prediction.UNVALIDATED_MESSAGE == item.value for item in app.info)
    assert "Predicted counting error" not in metrics_of(app)
    assert "Needs review flag" not in metrics_of(app)


# ---------------------------------------------------------------------------
# Evaluation tab
# ---------------------------------------------------------------------------
def test_evaluation_tab_keeps_placeholder_without_a_table(app) -> None:
    app.run()
    assert any("Planned for Week 4" in warning.value for warning in app.warning)


def test_evaluation_tab_renders_table_chart_and_agreement(tmp_path, monkeypatch) -> None:
    import csv as _csv
    import json

    from streamlit.testing.v1 import AppTest

    from core import evaluation as ev

    table = tmp_path / "evaluation_table.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = _csv.DictWriter(handle, fieldnames=list(ev.TABLE_COLUMNS))
        writer.writeheader()
        writer.writerow({
            "Item": "tl0011", "Condition": "night", "Ground Truth": 6, "YOLOv8n": 5,
            "YOLOv8s": 4, "RT-DETR-L": 8, "Sliced Count": 6, "Count Range": "4-8",
            "Agreement": 0.62, "Actual Error": 1,
        })
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({
        "warning": None,
        "condition_mae": {"night": {"YOLOv8n": {"mae": 1.0, "ci95": [0.5, 1.5], "n": 3}}},
        "labeller_passes": {
            "items_both_passes": 4, "covered_mean_abs_difference": 0.5,
            "covered_ci95": [0.1, 0.9], "all_vehicles_mean_abs_difference": 0.7,
            "identical_fraction": 0.5,
        },
    }), encoding="utf-8")

    monkeypatch.setattr("core.config.EVALUATION_TABLE_CSV", table)
    monkeypatch.setattr("core.config.EVALUATION_SUMMARY_JSON", summary)
    at = AppTest.from_file(APP_ENTRY, default_timeout=120)
    at.run()
    assert not at.exception
    labels = metrics_of(at)
    assert labels["Items labelled twice"] == "4"
    assert len(at.dataframe) >= 1, "the evaluation table should render"


def test_evaluation_tab_shows_the_refusal_when_every_row_was_rejected(tmp_path, monkeypatch) -> None:
    """A header only table plus a refusal warning must inform, not crash.

    This is the state the evaluator produces when every LABELS row still carries the
    unverified auto derived note, so the dashboard must say so rather than show numbers.
    """
    import csv as _csv
    import json

    from streamlit.testing.v1 import AppTest

    from core import evaluation as ev

    table = tmp_path / "evaluation_table.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        _csv.DictWriter(handle, fieldnames=list(ev.TABLE_COLUMNS)).writeheader()
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({
        "warning": "no error statistic computed: all 55 labelled row(s) were refused",
        "condition_mae": {},
        "labeller_passes": {"items_both_passes": 0, "covered_mean_abs_difference": None,
                            "covered_ci95": [None, None], "all_vehicles_mean_abs_difference": None,
                            "identical_fraction": None},
        "model_mae": {"YOLOv8n": {"covered": {"mae": None, "ci95": [None, None], "n": 0}}},
    }), encoding="utf-8")

    monkeypatch.setattr("core.config.EVALUATION_TABLE_CSV", table)
    monkeypatch.setattr("core.config.EVALUATION_SUMMARY_JSON", summary)
    at = AppTest.from_file(APP_ENTRY, default_timeout=120)
    at.run()
    assert not at.exception
    assert any("no rows yet" in note.value for note in at.info)
    assert any("refused" in warning.value for warning in at.warning)
    for frame in at.dataframe:
        assert frame.value.empty, "nothing should be tabulated when nothing was scored"
