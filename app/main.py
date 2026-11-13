"""TrafficLens Streamlit entry point.

Builds the sidebar controls, the four tab shell, and the persistent limitations note,
then delegates each tab to its renderer. The detector is cached with
``st.cache_resource`` so moving the confidence slider never reloads the model.

Run from the project root with: ``streamlit run app/main.py``.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# Ensure the project root is importable no matter how Streamlit is launched.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from core import dataset, evaluation
from app.ui import history_tab, image_tab, reliability_tab, video_tab
from core import config, detector
from core.detector import YoloDetector

st.set_page_config(page_title="TrafficLens", page_icon="TL", layout="wide")


@st.cache_resource(show_spinner="Loading the YOLOv8n model on first use")
def get_detector() -> YoloDetector:
    """Return a process wide cached detector. Loads weights once on first call."""
    return YoloDetector()


@st.cache_resource(show_spinner="Loading the fine-tuned rickshaw model on first use")
def get_finetuned_detector():
    """Return the separately cached fine-tuned detector."""
    return detector.build_detector(config.FINETUNED_MODEL_KEY)


@st.cache_resource(show_spinner="Loading the comparison models on first use")
def get_reliability_detectors() -> list:
    """Return one cached detector per registered model for the Reliability tab."""
    return [detector.build_detector(key) for key in config.PRETRAINED_MODEL_KEYS]


def _render_sidebar():
    """Draw the sidebar controls and return the current selections."""
    st.sidebar.header("Input")
    accepted = ", ".join(extension.lstrip(".").upper() for extension in config.ALLOWED_EXTENSIONS)
    limit_mb = config.MAX_UPLOAD_BYTES // (1024 * 1024)
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image",
        type=list(config.ALLOWED_EXTENSIONS),
        max_upload_size=limit_mb,
        help=f"{accepted} up to {limit_mb} MB.",
    )
    selected_classes = st.sidebar.multiselect(
        "Classes to count",
        options=list(config.TARGET_CLASS_NAMES),
        default=list(config.TARGET_CLASS_NAMES),
    )
    confidence_threshold = st.sidebar.slider(
        "Confidence threshold",
        min_value=float(config.MIN_CONFIDENCE),
        max_value=float(config.MAX_CONFIDENCE),
        value=float(config.DEFAULT_CONFIDENCE),
        step=0.05,
        help="Detections below this score are discarded. Range 0.05 to 0.95.",
    )
    st.sidebar.markdown("**Base model:** YOLOv8n (pretrained)")
    sliced = st.sidebar.toggle(
        "Sliced inference (Reliability tab)",
        value=False,
        help=(
            "On the Reliability tab, run each model on overlapping tiles and merge "
            "the duplicates instead of one full-frame pass."
        ),
    )
    return uploaded_file, selected_classes, confidence_threshold, sliced


def _render_limitations() -> None:
    """Show the persistent, honest limitations expander."""
    with st.sidebar.expander("Limitations", expanded=False):
        st.markdown(
            "\n".join(
                [
                    "* Occlusion and overlap between vehicles can hide vehicles or merge "
                    "their visual evidence.",
                    "* Night scenes and glare can degrade detections.",
                    "* Small and distant vehicles may be missed.",
                    "* Rickshaws and other local vehicle types have no COCO class and may "
                    "be missed or mislabelled.",
                    "* The optional fine-tuned rickshaw detector is a one-class pilot, "
                    "not a replacement for pretrained vehicle counts; its training-data "
                    "licence and citation were not provided.",
                    "* The fine-tuned detector overlaps with part of the test set; "
                    "overlapping images are excluded from its evaluation.",
                    "* Video tracking may assign duplicate or switched IDs, especially during "
                    "occlusion; video crossing totals can therefore be wrong.",
                    "* Video results are a demonstration, not an evaluation, and carry no "
                    "count guarantee. No provenance row is required for uploaded clips; "
                    "sample clips in data/samples/video/ still require PROVENANCE.csv.",
                    "* Results come from pretrained models applied to public Dhaka road "
                    "imagery; model outputs are not ground truth.",
                    "* Some dataset labels and annotation-referenced analyses are provisional "
                    "and unverified.",
                    "* Reliability signals are measured, but are not shown to predict error "
                    "unless the evaluation demonstrates that relationship.",
                ]
            )
        )


def _render_pilot_context() -> None:
    """Keep provenance, human-pass consistency, and fine-tuned status visible."""
    try:
        labels = evaluation.load_labels(config.LABELS_CSV)
        paired = [
            (item, data)
            for item, data in labels.items()
            if data.get("pass1") is not None and data.get("pass2") is not None
        ]
        sources = {
            dataset.parse_item_number(row.get("item_id", "")): row
            for row in csv.DictReader(
                config.SOURCES_CSV.open(encoding="utf-8-sig", newline="")
            )
        }
        differences = [
            (
                abs(
                    sum(int(data["pass1"].get(name, 0)) for name in config.LABEL_CLASS_NAMES)
                    - sum(int(data["pass2"].get(name, 0)) for name in config.LABEL_CLASS_NAMES)
                ),
                str(sources.get(item, {}).get("scene") or item),
            )
            for item, data in paired
        ]
        if differences:
            values = [value for value, _ in differences]
            scenes = [scene for _, scene in differences]
            consistency_ci = evaluation.cluster_bootstrap_ci(
                values, scenes, samples=2000, seed=config.SEED
            )
            consistency = (
                f"Human pass consistency: mean absolute total-count difference "
                f"{sum(values) / len(values):.2f} vehicles across {len(values)} paired "
                f"images ({len(set(scenes))} scenes); 95% scene-bootstrap interval "
                f"[{consistency_ci[0]:.2f}, {consistency_ci[1]:.2f}]."
            )
        else:
            consistency = "Human pass consistency is not available from the current labels."
        fine_summary_path = config.RESULTS_DIR / "finetuned_summary.json"
        if fine_summary_path.is_file():
            fine_summary = json.loads(fine_summary_path.read_text(encoding="utf-8"))
            contamination = fine_summary.get("contamination", {})
            evaluation_status = (
                f"The optional fine-tuned model has a separate pilot evaluation "
                f"(n={fine_summary.get('evaluation_first_pass_n', 'unknown')} labelled "
                f"evaluation images); {contamination.get('overlap_count', 'unknown')} of "
                f"{contamination.get('test_image_count', 'unknown')} test images overlap "
                "training data and are excluded."
            )
        else:
            evaluation_status = (
                "The optional fine-tuned model is separate; its pilot evaluation and "
                "contamination status are unavailable until results are generated."
            )
    except (OSError, ValueError, KeyError) as error:
        consistency = f"Human pass consistency could not be loaded ({error})."
        evaluation_status = "Fine-tuned pilot evaluation and contamination status unavailable."
    st.caption(
        "Results are a pilot on Dhaka imagery. "
        + consistency
        + " "
        + evaluation_status
    )


def main() -> None:
    """Assemble the full dashboard."""
    st.title("TrafficLens")
    st.caption("Image counts and experimental video crossings, with honest limits.")
    _render_pilot_context()

    uploaded_file, selected_classes, confidence_threshold, sliced = _render_sidebar()
    _render_limitations()

    tab_image, tab_video, tab_reliability, tab_history = st.tabs(
        ["Image", "Video", "Reliability", "History"]
    )
    with tab_image:
        image_tab.render(
            uploaded_file, selected_classes, confidence_threshold,
            get_detector, get_reliability_detectors, get_finetuned_detector,
        )
    with tab_video:
        video_tab.render()
    with tab_reliability:
        reliability_tab.render(
            uploaded_file, selected_classes, confidence_threshold, sliced,
            get_reliability_detectors,
        )
    with tab_history:
        history_tab.render()


if __name__ == "__main__":
    main()
