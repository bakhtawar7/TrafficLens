"""TrafficLens Streamlit entry point.

Builds the sidebar controls, the five tab shell, and the persistent limitations note,
then delegates each tab to its renderer. The detector is cached with
``st.cache_resource`` so moving the confidence slider never reloads the model.

Run from the project root with: ``streamlit run app/main.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable no matter how Streamlit is launched.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.ui import evaluation_tab, history_tab, image_tab, reliability_tab, video_tab
from core import config
from core.detector import YoloDetector

st.set_page_config(page_title="TrafficLens", page_icon="TL", layout="wide")


@st.cache_resource(show_spinner="Loading the YOLOv8n model on first use")
def get_detector() -> YoloDetector:
    """Return a process wide cached detector. Loads weights once on first call."""
    return YoloDetector()


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
    st.sidebar.markdown("**Model:** YOLOv8n (pretrained)")
    return uploaded_file, selected_classes, confidence_threshold


def _render_limitations() -> None:
    """Show the persistent, honest limitations expander."""
    with st.sidebar.expander("Limitations", expanded=False):
        st.markdown(
            "\n".join(
                [
                    "* Occlusion and overlap between vehicles cause undercounting.",
                    "* Night scenes and glare degrade detection quality.",
                    "* Small and distant vehicles are frequently missed.",
                    "* Rickshaws and other local vehicle types have no COCO class and may "
                    "be missed or mislabelled. TrafficLens never claims to detect rickshaws.",
                    "* Results come from pretrained models without verification against "
                    "ground truth in Week 1.",
                ]
            )
        )


def main() -> None:
    """Assemble the full dashboard."""
    st.title("TrafficLens")
    st.caption("Vehicle counting from a single image, with honest limits.")

    uploaded_file, selected_classes, confidence_threshold = _render_sidebar()
    _render_limitations()

    tab_image, tab_video, tab_reliability, tab_evaluation, tab_history = st.tabs(
        ["Image", "Video", "Reliability", "Evaluation", "History"]
    )
    with tab_image:
        image_tab.render(uploaded_file, selected_classes, confidence_threshold, get_detector)
    with tab_video:
        video_tab.render()
    with tab_reliability:
        reliability_tab.render()
    with tab_evaluation:
        evaluation_tab.render()
    with tab_history:
        history_tab.render()


if __name__ == "__main__":
    main()
