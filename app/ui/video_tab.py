"""Provenance-gated video counting with ByteTrack and a virtual line."""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import streamlit as st

from core import config, settings
from core.detector import YoloDetector
from core.video_counting import VideoProcessingError, process_video, verify_video_provenance

VIDEO_EXTENSIONS = ("avi", "mkv", "mov", "mp4")


@st.cache_resource(show_spinner="Loading the local frozen video model")
def _get_local_video_detector() -> YoloDetector:
    """Return the frozen YOLO model only when its weights are already on disk."""
    record = settings.settings_record(config.FINAL_SETTINGS_JSON)
    model_key = config.DEFAULT_MODEL_NAME
    if model_key not in record["models"]:
        raise VideoProcessingError(
            f"{config.FINAL_SETTINGS_JSON.name} has no settings for {model_key}."
        )
    spec = config.MODEL_REGISTRY[model_key]
    weights = config.PROJECT_ROOT / spec.weights_file
    if not weights.is_file():
        raise VideoProcessingError(
            f"Local model weights are missing: {weights}. Video mode never downloads weights."
        )
    return YoloDetector(weights=str(weights), name=spec.display_name)


def render() -> None:
    """Render the video upload, provenance gate, line setup, and count results."""
    st.markdown(
        "**Video results are not part of the evaluation and carry no accuracy claim.** "
        "Counts depend on detections and track identity continuity."
    )
    st.caption(
        "Tracking limitations: vehicles may receive duplicate or switched IDs, especially "
        "during occlusion. These errors can change crossing totals."
    )
    st.markdown(
        "Add a matching row to `data/samples/video/PROVENANCE.csv` before processing. "
        "Required columns: `filename`, `source_url`, `author`, `licence_text`, "
        "`download_date` (YYYY-MM-DD). The app does not download videos or model weights."
    )

    uploaded = st.file_uploader(
        "Upload a licensed video",
        type=list(VIDEO_EXTENSIONS),
        key="video_upload",
    )
    if uploaded is None:
        st.info("Upload a video with a complete provenance row to configure a counting line.")
        return

    try:
        verify_video_provenance(uploaded.name)
    except VideoProcessingError as error:
        st.error(str(error))
        return

    with tempfile.TemporaryDirectory(prefix="trafficlens_video_") as temporary:
        input_path = Path(temporary) / Path(uploaded.name).name
        input_path.write_bytes(uploaded.getvalue())
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            capture.release()
            st.error("Could not read the uploaded video.")
            return
        ok, preview = capture.read()
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        if not ok or width <= 0 or height <= 0:
            st.error("The uploaded video has no readable first frame.")
            return

        st.image(
            cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
            caption=f"Frame size: {width} × {height}",
        )
        st.caption(
            "Coordinates use the original frame's pixels. Inbound is movement from the "
            "negative to positive side of the directed line; reversing the endpoints "
            "reverses the direction labels."
        )
        col1, col2, col3, col4 = st.columns(4)
        line_start = (
            int(col1.number_input(
                "Line start X", min_value=0, max_value=width - 1, value=0, key="video_x1"
            )),
            int(col2.number_input(
                "Line start Y", min_value=0, max_value=height - 1, value=height // 2, key="video_y1"
            )),
        )
        line_end = (
            int(col3.number_input(
                "Line end X", min_value=0, max_value=width - 1, value=width - 1, key="video_x2"
            )),
            int(col4.number_input(
                "Line end Y", min_value=0, max_value=height - 1, value=height // 2, key="video_y2"
            )),
        )

        if not st.button("Count video crossings", type="primary"):
            return

        output_path = Path(temporary) / "trafficlens_annotated.mp4"
        crossings_path = Path(temporary) / "trafficlens_crossings.csv"
        try:
            detector = _get_local_video_detector()
            result = process_video(
                input_path,
                output_path,
                crossings_path,
                detector,
                line_start,
                line_end,
            )
        except (VideoProcessingError, OSError, ValueError) as error:
            st.error(str(error))
            return

        annotated_video = output_path.read_bytes()
        crossing_csv = crossings_path.read_bytes()
        st.success(f"Processed {result.frames_processed} frames with {result.model}.")
        inbound_col, outbound_col = st.columns(2)
        inbound_col.metric("Inbound crossings", result.inbound)
        outbound_col.metric("Outbound crossings", result.outbound)
        st.video(annotated_video)
        st.download_button(
            "Download annotated video",
            annotated_video,
            file_name="trafficlens_annotated.mp4",
            mime="video/mp4",
        )
        st.download_button(
            "Download crossing CSV",
            crossing_csv,
            file_name="trafficlens_crossings.csv",
            mime="text/csv",
        )
