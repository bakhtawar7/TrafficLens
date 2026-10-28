"""Image tab: the fully working Week 1 counting screen.

Takes the sidebar selections, runs the cached detector, and renders the annotated
result, a metrics row, a detections table, and the JSON and CSV downloads. All failures
are surfaced as friendly messages; raw tracebacks never reach the user.
"""
from __future__ import annotations

import cv2
import pandas as pd
import streamlit as st

from core import annotate, counting, export, validation
from core.counting import CountResult


def _to_rgb(image_bgr):
    """Convert a BGR array to RGB for correct Streamlit display."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _render_metrics(result: CountResult) -> None:
    """Show the total, per class counts, and inference time as metric tiles."""
    columns = st.columns(len(result.selected_classes) + 2)
    columns[0].metric("Total vehicles", result.total_count)
    for index, class_name in enumerate(result.selected_classes, start=1):
        columns[index].metric(class_name.capitalize(), result.counts_by_class[class_name])
    columns[-1].metric("Inference (ms)", f"{result.inference_ms:.0f}")


def _render_table(result: CountResult) -> None:
    """Render the detections table, or a note when nothing was found."""
    if not result.detections:
        st.caption("No detections to list.")
        return
    rows = [
        {
            "class": d.class_name,
            "confidence": round(d.confidence, 3),
            "x_min": round(d.xyxy[0], 1),
            "y_min": round(d.xyxy[1], 1),
            "x_max": round(d.xyxy[2], 1),
            "y_max": round(d.xyxy[3], 1),
        }
        for d in result.detections
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_downloads(result: CountResult, input_filename: str) -> None:
    """Add the JSON and CSV download buttons for this run."""
    payload = export.build_payload(result, input_filename)
    columns = st.columns(2)
    columns[0].download_button(
        "Download JSON",
        data=export.to_json_bytes(payload),
        file_name=export.make_json_filename(input_filename, payload["timestamp_utc"]),
        mime="application/json",
        use_container_width=True,
    )
    columns[1].download_button(
        "Download CSV",
        data=export.to_csv_bytes(payload),
        file_name=export.make_csv_filename(input_filename, payload["timestamp_utc"]),
        mime="text/csv",
        use_container_width=True,
    )


def render(uploaded_file, selected_classes, confidence_threshold, get_detector) -> None:
    """Render the Image tab for the current sidebar selections.

    Args:
        uploaded_file: The Streamlit UploadedFile, or ``None`` when nothing is chosen.
        selected_classes: Vehicle class names to count.
        confidence_threshold: Slider value for the detector.
        get_detector: Callable returning the cached detector instance.
    """
    if uploaded_file is None:
        st.info("Upload a JPG, JPEG, PNG, or WEBP image in the sidebar to begin counting.")
        return

    if not selected_classes:
        st.warning("Select at least one vehicle class to count.")
        return

    try:
        data = uploaded_file.getvalue()
        validation.validate_image_input(uploaded_file.name, data)
        image_bgr = validation.decode_image(data)
    except validation.TrafficLensInputError as error:
        st.error(str(error))
        return
    except Exception:
        st.error("Something went wrong while reading the image. Please try another file.")
        return

    try:
        result = counting.count_vehicles(image_bgr, get_detector(), selected_classes, confidence_threshold)
    except Exception:
        st.error(
            "The detector could not run on this image. This can happen if the model "
            "weights are unavailable. Please check the network and try again."
        )
        return

    if result.total_count == 0:
        st.info("0 vehicles detected for the selected classes at this confidence threshold.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Original image**")
        st.image(_to_rgb(image_bgr), use_container_width=True)
    with right:
        st.markdown("**Annotated image**")
        st.image(_to_rgb(annotate.annotate(image_bgr, result.detections)), use_container_width=True)

    _render_metrics(result)
    st.markdown("**Detections**")
    _render_table(result)
    _render_downloads(result, uploaded_file.name)
