"""Image tab: the fully working Week 1 counting screen.

Takes the sidebar selections, runs the cached detector, and renders the annotated
result, a metrics row, a detections table, and the JSON and CSV downloads. All failures
are surfaced as friendly messages; raw tracebacks never reach the user.
"""
from __future__ import annotations

import cv2
import pandas as pd
import streamlit as st

from core import annotate, config, counting, export, validation, zones
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
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_downloads(result: CountResult, input_filename: str) -> None:
    """Add the JSON and CSV download buttons for this run."""
    payload = export.build_payload(result, input_filename)
    columns = st.columns(2)
    columns[0].download_button(
        "Download JSON",
        data=export.to_json_bytes(payload),
        file_name=export.make_json_filename(input_filename, payload["timestamp_utc"]),
        mime="application/json",
        width="stretch",
    )
    columns[1].download_button(
        "Download CSV",
        data=export.to_csv_bytes(payload),
        file_name=export.make_csv_filename(input_filename, payload["timestamp_utc"]),
        mime="text/csv",
        width="stretch",
    )


def _render_zone_analysis(
    image_bgr,
    result: CountResult,
    selected_classes: list[str],
    confidence_threshold: float,
    get_detector,
    get_comparison_detectors,
) -> None:
    if not st.checkbox(
        "Define an analysis zone",
        help="Enter polygon vertices as image-pixel x,y coordinates, one vertex per line.",
    ):
        return

    st.caption(
        "Coordinates use the uploaded image's pixel dimensions. "
        "The polygon must be inside the image."
    )
    coordinates = st.text_area(
        "Zone polygon coordinates (x,y per line)",
        placeholder="20,20\n300,20\n300,200\n20,200",
        key="zone_polygon_coordinates",
    )
    if not coordinates.strip():
        st.info("Enter at least three polygon vertices to calculate zone metrics.")
        return
    try:
        polygon = zones.parse_polygon_coordinates(
            coordinates, image_bgr.shape[1], image_bgr.shape[0]
        )
    except ValueError as error:
        st.warning(str(error))
        return

    try:
        weights, pcu_message, pcu_source = zones.load_pcu_weights()
    except zones.ZoneConfigurationError as error:
        weights, pcu_message, pcu_source = None, None, None
        st.error(str(error))
    try:
        occupancy_threshold, threshold_justification, _ = zones.load_occupancy_threshold()
    except zones.ZoneConfigurationError as error:
        occupancy_threshold, threshold_justification = None, None
        st.error(str(error))

    primary_name = getattr(get_detector(), "name", "YOLOv8n")
    if primary_name not in {spec.display_name for spec in config.MODEL_REGISTRY.values()}:
        primary_name = "YOLOv8n"
    model_detections = {primary_name: result.detections}
    comparison_models = get_comparison_detectors()
    for model in comparison_models:
        if model.name == primary_name:
            continue
        try:
            model_detections[model.name] = [
                detection
                for detection in model.detect(image_bgr, confidence_threshold)
                if detection.class_name in selected_classes
            ]
        except Exception as error:
            st.error(f"Could not calculate zone metrics for {model.name}: {error}")
            return

    table: list[dict[str, object]] = []
    for model_name, detections in model_detections.items():
        summary = zones.summarize_zone(detections, polygon, selected_classes)
        row: dict[str, object] = {
            "Model": model_name,
            "Vehicles in zone": summary["vehicle_count"],
            "Occupancy": f"{summary['occupancy']:.1%}",
        }
        row.update({
            f"{name.capitalize()} in zone": count
            for name, count in summary["counts_by_class"].items()
        })
        if weights is not None:
            row["PCU in zone"] = round(
                sum(
                    summary["counts_by_class"][name] * weights[name]
                    for name in selected_classes
                ),
                3,
            )
        if occupancy_threshold is not None:
            row["Congestion indicator"] = (
                "At or above configured threshold"
                if summary["occupancy"] >= occupancy_threshold
                else "Below configured threshold"
            )
        table.append(row)

    st.markdown("**Standard-mode zone metrics by model**")
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)
    if weights is None and pcu_message:
        st.info(pcu_message)
    if pcu_source is not None:
        st.caption(f"PCU weight source: {pcu_source}")
    if occupancy_threshold is not None:
        st.caption(
            "Congestion indicator: indicative and not calibrated against ground truth. "
            f"Configured justification: {threshold_justification}"
        )
    st.caption(
        "In-zone vehicle counts use box centres; occupancy is the union area of "
        "vehicle boxes clipped to the zone."
    )


def render(
    uploaded_file,
    selected_classes,
    confidence_threshold,
    get_detector,
    get_comparison_detectors,
) -> None:
    """Render the Image tab for the current sidebar selections.

    Args:
        uploaded_file: The Streamlit UploadedFile, or ``None`` when nothing is chosen.
        selected_classes: Vehicle class names to count.
        confidence_threshold: Slider value for the detector.
        get_detector: Callable returning the cached detector instance.
        get_comparison_detectors: Callable returning standard-mode model detectors.
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
        st.image(_to_rgb(image_bgr), width="stretch")
    with right:
        st.markdown("**Annotated image**")
        st.image(_to_rgb(annotate.annotate(image_bgr, result.detections)), width="stretch")

    _render_metrics(result)
    _render_zone_analysis(
        image_bgr, result, selected_classes, confidence_threshold,
        get_detector, get_comparison_detectors,
    )
    st.markdown("**Detections**")
    _render_table(result)
    _render_downloads(result, uploaded_file.name)
