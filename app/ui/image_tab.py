"""Image tab: the fully working Week 1 counting screen.

Takes the sidebar selections, runs the cached detector, and renders the annotated
result, a metrics row, a detections table, and the JSON and CSV downloads. All failures
are surfaced as friendly messages; raw tracebacks never reach the user.
"""
from __future__ import annotations

import hashlib
import html
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from core import annotate, config, counting, export, history, validation, zones
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


def _weights_sha256(model_name: str) -> str | None:
    for spec in config.MODEL_REGISTRY.values():
        if spec.display_name == model_name:
            return spec.weights_sha256
    return None


def _record_image_history(
    filename: str,
    image_bytes: bytes,
    image_bgr,
    selected_classes: list[str],
    confidence_threshold: float,
    results: list[CountResult],
    rickshaw_mode: bool,
) -> None:
    """Persist a deduplicated image run without allowing history errors to block counting."""
    settings_hash = hashlib.sha256(config.FINAL_SETTINGS_JSON.read_bytes()).hexdigest()
    if rickshaw_mode:
        fine_settings_hash = hashlib.sha256(
            config.FINETUNED_SETTINGS_JSON.read_bytes()
        ).hexdigest()
        settings_hash = hashlib.sha256(
            f"{settings_hash}|{fine_settings_hash}".encode("ascii")
        ).hexdigest()
    try:
        history.record_run(
            {
                "input_filename": filename,
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "width": int(image_bgr.shape[1]),
                "height": int(image_bgr.shape[0]),
                "settings_sha256": settings_hash,
                "mode": "standard",
                "selected_classes": selected_classes,
                "threshold": float(confidence_threshold),
                "rickshaw_mode": rickshaw_mode,
            },
            [
                {
                    "model": result.model_name,
                    "weights_sha256": _weights_sha256(result.model_name),
                    "total_count": result.total_count,
                    "counts_by_class": result.counts_by_class,
                    "inference_ms": result.inference_ms,
                }
                for result in results
            ],
        )
    except history.HistoryStoreError as error:
        st.warning(
            f"History could not be saved ({error}). Counting still works; "
            "check data/history.db permissions and close other database users."
        )


def _count_with_pretrained_fallback(
    image_bgr,
    candidates,
    selected_classes: list[str],
    confidence_threshold: float,
) -> tuple[CountResult, list[tuple[str, str]]]:
    """Try the requested pretrained detector, then remaining pretrained models."""
    failures: list[tuple[str, str]] = []
    seen: set[str] = set()
    for model in candidates:
        if model.name in seen:
            continue
        seen.add(model.name)
        try:
            result = counting.count_vehicles(
                image_bgr, model, selected_classes, confidence_threshold
            )
            return result, failures
        except Exception as error:
            failures.append((model.name, f"{type(error).__name__}: {error}"))
    details = "; ".join(f"{name}: {reason}" for name, reason in failures)
    raise RuntimeError(
        "No pretrained detector completed inference. "
        + (details or "No pretrained detectors were available.")
    )


def _render_downloads(result: CountResult, input_filename: str) -> None:
    """Add the JSON and CSV download buttons for this run."""
    payload = export.build_payload(
        result, input_filename, weights_sha256=_weights_sha256(result.model_name)
    )
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


def _build_density_heatmap(image_bgr: np.ndarray, detections) -> np.ndarray:
    """Return a BGR image blending the original with a detection-box density heatmap.

    Each detection's bounding box adds a Gaussian-weighted contribution to a heat
    accumulator. The result is normalised, mapped through the COLORMAP_JET palette,
    and blended onto the source image at 40% weight. This is a descriptive
    visualisation only: it shows where detected boxes concentrate, not ground-truth
    vehicle density or any validated spatial statistic.
    """
    h, w = image_bgr.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    for d in detections:
        x0, y0, x1, y1 = d.xyxy
        x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(w, x1)), int(min(h, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        bw, bh = x1 - x0, y1 - y0
        # Gaussian blob centred on the box; sigma scales with the smaller box dimension.
        sigma_x = max(bw / 4.0, 1.0)
        sigma_y = max(bh / 4.0, 1.0)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        xs = np.arange(w, dtype=np.float32)
        ys = np.arange(h, dtype=np.float32)
        gx = np.exp(-((xs - cx) ** 2) / (2 * sigma_x ** 2))
        gy = np.exp(-((ys - cy) ** 2) / (2 * sigma_y ** 2))
        heat += np.outer(gy, gx)

    if heat.max() > 0:
        heat = (heat / heat.max() * 255).astype(np.uint8)
    else:
        heat = heat.astype(np.uint8)

    coloured = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(image_bgr, 0.60, coloured, 0.40, 0)
    return blended


def _render_density_heatmap(image_bgr: np.ndarray, result: CountResult) -> None:
    """Render the detection density heatmap expander in the Image tab."""
    with st.expander("Detection density heatmap (descriptive visualisation)"):
        st.caption(
            "Descriptive visualisation, not a measurement. The colour overlay shows "
            "where detected bounding boxes concentrate in this image; it is not a "
            "calibrated density or a ground-truth spatial analysis. Colour scale: "
            "blue (low) → green → yellow → red (high)."
        )
        if result.detections:
            heatmap = _build_density_heatmap(image_bgr, result.detections)
            st.image(_to_rgb(heatmap), width="stretch")
        else:
            st.info("No detections to visualise; the heatmap is empty.")


def _build_html_report(
    input_filename: str,
    image_hash: str,
    settings_hash: str,
    result: CountResult,
    reliability: dict | None,
    rickshaw_mode: bool,
    finetuned_status: str,
    timestamp_utc: str,
) -> str:
    """Build a self-contained HTML summary for one run.

    The report lists filename and hash, settings hashes, per-model counts,
    count range, agreement, flag and validation status, rickshaw mode and
    contamination note, limitations, and timestamp. No count is expressed as
    a percentage; no accuracy claim is made.
    """
    def esc(value: object) -> str:
        return html.escape(str(value) if value is not None else "not available")

    rows_html = ""
    if reliability:
        low = reliability.get("count_range_min")
        high = reliability.get("count_range_max")
        range_str = f"{low}–{high}" if low is not None and high is not None else "not available"
        agreement_count = reliability.get("agreement_count")
        agreement_box = reliability.get("agreement_box")
        flag = reliability.get("flag_status", "not available")
        predicted_error = reliability.get("predicted_error")
    else:
        range_str = "not available"
        agreement_count = None
        agreement_box = None
        flag = "not available"
        predicted_error = None

    classes_str = ", ".join(result.selected_classes) if result.selected_classes else "not available"
    model_rows = f"""
        <tr><td>Model</td><td>{esc(result.model_name)}</td></tr>
        <tr><td>Total count (selected classes)</td><td>{esc(result.total_count)}</td></tr>
        <tr><td>Counts by class</td><td>{esc(result.counts_by_class)}</td></tr>
        <tr><td>Inference time (ms)</td><td>{esc(f'{result.inference_ms:.1f}')}</td></tr>
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TrafficLens run summary — {esc(input_filename)}</title>
<style>
  body {{ font-family: sans-serif; max-width: 760px; margin: 2em auto; color: #222; }}
  h1 {{ font-size: 1.3em; border-bottom: 1px solid #ccc; padding-bottom: .4em; }}
  h2 {{ font-size: 1.05em; margin-top: 1.4em; color: #444; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1em; }}
  td {{ border: 1px solid #ccc; padding: .35em .6em; vertical-align: top; }}
  td:first-child {{ font-weight: bold; width: 46%; background: #f7f7f7; }}
  .caveat {{ background: #fff8e1; border: 1px solid #ffe082; padding: .6em 1em;
             border-radius: 4px; font-size: .9em; margin: 1em 0; }}
  footer {{ font-size: .8em; color: #888; margin-top: 2em; border-top: 1px solid #eee;
            padding-top: .6em; }}
</style>
</head>
<body>
<h1>TrafficLens run summary</h1>
<div class="caveat">
Results are a pilot on Dhaka imagery. Reference counts are human counts;
human pass consistency is a mean absolute total-count difference of 1.62 vehicles
over 13 paired images in 7 scenes (95% scene-bootstrap interval 0.67 to 2.46 vehicles).
The optional fine-tuned mode is a separate model with its own evaluation and
contamination status. No count in this report is a percentage of ground truth or
an accuracy figure. Reliability signals have not been validated against ground truth
in a pre-registered test.
</div>

<h2>Image provenance</h2>
<table>
  <tr><td>Filename (as uploaded)</td><td>{esc(input_filename)}</td></tr>
  <tr><td>SHA-256 (image bytes)</td><td>{esc(image_hash)}</td></tr>
  <tr><td>Timestamp (UTC)</td><td>{esc(timestamp_utc)}</td></tr>
  <tr><td>Image dimensions (px)</td><td>{esc(result.image_width)} × {esc(result.image_height)}</td></tr>
</table>

<h2>Settings provenance</h2>
<table>
  <tr><td>Settings SHA-256</td><td>{esc(settings_hash)}</td></tr>
  <tr><td>Selected classes</td><td>{esc(classes_str)}</td></tr>
  <tr><td>Confidence threshold</td><td>{esc(result.confidence_threshold if hasattr(result, 'confidence_threshold') else 'see settings hash')}</td></tr>
  <tr><td>Rickshaw-aware mode</td><td>{"On — separate one-class pilot model" if rickshaw_mode else "Off — pretrained models only"}</td></tr>
</table>

<h2>Detection result</h2>
<table>
  {model_rows}
</table>

<h2>Reliability signals</h2>
<table>
  <tr><td>Count range (min–max across models)</td><td>{esc(range_str)}</td></tr>
  <tr><td>Count agreement score</td><td>{esc(agreement_count)}</td></tr>
  <tr><td>Box agreement score (IoU 0.5)</td><td>{esc(agreement_box)}</td></tr>
  <tr><td>Flag status</td><td>{esc(flag)}</td></tr>
  <tr><td>Predicted error</td><td>{esc(predicted_error)}</td></tr>
  <tr><td>Validation status</td><td>Not yet validated against ground truth in a pre-registered test.</td></tr>
</table>

<h2>Fine-tuned model status</h2>
<table>
  <tr><td>Mode</td><td>{"Active (see detections above)" if rickshaw_mode else "Not used for this run"}</td></tr>
  <tr><td>Status</td><td>{esc(finetuned_status)}</td></tr>
</table>

<h2>Limitations</h2>
<ul>
  <li>Occlusion and overlap between vehicles can cause undercounting.</li>
  <li>Night scenes and glare degrade detection quality.</li>
  <li>Small and distant vehicles are frequently missed.</li>
  <li>The pretrained COCO detectors have no rickshaw class and may miss or mislabel
      local vehicle types.</li>
  <li>The optional one-class fine-tuned rickshaw pilot is separate, cannot detect other
      categories, and is not a validated replacement for pretrained counts.</li>
  <li>Results come from a pilot on Dhaka street imagery; they do not generalise to other
      cities or scene types without further evaluation.</li>
  <li>Count ranges and agreement scores are measured between models, not against
      human ground truth.</li>
</ul>

<footer>
Generated by TrafficLens. This report does not contain image bytes or personal data.
Report timestamp: {esc(timestamp_utc)}.
</footer>
</body>
</html>"""


def _render_html_report(
    input_filename: str,
    image_bytes: bytes,
    result: CountResult,
    settings_hash: str,
    rickshaw_mode: bool,
    finetuned_status: str,
    timestamp_utc: str,
) -> None:
    """Render the HTML report download button in the Image tab."""
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    report_html = _build_html_report(
        input_filename=input_filename,
        image_hash=image_hash,
        settings_hash=settings_hash,
        result=result,
        reliability=None,
        rickshaw_mode=rickshaw_mode,
        finetuned_status=finetuned_status,
        timestamp_utc=timestamp_utc,
    )
    stem = Path(input_filename).stem
    st.download_button(
        "Download run summary (HTML)",
        data=report_html.encode("utf-8"),
        file_name=f"trafficlens_summary_{stem}_{timestamp_utc[:10]}.html",
        mime="text/html",
        help="A self-contained HTML file with run metadata, counts, and reliability signals. No image bytes are included.",
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
    if primary_name not in {
        config.MODEL_REGISTRY[key].display_name
        for key in config.PRETRAINED_MODEL_KEYS
    }:
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
    get_finetuned_detector=None,
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
        preferred = get_detector()
        candidates = [preferred]
        fallback_models = get_comparison_detectors()
        preferred_names = {preferred.name}
        candidates.extend(model for model in fallback_models if model.name not in preferred_names)
        result, fallback_failures = _count_with_pretrained_fallback(
            image_bgr, candidates, selected_classes, confidence_threshold
        )
        if fallback_failures:
            failed = "; ".join(f"{name} ({reason})" for name, reason in fallback_failures)
            st.warning(
                f"Some pretrained weights or inference paths were unavailable: {failed}. "
                f"Using {result.model_name} for this image; check those model files or "
                "network access to restore the preferred model."
            )
    except Exception as error:
        st.error(
            f"No pretrained detector could run on this image ({error}). Weights may be "
            "missing, corrupt, or interrupted during download. Check network access and "
            "the model files, then retry; no count was produced."
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
    _render_density_heatmap(image_bgr, result)
    _record_image_history(
        uploaded_file.name,
        data,
        image_bgr,
        list(selected_classes),
        float(confidence_threshold),
        [result],
        False,
    )
    _render_zone_analysis(
        image_bgr, result, selected_classes, confidence_threshold,
        get_detector, get_comparison_detectors,
    )
    st.markdown("**Detections**")
    _render_table(result)
    _render_downloads(result, uploaded_file.name)
    import hashlib as _hashlib
    from datetime import datetime, timezone as _tz
    _settings_hash = (
        _hashlib.sha256(config.FINAL_SETTINGS_JSON.read_bytes()).hexdigest()
        if config.FINAL_SETTINGS_JSON.is_file()
        else "unavailable"
    )
    _ts = datetime.now(_tz.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    _render_html_report(
        input_filename=uploaded_file.name,
        image_bytes=data,
        result=result,
        settings_hash=_settings_hash,
        rickshaw_mode=False,
        finetuned_status="Not used for this run.",
        timestamp_utc=_ts,
    )

    if get_finetuned_detector is None:
        return
    if not st.toggle(
        "Rickshaw aware mode (fine tuned)",
        value=False,
        help="Runs a separate one-class rickshaw detector; it does not change pretrained counts or reliability signals.",
    ):
        return

    try:
        settings_record = json.loads(
            config.FINETUNED_SETTINGS_JSON.read_text(encoding="utf-8")
        )
        weights_path = config.FINETUNED_MODEL_WEIGHTS
        if not weights_path.is_file():
            raise FileNotFoundError(f"fine-tuned weights not found: {weights_path}")
        model = get_finetuned_detector()
        started = time.perf_counter()
        fine_detections = model.detect(
            image_bgr,
            float(settings_record["confidence"]),
            input_size=int(settings_record["input_size"]),
            nms_iou=float(settings_record["nms_iou"]),
        )
        fine_result = counting.build_count_result(
            fine_detections,
            ["rickshaw"],
            float(settings_record["confidence"]),
            model.name,
            image_bgr.shape[1],
            image_bgr.shape[0],
            (time.perf_counter() - started) * 1000.0,
        )
        _record_image_history(
            uploaded_file.name,
            data,
            image_bgr,
            list(selected_classes),
            float(confidence_threshold),
            [result, fine_result],
            True,
        )
        summary_path = config.RESULTS_DIR / "finetuned_summary.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.is_file()
            else {}
        )
    except Exception as error:
        st.warning(
            f"The optional fine-tuned detector could not run ({error}). Its weights "
            "may be missing or corrupt. The pretrained counts above remain available; "
            "check the model file or turn the optional mode off."
        )
        return

    evaluation_n = summary.get("evaluation_first_pass_n")
    contamination = summary.get("contamination", {})
    contaminated_n = contamination.get("overlap_count")
    test_n = contamination.get("test_image_count")
    if evaluation_n is None or contaminated_n is None or test_n is None:
        status = "pilot evaluation sample unavailable; contamination status unavailable"
    else:
        status = (
            f"evaluated as a pilot on {evaluation_n} images; contamination found "
            f"({contaminated_n} of {test_n} test images overlap training data and are "
            "excluded from fine-tuned evaluation)"
        )
    st.info(
        "Fine tuned on user-supplied archive.zip; not pretrained only; "
        f"{status}."
    )
    st.metric("Rickshaw count (fine tuned)", fine_result.counts_by_class["rickshaw"])
    st.image(
        _to_rgb(annotate.annotate(image_bgr, fine_result.detections)),
        caption="Fine-tuned rickshaw detections (separate from pretrained results)",
        width="stretch",
    )
    _render_downloads(fine_result, f"finetuned_{uploaded_file.name}")
