"""Reliability tab: model agreement measured on the uploaded image.

Runs every registered detector on the current image (optionally inside the
sliced wrapper), then reports the per model counts, the spread between those
counts, and the two agreement measures defined in :mod:`core.reliability`:
normalised per class count disagreement and class-aware box agreement.

Everything on this screen is measured **between the models**. It says how much
the detectors differ from one another and deliberately offers no verdict about
whether any count is correct or trustworthy; that judgement needs ground truth,
which lives on the Evaluation tab.
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from core import config, counting, detector, features, history, reliability, reliability_prediction, settings, validation
from core.reliability import ReliabilityReport

PREDICTOR_REPORT_PATH = config.RESULTS_DIR / "predictor_report.json"
WEEK3_SUMMARY_PATH = config.RESULTS_DIR / "week3_summary.md"


def _run_models(image_bgr, detectors, selected_classes, threshold, sliced):
    """Run frozen standard inference for features and the selected mode for display.

    Returns:
        Rows, selected-mode detections, family names, and raw standard detections
        keyed by registry model id for :func:`core.features.image_features`.
    """
    rows: list[dict] = []
    detections_by_model: dict[str, list] = {}
    families_by_model: dict[str, str] = {}
    raw_by_model: dict[str, list] = {}
    inference_ms_by_model: dict[str, float] = {}
    failures: list[tuple[str, str]] = []
    frozen = settings.load_settings(config.FINAL_SETTINGS_JSON)
    model_settings = frozen.get("models", {})
    tile_settings = frozen.get("sliced_tiles", {})
    for model in detectors:
        model_key = next(
            (key for key in config.PRETRAINED_MODEL_KEYS
             if config.MODEL_REGISTRY[key].display_name == model.name),
            None,
        )
        if model_key is None:
            raise ValueError(f"unregistered reliability model: {model.name}")
        model_config = model_settings.get(model_key, {})
        input_size = model_config.get("input_size")
        nms_iou = model_config.get("nms_iou") if model_config.get("uses_nms") else None
        started = time.perf_counter()
        try:
            raw_standard = model.detect(
                image_bgr,
                min(config.THRESHOLD_SWEEP),
                input_size=input_size,
                nms_iou=nms_iou,
            )
            if sliced:
                runner = detector.SlicedDetector(
                    model,
                    tile_size=int(tile_settings.get("tile_size", config.SLICE_TILE_SIZE)),
                    overlap_ratio=float(tile_settings.get("overlap_ratio", config.SLICE_OVERLAP_RATIO)),
                    iou_threshold=float(tile_settings.get("iou_threshold", config.SLICE_IOU_THRESHOLD)),
                    input_size=input_size,
                    nms_iou=nms_iou,
                )
                displayed = runner.detect(image_bgr, threshold)
            else:
                displayed = [
                    detection for detection in raw_standard
                    if detection.confidence >= threshold
                ]
        except Exception as error:
            failures.append((model.name, f"{type(error).__name__}: {error}"))
            continue
        raw_by_model[model_key] = raw_standard
        inference_ms_by_model[model.name] = (time.perf_counter() - started) * 1000.0
        filtered = counting.filter_by_classes(displayed, selected_classes)
        counts = counting.count_by_class(filtered, selected_classes)
        detections_by_model[model.name] = filtered
        families_by_model[model.name] = getattr(model, "family", "other")
        row = {"Model": model.name, "Total": len(filtered)}
        row.update({name: counts[name] for name in selected_classes})
        rows.append(row)
    return (
        rows,
        detections_by_model,
        families_by_model,
        raw_by_model,
        inference_ms_by_model,
        failures,
    )


def _render_needs_review(feature_row: dict) -> dict:
    """Render a stored validated flag, or the exact unvalidated message."""
    configuration = reliability_prediction.load_flag_configuration(
        PREDICTOR_REPORT_PATH, WEEK3_SUMMARY_PATH
    )
    if configuration is None:
        st.info(reliability_prediction.UNVALIDATED_MESSAGE)
        return {
            "flag_status": "not validated",
            "flag_source": "not validated",
            "predicted_error": None,
        }
    prediction = reliability_prediction.score_features(feature_row, configuration)
    if prediction is None:
        st.info(reliability_prediction.UNVALIDATED_MESSAGE)
        return {
            "flag_status": "not validated",
            "flag_source": "not validated",
            "predicted_error": None,
        }

    st.markdown("**Needs review**")
    st.metric("Predicted counting error", f"{prediction['predicted_error']:.2f} vehicles")
    st.metric("Needs review flag", "Yes" if prediction["flagged"] else "No")
    if configuration["verdict"] == "supported":
        st.caption(
            "Source: learned ridge predictor, using the pre-registered highest-predicted-error "
            "quarter rule."
        )
    else:
        st.caption(
            f"Source: simple `{configuration['source']}` baseline named in the stored results; "
            "the highest-predicted-error quarter rule is applied."
        )

    precision = configuration.get("precision")
    recall = configuration.get("recall")
    precision_value = "not available" if precision is None else f"{float(precision):.3f}"
    recall_value = "not available" if recall is None else f"{float(recall):.3f}"
    intervals = configuration["intervals"]
    status = (
        f"H1: {configuration['verdict']} | Audited images: {configuration['audited_images']} | "
        f"Exploratory: {'yes' if configuration['exploratory'] else 'no'} | "
        f"Precision: {precision_value} (95% scene interval "
        f"{reliability_prediction.format_interval(intervals, 'precision')}) | "
        f"Recall: {recall_value} (95% scene interval "
        f"{reliability_prediction.format_interval(intervals, 'recall')})"
    )
    st.caption(status)
    st.caption(
        f"estimated from {configuration['audited_images']} audited images on Dhaka street "
        "imagery; it may not transfer to other scenes."
    )
    st.caption("Estimated signal only; review the image yourself.")
    st.markdown("Top contributing features")
    for item in prediction["top_features"]:
        st.write(f"- {item['feature']}: {item['contribution']:+.3f}")
    return {
        "flag_status": "flagged" if prediction["flagged"] else "not flagged",
        "flag_source": str(configuration["source"]),
        "predicted_error": float(prediction["predicted_error"]),
    }


def _record_reliability_history(
    uploaded_file,
    image_bytes: bytes,
    image_bgr,
    selected_classes: list[str],
    threshold: float,
    sliced: bool,
    rows: list[dict],
    inference_ms_by_model: dict[str, float],
    report: ReliabilityReport,
    flag_record: dict,
) -> None:
    """Save displayed model results and signals once; history failures are non-fatal."""
    import hashlib
    import statistics

    totals = [int(row["Total"]) for row in rows]
    pair_values = [pair.agreement for pair in report.pairs]
    setting_hash = hashlib.sha256(config.FINAL_SETTINGS_JSON.read_bytes()).hexdigest()
    try:
        history.record_run(
            {
                "input_filename": uploaded_file.name,
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "width": int(image_bgr.shape[1]),
                "height": int(image_bgr.shape[0]),
                "settings_sha256": setting_hash,
                "mode": "sliced" if sliced else "standard",
                "selected_classes": selected_classes,
                "threshold": threshold,
                "rickshaw_mode": False,
            },
            [
                {
                    "model": row["Model"],
                    "weights_sha256": next(
                        (
                            spec.weights_sha256
                            for spec in config.MODEL_REGISTRY.values()
                            if spec.display_name == row["Model"]
                        ),
                        None,
                    ),
                    "total_count": int(row["Total"]),
                    "counts_by_class": {
                        name: int(row[name]) for name in selected_classes
                    },
                    "inference_ms": float(inference_ms_by_model[row["Model"]]),
                }
                for row in rows
            ],
            {
                "count_range_min": min(totals),
                "count_range_max": max(totals),
                "agreement_count": (
                    statistics.fmean(item.count_disagreement for item in pair_values)
                    if pair_values else None
                ),
                "agreement_box": (
                    statistics.fmean(item.box_agreement for item in pair_values)
                    if pair_values else None
                ),
                **flag_record,
            },
        )
    except history.HistoryStoreError as error:
        st.warning(
            f"History could not be saved ({error}). Model comparison still works; "
            "check data/history.db permissions and close other database users."
        )


def _render_counts(rows: list[dict], selected_classes: list[str]) -> None:
    """Show the per-model count table and the count range across models."""
    st.markdown("**Per model counts** (measured from the models)")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    totals = [row["Total"] for row in rows]
    low, high = min(totals), max(totals)
    columns = st.columns(3)
    columns[0].metric("Lowest model total", low)
    columns[1].metric("Highest model total", high)
    columns[2].metric("Count range", f"{low}-{high}")


def _render_agreement(report: ReliabilityReport) -> None:
    """Show the pairwise agreement table and the within/cross family aggregates."""
    st.markdown("**Model agreement** (measured from the models)")
    if not report.pairs:
        st.caption("Agreement needs at least two models; only one is available.")
        return

    pair_rows = [
        {
            "Pair": f"{pair.models[0]} vs {pair.models[1]}",
            "Relation": pair.family_relation,
            "Count disagreement": round(pair.agreement.count_disagreement, 3),
            "Box agreement": round(pair.agreement.box_agreement, 3),
        }
        for pair in report.pairs
    ]
    st.dataframe(pd.DataFrame(pair_rows), width="stretch", hide_index=True)

    st.caption(
        "Count disagreement is the mean normalised per-class difference in [0, 1] "
        "(0 means identical counts); box agreement is class-aware IoU matching at "
        f"IoU {report.iou_threshold:.2f} (1 means the same boxes). Both are measured "
        "between models, not against real traffic."
    )

    columns = st.columns(4)
    for index, (label, agreement) in enumerate(
        (("Within family", report.within_family), ("Cross family", report.cross_family))
    ):
        if agreement is None:
            columns[index * 2].metric(f"{label}: count disagreement", "n/a")
            columns[index * 2 + 1].metric(f"{label}: box agreement", "n/a")
        else:
            columns[index * 2].metric(f"{label}: count disagreement", f"{agreement.count_disagreement:.3f}")
            columns[index * 2 + 1].metric(f"{label}: box agreement", f"{agreement.box_agreement:.3f}")
    if report.best_yolo is not None:
        st.caption(f"Cross-family aggregate uses best YOLO = {report.best_yolo}.")


def render(uploaded_file, selected_classes, confidence_threshold, sliced, get_detectors) -> None:
    """Render the Reliability tab for the current sidebar selections.

    Args:
        uploaded_file: The Streamlit UploadedFile, or ``None`` when nothing is chosen.
        selected_classes: Vehicle class names the models are compared on.
        confidence_threshold: Slider value every model runs at.
        sliced: When true, every model runs inside the sliced inference wrapper.
        get_detectors: Callable returning the cached list of registry detectors.
    """
    st.markdown(
        "This tab compares the detectors with each other on your image. Every number "
        "here is **measured from the models**: it shows how far they differ, and says "
        "nothing about whether any of them is right."
    )

    if uploaded_file is None:
        if reliability_prediction.load_flag_configuration(
            PREDICTOR_REPORT_PATH, WEEK3_SUMMARY_PATH
        ) is None:
            st.info(reliability_prediction.UNVALIDATED_MESSAGE)
        st.info("Upload a JPG, JPEG, PNG, or WEBP image in the sidebar to compare the models.")
        return
    if not selected_classes:
        st.warning("Select at least one vehicle class to compare.")
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
        detectors = list(get_detectors())
        (
            rows,
            detections_by_model,
            families_by_model,
            raw_by_model,
            inference_ms_by_model,
            model_failures,
        ) = _run_models(
            image_bgr, detectors, selected_classes, float(confidence_threshold), sliced
        )
    except Exception as error:
        st.error(
            f"No pretrained comparison model could run ({error}). Weights may be missing, "
            "corrupt, or interrupted during download. Check the model files or network "
            "access, then retry."
        )
        return
    if model_failures:
        details = "; ".join(f"{name} ({reason})" for name, reason in model_failures)
        st.warning(
            f"Unavailable pretrained models were omitted: {details}. The table and "
            "agreement below use only models that ran; restore the missing weights to "
            "compare the full pretrained set."
        )
    if not rows:
        st.error("No pretrained comparison models returned results. Check weights and retry.")
        return

    mode = "sliced (tiled) inference" if sliced else "standard full-frame inference"
    st.caption(
        f"Confidence {float(confidence_threshold):.2f} | mode: {mode} | "
        f"models: {len(rows)} | classes: {', '.join(selected_classes)}"
    )

    _render_counts(rows, selected_classes)
    built = reliability.build_reliability_report(
        detections_by_model, families_by_model, float(confidence_threshold),
        selected_classes=list(selected_classes),
    )
    _render_agreement(built)
    feature_row = features.image_features(
        raw_by_model, image_bgr, confidence=config.DEFAULT_CONFIDENCE
    )
    flag_record = _render_needs_review(feature_row)
    _record_reliability_history(
        uploaded_file,
        data,
        image_bgr,
        list(selected_classes),
        float(confidence_threshold),
        bool(sliced),
        rows,
        inference_ms_by_model,
        built,
        flag_record,
    )
