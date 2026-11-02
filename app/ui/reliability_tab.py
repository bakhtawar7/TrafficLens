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

import pandas as pd
import streamlit as st

from core import counting, detector, reliability, validation
from core.reliability import ReliabilityReport


def _run_models(image_bgr, detectors, selected_classes, threshold, sliced):
    """Detect with every model once and return display rows plus the raw results.

    Returns:
        ``(rows, detections_by_model, families_by_model)`` where ``rows`` is one
        per-model record for the counts table, and the two mappings feed
        :func:`core.reliability.build_reliability_report`. Keys are the base
        model names, so the report reads the same in standard and sliced mode.
    """
    rows: list[dict] = []
    detections_by_model: dict[str, list] = {}
    families_by_model: dict[str, str] = {}
    for model in detectors:
        runner = detector.SlicedDetector(model) if sliced else model
        raw = runner.detect(image_bgr, threshold)
        filtered = counting.filter_by_classes(raw, selected_classes)
        counts = counting.count_by_class(filtered, selected_classes)
        detections_by_model[model.name] = filtered
        families_by_model[model.name] = getattr(model, "family", "other")
        row = {"Model": model.name, "Total": len(filtered)}
        row.update({name: counts[name] for name in selected_classes})
        rows.append(row)
    return rows, detections_by_model, families_by_model


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
        rows, detections_by_model, families_by_model = _run_models(
            image_bgr, detectors, selected_classes, float(confidence_threshold), sliced
        )
    except Exception:
        st.error(
            "The comparison models could not run on this image. This can happen if a "
            "model's weights are unavailable. Please check the network and try again."
        )
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
