"""Evaluation tab: counting quality against the labelled local road scene benchmark.

Reads the artefacts produced by :func:`core.evaluation.evaluate`. If
``results/evaluation_table.csv`` does not exist yet the tab keeps the honest
Week 4 placeholder, because there is nothing to show. When it does exist, the
tab renders the per item table, the mean absolute error by condition as a Plotly
chart, and the agreement between the first and second human labelling passes.

Every number here is measured **against human ground truth** with the
development subset excluded, so it can be read next to the model-to-model
figures on the Reliability tab without conflating the two.
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from core import config

from . import planned_panel


def _read_summary() -> dict:
    """Load ``results/summary.json`` if present, otherwise an empty mapping."""
    path = config.EVALUATION_SUMMARY_JSON
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _condition_frame(condition_mae: dict) -> pd.DataFrame:
    """Flatten the summary's per condition, per model MAE into tidy chart rows."""
    rows = []
    for tag, models in condition_mae.items():
        for model, stats in models.items():
            rows.append({"Condition": tag, "Model": model, "MAE": stats.get("mae"),
                         "N": stats.get("n")})
    return pd.DataFrame(rows)


def _render_condition_chart(summary: dict) -> None:
    """Plot mean absolute error per condition and model, or note its absence."""
    condition_mae = summary.get("condition_mae") or {}
    if not condition_mae:
        st.caption("No per-condition error figures are available in the summary yet.")
        return
    frame = _condition_frame(condition_mae)
    figure = px.bar(
        frame, x="Condition", y="MAE", color="Model", barmode="group",
        title="Mean absolute error by condition (model count vs ground truth)",
    )
    figure.update_layout(yaxis_title="Mean absolute error (vehicles)", xaxis_title="Condition tag")
    st.plotly_chart(figure, width="stretch")


def _render_labeller_agreement(summary: dict) -> None:
    """Show the first versus second labelling pass agreement figures."""
    labeller = summary.get("labeller_passes") or {}
    if not labeller:
        return
    st.markdown("**Labelling pass agreement** (first pass vs second pass, human labels)")
    columns = st.columns(4)
    columns[0].metric("Items labelled twice", labeller.get("items_both_passes", 0))
    columns[1].metric("Covered mean abs diff", labeller.get("covered_mean_abs_difference", "n/a"))
    columns[2].metric("Identical fraction", labeller.get("identical_fraction", "n/a"))
    confidence_interval = labeller.get("covered_ci95")
    if confidence_interval and confidence_interval[0] is not None:
        columns[3].metric("Covered 95% CI", f"[{confidence_interval[0]}, {confidence_interval[1]}]")
    else:
        columns[3].metric("Covered 95% CI", "n/a")


def render() -> None:
    """Render the Evaluation tab, or the placeholder when no evaluation exists."""
    table_path = config.EVALUATION_TABLE_CSV
    if not table_path.exists():
        planned_panel(
            week=4,
            summary="Measure counting quality against the labelled local road scene benchmark.",
            items=[
                "Compare predicted counts to ground truth for every test set item.",
                "Report mean absolute error and error distribution by condition tag.",
                "Break down results for night, glare, dense, sparse, and distant scenes.",
                "Quantify how often local vehicle types are missed or mislabelled.",
            ],
        )
        return

    st.markdown(
        "These figures are measured **against human ground truth** with the development "
        "subset excluded. They come from `core.evaluation`; run the full benchmark and "
        "the labelling pass to populate them."
    )

    try:
        table = pd.read_csv(table_path)
    except (OSError, pd.errors.ParserError):
        st.error("The evaluation table exists but could not be read. Rerun the evaluation.")
        return

    if table.empty:
        st.info(
            "The evaluation table is present but has no rows yet. The full benchmark and "
            "a completed labelling pass are needed before any item can be scored."
        )
    else:
        st.dataframe(table, width="stretch", hide_index=True)

    summary = _read_summary()
    if summary.get("warning"):
        st.warning(summary["warning"])

    st.markdown("**Error by condition**")
    _render_condition_chart(summary)
    _render_labeller_agreement(summary)
