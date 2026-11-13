"""Browsable local history of image counts and reliability runs."""
from __future__ import annotations

import io
import json

import pandas as pd
import streamlit as st

from core import config, history


def _render_run_detail(run: dict) -> None:
    st.markdown("#### Run details")
    st.write(
        {
            "Run": run["id"],
            "Timestamp (UTC)": run["timestamp_utc"],
            "Filename": run["input_filename"],
            "Image SHA-256": run["image_sha256"],
            "Dimensions": f"{run['width']} x {run['height']}",
            "Settings SHA-256": run["settings_sha256"],
            "Mode": run["mode"],
            "Classes": ", ".join(json.loads(run["selected_classes"])),
            "Threshold": run["threshold"],
            "Rickshaw-aware mode": "On" if run["rickshaw_mode"] else "Off",
        }
    )
    results = run.get("model_results", [])
    if results:
        rows = []
        for result in results:
            rows.append(
                {
                    "Model": result["model"],
                    "Total": result["total_count"],
                    "Counts by class": json.loads(result["counts_by_class"]),
                    "Inference (ms)": result["inference_ms"],
                    "Weights SHA-256": result["weights_sha256"],
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if run.get("flag_status") is not None:
        st.write(
            {
                "Count range": (
                    None if run["count_range_min"] is None
                    else f"{run['count_range_min']}-{run['count_range_max']}"
                ),
                "Count agreement": run["agreement_count"],
                "Box agreement": run["agreement_box"],
                "Flag": run["flag_status"],
                "Flag source": run["flag_source"],
                "Predicted error": run["predicted_error"],
            }
        )


def render() -> None:
    """Render the local history browser, details, export, and deletion controls."""
    st.markdown("### Run history")
    st.caption(
        "Local history stores uploaded filenames, image hashes, settings hashes, counts, "
        "and reliability metadata. Image bytes are never stored. Results are a pilot on "
        "Dhaka imagery; reference counts are human counts with consistency reported in "
        "the evaluation documentation. The optional fine-tuned model has a separate "
        "evaluation and contamination status."
    )
    st.caption(f"Database file: `{config.HISTORY_DB.name}`")

    left, right = st.columns(2)
    input_filter = left.text_input("Filter by filename", key="history_filename_filter")
    model_filter = right.text_input("Filter by model", key="history_model_filter")
    try:
        runs = history.list_runs(input_filter=input_filter, model_filter=model_filter)
    except history.HistoryStoreError as error:
        st.warning(
            f"History is unavailable ({error}). Counting remains available; check "
            "database permissions and close other database users."
        )
        return

    if not runs:
        st.info("No matching image runs are stored yet.")
    else:
        table = pd.DataFrame(
            [
                {
                    "Timestamp (UTC)": run["timestamp_utc"],
                    "Filename": run["input_filename"],
                    "Mode": run["mode"],
                    "Rickshaw mode": "On" if run["rickshaw_mode"] else "Off",
                    "Flag": run["flag_status"] or "not recorded",
                    "Run ID": run["id"],
                }
                for run in runs
            ]
        )
        st.dataframe(table, width="stretch", hide_index=True)
        selected_id = st.selectbox(
            "View run details",
            options=[run["id"] for run in runs],
            format_func=lambda value: next(
                f"{item['timestamp_utc']} | {item['input_filename']} | {value[:8]}"
                for item in runs if item["id"] == value
            ),
        )
        if selected_id:
            try:
                selected = history.get_run(selected_id)
                if selected is not None:
                    _render_run_detail(selected)
            except history.HistoryStoreError as error:
                st.warning(f"The selected run could not be loaded: {error}")

        try:
            export_rows = history.export_rows()
            buffer = io.StringIO(newline="")
            if export_rows:
                pd.DataFrame(export_rows).to_csv(buffer, index=False)
            st.download_button(
                "Export history as CSV",
                data=buffer.getvalue().encode("utf-8"),
                file_name="trafficlens_history.csv",
                mime="text/csv",
                width="content",
            )
        except history.HistoryStoreError as error:
            st.warning(f"History export is unavailable: {error}")

        st.markdown("#### Delete history")
        delete_id = st.selectbox(
            "Select a run to delete",
            options=[run["id"] for run in runs],
            key="history_delete_id",
            format_func=lambda value: next(
                f"{item['input_filename']} | {value[:8]}"
                for item in runs if item["id"] == value
            ),
        )
        confirm_delete = st.checkbox(
            "Confirm deletion of the selected run", key="history_confirm_delete"
        )
        if st.button("Delete selected run", disabled=not confirm_delete):
            try:
                history.delete_run(delete_id)
                st.session_state["history_confirm_delete"] = False
                st.rerun()
            except history.HistoryStoreError as error:
                st.error(f"Could not delete the run: {error}")

    confirm_clear = st.checkbox(
        "Confirm that all local run history should be permanently cleared",
        key="history_confirm_clear",
    )
    if st.button("Clear history", disabled=not confirm_clear):
        try:
            history.clear_history()
            st.session_state["history_confirm_clear"] = False
            st.rerun()
        except history.HistoryStoreError as error:
            st.error(f"Could not clear history: {error}")
