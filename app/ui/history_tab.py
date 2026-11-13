"""Browsable local history of image counts and reliability runs."""
from __future__ import annotations

import io
import hashlib
import json

import cv2
import pandas as pd
import streamlit as st

from core import config, history, validation


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


def _render_history_browser() -> None:
    st.markdown("#### Stored runs")
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
            key="history_view_run_id",
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


def _render_model_comparison(run: dict, correction: dict) -> None:
    comparison = history.correction_comparison(run, correction)
    st.markdown("**Latest user correction**")
    st.write({
        "Reviewer": correction["reviewer"],
        "Saved (UTC)": correction["timestamp_utc"],
        "Corrected counts": correction["corrected_counts"],
        "Note": correction["note"],
    })
    st.caption(
        "Signed error is model count minus the user-reported corrected total for the "
        "run's selected categories. Range status compares that corrected total with the "
        "stored model-count range."
    )
    st.dataframe(pd.DataFrame(comparison["models"]), width="stretch", hide_index=True)


def _render_correction_summary() -> None:
    try:
        aggregates = history.correction_aggregates()
        st.markdown("#### Correction comparison summary")
        st.caption(
            "Small sample, descriptive only. Mean absolute pre-review model-count "
            "error is calculated against the latest user-reported corrections; it is "
            "not a benchmark or model update."
        )
        if any(values["reviewed_runs"] for values in aggregates.values()):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Flag group": group,
                            **values,
                        }
                        for group, values in aggregates.items()
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("No reviewed runs yet; flagged versus unflagged error is unavailable.")
        correction_rows = history.correction_export_rows()
        if correction_rows:
            output = io.StringIO(newline="")
            pd.DataFrame(correction_rows).to_csv(output, index=False)
            st.download_button(
                "Export corrections as CSV",
                data=output.getvalue().encode("utf-8"),
                file_name="trafficlens_corrections.csv",
                mime="text/csv",
                key="export_corrections_csv",
            )
    except history.HistoryStoreError as error:
        st.warning(f"Correction summary or export is unavailable: {error}")


def _render_review_queue() -> None:
    st.markdown("#### Blind review queue")
    st.caption(
        "Queue results are a pilot on Dhaka imagery. References are human counts with "
        "mean absolute pass difference 1.62 vehicles per image (95% scene interval "
        "0.67–2.46; 13 paired images, 7 scenes). The optional fine-tuned model remains "
        "separate, with its own evaluation and contamination exclusions."
    )
    st.info(
        "Corrections are user-reported counts. They are never written to LABELS.csv and "
        "never used to change a model or setting. Stored filenames are visible to the "
        "local user; images must be uploaded again for review."
    )
    _render_correction_summary()
    filter_col, status_col = st.columns(2)
    filename_filter = filter_col.text_input(
        "Filter review queue by filename", key="review_filename_filter"
    )
    queue_filter = status_col.selectbox(
        "Review status",
        ["All", "Unreviewed", "Reviewed"],
        key="review_status_filter",
    )
    try:
        queue = history.review_queue(
            filename_filter=filename_filter, queue_filter=queue_filter
        )
    except history.HistoryStoreError as error:
        st.warning(f"The review queue is unavailable: {error}")
        return

    st.caption(
        "Order: flagged runs first, then others oldest first. When a flag is not "
        "validated, ordering is by model disagreement (count-range width)."
    )
    safe_rows = history.blind_review_rows(queue)
    if safe_rows:
        st.dataframe(pd.DataFrame(safe_rows), width="stretch", hide_index=True)
    else:
        st.info("No runs match this review filter.")
        return

    labels = {
        run["id"]: (
            f"{run['timestamp_utc']} | {run['input_filename']} | "
            f"{run['id'][:8]}"
        )
        for run in queue
    }
    selected_id = st.selectbox(
        "Select a run to review",
        [run["id"] for run in queue],
        format_func=lambda value: labels[value],
        key="review_selected_run",
    )
    selected = next(run for run in queue if run["id"] == selected_id)
    st.caption(
        f"Selected file: {selected['input_filename']}. The database does not retain "
        "image bytes; upload the matching image to count it."
    )
    image_file = st.file_uploader(
        "Upload the matching image for blind review",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"review_image_{selected_id}",
        help="The uploaded image is used in memory for review and is not saved to history.",
    )
    if image_file is None:
        st.info("Upload the original image with the same SHA-256 before entering a correction.")
        return
    image_bytes = image_file.getvalue()
    if hashlib.sha256(image_bytes).hexdigest() != selected["image_sha256"]:
        st.error(
            "This image does not match the selected run's SHA-256. Choose the exact "
            "original file so the correction is attached to the right run."
        )
        return
    try:
        validation.validate_image_input(image_file.name, image_bytes)
        image_bgr = validation.decode_image(image_bytes)
    except validation.TrafficLensInputError as error:
        st.error(f"The review image could not be decoded ({error}). Choose a supported image.")
        return
    st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Blind review image")

    draft_key = f"review_draft_{selected_id}"
    with st.form(f"review_counts_{selected_id}"):
        reviewer = st.text_input("Reviewer", value="anonymous")
        corrected_counts = {
            name: int(st.number_input(
                name.replace("_", " ").capitalize(),
                min_value=0,
                value=0,
                step=1,
                key=f"review_{selected_id}_{name}",
            ))
            for name in config.LABEL_CLASS_NAMES
        }
        note = st.text_input("Review note (optional)")
        reveal = st.form_submit_button("Reveal model results for comparison")
    if reveal:
        st.session_state[draft_key] = {
            "reviewer": reviewer,
            "corrected_counts": corrected_counts,
            "note": note,
        }
        st.session_state[f"review_revealed_{selected_id}"] = True

    draft = st.session_state.get(draft_key)
    revealed = st.session_state.get(f"review_revealed_{selected_id}", False)
    if revealed and draft:
        try:
            run = history.get_run(selected_id)
        except history.HistoryStoreError as error:
            st.warning(f"Model results could not be revealed: {error}")
            return
        if run is None:
            st.warning("The selected run no longer exists.")
            return
        draft_correction = {
            **draft,
            "timestamp_utc": "unsaved correction",
        }
        _render_model_comparison(run, draft_correction)
        if st.button("Save correction (append-only)", key=f"save_review_{selected_id}"):
            try:
                history.save_correction(
                    selected_id,
                    draft["corrected_counts"],
                    draft["reviewer"],
                    draft["note"],
                )
                st.session_state[f"review_saved_{selected_id}"] = True
                st.success(
                    "Correction appended to local history. LABELS.csv and all models "
                    "and settings remain unchanged."
                )
            except (history.HistoryStoreError, ValueError) as error:
                st.error(f"The correction was not saved: {error}")
        if st.session_state.get(f"review_saved_{selected_id}"):
            latest_run = history.get_run(selected_id)
            if latest_run and latest_run["corrections"]:
                latest = latest_run["corrections"][-1]
                latest["corrected_counts"] = json.loads(latest["corrected_counts"])
                _render_model_comparison(latest_run, latest)

def render() -> None:
    """Render run history and the blind human review loop."""
    st.markdown("### History and review")
    st.caption(
        "Results are a pilot on Dhaka imagery. Reference counts are human counts; "
        "consistency is summarized as a mean absolute pass difference of 1.62 vehicles "
        "(95% scene interval 0.67–2.46; 13 paired images, 7 scenes). The optional "
        "fine-tuned model is separate, with its own evaluation and contamination status."
    )
    st.caption(
        "Local history stores uploaded filenames, image hashes, settings hashes, counts, "
        "and reliability metadata. Image bytes are never stored."
    )
    st.caption(f"Database file: `{config.HISTORY_DB.name}`")
    browse_tab, review_tab = st.tabs(["Run history", "Human review"])
    with browse_tab:
        _render_history_browser()
    with review_tab:
        _render_review_queue()
