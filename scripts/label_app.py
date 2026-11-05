"""Manual count audit app for the TrafficLens test set.

Run with ``streamlit run scripts/label_app.py``. This is a stand-alone labelling
harness for a human to count vehicles in each test-set image; it is completely
separate from the counting dashboard under ``app/`` and never imports it.

Design rules that the code below must keep honouring:

* The labeller counts blind. Only the dataset's own ground-truth annotation boxes
  are ever overlaid (from ``results/audit_overlays``). No model detections and no
  model-derived counts are shown anywhere, so nothing biases the human tally.
* Nothing is pre-filled from annotation counts or from any model. A category field
  starts blank, or at a previously saved *human* value, never a machine value.
* Writes are atomic and guarded. Every save first backs up ``LABELS.csv`` once per
  session, then rewrites the whole file through a temporary file and ``os.replace``.
  An already-saved row is never overwritten without an explicit confirmation click.

Everything that touches the file system is broken into small, side-effect-free
helpers that take their paths as arguments, so the tests can drive them against a
temporary directory without launching the Streamlit runtime.
"""
from __future__ import annotations

import datetime as _dt
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import cv2
import pandas as pd
import streamlit as st

from core import config, dataset

# ---------------------------------------------------------------------------
# Fixed column layout
# ---------------------------------------------------------------------------
#: Column order of ``LABELS.csv``. Matches the header-only scaffold on disk.
LABEL_COLUMNS: tuple[str, ...] = (
    "item_id",
    *config.LABEL_CLASS_NAMES,
    "labeller_pass",
    "notes",
)
#: Categories the labeller fills, in protocol order.
CATEGORIES: tuple[str, ...] = config.LABEL_CLASS_NAMES
#: Value written into ``tag_confirmed`` once a human has reviewed the tags.
TAG_CONFIRMED_VALUE = "yes"
#: Provenance recorded in ``tag_source`` once a human confirms or edits the tags.
MANUAL_TAG_SOURCE = "manual"
#: Script run by the "Run checker" button.
CHECK_SCRIPT: Path = config.PROJECT_ROOT / "scripts" / "check_testset.py"


# ---------------------------------------------------------------------------
# Reading helpers
# ---------------------------------------------------------------------------
def read_table(path: Path | str) -> pd.DataFrame:
    """Read a CSV keeping every cell as a plain string with no NaN coercion."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_labels(path: Path | str | None = None) -> pd.DataFrame:
    """Return ``LABELS.csv`` with the fixed columns, adding any that are missing."""
    path = Path(path) if path is not None else config.LABELS_CSV
    if not path.exists():
        return pd.DataFrame(columns=list(LABEL_COLUMNS))
    frame = read_table(path)
    for column in LABEL_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def load_sources(path: Path | str | None = None) -> pd.DataFrame:
    """Return ``SOURCES.csv`` as strings; an empty frame when the file is absent."""
    path = Path(path) if path is not None else config.SOURCES_CSV
    if not path.exists():
        return pd.DataFrame()
    return read_table(path)


def find_rows(frame: pd.DataFrame, item_id: str, labeller_pass: int) -> pd.DataFrame:
    """Rows of ``frame`` saved for ``item_id`` at the given labeller pass."""
    if "item_id" not in frame.columns:
        return frame
    mask = frame["item_id"].astype(str).str.strip() == str(item_id).strip()
    if "labeller_pass" in frame.columns:
        mask = mask & (frame["labeller_pass"].astype(str).str.strip() == str(labeller_pass))
        return frame[mask]
    return frame[mask]


def row_exists(frame: pd.DataFrame, item_id: str, labeller_pass: int) -> bool:
    """True when a count row is already saved for this item at this pass."""
    return not find_rows(frame, item_id, labeller_pass).empty


def saved_counts(frame: pd.DataFrame, item_id: str, labeller_pass: int) -> dict[str, int] | None:
    """The saved per-category human counts for an item/pass, or ``None`` if unsaved.

    Only ever reads a row the labeller wrote themselves; it never touches
    annotation counts or model output.
    """
    rows = find_rows(frame, item_id, labeller_pass)
    if rows.empty:
        return None
    row = rows.iloc[-1]
    counts: dict[str, int] = {}
    for category in CATEGORIES:
        raw = str(row.get(category, "")).strip()
        try:
            counts[category] = int(float(raw)) if raw else 0
        except ValueError:
            counts[category] = 0
    return counts


# ---------------------------------------------------------------------------
# Writing helpers (atomic + backup)
# ---------------------------------------------------------------------------
def write_atomic(frame: pd.DataFrame, path: Path | str) -> Path:
    """Write ``frame`` to ``path`` atomically via a temporary file and os.replace.

    The destination is only swapped once the full CSV has been flushed to a
    sibling temporary file, so a crash mid-write can never leave a half-written
    ``LABELS.csv``. Returns the destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False)
        os.replace(temp_name, path)
    except BaseException:
        if os.path.exists(temp_name):
            os.remove(temp_name)
        raise
    return path


def make_backup(
    labels_path: Path | str | None = None,
    backups_dir: Path | str | None = None,
    *,
    timestamp: _dt.datetime | None = None,
) -> Path | None:
    """Copy the current ``LABELS.csv`` into ``results/label_backups``.

    Returns the backup path, or ``None`` when there is no labels file to back up
    yet. The backup name is timestamped so each session's first write leaves a
    distinct, restorable snapshot.
    """
    labels_path = Path(labels_path) if labels_path is not None else config.LABELS_CSV
    backups_dir = Path(backups_dir) if backups_dir is not None else config.LABEL_BACKUPS_DIR
    if not labels_path.exists():
        return None
    moment = timestamp or _dt.datetime.now()
    stamp = moment.strftime("%Y%m%d_%H%M%S_%f")
    backups_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backups_dir / f"{labels_path.stem}_{stamp}.csv"
    backup_path.write_bytes(labels_path.read_bytes())
    return backup_path


def upsert_label(
    frame: pd.DataFrame,
    item_id: str,
    counts: dict[str, int],
    notes: str,
    labeller_pass: int,
    *,
    allow_overwrite: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Insert or update one count row, returning ``(frame, status)``.

    ``status`` is ``"created"`` for a brand new row, ``"overwrite_saved"`` when an
    existing row was replaced, or ``"needs_confirmation"`` when a row already
    exists and ``allow_overwrite`` is False (the frame is then returned unchanged
    so the caller can ask the labeller to confirm before retrying).
    """
    item_id = str(item_id).strip()
    row = {"item_id": item_id, "notes": (notes or "").strip(), "labeller_pass": str(labeller_pass)}
    for category in CATEGORIES:
        row[category] = str(int(counts.get(category, 0)))

    frame = frame.copy()
    for column in LABEL_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""

    existing = find_rows(frame, item_id, labeller_pass)
    if not existing.empty:
        if not allow_overwrite:
            return frame, "needs_confirmation"
        index = existing.index[-1]
        for column, value in row.items():
            frame.loc[index, column] = value
        return frame, "overwrite_saved"

    frame.loc[len(frame)] = {column: row.get(column, "") for column in frame.columns}
    return frame, "created"


# ---------------------------------------------------------------------------
# Pass 2 visibility
# ---------------------------------------------------------------------------
def pass2_saved(frame: pd.DataFrame, item_id: str) -> bool:
    """True once a pass 2 row has been saved for the item."""
    return row_exists(frame, item_id, 2)


def should_reveal_pass1(frame: pd.DataFrame, item_id: str) -> bool:
    """Whether the pass 1 counts may be shown during the pass 2 review.

    The second-pass procedure says re-label from scratch without looking at the
    first-pass number, so the pass 1 counts stay hidden until the pass 2 entry for
    that same item has been saved.
    """
    return row_exists(frame, item_id, 1) and row_exists(frame, item_id, 2)


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------
def item_tags(tags_raw: object) -> list[str]:
    """Valid condition tags for a SOURCES ``condition_tags`` cell, in canonical order."""
    valid, _invalid = dataset.split_tags(tags_raw)
    return valid


def order_rank(tags: list[str]) -> tuple[int, str]:
    """Grouping sort key: by primary tag (canonical order) then item id.

    Tagless items sort into a final group so evaluation images sharing a condition
    sit together while untagged ones do not scatter through the tagged batches.
    """
    primary = tags[0] if tags else None
    if primary is None or primary not in dataset.CONDITION_TAGS:
        return (len(dataset.CONDITION_TAGS), "")
    return (dataset.CONDITION_TAGS.index(primary), primary)


def ordered_items(
    sources: pd.DataFrame, plan: dict | None
) -> list[dict]:
    """Build the labelling order: development images first, then evaluation grouped by tag.

    Each entry is ``{"item_id", "filename", "tags", "group", "is_dev", "in_second_pass"}``
    where ``group`` is ``"development"`` or ``"evaluation"``. ``plan`` supplies the
    development subset and the second-pass sample; both default to empty when absent.
    """
    dev_ids: set[str] = set()
    second_ids: set[str] = set()
    if plan:
        dev_ids = {dataset.format_item_id(dataset.parse_item_number(x)) for x in plan.get("development_subset", [])}
        second_ids = {dataset.format_item_id(dataset.parse_item_number(x)) for x in plan.get("second_pass", [])}
        dev_ids.discard(None)
        second_ids.discard(None)

    entries: list[dict] = []
    seen: set[str] = set()
    for _, row in sources.iterrows():
        item_id = dataset.normalise_item_id(row.get("item_id", ""))
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        tags = item_tags(row.get("condition_tags", ""))
        is_dev = item_id in dev_ids
        entries.append(
            {
                "item_id": item_id,
                "filename": str(row.get("filename", "")).strip(),
                "tags": tags,
                "group": "development" if is_dev else "evaluation",
                "is_dev": is_dev,
                "in_second_pass": item_id in second_ids,
            }
        )

    development = sorted((e for e in entries if e["is_dev"]), key=lambda e: e["item_id"])
    evaluation = sorted(
        (e for e in entries if not e["is_dev"]),
        key=lambda e: (order_rank(e["tags"]), e["item_id"]),
    )
    return development + evaluation


def filter_second_pass(items: Iterable[dict]) -> list[dict]:
    """The subset of ``items`` marked for the second pass, preserving order."""
    return [item for item in items if item["in_second_pass"]]


def unlabelled_items(items: Iterable[dict], labels: pd.DataFrame, labeller_pass: int) -> list[str]:
    """Item ids in ``items`` that have no saved count row at ``labeller_pass``."""
    return [item["item_id"] for item in items if not row_exists(labels, item["item_id"], labeller_pass)]


# ---------------------------------------------------------------------------
# Tag confirmation
# ---------------------------------------------------------------------------
def update_item_tags(
    sources: pd.DataFrame,
    item_id: str,
    tags: Iterable[str],
    *,
    tag_confirmed: str = TAG_CONFIRMED_VALUE,
) -> pd.DataFrame:
    """Set a human-reviewed tag set on one SOURCES row and mark it confirmed.

    ``condition_tags`` is normalised through the canonical merge, ``tag_source``
    becomes ``"manual"``, and ``tag_confirmed`` is written so the checker stops
    counting the row as an unconfirmed auto proposal. All other columns are kept.
    """
    item_id = str(item_id).strip()
    frame = sources.copy()
    for column in ("condition_tags", "tag_source", "tag_confirmed"):
        if column not in frame.columns:
            frame[column] = ""
    mask = frame["item_id"].astype(str).str.strip() == item_id
    merged = dataset.merge_tags(tags)
    frame.loc[mask, "condition_tags"] = merged
    frame.loc[mask, "tag_source"] = MANUAL_TAG_SOURCE
    frame.loc[mask, "tag_confirmed"] = tag_confirmed
    return frame


def current_tags(sources: pd.DataFrame, item_id: str) -> list[str]:
    """The condition tags currently recorded for ``item_id``."""
    if "item_id" not in sources.columns:
        return []
    rows = sources[sources["item_id"].astype(str).str.strip() == str(item_id).strip()]
    if rows.empty:
        return []
    return item_tags(rows.iloc[-1].get("condition_tags", ""))


# ---------------------------------------------------------------------------
# Image loading (ground-truth overlay only)
# ---------------------------------------------------------------------------
def plain_image_path(item: dict, images_dir: Path | str | None = None) -> Path:
    images_dir = Path(images_dir) if images_dir is not None else config.TESTSET_IMAGES_DIR
    return images_dir / item["filename"]


def overlay_image_path(item: dict, overlays_dir: Path | str | None = None) -> Path:
    overlays_dir = Path(overlays_dir) if overlays_dir is not None else config.AUDIT_OVERLAYS_DIR
    return overlays_dir / f"{item['item_id']}.jpg"


def load_display_image(
    item: dict, *, show_boxes: bool, images_dir: Path | str | None = None,
    overlays_dir: Path | str | None = None,
) -> tuple["np.ndarray | None", str]:
    """Return ``(rgb_array, note)`` for the current item.

    With ``show_boxes`` the pre-rendered ground-truth overlay is used; otherwise the
    plain copied image is shown. Neither carries a model detection. A missing file
    yields ``(None, message)`` so the caller can warn instead of crashing.
    """
    if show_boxes:
        path = overlay_image_path(item, overlays_dir)
        if not path.exists():
            path = plain_image_path(item, images_dir)
    else:
        path = plain_image_path(item, images_dir)
    if not path.exists():
        return None, f"Image file not found: {path.name}"
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        return None, f"Could not decode image: {path.name}"
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return rgb, ("ground-truth boxes" if show_boxes else "no boxes")


def zoom_image(rgb: "np.ndarray", factor: float) -> "np.ndarray":
    """Scale an RGB array by ``factor`` for the zoom control (1.0 leaves it as is)."""
    if factor is None or abs(factor - 1.0) < 1e-6:
        return rgb
    height, width = rgb.shape[:2]
    new_size = (max(1, int(width * factor)), max(1, int(height * factor)))
    return cv2.resize(rgb, new_size, interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Checker integration
# ---------------------------------------------------------------------------
def run_checker(cwd: Path | str | None = None) -> tuple[int, str]:
    """Run ``scripts/check_testset.py`` and return ``(return_code, combined_output)``."""
    cwd = Path(cwd) if cwd is not None else config.PROJECT_ROOT
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=str(cwd),
    )
    output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
    return result.returncode, output.strip()


# ---------------------------------------------------------------------------
# Streamlit application
# ---------------------------------------------------------------------------
def _session_key(item_id: str, pass_no: int, field: str) -> str:
    return f"label::{item_id}::p{pass_no}::{field}"


def render() -> None:
    """Render the whole labelling screen. Streamlit re-runs this on every interaction."""
    st.set_page_config(page_title="TrafficLens manual count audit", layout="wide")
    st.title("TrafficLens manual count audit")
    st.caption("Count vehicles blind. Only dataset ground-truth boxes are shown, never model detections.")

    sources = load_sources()
    plan = _load_plan()
    full_order = ordered_items(sources, plan)
    if not full_order:
        st.error("No test-set items found in SOURCES.csv. Run the build first.")
        return

    # -- sidebar -------------------------------------------------------------
    with st.sidebar:
        st.header("Audit controls")
        mode = st.radio("Labelling pass", ("Pass 1 (first count)", "Pass 2 (second-pass review)"), index=0)
        pass_no = 1 if mode.startswith("Pass 1") else 2
        show_boxes = st.checkbox("Overlay dataset annotation boxes", value=True)
        zoom = st.slider("Zoom", 1.0, 3.0, 1.0, 0.1, help="Enlarge the displayed image for fine counting.")

    # Choose the working list. Pass 2 shows only second-pass items and hides pass 1
    # counts until the pass 2 entry for that item has been saved.
    if pass_no == 2:
        items = filter_second_pass(full_order)
        if not items:
            st.warning("No items are marked for the second pass in the current plan.")
            return
    else:
        items = full_order

    if "order_index" not in st.session_state:
        st.session_state.order_index = 0
    st.session_state.order_index = max(0, min(st.session_state.order_index, len(items) - 1))

    labels = load_labels()

    # -- navigation ----------------------------------------------------------
    nav_left, nav_center, nav_right = st.columns([1, 2, 1])
    if nav_left.button("Previous", disabled=st.session_state.order_index == 0):
        st.session_state.order_index -= 1
        st.rerun()
    current = items[st.session_state.order_index]
    nav_center.markdown(
        f"**{current['item_id']}** &nbsp; ({current['group']}"
        + (", second pass" if current["in_second_pass"] else "")
        + ")",
    )
    if nav_right.button("Next", disabled=st.session_state.order_index >= len(items) - 1):
        st.session_state.order_index += 1
        st.rerun()

    position = st.session_state.order_index + 1
    st.progress(position / len(items), text=f"Item {position} of {len(items)}")

    # -- image ---------------------------------------------------------------
    rgb, note = load_display_image(current, show_boxes=show_boxes)
    if rgb is None:
        st.error(note)
    else:
        st.image(zoom_image(rgb, zoom), channels="RGB", use_container_width=True)
        st.caption(f"Display: {note}.")

    # -- count inputs --------------------------------------------------------
    st.subheader(f"Counts ({'pass ' + str(pass_no)})")
    saved = saved_counts(labels, current["item_id"], pass_no)
    columns = st.columns(len(CATEGORIES))
    values: dict[str, int] = {}
    for column, category in zip(columns, CATEGORIES):
        key = _session_key(current["item_id"], pass_no, category)
        default = saved[category] if saved else 0
        values[category] = column.number_input(
            category.replace("_", " ").capitalize(), min_value=0, max_value=99999,
            step=1, value=int(default), key=key,
        )
    notes_key = _session_key(current["item_id"], pass_no, "notes")
    notes_default = ""
    if saved is not None:
        existing_rows = find_rows(labels, current["item_id"], pass_no)
        if not existing_rows.empty:
            notes_default = str(existing_rows.iloc[-1].get("notes", ""))
    notes = st.text_input("Notes", value=notes_default, key=notes_key)

    # -- save ----------------------------------------------------------------
    _render_save(current, values, notes, pass_no, labels)

    # -- tag confirmation (only after a count row exists for this pass) ------
    if row_exists(labels, current["item_id"], pass_no):
        _render_tag_control(current)

    # -- pass 2 comparison reveal -------------------------------------------
    if pass_no == 2:
        _render_pass1_reveal(current, labels)

    # -- checker + unlabelled list ------------------------------------------
    st.divider()
    _render_checker(pass_no)


def _render_save(current: dict, values: dict[str, int], notes: str, pass_no: int, labels: pd.DataFrame) -> None:
    """Persist the current counts, backing up first and confirming any overwrite."""
    item_id = current["item_id"]
    exists = row_exists(labels, item_id, pass_no)

    if exists:
        confirm_key = _session_key(item_id, pass_no, "confirm_overwrite")
        st.warning("A count row is already saved for this item at this pass.")
        confirmed = st.checkbox("Yes, overwrite the saved row", key=confirm_key)
        if st.button("Save counts", type="primary"):
            if not confirmed:
                st.error("Confirm the overwrite checkbox before saving over a saved row.")
                return
            _do_save(item_id, values, notes, pass_no)
    else:
        if st.button("Save counts", type="primary"):
            _do_save(item_id, values, notes, pass_no)


def _do_save(item_id: str, values: dict[str, int], notes: str, pass_no: int) -> None:
    """Back up once this session, then atomically write the updated LABELS.csv."""
    if not st.session_state.get("_backup_done"):
        backup = make_backup()
        st.session_state["_backup_done"] = True
        if backup is not None:
            st.session_state["_last_backup"] = str(backup)
    labels = load_labels()
    frame, status = upsert_label(labels, item_id, values, notes, pass_no, allow_overwrite=True)
    ordered = frame[list(LABEL_COLUMNS)] if set(LABEL_COLUMNS).issubset(frame.columns) else frame
    write_atomic(ordered, config.LABELS_CSV)
    if status == "overwrite_saved":
        st.success(f"Updated saved counts for {item_id} (pass {pass_no}).")
    else:
        st.success(f"Saved counts for {item_id} (pass {pass_no}).")
    # Any newly saved pass 2 row reveals its pass 1 counterpart, so drop the
    # overwrite confirmation for this item/pass to keep the box un-ticked next view.
    st.session_state.pop(_session_key(item_id, pass_no, "confirm_overwrite"), None)


def _render_tag_control(current: dict) -> None:
    """Show condition tags with a confirm / edit control once counts are saved."""
    item_id = current["item_id"]
    sources = load_sources()
    tags = current_tags(sources, item_id)
    st.subheader("Condition tags")
    st.caption("Confirm the tags the build assigned, or edit them. Saving writes tag_confirmed and sets tag_source to manual.")
    selection = st.multiselect(
        "Condition tags", options=list(dataset.CONDITION_TAGS), default=tags,
        key=_session_key(item_id, 0, "tags"),
    )
    confirmed_cell = ""
    if "tag_confirmed" in sources.columns:
        rows = sources[sources["item_id"].astype(str).str.strip() == item_id]
        if not rows.empty:
            confirmed_cell = str(rows.iloc[-1].get("tag_confirmed", "")).strip()
    if confirmed_cell:
        st.info("These tags are already confirmed.")
    control_left, control_right = st.columns(2)
    if control_left.button("Confirm tags"):
        _write_tags(item_id, tags)
    if control_right.button("Save edited tags", disabled=sorted(selection) == sorted(tags)):
        _write_tags(item_id, selection)


def _write_tags(item_id: str, tags: list[str]) -> None:
    """Atomically persist a tag confirmation / edit into SOURCES.csv."""
    sources = load_sources()
    updated = update_item_tags(sources, item_id, tags)
    write_atomic(updated, config.SOURCES_CSV)
    st.success(f"Tags for {item_id} saved as manual (confirmed).")


def _render_pass1_reveal(current: dict, labels: pd.DataFrame) -> None:
    """During pass 2, reveal the pass 1 counts only once the pass 2 row is saved."""
    item_id = current["item_id"]
    if not should_reveal_pass1(labels, item_id):
        st.info("Pass 1 counts are hidden until you save this item's pass 2 entry.")
        return
    pass1 = saved_counts(labels, item_id, 1)
    if pass1:
        st.markdown("**Pass 1 counts (for comparison)**")
        st.dataframe(pd.DataFrame([{"category": c, "pass1": pass1[c]} for c in CATEGORIES]),
                     hide_index=True, width="stretch")


def _render_checker(pass_no: int) -> None:
    """Run the test-set checker on demand and list items still unlabelled."""
    if st.button("Run checker (check_testset.py)"):
        code, output = run_checker()
        if code == 0:
            st.success("check_testset.py exited 0.")
        else:
            st.error(f"check_testset.py exited {code}.")
        st.code(output or "(no output)")

    sources = load_sources()
    plan = _load_plan()
    items = ordered_items(sources, plan)
    working = filter_second_pass(items) if pass_no == 2 else items
    labels = load_labels()
    pending = unlabelled_items(working, labels, pass_no)
    st.markdown(f"**Items still unlabelled (pass {pass_no}): {len(pending)}**")
    if pending:
        st.write(", ".join(pending))
    else:
        st.success("Every item in this pass has a saved count row.")


def _load_plan() -> dict | None:
    """Load the test-set plan (development subset + second pass) if present."""
    path = config.TESTSET_PLAN_JSON
    if not path.exists():
        return None
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


if __name__ == "__main__":
    render()
