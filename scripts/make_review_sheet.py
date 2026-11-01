"""Build the human verification package for a drafted test set.

Nothing here decides whether a label is correct. It makes the decision cheap to make.
Every chosen item is drawn with its ground truth boxes, grouped into contact sheets, and
listed in ``data/testset/review/REVIEW.csv`` with one row to fill in. Open a sheet, compare
the drawn boxes against the number in the caption, and record the answer in the CSV. The
companion script ``scripts/apply_review.py`` folds those answers back into ``LABELS.csv``.

Run from the project root::

    python scripts/make_review_sheet.py
    python scripts/make_review_sheet.py --per-sheet 4 --max-edge 900
"""
from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config, dataset  # noqa: E402

#: Categories a COCO detector can never report, drawn in grey so the gap is visible.
NON_DETECTABLE: frozenset[str] = frozenset({"rickshaw", "loader_rickshaw", "other"})

CAPTION_LINES = 3
TEXT_COLOR: tuple[int, int, int] = (233, 238, 245)
PANEL_COLOR: tuple[int, int, int] = (14, 26, 43)
BOX_THICKNESS = 2


def category_color(label: str) -> tuple[int, int, int]:
    """Pick a BGR colour for a ground truth category, stable across runs."""
    category, _known = dataset.map_label(label)
    if category in config.CLASS_COLORS_RGB:
        red, green, blue = config.CLASS_COLORS_RGB[category]
        return (blue, green, red)
    return (128, 128, 128)


def summary_text(counts: dict[str, int]) -> str:
    """Render the per category ground truth counts as one compact line."""
    parts = [
        f"{category[:4]} {int(counts.get(category, 0))}"
        for category in dataset.LABEL_CATEGORIES
        if int(counts.get(category, 0)) > 0
    ]
    return "  ".join(parts) if parts else "no vehicles"


def draw_annotated(
    image: np.ndarray,
    objects: Iterable[dataset.VocObject],
    counts: dict[str, int],
    caption: str,
) -> np.ndarray:
    """Draw ground truth boxes plus a caption panel onto a copy of ``image``."""
    canvas = image.copy()
    for obj in objects:
        x_min, y_min, x_max, y_max = (int(round(value)) for value in obj.xyxy)
        color = category_color(obj.label)
        cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), color, BOX_THICKNESS)
        label = dataset.map_label(obj.label)[0]
        cv2.putText(canvas, label, (x_min, max(y_min - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    width = canvas.shape[1]
    panel = np.full((CAPTION_LINES * 22 + 14, width, 3), PANEL_COLOR, dtype=np.uint8)
    lines = [caption, summary_text(counts), "grey boxes are categories a COCO detector cannot report"]
    for index, line in enumerate(lines):
        cv2.putText(panel, line[:220], (8, 22 + index * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    return np.vstack([canvas, panel])


def build_sheet(items: list[tuple[str, np.ndarray]], columns: int = 2) -> np.ndarray:
    """Tile annotated thumbnails into one contact sheet image."""
    if not items:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    thumbnails = [image for _name, image in items]
    target_width = max(thumb.shape[1] for thumb in thumbnails)
    resized = [
        cv2.resize(thumb, (target_width, int(thumb.shape[0] * target_width / thumb.shape[1])), interpolation=cv2.INTER_AREA)
        for thumb in thumbnails
    ]
    rows = math.ceil(len(resized) / columns)
    row_heights = [
        max(resized[row * columns + col].shape[0] for col in range(columns) if row * columns + col < len(resized))
        for row in range(rows)
    ]
    sheet = np.full((sum(row_heights) + 10 * (rows + 1), target_width * columns + 10 * (columns + 1), 3), 8, dtype=np.uint8)

    y = 10
    for row, height in enumerate(row_heights):
        x = 10
        for col in range(columns):
            index = row * columns + col
            if index >= len(resized):
                break
            thumb = resized[index]
            sheet[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
            x += thumb.shape[1] + 10
        y += height + 10
    return sheet


def second_pass_draw(item_ids: list[str], share: float = 0.2) -> set[str]:
    """Choose the review sample required by the labelling protocol.

    The draw is deterministic across runs, rounded up, and spread over the sorted ids so
    easy scenes cannot be picked in preference to hard ones.
    """
    if not item_ids:
        return set()
    need = max(1, math.ceil(len(item_ids) * share))
    ranked = sorted(item_ids, key=lambda item_id: (dataset.stable_seed(item_id), item_id))
    return set(ranked[:need])


def load_inputs(inventory_path: Path, selection_path: Path, sources_path: Path, labels_path: Path) -> list[dict[str, object]]:
    """Join the selection log, inventory, sources, and labels into one record per item."""
    selection = pd.read_csv(selection_path)
    inventory = pd.read_csv(inventory_path)
    sources = pd.read_csv(sources_path)
    labels = pd.read_csv(labels_path)

    boxes_by_name = dict(zip(inventory["original_filename"].astype(str), inventory["boxes_json"]))
    rows: list[dict[str, object]] = []
    chosen = selection[selection["decision"] == "selected"]
    for selection_row in chosen.to_dict("records"):
        original = str(selection_row["original_filename"])
        item_id = str(selection_row["item_id"])
        source_row = sources[sources["item_id"] == item_id]
        label_row = labels[labels["item_id"] == item_id]
        if source_row.empty or label_row.empty:
            continue
        counts = {category: int(label_row.iloc[0][category]) for category in dataset.LABEL_CATEGORIES}
        rows.append(
            {
                "item_id": item_id,
                "filename": str(source_row.iloc[0]["filename"]),
                "original_filename": original,
                "condition_tags": str(source_row.iloc[0]["condition_tags"]),
                "location_area": str(source_row.iloc[0]["location_area"]),
                "counts": counts,
                "objects": dataset.objects_from_json(boxes_by_name.get(original, "")),
            }
        )
    return rows


def review_frame(rows: list[dict[str, object]], second_pass: set[str]) -> pd.DataFrame:
    """Create the checklist the operator fills in."""
    records = []
    for row in rows:
        counts = row["counts"]
        records.append(
            {
                "item_id": row["item_id"],
                "filename": row["filename"],
                "keep": "yes",
                "tags_are_correct": "",
                "corrected_condition_tags": row["condition_tags"],
                "counts_are_correct": "",
                "privacy_ok": "",
                "second_pass_due": "yes" if row["item_id"] in second_pass else "no",
                **{category: counts[category] for category in dataset.LABEL_CATEGORIES},
                "reviewer_initials": "",
                "notes": "",
            }
        )
    columns = [
        "item_id",
        "filename",
        "keep",
        "tags_are_correct",
        "corrected_condition_tags",
        "counts_are_correct",
        "privacy_ok",
        "second_pass_due",
        *dataset.LABEL_CATEGORIES,
        "reviewer_initials",
        "notes",
    ]
    return pd.DataFrame(records, columns=columns)


def clear_directory(directory: Path) -> int:
    """Delete previously rendered images so a smaller run never leaves stale sheets."""
    removed = 0
    for path in directory.glob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".png"}:
            path.unlink()
            removed += 1
    return removed


def main(argv: list[str] | None = None) -> int:
    """Render annotated images and contact sheets, then write the review checklist."""
    parser = argparse.ArgumentParser(description="Build the review package for the drafted test set.")
    parser.add_argument("--per-sheet", type=int, default=6, help="Items per contact sheet.")
    parser.add_argument("--max-edge", type=int, default=760, help="Longest sheet thumbnail edge in pixels.")
    parser.add_argument("--columns", type=int, default=2, help="Thumbnail columns on a sheet.")
    args = parser.parse_args(argv)

    for path in (config.TESTSET_INVENTORY_CSV, config.TESTSET_SELECTION_CSV, config.SOURCES_CSV, config.LABELS_CSV):
        if not path.exists():
            print(f"ERROR: {path} is missing. Run index_dataset.py and select_testset.py first.")
            return 1

    rows = load_inputs(
        config.TESTSET_INVENTORY_CSV,
        config.TESTSET_SELECTION_CSV,
        config.SOURCES_CSV,
        config.LABELS_CSV,
    )
    if not rows:
        print("ERROR: nothing to review. The selection log and CSV files do not agree.")
        return 1

    annotated_dir = config.TESTSET_REVIEW_DIR / "annotated"
    sheets_dir = config.TESTSET_REVIEW_DIR / "sheets"
    stale = clear_directory(annotated_dir) + clear_directory(sheets_dir)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    sheet_items: list[tuple[str, np.ndarray]] = []
    full_rendered = 0
    missing = 0
    for row in rows:
        source = config.TESTSET_IMAGES_DIR / str(row["filename"])
        image = dataset.load_image(source)
        if image is None:
            missing += 1
            print(f"  WARNING: cannot read {source}")
            continue
        caption = f"{row['item_id']}  {row['location_area']}  {row['condition_tags']}"
        annotated = draw_annotated(image, row["objects"], row["counts"], caption)
        cv2.imwrite(str(annotated_dir / f"{row['item_id']}.jpg"), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        full_rendered += 1

        thumb = annotated
        scale = args.max_edge / max(thumb.shape[0], thumb.shape[1])
        if scale < 1:
            thumb = cv2.resize(thumb, (int(thumb.shape[1] * scale), int(thumb.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        sheet_items.append((str(row["item_id"]), thumb))

    per_sheet = max(1, args.per_sheet)
    sheet_count = 0
    for start in range(0, len(sheet_items), per_sheet):
        sheet_count += 1
        sheet = build_sheet(sheet_items[start : start + per_sheet], columns=max(1, args.columns))
        cv2.imwrite(str(sheets_dir / f"sheet_{sheet_count:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])

    second_pass = second_pass_draw([str(row["item_id"]) for row in rows])
    checklist = review_frame(rows, second_pass)
    checklist.to_csv(config.TESTSET_REVIEW_CHECKLIST_CSV, index=False)

    print(f"Rendered {full_rendered} annotated image(s) into {annotated_dir}")
    if stale:
        print(f"Removed {stale} previously rendered image(s) before this run.")
    if missing:
        print(f"Skipped {missing} item(s) whose image file could not be read.")
    print(f"Wrote {sheet_count} contact sheet(s) into {sheets_dir}")
    print(f"Wrote the review checklist to {config.TESTSET_REVIEW_CHECKLIST_CSV}")
    print(f"{len(second_pass)} item(s) are marked second_pass_due for the review pass.")
    print("\nNext: open the sheets, confirm each count, and fill REVIEW.csv. Then run scripts/apply_review.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
