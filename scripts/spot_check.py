"""Render the eight items where the draft counts disagree most with the model counts.

This is a diagnostic picture, nothing more. Each output file shows one test set image with
the dataset's own annotation boxes in a single colour, each model's detections in its own
colour, and the draft count plus every model count printed in a band above the image. The
selection of eight is arithmetic: the largest absolute gap between the draft covered total
and the median of the three standard model counts, taken from the archived draft labels and
the existing detection cache. No detector is run, no threshold is changed, and no setting is
chosen from these images. They exist so a person can look at them and decide whether the
label, the models, or the image selection is at fault.

Run from the project root with:

    python scripts/spot_check.py --source "C:/path/to/archive"

``--source`` may be omitted when ``results/curation_log.csv`` records the archive folder the
operator supplied; that path is then recovered from the log.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config, dataset, evaluation as ev  # noqa: E402  (path setup first)

#: How many disagreements to render.
TOP_N: int = 8
#: Where the pictures go, one per item.
SPOT_CHECK_DIR: Path = config.RESULTS_DIR / "spot_check"
#: Archived draft label copy. Used when it exists so the picture shows the unverified draft
#: that is under suspicion; falls back to the live LABELS.csv when the copy is absent.
DRAFT_LABELS_CSV: Path = config.RESULTS_DIR / "draft_auto_labels.csv"
#: Image extensions searched for in the archive when matching a copied file back to its XML.
_IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})

#: Dataset annotation boxes: one colour for every category, thick line, no offset.
DATASET_COLOR_BGR: tuple[int, int, int] = (255, 0, 255)
#: Model detection colours, keyed by registry key, thin lines offset so overlaps stay visible.
MODEL_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "yolov8n": (60, 220, 60),
    "yolov8s": (40, 140, 255),
    "rtdetr-l": (235, 80, 60),
}
#: The same colours named in words for the printed legend.
MODEL_COLOR_NAMES: dict[str, str] = {"yolov8n": "green", "yolov8s": "orange", "rtdetr-l": "blue"}
_DEFAULT_MODEL_COLOR_BGR: tuple[int, int, int] = (200, 200, 200)

#: Category initials printed beside a dataset box.
_CATEGORY_INITIALS: dict[str, str] = {
    "car": "c", "bus": "B", "truck": "t", "motorcycle": "m",
    "rickshaw": "r", "loader_rickshaw": "L", "other": "o",
}


def _to_int(value: object) -> int:
    """Read a count cell as an int, treating anything unreadable as zero."""
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def read_draft_counts(path: Path) -> dict[int, int]:
    """Map item number to the draft covered vehicle total, skipping archive comment lines."""
    counts: dict[int, int] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    for row in csv.DictReader(lines):
        number = dataset.parse_item_number(row.get("item_id", ""))
        if number is None:
            continue
        per_class = {name: _to_int(row.get(name)) for name in ev.COVERED_CLASSES}
        counts[number] = sum(per_class.values())
    return counts


def model_counts(
    cache: dict[tuple[int, str, str], list], item: int
) -> dict[str, int]:
    """Covered-class detection count per registry model for one item in standard mode."""
    return {
        key: len(cache.get((item, key, "standard"), []))
        for key in config.MODEL_REGISTRY
    }


def all_gaps(draft: dict[int, int],
             cache: dict[tuple[int, str, str], list]) -> list[dict[str, object]]:
    """Compute every item's draft versus median gap, largest gap first, id as tie break.

    Items with no draft row, or with no cached detections, are skipped: there is nothing to
    compare, and inventing a gap of zero would bury the real ones.
    """
    scored: list[dict[str, object]] = []
    for item in sorted(set(draft) | {key[0] for key in cache}):
        counts = model_counts(cache, item)
        if item not in draft or not counts or all(
            (item, key, "standard") not in cache for key in config.MODEL_REGISTRY
        ):
            continue
        median = round(statistics.median(counts.values()))
        scored.append({
            "item": item,
            "draft": draft[item],
            "median": median,
            "gap": abs(draft[item] - median),
            "counts": counts,
        })
    scored.sort(key=lambda row: (-row["gap"], row["item"]))
    return scored


def rank_disagreements(draft: dict[int, int], cache: dict[tuple[int, str, str], list],
                      top: int = TOP_N) -> list[dict[str, object]]:
    """Return the items whose draft total disagrees most with the median model total."""
    return all_gaps(draft, cache)[:top]


def archive_root_from_log(log_path: Path = config.CURATION_LOG_CSV) -> Path | None:
    """Recover the archive folder the operator supplied from every logged candidate path.

    The log records absolute paths across all splits, so their common ancestor is the folder
    that was passed to ``--source`` when the test set was built. Only the rejected and
    screened rows are logged, which is why the selected images themselves are not found by
    name here and have to be matched by content.
    """
    if not Path(log_path).exists():
        return None
    paths: list[str] = []
    with Path(log_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = (row.get("image_path") or "").strip()
            if value:
                paths.append(value)
    if not paths:
        return None
    return Path(os.path.commonpath(paths))


def index_by_size(root: Path) -> dict[int, list[Path]]:
    """Group every archive image by byte size so a copy can be matched cheaply."""
    index: dict[int, list[Path]] = {}
    for directory, _dirs, files in os.walk(root):
        for name in files:
            if os.path.splitext(name)[1].lower() not in _IMAGE_SUFFIXES:
                continue
            path = Path(directory) / name
            try:
                size = path.stat().st_size
            except OSError:
                continue
            index.setdefault(size, []).append(path)
    return index


def sha256(path: Path) -> str:
    """Digest of a file, used to confirm a size match is the same image."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_archive_source(image_path: Path, root: Path, index: dict[int, list[Path]]) -> Path | None:
    """Locate the archive file this test set copy was taken from, by size then digest.

    ``scripts/build_testset.py`` copies selected images unchanged, so the bytes still match
    and the sibling XML still holds the annotation that produced the dataset counts.
    """
    target = Path(image_path)
    candidates = index.get(target.stat().st_size, [])
    if not candidates:
        return None
    wanted = sha256(target)
    for candidate in candidates:
        if sha256(candidate) == wanted:
            return candidate
    return None


def _draw_band(canvas: np.ndarray, lines: list[str]) -> np.ndarray:
    """Return a new image with a dark text band of one line per message above the picture.

    Font size scales with the image width so the numbers stay readable on a 4000 pixel
    frame as well as on a 1280 pixel one, and each line is cut to what actually fits.
    """
    width = canvas.shape[1]
    scale = max(0.55, round(width / 1280.0 * 0.62, 2))
    per_char = 17.7 * scale
    line_height = int(30 * scale)
    limit = max(1, int((width - 16) / per_char))
    rows: list[str] = []
    for text in lines:
        # Wrap rather than truncate: a cut off number is worse than a second line.
        while text:
            rows.append(text[:limit])
            text = text[limit:].lstrip()
    band = np.full((line_height * len(rows) + int(16 * scale), width, 3), 28, np.uint8)
    for offset, text in enumerate(rows):
        cv2.putText(band, text, (int(8 * scale), int(line_height * (offset + 0.75))),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 235, 235),
                    max(1, int(scale * 1.6)), cv2.LINE_AA)
    return np.vstack([band, canvas])


def render(image: np.ndarray, objects: list[tuple[float, float, float, float, str]],
          per_model: list[tuple[str, list[tuple[float, float, float, float]]]],
          lines: list[str]) -> np.ndarray:
    """Draw dataset boxes in one colour and each model's boxes in another."""
    canvas = image.copy()
    box_scale = max(1, int(round(canvas.shape[1] / 1280.0)))
    for x_min, y_min, x_max, y_max, category in objects:
        cv2.rectangle(canvas, (int(x_min), int(y_min)), (int(x_max), int(y_max)),
                      DATASET_COLOR_BGR, 3 * box_scale)
        cv2.putText(canvas, _CATEGORY_INITIALS.get(category, "?"),
                    (int(x_min), max(int(y_min) - 4, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * box_scale, DATASET_COLOR_BGR,
                    box_scale, cv2.LINE_AA)
    for position, (model, boxes) in enumerate(per_model):
        color = MODEL_COLORS_BGR.get(model, _DEFAULT_MODEL_COLOR_BGR)
        inset = 2 * box_scale * (position + 1)
        for x_min, y_min, x_max, y_max in boxes:
            cv2.rectangle(canvas, (int(x_min) + inset, int(y_min) + inset),
                          (int(x_max) - inset, int(y_max) - inset), color, box_scale)
    return _draw_band(canvas, lines)


def _testset_image(item: int) -> Path | None:
    """Return the test set image file for an item number, or None if it is not on disk."""
    directory = config.TESTSET_IMAGES_DIR
    if not directory.exists():
        return None
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            if dataset.parse_item_number(path.name.split("_")[0]) == item:
                return path
    return None


def main(argv: list[str] | None = None) -> int:
    """Write the spot check pictures and print the numbers they show."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=None,
                        help="archive folder holding the original images and their XML")
    parser.add_argument("--draft", default=str(DRAFT_LABELS_CSV),
                        help="draft label CSV to compare against (default the archive copy)")
    parser.add_argument("--labels", default=str(config.LABELS_CSV),
                        help="fallback label CSV when the draft archive is missing")
    parser.add_argument("--detections", default=str(config.RAW_DETECTIONS_JSONL),
                        help="cached raw detections JSONL")
    parser.add_argument("--confidence", type=float, default=config.DEFAULT_CONFIDENCE,
                        help="threshold applied to the cache, matching the evaluator")
    parser.add_argument("--out", default=str(SPOT_CHECK_DIR), help="folder for the pictures")
    args = parser.parse_args(argv)

    draft_path = Path(args.draft)
    if not draft_path.exists():
        draft_path = Path(args.labels)
    if not draft_path.exists():
        print(f"ERROR: no draft label file at {args.draft} or {args.labels}")
        return 1
    cache_path = Path(args.detections)
    if not cache_path.exists():
        print(f"ERROR: detection cache missing: {cache_path}")
        return 1

    root = Path(args.source) if args.source else archive_root_from_log()
    if root is None or not root.exists():
        print("ERROR: archive folder not found; pass --source with the folder the images "
              "were selected from so the annotation XML can be read.")
        return 1

    draft = read_draft_counts(draft_path)
    cache = ev.load_detection_cache(cache_path, args.confidence)
    conditions = ev.load_conditions(config.SOURCES_CSV) if config.SOURCES_CSV.exists() else {}
    ranked = rank_disagreements(draft, cache)
    if not ranked:
        print("Nothing to render: no item has both a draft count and cached detections.")
        return 0

    print(f"Archive: {root} (pass --source to override)")
    print(f"Rendering {len(ranked)} of {len(draft)} draft rows, largest gap first.")
    cutoff = ranked[-1]
    tied = [
        row for row in all_gaps(draft, cache)[TOP_N:]
        if row["gap"] == cutoff["gap"]
    ]
    if tied:
        print(f"Not rendered, same gap of {cutoff['gap']}: "
              f"{', '.join(dataset.format_item_id(int(r['item'])) for r in tied)}")
    index = index_by_size(root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0

    for entry in ranked:
        item = int(entry["item"])
        item_id = dataset.format_item_id(item)
        image_path = _testset_image(item)
        image = dataset.load_image(image_path) if image_path else None
        if image is None:
            print(f"  {item_id}: SKIP, no test set image on disk")
            failures += 1
            continue

        source = find_archive_source(image_path, root, index)
        objects: list[tuple[float, float, float, float, str]] = []
        archive_covered = 0
        if source is not None:
            annotation, _ = dataset.parse_voc_annotation(Path(source).with_suffix(".xml"))
            kept = dataset.surviving_objects(annotation.objects if annotation else ())
            for obj in kept:
                category, _ = dataset.map_label(obj.label)
                objects.append((*obj.xyxy, category))
            archive_covered = sum(
                1 for *_unused, category in objects if category in ev.COVERED_CLASSES
            )

        counts: dict[str, int] = entry["counts"]  # type: ignore[assignment]
        per_model = [
            (key, [detection.xyxy for detection in cache.get((item, key, "standard"), [])])
            for key in config.MODEL_REGISTRY
        ]
        legend = "  ".join(
            f"{config.MODEL_REGISTRY[key].display_name}={counts[key]}"
            for key in config.MODEL_REGISTRY if key in counts
        )
        colour_key = "  ".join(
            f"{config.MODEL_REGISTRY[key].display_name}={MODEL_COLOR_NAMES.get(key, 'grey')}"
            for key in config.MODEL_REGISTRY
        )
        lines = [
            f"{item_id}  draft={entry['draft']}  median={entry['median']}  gap={entry['gap']}",
            f"{legend}  annotation_covered={archive_covered}  annotation_boxes={len(objects)}",
            f"colours: dataset=magenta (c b t m r L o)  {colour_key}",
            f"condition={conditions.get(item) or 'none recorded'}  "
            f"rickshaw_and_loader_boxes={sum(1 for *_xy, c in objects if c not in ev.COVERED_CLASSES)}",
        ]
        canvas = render(image, objects, per_model, lines)
        out_path = out_dir / f"{item_id}.png"
        cv2.imwrite(str(out_path), canvas)
        print(f"  {item_id}: draft={entry['draft']} median={entry['median']} "
              f"gap={entry['gap']} archive={archive_covered} "
              f"dataset_boxes={len(objects)} -> {out_path.name}")
        if source is None:
            print(f"           WARNING no archive file matched this image; dataset boxes absent")

    print(f"\nWrote {len(ranked) - failures} picture(s) to {out_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
