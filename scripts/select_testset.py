"""Turn an inventory into a reviewable TrafficLens test set.

The selection is deliberately mechanical so it can be repeated and argued about: unfit
items are dropped by the screening rules in :mod:`core.dataset`, the remaining items are
ranked, and a greedy pass keeps the set spread across areas, condition combinations, and
near duplicate frames. Ground truth counts are taken from the archive annotations rather
than invented, which is why every generated row is marked as awaiting a labeller pass.

The archive is read only. Chosen images are copied into ``data/testset/images`` using the
naming convention from ``docs/test_set_guide.md``.

Run from the project root::

    python scripts/select_testset.py                      # print the plan, change nothing
    python scripts/select_testset.py --apply              # write 55 items and the CSV files
    python scripts/select_testset.py --apply --target 60 --clean
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config, dataset  # noqa: E402

REVIEW_NOTE = (
    "counts and condition tags auto derived from the archive annotation; "
    "requires labeller verification pass"
)

SOURCES_COLUMNS: tuple[str, ...] = (
    "item_id",
    "filename",
    "type",
    "source",
    "permission",
    "date_captured",
    "location_area",
    "condition_tags",
    "notes",
)

LABELS_COLUMNS: tuple[str, ...] = ("item_id", *dataset.LABEL_CATEGORIES, "labeller_pass", "notes")


def _text(value: object) -> str:
    """Return an empty string for the missing values pandas stores as NaN."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text


def _number(value: object, cast: type = float) -> object:
    """Cast a CSV cell to a number, falling back to zero for missing values."""
    text = _text(value)
    if not text:
        return cast(0)
    try:
        return cast(float(text))
    except ValueError:
        return cast(0)


def load_candidates(inventory: Path) -> list[dataset.Candidate]:
    """Rebuild candidates from a previously written inventory CSV."""
    frame = pd.read_csv(inventory)
    candidates: list[dataset.Candidate] = []
    for row in frame.to_dict("records"):
        tags = [tag for tag in _text(row.get("condition_tags")).split(";") if tag]
        candidates.append(
            dataset.Candidate(
                image_path=_text(row["image_path"]),
                xml_path=_text(row.get("xml_path")),
                original_filename=_text(row["original_filename"]),
                source_split=_text(row.get("source_split")),
                width=_number(row.get("width"), int),
                height=_number(row.get("height"), int),
                area=_text(row.get("area")) or "unknown",
                area_origin=_text(row.get("area_origin")) or "batch",
                counts=json.loads(row["counts_json"]) if _text(row.get("counts_json")) else {},
                kept_objects=_number(row.get("kept_objects"), int),
                dropped_small=_number(row.get("dropped_small"), int),
                unmapped_labels=[t for t in _text(row.get("unmapped_labels")).split(";") if t],
                tags=tags,
                mean_luma=_number(row.get("mean_luma")),
                sky_luma=_number(row.get("sky_luma")),
                lit_fraction=_number(row.get("lit_fraction")),
                highlight_fraction=_number(row.get("highlight_fraction")),
                sharpness=_number(row.get("sharpness")),
                phash=_text(row.get("phash")),
                size_mismatch=bool(row.get("size_mismatch")),
                parse_error=_text(row.get("parse_error")),
                reject_reasons=[],
                objects=dataset.objects_from_json(row.get("boxes_json")),
            )
        )
    return candidates


def sources_row(item_id: str, candidate: dataset.Candidate, filename: str, source_name: str, permission: str) -> dict[str, str]:
    """Build one ``SOURCES.csv`` row for a chosen candidate."""
    notes = [REVIEW_NOTE, f"original filename {candidate.original_filename}"]
    if candidate.area_origin == "batch":
        notes.append("area taken from the archive batch name, not a verified location")
    if candidate.unmapped_labels:
        notes.append("unmapped archive labels: " + ", ".join(candidate.unmapped_labels))
    if candidate.dropped_small:
        notes.append(f"{candidate.dropped_small} object(s) excluded by the 12px rule")
    return {
        "item_id": item_id,
        "filename": filename,
        "type": "image",
        "source": source_name,
        "permission": permission,
        "date_captured": dataset.extract_date(candidate.original_filename),
        "location_area": candidate.area,
        "condition_tags": dataset.merge_tags(candidate.tags),
        "notes": "; ".join(notes),
    }


def labels_row(item_id: str, candidate: dataset.Candidate) -> dict[str, object]:
    """Build the draft ``LABELS.csv`` row for a chosen candidate."""
    row: dict[str, object] = {"item_id": item_id}
    for category in dataset.LABEL_CATEGORIES:
        row[category] = int(candidate.counts.get(category, 0))
    row["labeller_pass"] = 1
    row["notes"] = REVIEW_NOTE
    return row


def write_log(path: Path, accepted: list[dataset.Candidate], rejected: list[dataset.Candidate]) -> None:
    """Record every candidate with its decision so the selection can be audited."""
    rows = []
    for index, candidate in enumerate(accepted, start=1):
        rows.append(
            {
                "decision": "selected",
                "item_id": f"tl{index:03d}",
                "original_filename": candidate.original_filename,
                "area": candidate.area,
                "condition_tags": dataset.merge_tags(candidate.tags),
                "total": candidate.total,
                "detectable_total": candidate.detectable_total,
                "score": round(dataset.candidate_score(candidate), 4),
                "reasons": "",
            }
        )
    for candidate in rejected:
        rows.append(
            {
                "decision": "rejected",
                "item_id": "",
                "original_filename": candidate.original_filename,
                "area": candidate.area,
                "condition_tags": dataset.merge_tags(candidate.tags),
                "total": candidate.total,
                "detectable_total": candidate.detectable_total,
                "score": round(dataset.candidate_score(candidate), 4),
                "reasons": "; ".join(candidate.reject_reasons),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def report(accepted: list[dataset.Candidate], rejected: list[dataset.Candidate], tag_quotas: dict[str, int] | None = None) -> None:
    """Print the shape of the proposed set so the plan is readable before it is applied."""
    print(f"Proposed test set: {len(accepted)} items. Rejected candidates: {len(rejected)}.")

    tags = {tag: 0 for tag in dataset.CONDITION_TAGS}
    categories = {category: 0 for category in dataset.LABEL_CATEGORIES}
    items_with = {category: 0 for category in dataset.LABEL_CATEGORIES}
    areas: dict[str, int] = {}
    for candidate in accepted:
        for tag in candidate.tags:
            tags[tag] = tags.get(tag, 0) + 1
        for category in dataset.LABEL_CATEGORIES:
            value = int(candidate.counts.get(category, 0))
            categories[category] += value
            if value > 0:
                items_with[category] += 1
        areas[candidate.area] = areas.get(candidate.area, 0) + 1

    print("\n  Condition coverage:")
    for tag, count in tags.items():
        quota = (tag_quotas or {}).get(tag)
        marker = ""
        if quota:
            marker = "  (quota met)" if count >= quota else f"  (quota {quota} NOT reachable)"
        print(f"    {tag:<8}: {count}{marker}")
    print("\n  Ground truth volume per category (items containing at least one):")
    for category in dataset.LABEL_CATEGORIES:
        print(f"    {category:<16}: {categories[category]:>5} in {items_with[category]} item(s)")
    print("\n  Areas represented:")
    for area, count in sorted(areas.items(), key=lambda item: (-item[1], item[0])):
        print(f"    {area:<16}: {count}")


def main(argv: list[str] | None = None) -> int:
    """Select items and, with ``--apply``, copy them out and write the CSV files."""
    parser = argparse.ArgumentParser(description="Select the TrafficLens test set from an inventory.")
    parser.add_argument("--inventory", default=str(config.TESTSET_INVENTORY_CSV), help="Inventory CSV.")
    parser.add_argument("--target", type=int, default=55, help="Number of items to select.")
    parser.add_argument("--max-per-area", type=int, default=8, help="Cap on items from one area.")
    parser.add_argument(
        "--max-per-signature",
        type=int,
        default=6,
        help="Cap on items sharing one identical condition tag combination.",
    )
    parser.add_argument("--source-name", default="user supplied annotated archive", help="Recorded in SOURCES.csv.")
    parser.add_argument(
        "--min-tag",
        action="append",
        default=[],
        metavar="TAG=COUNT",
        help="Reserve at least COUNT items carrying a condition tag, for example --min-tag night=12. Repeatable.",
    )
    parser.add_argument(
        "--permission",
        default="UNVERIFIED: supplied by operator, confirm usage rights before publication",
        help="Recorded in SOURCES.csv; must be replaced with a real grant.",
    )
    parser.add_argument("--apply", action="store_true", help="Copy images and write the CSV files.")
    parser.add_argument("--clean", action="store_true", help="Empty the images folder before applying.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing SOURCES and LABELS rows.")
    args = parser.parse_args(argv)

    inventory = Path(args.inventory)
    if not inventory.exists():
        print(f"ERROR: {inventory} not found. Run scripts/index_dataset.py first.")
        return 1

    candidates = load_candidates(inventory)
    tag_quotas: dict[str, int] = {}
    try:
        for expression in args.min_tag:
            tag, count = dataset.parse_tag_quota(expression)
            tag_quotas[tag] = count
    except ValueError as exc:
        print(f"ERROR: invalid --min-tag value: {exc}")
        return 1

    accepted, rejected = dataset.select_candidates(
        candidates,
        target_size=args.target,
        max_per_area=args.max_per_area,
        max_per_tag_signature=args.max_per_signature,
        tag_quotas=tag_quotas,
    )
    report(accepted, rejected, tag_quotas)

    config.TESTSET_DIR.mkdir(parents=True, exist_ok=True)
    write_log(config.TESTSET_SELECTION_CSV, accepted, rejected)
    print(f"\nSelection log written to {config.TESTSET_SELECTION_CSV}")

    if not args.apply:
        print("Dry run only. Add --apply to copy images and write SOURCES.csv and LABELS.csv.")
        return 0

    if args.clean:
        removed = 0
        media_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".avi", ".mkv"}
        for existing in config.TESTSET_IMAGES_DIR.glob("*"):
            if existing.is_file() and existing.suffix.lower() in media_suffixes:
                existing.unlink()
                removed += 1
        print(f"Cleared {removed} media file(s) from {config.TESTSET_IMAGES_DIR}")

    if not args.force:
        for path in (config.SOURCES_CSV, config.LABELS_CSV):
            if path.exists() and len(pd.read_csv(path)) > 0:
                print(f"ERROR: {path.name} already holds rows. Use --force to replace them.")
                return 1

    config.TESTSET_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    sources_rows: list[dict[str, str]] = []
    labels_rows: list[dict[str, object]] = []
    copied = 0
    for index, candidate in enumerate(accepted, start=1):
        item_id = f"tl{index:03d}"
        filename = dataset.make_item_filename(item_id, candidate.area, candidate.tags, Path(candidate.image_path).suffix.lower())
        destination = config.TESTSET_IMAGES_DIR / filename
        shutil.copy2(candidate.image_path, destination)
        copied += 1
        sources_rows.append(sources_row(item_id, candidate, filename, args.source_name, args.permission))
        labels_rows.append(labels_row(item_id, candidate))

    pd.DataFrame(sources_rows, columns=list(SOURCES_COLUMNS)).to_csv(config.SOURCES_CSV, index=False)
    pd.DataFrame(labels_rows, columns=list(LABELS_COLUMNS)).to_csv(config.LABELS_CSV, index=False)
    print(f"Copied {copied} image(s) into {config.TESTSET_IMAGES_DIR}")
    print(f"Wrote {config.SOURCES_CSV} and {config.LABELS_CSV}")
    print("\nEvery row is a draft. Run scripts/make_review_sheet.py and verify before use.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
