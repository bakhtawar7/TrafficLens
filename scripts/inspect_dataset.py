"""Inspect a raw dataset and propose a class mapping for TrafficLens.

Reads the archive through :mod:`core.curation` and reports, for every class the dataset
actually contains, how many images feature it and how many annotations it has. It then
writes ``configs/class_mapping.json`` with a PROPOSED mapping of each class onto the seven
TrafficLens ground truth categories plus ``ignore``.

The archive is never copied, filtered, or modified. The proposal reuses the existing
``core.dataset.LABEL_TO_CATEGORY`` so this JSON stays consistent with the code rather than
becoming a competing source of truth. It is a review artefact: nothing in the running
detector reads it yet.

Run from the project root::

    python scripts/inspect_dataset.py --source "C:/path/to/archive"
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config, curation, dataset  # noqa: E402

#: Every category a class may be proposed into, matching the labelling protocol plus ignore.
TARGET_CATEGORIES: tuple[str, ...] = dataset.LABEL_CATEGORIES + ("ignore",)


def class_counts(records: list[curation.AnnotationRecord]) -> dict[str, dict[str, int]]:
    """Return per class image and annotation counts.

    ``annotations`` counts every box; ``images`` counts distinct image paths that contain
    at least one box of the class.
    """
    annotations: Counter[str] = Counter()
    images: dict[str, set[str]] = defaultdict(set)
    for record in records:
        annotations[record.class_name] += 1
        images[record.class_name].add(record.image_path)
    return {
        name: {"images": len(images[name]), "annotations": count}
        for name, count in annotations.items()
    }


def propose_mapping(class_name: str) -> dict[str, object]:
    """Propose a target category for one dataset class using the existing code mapping."""
    category, known = dataset.map_label(class_name)
    return {
        "proposed": category,
        "known_mapping": known,
        "mapping_source": (
            "core.dataset.LABEL_TO_CATEGORY" if known else "fallback:other (review needed)"
        ),
    }


def build_mapping_payload(source: Path, records: list[curation.AnnotationRecord]) -> dict[str, object]:
    """Assemble the full class mapping document from the loaded records."""
    counts = class_counts(records)
    classes: dict[str, object] = {}
    for class_name in sorted(counts):
        entry = propose_mapping(class_name)
        entry["images"] = counts[class_name]["images"]
        entry["annotations"] = counts[class_name]["annotations"]
        classes[class_name] = entry
    return {
        "_note": (
            "PROPOSED mapping for human review, not applied to any detector. "
            "Derived from core.dataset.LABEL_TO_CATEGORY. Verify before use."
        ),
        "dataset_path": str(source),
        "target_categories": list(TARGET_CATEGORIES),
        "totals": {
            "classes": len(classes),
            "images_with_annotations": len({r.image_path for r in records}),
            "annotations": len(records),
        },
        "classes": classes,
    }


def print_report(payload: dict[str, object]) -> None:
    """Print every dataset class with its image and annotation counts."""
    totals = payload["totals"]
    print(f"Dataset: {payload['dataset_path']}")
    print(
        f"Classes: {totals['classes']}  "
        f"Images with annotations: {totals['images_with_annotations']}  "
        f"Annotations: {totals['annotations']}"
    )
    print("\n  class                         images  annotations  proposed")
    print("  " + "-" * 66)
    classes = payload["classes"]  # type: ignore[index]
    ordered = sorted(classes, key=lambda name: (-classes[name]["annotations"], name))  # type: ignore[index]
    for class_name in ordered:
        row = classes[class_name]  # type: ignore[index]
        marker = "" if row["known_mapping"] else "  (unmapped, review)"
        print(
            f"  {class_name:<28} {row['images']:>6}  {row['annotations']:>11}  "
            f"{row['proposed']}{marker}"
        )


def main(argv: list[str] | None = None) -> int:
    """Load the dataset, print the class report, and write the proposed mapping JSON."""
    parser = argparse.ArgumentParser(description="Inspect a dataset and propose a class mapping.")
    parser.add_argument("--source", required=True, help="Folder holding the raw dataset.")
    parser.add_argument(
        "--out",
        default=str(config.CLASS_MAPPING_JSON),
        help="Path for the proposed mapping JSON (default configs/class_mapping.json).",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="Print the report without writing the mapping."
    )
    args = parser.parse_args(argv)

    source = Path(args.source).expanduser().resolve()
    try:
        records = curation.load_records(source)
    except curation.CurationError as error:
        print(f"ERROR: {error}")
        return 1

    payload = build_mapping_payload(source, records)
    print_report(payload)

    if args.no_write:
        print("\nMapping not written (--no-write).")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote proposed mapping to {out}")
    print("Review it before any use. It is not read by the detector.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
