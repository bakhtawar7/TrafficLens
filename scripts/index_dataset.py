"""Index a raw annotated archive into a machine readable inventory.

The archive is never modified. Every image is paired with its PASCAL VOC XML, decoded,
measured, mapped onto the seven TrafficLens ground truth categories, and written as one
row in ``data/testset/inventory.csv``. Rows carry the automatic reject reasons too, so the
next step can explain why an item did or did not become a test case.

Run from the project root::

    python scripts/index_dataset.py --source "C:/path/to/archive"
    python scripts/index_dataset.py --source "C:/path/to/archive" --limit 300
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config, dataset  # noqa: E402

IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")

COLUMNS: tuple[str, ...] = (
    "image_path",
    "xml_path",
    "original_filename",
    "source_split",
    "width",
    "height",
    "area",
    "area_origin",
    "counts_json",
    "total",
    "detectable_total",
    "kept_objects",
    "dropped_small",
    "unmapped_labels",
    "condition_tags",
    "boxes_json",
    "mean_luma",
    "sky_luma",
    "lit_fraction",
    "highlight_fraction",
    "sharpness",
    "phash",
    "size_mismatch",
    "parse_error",
    "reject_reasons",
)


def find_pairs(source: Path) -> list[tuple[Path, Path | None, str]]:
    """Pair every image under ``source`` with the XML file of the same stem.

    Returns:
        A sorted list of ``(image, xml or None, split name)`` triples. The split name is
        the archive subfolder the item came from.
    """
    pairs: list[tuple[Path, Path | None, str]] = []
    seen: set[str] = set()
    for image in sorted(source.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stem_key = f"{image.parent}::{image.stem}".lower()
        if stem_key in seen:
            continue
        seen.add(stem_key)
        xml = next((candidate for candidate in (image.with_suffix(".xml"),) if candidate.exists()), None)
        split = image.parent.name if image.parent != source else "root"
        pairs.append((image, xml, split))
    return pairs


def build_candidate(image: Path, xml: Path | None, split: str) -> dataset.Candidate:
    """Measure one archive image and package the result as a :class:`Candidate`."""
    candidate = dataset.Candidate(
        image_path=str(image),
        xml_path=str(xml) if xml else "",
        original_filename=image.name,
        source_split=split,
    )

    if xml is None:
        candidate.parse_error = "no XML annotation beside the image"
        candidate.reject_reasons = dataset.screen_candidate(candidate)
        return candidate

    annotation, error = dataset.parse_voc_annotation(xml)
    if annotation is None:
        candidate.parse_error = error
        candidate.reject_reasons = dataset.screen_candidate(candidate)
        return candidate

    image_array = dataset.load_image(image)
    if image_array is None:
        candidate.parse_error = f"{image.name} could not be decoded"
        candidate.reject_reasons = dataset.screen_candidate(candidate)
        return candidate

    declared = (annotation.width, annotation.height)
    actual = (image_array.shape[1], image_array.shape[0])
    candidate.size_mismatch = declared != actual and declared != (0, 0)
    objects = (
        dataset.scale_objects(annotation.objects, declared, actual)
        if candidate.size_mismatch
        else annotation.objects
    )

    result = dataset.count_by_category(objects)
    metrics = dataset.measure_image(image_array)

    candidate.width = metrics.width
    candidate.height = metrics.height
    candidate.mean_luma = metrics.mean_luma
    candidate.sky_luma = metrics.sky_luma
    candidate.lit_fraction = metrics.lit_fraction
    candidate.highlight_fraction = metrics.highlight_fraction
    candidate.sharpness = metrics.sharpness
    candidate.phash = dataset.perceptual_hash(image_array)
    candidate.counts = result["counts"]
    candidate.kept_objects = result["kept"]
    candidate.dropped_small = result["dropped_small"]
    candidate.unmapped_labels = result["unmapped"]
    candidate.area, candidate.area_origin = dataset.infer_area(image.name, split)
    candidate.objects = dataset.surviving_objects(objects)
    candidate.tags = dataset.suggest_condition_tags(metrics, candidate.counts, candidate.objects)
    candidate.reject_reasons = dataset.screen_candidate(candidate)
    return candidate


def to_row(candidate: dataset.Candidate) -> dict[str, object]:
    """Flatten a candidate into inventory columns."""
    return {
        "image_path": candidate.image_path,
        "xml_path": candidate.xml_path,
        "original_filename": candidate.original_filename,
        "source_split": candidate.source_split,
        "width": candidate.width,
        "height": candidate.height,
        "area": candidate.area,
        "area_origin": candidate.area_origin,
        "counts_json": json.dumps(candidate.counts),
        "total": candidate.total,
        "detectable_total": candidate.detectable_total,
        "kept_objects": candidate.kept_objects,
        "dropped_small": candidate.dropped_small,
        "unmapped_labels": ";".join(candidate.unmapped_labels),
        "condition_tags": dataset.merge_tags(candidate.tags),
        "boxes_json": dataset.objects_to_json(candidate.objects),
        "mean_luma": round(candidate.mean_luma, 2),
        "sky_luma": round(candidate.sky_luma, 2),
        "lit_fraction": round(candidate.lit_fraction, 4),
        "highlight_fraction": round(candidate.highlight_fraction, 5),
        "sharpness": round(candidate.sharpness, 2),
        "phash": candidate.phash,
        "size_mismatch": candidate.size_mismatch,
        "parse_error": candidate.parse_error,
        "reject_reasons": "; ".join(candidate.reject_reasons),
    }


def summarise(rows: list[dict[str, object]]) -> None:
    """Print what the archive contains and why items were screened out."""
    frame = pd.DataFrame(rows)
    eligible = frame[frame["reject_reasons"].astype(str) == ""]
    print(f"Indexed {len(frame)} images, {len(eligible)} eligible for selection.")

    print("\n  Ground truth vehicles recorded across eligible items:")
    totals = Counter()
    for counts in eligible["counts_json"]:
        for category, value in json.loads(counts).items():
            totals[category] += int(value)
    for category in dataset.LABEL_CATEGORIES:
        print(f"    {category:<16}: {totals.get(category, 0)}")

    print("\n  Suggested condition tags across eligible items:")
    tag_counts = Counter()
    for tags in eligible["condition_tags"]:
        for tag in str(tags).split(";"):
            if tag:
                tag_counts[tag] += 1
    for tag in dataset.CONDITION_TAGS:
        print(f"    {tag:<8}: {tag_counts.get(tag, 0)}")

    print("\n  Top reject reasons across the whole archive:")
    reasons = Counter()
    for text in frame["reject_reasons"]:
        for reason in str(text).split("; "):
            if reason:
                reasons[reason.split(" (")[0]] += 1
    for reason, count in reasons.most_common(8):
        print(f"    {count:>5}  {reason}")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, index the archive, and write the inventory CSV."""
    parser = argparse.ArgumentParser(description="Index an annotated archive into inventory.csv.")
    parser.add_argument("--source", required=True, help="Folder holding the raw archive.")
    parser.add_argument(
        "--out",
        default=str(config.TESTSET_INVENTORY_CSV),
        help="Inventory CSV to write (default data/testset/inventory.csv).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Index at most this many images.")
    parser.add_argument("--verbose", action="store_true", help="Print a line per image.")
    args = parser.parse_args(argv)

    source = Path(args.source).expanduser().resolve()
    if not source.is_dir():
        print(f"ERROR: no such archive folder: {source}")
        return 1

    pairs = find_pairs(source)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    if not pairs:
        print(f"ERROR: no images found under {source}")
        return 1

    rows: list[dict[str, object]] = []
    for index, (image, xml, split) in enumerate(pairs, start=1):
        rows.append(to_row(build_candidate(image, xml, split)))
        if args.verbose and index % 100 == 0:
            print(f"  indexed {index}/{len(pairs)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(COLUMNS)).to_csv(out, index=False)
    print(f"Wrote {len(rows)} rows to {out}")
    summarise(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
