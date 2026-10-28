"""Validate the test set CSV files against the images and clips on disk.

Reports files present on disk but missing from SOURCES.csv, SOURCES rows that point at
a missing file, missing permission fields, and invalid condition tags. It also prints a
summary count per condition tag. Exits non zero when any error is found so it can gate
a data collection session.

Run from the project root with: ``python scripts/check_testset.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config  # noqa: E402  (path setup must run first)

VALID_CONDITION_TAGS: set[str] = {"day", "night", "glare", "dense", "sparse", "distant"}
_IMAGE_SUFFIXES: set[str] = {".jpg", ".jpeg", ".png", ".webp"}
_CLIP_SUFFIXES: set[str] = {".mp4", ".mov", ".avi", ".mkv"}


def _on_disk_media() -> set[str]:
    """Return media filenames present in the images and clips directories."""
    found: set[str] = set()
    for directory, suffixes in ((config.TESTSET_IMAGES_DIR, _IMAGE_SUFFIXES),
                                (config.TESTSET_CLIPS_DIR, _CLIP_SUFFIXES)):
        if directory.exists():
            for path in directory.iterdir():
                if path.is_file() and path.suffix.lower() in suffixes:
                    found.add(path.name)
    return found


def _split_tags(raw: object) -> list[str]:
    """Split a condition tag cell into normalised individual tags."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    parts = str(raw).replace("|", ";").split(";")
    return [part.strip().lower() for part in parts if part.strip()]


def check(sources: pd.DataFrame, labels: pd.DataFrame, media: set[str]) -> tuple[list[str], list[int]]:
    """Run every validation and return error messages plus a condition histogram.

    Returns:
        A tuple of (error messages, counts per condition) where counts align with
        ``sorted(VALID_CONDITION_TAGS)``.
    """
    errors: list[str] = []
    condition_counts = {tag: 0 for tag in sorted(VALID_CONDITION_TAGS)}

    listed = set(sources["filename"].dropna().astype(str)) if "filename" in sources else set()

    for name in sorted(media - listed):
        errors.append(f"File '{name}' is on disk but missing from SOURCES.csv.")
    for name in sorted(listed - media):
        errors.append(f"SOURCES.csv row references '{name}' but no such file exists.")

    if "permission" in sources.columns:
        for index, row in sources.iterrows():
            if pd.isna(row["permission"]) or str(row["permission"]).strip() == "":
                item = row.get("item_id", f"row {index}")
                errors.append(f"SOURCES.csv item '{item}' has no permission recorded.")

    if "condition_tags" in sources.columns:
        for index, row in sources.iterrows():
            tags = _split_tags(row["condition_tags"])
            for tag in tags:
                if tag in condition_counts:
                    condition_counts[tag] += 1
                else:
                    item = row.get("item_id", f"row {index}")
                    errors.append(
                        f"SOURCES.csv item '{item}' has invalid condition tag '{tag}'."
                    )

    if "item_id" in sources.columns and "item_id" in labels.columns:
        source_ids = set(sources["item_id"].dropna().astype(str))
        for item in sorted(set(labels["item_id"].dropna().astype(str)) - source_ids):
            errors.append(f"LABELS.csv item '{item}' has no matching row in SOURCES.csv.")

    return errors, [condition_counts[tag] for tag in sorted(VALID_CONDITION_TAGS)]


def main() -> int:
    """Load the CSV files, run checks, print a report, and return the exit code."""
    if not config.SOURCES_CSV.exists():
        print(f"ERROR: missing {config.SOURCES_CSV}")
        return 1

    sources = pd.read_csv(config.SOURCES_CSV)
    labels = (
        pd.read_csv(config.LABELS_CSV)
        if config.LABELS_CSV.exists()
        else pd.DataFrame(columns=["item_id"])
    )
    media = _on_disk_media()

    errors, counts = check(sources, labels, media)
    tags = sorted(VALID_CONDITION_TAGS)

    print("TrafficLens test set check")
    print(f"  SOURCES rows : {len(sources)}")
    print(f"  LABELS rows  : {len(labels)}")
    print(f"  Media on disk: {len(media)}")
    print("  Condition summary:")
    for tag, count in zip(tags, counts):
        print(f"    {tag:<8}: {count}")

    if errors:
        print(f"\nFound {len(errors)} problem(s):")
        for message in errors:
            print(f"  - {message}")
        return 1

    print("\nNo problems found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
