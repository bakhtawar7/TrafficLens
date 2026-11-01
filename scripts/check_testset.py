"""Validate the test set CSV files against the images and clips on disk.

Reports files present on disk but missing from SOURCES.csv, SOURCES rows that point at
a missing file, missing permission or provenance fields (dataset, split, licence, and a
malformed url), and invalid condition tags. It also validates LABELS.csv values: every
category count must be a non-negative integer and labeller_pass must be 1 or 2. Blank
cells in those files are treated as "not recorded yet" and skipped, so a partially
labelled sheet or an empty scaffold still validates cleanly. A summary count per
condition tag is printed. Exits non zero when any error is found so it can gate a data
collection session.

Run from the project root with: ``python scripts/check_testset.py``.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config, dataset  # noqa: E402  (path setup must run first)

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


def _is_blank(value: object) -> bool:
    """True for a cell that carries no value, which is treated as 'not recorded yet'."""
    return value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == ""


def _count_is_valid(value: object) -> bool:
    """True when a LABELS count cell is a non-negative integer.

    Native ints and whole-number floats are both accepted because pandas widens a column
    to float when it also holds blanks, so a filled ``3`` can read back as ``3.0``.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value >= 0
    if isinstance(value, float):
        return value.is_integer() and value >= 0
    text = str(value).strip()
    return bool(re.fullmatch(r"\d+", text) or re.fullmatch(r"\d+\.0*", text))


def _labeller_pass_is_valid(value: object) -> bool:
    """True when a labeller pass cell is the integer 1 or 2."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value in (1, 2)
    if isinstance(value, float):
        return value.is_integer() and value in (1.0, 2.0)
    return str(value).strip() in {"1", "2", "1.0", "2.0"}


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

    # Provenance columns written by build_testset: every item must record where it came
    # from and under what usage rights. These fields identify the source, so a blank is an
    # error; url alone is optional because local archive images carry no public link.
    for column in ("dataset", "split", "licence"):
        if column in sources.columns:
            for index, row in sources.iterrows():
                if _is_blank(row[column]):
                    item = row.get("item_id", f"row {index}")
                    errors.append(f"SOURCES.csv item '{item}' has no {column} recorded.")

    if "url" in sources.columns:
        for index, row in sources.iterrows():
            value = row["url"]
            if _is_blank(value):
                continue
            if not str(value).strip().lower().startswith(("http://", "https://")):
                item = row.get("item_id", f"row {index}")
                errors.append(f"SOURCES.csv item '{item}' has an invalid url '{str(value).strip()}'.")

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

    # LABELS value validation: category counts must be non-negative integers and the
    # labeller pass must be 1 or 2. Blank cells are skipped so a partially labelled sheet
    # still passes, and a scaffold with only a header and no rows raises no error.
    count_columns = [column for column in dataset.LABEL_CATEGORIES if column in labels.columns]
    for index, row in labels.iterrows():
        item = row.get("item_id", f"row {index}")
        for column in count_columns:
            value = row[column]
            if _is_blank(value):
                continue
            if not _count_is_valid(value):
                errors.append(
                    f"LABELS.csv item '{item}' count '{column}' must be a non-negative "
                    f"integer, got '{str(value).strip()}'."
                )
        if "labeller_pass" in labels.columns:
            pass_value = row["labeller_pass"]
            if not _is_blank(pass_value) and not _labeller_pass_is_valid(pass_value):
                errors.append(
                    f"LABELS.csv item '{item}' labeller_pass must be 1 or 2, "
                    f"got '{str(pass_value).strip()}'."
                )

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
