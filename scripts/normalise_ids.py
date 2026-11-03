"""Rewrite legacy item ids into the canonical four digit form, one file at a time.

The test set was assembled by ``scripts/build_testset.py``, which mints ids as ``tl0001``.
The draft ``LABELS.csv`` inherited from the earlier manual selection uses ``tl001``, so the
two files name the same item differently and any join that compares strings fails. This
script makes the spelling uniform without hiding anything: every id it changes is printed as
``old -> new``, the count per file is printed, and a cell that is not a TrafficLens id at all
is reported and left untouched rather than guessed at.

Nothing is written unless ``--apply`` is given. The default run is a preview.

``data/testset/selection_log.csv`` is deliberately not a target. It records the discarded
manual selection, where ``tl001`` meant whatever archive frame the operator picked at the
time, not the current item 1. Rewriting its ids would make that history look like it
describes today's test set, which would be a worse lie than the mismatch it fixes.

Run from the project root with:

    python scripts/normalise_ids.py                    # preview every file
    python scripts/normalise_ids.py --apply            # rewrite every file
    python scripts/normalise_ids.py --apply --only SOURCES.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config, dataset  # noqa: E402  (path setup must run first)

#: Files whose item ids are part of the current test set, keyed by the name used with
#: ``--only``. Each is rewritten in its ``item_id`` column only.
TARGETS: dict[str, Path] = {
    "SOURCES.csv": config.SOURCES_CSV,
    "LABELS.csv": config.LABELS_CSV,
    "annotation_counts.csv": config.ANNOTATION_COUNTS_CSV,
    "audit_sheet.csv": config.AUDIT_SHEET_CSV,
}

#: Column that carries the item id in every target file.
ID_COLUMN = "item_id"


def plan_file(path: Path) -> tuple[list[str], list[str], int]:
    """Return the rewrites, the unreadable cells, and the row count for one file.

    Each rewrite is a printable ``row N: old -> new`` message. The file is not modified.
    """
    rewrites: list[str] = []
    problems: list[str] = []
    rows = 0
    if not path.exists():
        return [], [f"{path.name} does not exist"], 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or ID_COLUMN not in reader.fieldnames:
            return [], [f"{path.name} has no {ID_COLUMN} column"], 0
        for index, row in enumerate(reader, start=2):
            rows += 1
            raw = (row.get(ID_COLUMN) or "").strip()
            if not raw:
                continue
            canonical = dataset.normalise_item_id(raw)
            if canonical is None:
                problems.append(f"{path.name} row {index}: '{raw}' is not a TrafficLens item id")
            elif canonical != raw:
                rewrites.append(f"{path.name} row {index}: {raw} -> {canonical}")
    return rewrites, problems, rows


def apply_file(path: Path) -> int:
    """Rewrite the item id column of one file in place and return how many cells changed.

    Cells outside the id column are copied through exactly as read, so quoting, blank
    fields, and column order survive the round trip.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    changed = 0
    for row in rows:
        raw = (row.get(ID_COLUMN) or "").strip()
        canonical = dataset.normalise_item_id(raw) if raw else None
        if canonical is not None and canonical != raw:
            row[ID_COLUMN] = canonical
            changed += 1
    if changed:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    return changed


def main(apply: bool = False, only: str | None = None) -> int:
    """Preview or run the migration and print every id that changes."""
    names = list(TARGETS)
    if only:
        if only not in TARGETS:
            print(f"ERROR: unknown file '{only}'. Choose from: {', '.join(names)}")
            return 2
        names = [only]

    total = 0
    for name in names:
        path = TARGETS[name]
        rewrites, problems, rows = plan_file(path)
        print(f"{name} ({rows} rows)")
        if problems:
            for message in problems:
                print(f"  PROBLEM {message}")
        if not rewrites:
            print("  no ids to rewrite: already in the four digit form")
            continue
        for message in rewrites:
            print(f"  {message}")
        print(f"  {len(rewrites)} id(s) would {'be rewritten' if apply else 'change'}")
        if apply:
            changed = apply_file(path)
            print(f"  {changed} id(s) rewritten in place")
            total += changed
        else:
            total += len(rewrites)

    print(f"\n{'Applied' if apply else 'Preview'}: {total} id(s), files: {', '.join(names)}")
    if not apply and total:
        print("Re-run with --apply to write these changes.")
    print("Not a target: selection_log.csv (its ids name the discarded manual selection).")
    return 1 if any(plan_file(TARGETS[name])[1] for name in names) else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="rewrite the files instead of previewing")
    parser.add_argument("--only", help="restrict the run to one target file name")
    args = parser.parse_args()
    sys.exit(main(apply=args.apply, only=args.only))
