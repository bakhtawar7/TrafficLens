"""Clean a raw dataset and log every dropped image.

Runs :func:`core.curation.clean_dataset` over a folder, writes ``results/curation_log.csv``
with one row per dropped image and its reason, and prints the counts. The dataset folder is
only read, never modified, so the originals stay exactly as downloaded.

Run from the project root::

    python scripts/clean_dataset.py --source "C:/path/to/archive"
    python scripts/clean_dataset.py --source "C:/path/to/archive" --hamming 10 --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config, curation, dataset  # noqa: E402


def print_counts(report: curation.CleaningReport, log_path: Path, dry_run: bool) -> None:
    """Print loaded, dropped by reason, duplicates removed, and remaining."""
    print("Curation cleaning report")
    print(f"  loaded:            {report.loaded}")
    print("  dropped by reason:")
    if report.dropped_by_reason:
        for reason in sorted(report.dropped_by_reason):
            print(f"    {reason:<24}: {report.dropped_by_reason[reason]}")
    else:
        print("    (none)")
    print(f"  dropped total:     {report.dropped_total}")
    print(f"  duplicates removed:{report.duplicates_removed}")
    print(f"  remaining:         {report.remaining}")
    print(f"  log:               {log_path}" + ("  (dry run, not written)" if dry_run else ""))


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, clean the dataset, and report the counts."""
    parser = argparse.ArgumentParser(description="Clean a dataset and log dropped images.")
    parser.add_argument("--source", required=True, help="Raw dataset folder to clean.")
    parser.add_argument(
        "--hamming",
        type=int,
        default=dataset.DUPLICATE_HASH_DISTANCE,
        help=f"Hamming distance for near duplicates (default {dataset.DUPLICATE_HASH_DISTANCE}).",
    )
    parser.add_argument(
        "--mapping",
        default=str(config.CLASS_MAPPING_JSON),
        help="Reviewed class mapping JSON (default configs/class_mapping.json).",
    )
    parser.add_argument(
        "--log",
        default=str(config.CURATION_LOG_CSV),
        help="Drop log CSV to write (default results/curation_log.csv).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Compute the report without writing the log."
    )
    args = parser.parse_args(argv)

    try:
        report = curation.clean_dataset(
            Path(args.source),
            hamming_threshold=args.hamming,
            mapping_path=Path(args.mapping),
            log_path=Path(args.log),
            write_log=not args.dry_run,
        )
    except curation.CurationError as error:
        print(f"ERROR: {error}")
        return 1

    print_counts(report, Path(args.log), args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
