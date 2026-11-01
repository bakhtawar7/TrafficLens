"""Measure the cleaned pool and propose condition tags from quartile cut-offs.

Runs after ``clean_dataset``. For every remaining image it measures mean brightness, the
share of saturated pixels, the annotated vehicle count, the median annotated box area as a
fraction of the image, and whether a rickshaw or CNG class is present. All values come from
the pixels and the ground truth annotation, never from model detections.

Quartile cut-offs are computed on the full pool and written to ``configs/tag_cutoffs.json``
before any sampling happens, then the distribution of each feature and the number of images
per proposed tag are printed. The dataset folder is only read.

Run from the project root::

    python scripts/tag_dataset.py --source "C:/path/to/archive"
    python scripts/tag_dataset.py --source "C:/path/to/archive" --hamming 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config, curation, dataset  # noqa: E402


def _print_histogram(values: np.ndarray, bins: int = 10) -> None:
    """Print a compact text histogram of ``values`` across ``bins`` buckets."""
    if values.size == 0:
        print("    (no data)")
        return
    counts, edges = np.histogram(values, bins=bins)
    peak = max(counts.max(), 1)
    for count, low, high in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(round(24 * count / peak))
        print(f"    [{low:8.3f}, {high:8.3f}) {bar} {count}")


def print_distributions(features: list[curation.ImageFeature]) -> None:
    """Print the five number summary and a histogram for each numeric feature."""
    print("Feature distributions")
    for name in curation.FEATURE_NAMES:
        series = np.asarray([getattr(f, name) for f in features], dtype=float)
        summary = curation.summarize_distribution(series)
        print(
            f"  {name}: n={summary['count']:.0f} min={summary['min']:.3f} "
            f"q1={summary['q1']:.3f} median={summary['median']:.3f} "
            f"q3={summary['q3']:.3f} max={summary['max']:.3f} "
            f"mean={summary['mean']:.3f} std={summary['std']:.3f}"
        )
        _print_histogram(series)
    dist = curation.feature_distributions(features)["rickshaw_or_cng_rate"]
    print(
        f"  rickshaw_or_cng: {dist['images']:.0f} images "
        f"({dist['rate'] * 100:.1f}% of the pool)"
    )


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, compute features and cut-offs, and print the report."""
    parser = argparse.ArgumentParser(description="Propose condition tags from quartiles.")
    parser.add_argument("--source", required=True, help="Raw dataset folder to measure.")
    parser.add_argument(
        "--hamming",
        type=int,
        default=dataset.DUPLICATE_HASH_DISTANCE,
        help=f"Hamming distance used to drop near duplicates (default {dataset.DUPLICATE_HASH_DISTANCE}).",
    )
    parser.add_argument(
        "--mapping",
        default=str(config.CLASS_MAPPING_JSON),
        help="Reviewed class mapping JSON used during cleaning.",
    )
    parser.add_argument(
        "--cutoffs",
        default=str(config.TAG_CUTOFFS_JSON),
        help="Where to write the proposed quartile cut-offs.",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="Print the report without writing the cut-off file."
    )
    args = parser.parse_args(argv)

    try:
        cleaning = curation.clean_dataset(
            Path(args.source),
            hamming_threshold=args.hamming,
            mapping_path=Path(args.mapping),
            write_log=False,
        )
    except curation.CurationError as error:
        print(f"ERROR: {error}")
        return 1

    features = curation.compute_features(cleaning.kept_paths)
    if not features:
        print("ERROR: no readable, annotated images remain to measure.")
        return 1

    cutoffs = curation.compute_cutoffs(features)
    # Cut-offs are persisted before any sampling so later stages share one fixed reference.
    if not args.no_write:
        curation.write_tag_cutoffs(
            cutoffs, source=str(Path(args.source).resolve()),
            images_used=len(features), path=Path(args.cutoffs),
        )

    counts = curation.tag_counts(features, cutoffs)
    print(f"Pool measured: {len(features)} images (remaining after cleaning)")
    print_distributions(features)
    print("Images per proposed tag")
    for tag in dataset.CONDITION_TAGS:
        print(f"  {tag:<8}: {counts[tag]}")
    print(f"cut-offs: {args.cutoffs}" + ("  (dry run, not written)" if args.no_write else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
