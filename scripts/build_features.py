"""Build the label free reliability feature table from the cache and the pixels.

Reads only ``results/raw_detections.jsonl``, the test set images, and the
settings file. It never opens ``LABELS.csv``, ``annotation_counts.csv``, or any
dataset annotation: every feature is available at inference time (defined in
``core/features.py`` and ``docs/metrics.md``).

The table carries the settings file's SHA-256 on every row. An existing
``results/features.csv`` stamped with the current hash is reused; a changed
hash triggers a full recompute. Only aggregate summaries are printed (feature
means, ranges, and missing counts), never per-image detections or counts.

Usage::

    python scripts/build_features.py
    python scripts/build_features.py --settings configs/default_settings.json
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config, dataset, features, settings  # noqa: E402
from core import evaluation as ev  # noqa: E402


def image_paths_by_item(images_dir: Path) -> dict[int, Path]:
    """Map item numbers to their image files; the first sorted file wins."""
    if not images_dir.exists():
        raise FileNotFoundError(f"images folder missing: {images_dir}")
    paths: dict[int, Path] = {}
    for path in sorted(images_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in config.ALLOWED_EXTENSIONS:
            continue
        number = dataset.parse_item_number(path.stem.split("_")[0])
        if number is not None and number not in paths:
            paths[number] = path
    return paths


def main(argv: list[str] | None = None) -> int:
    """Compute (or reuse) the feature table and print aggregate summaries only."""
    parser = argparse.ArgumentParser(
        description="Label free reliability features from cached detections and image pixels.")
    parser.add_argument("--cache", default=str(config.RAW_DETECTIONS_JSONL),
                        help="benchmark JSONL cache produced by run_benchmark.py")
    parser.add_argument("--images-dir", default=str(config.TESTSET_IMAGES_DIR),
                        help="folder of test set images")
    parser.add_argument("--settings", default=str(config.DEFAULT_SETTINGS_JSON),
                        help="settings file whose hash is stamped on every row")
    parser.add_argument("--output", default=str(config.FEATURES_CSV),
                        help="feature table CSV (rewritten only when the hash changes)")
    args = parser.parse_args(argv)

    try:
        settings_hash = settings.settings_sha256(args.settings)
    except FileNotFoundError as error:
        print(f"ERROR: {error}")
        return 1

    output_path = Path(args.output)
    if not features.needs_recompute(output_path, settings_hash):
        print(f"Features already computed for settings sha256 {settings_hash}; reusing {output_path}")
        return 0

    cache = ev.load_detection_cache(Path(args.cache), min(config.THRESHOLD_SWEEP))
    if not cache:
        print(f"ERROR: no cached detections in {args.cache}; run scripts/run_benchmark.py first")
        return 1
    try:
        image_paths = image_paths_by_item(Path(args.images_dir))
    except FileNotFoundError as error:
        print(f"ERROR: {error}")
        return 1

    rows = features.build_feature_table(cache, image_paths, settings_hash)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(features.TABLE_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Features: {len(rows)} images written to {output_path}")
    print(f"Settings: {Path(args.settings).name} (sha256 {settings_hash})")
    print("Inputs: cached standard-mode detections and image pixels only; no labels read")
    print("Scope: every cached image, development subset included; the features use no labels")
    print("Aggregate summary (mean, range, missing per feature):")
    for column, stats in features.summarise_features(rows).items():
        mean = "n/a" if stats["mean"] is None else f"{stats['mean']:.6g}"
        low = "n/a" if stats["min"] is None else f"{stats['min']:.6g}"
        high = "n/a" if stats["max"] is None else f"{stats['max']:.6g}"
        print(f"  {column}: mean={mean}, range=[{low}, {high}], missing={stats['missing']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())