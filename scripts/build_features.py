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
import json
import sys
from pathlib import Path
from typing import Any

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


def _read_feature_rows(path: Path) -> list[dict[str, Any]]:
    """Read an existing feature table for aggregate-only reporting."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows: list[dict[str, Any]] = []
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = {
                "item_id": row["item_id"],
                "settings_sha256": row["settings_sha256"],
            }
            for column in features.TABLE_COLUMNS[2:]:
                value = row[column]
                parsed[column] = None if value == "" else float(value)
            rows.append(parsed)
    return rows


def _print_aggregate_summary(rows: list[dict[str, Any]]) -> None:
    """Print means, ranges, and missing values without exposing per-image data."""
    print("Aggregate summary (mean, range, missing per feature):")
    for column, stats in features.summarise_features(rows).items():
        mean = "n/a" if stats["mean"] is None else f"{stats['mean']:.6g}"
        low = "n/a" if stats["min"] is None else f"{stats['min']:.6g}"
        high = "n/a" if stats["max"] is None else f"{stats['max']:.6g}"
        print(f"  {column}: mean={mean}, range=[{low}, {high}], missing={stats['missing']}")


def _cache_validation_error(path: Path, settings_hash: str) -> str | None:
    """Require a settings-matched, low-threshold standard cache for all models."""
    if not path.is_file():
        return f"cached detections file missing: {path}"

    required_models = set(config.PRETRAINED_MODEL_KEYS)
    models_by_item: dict[int, set[str]] = {}
    found_matching_entries = False
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                return f"invalid JSON in cached detections at line {line_number}: {error}"
            if entry.get("settings_sha256") != settings_hash or entry.get("mode") != "standard":
                continue
            model = entry.get("model")
            item = ev.normalise_item_id(str(entry.get("image", "")).split("_")[0])
            if model not in required_models or item is None:
                continue
            found_matching_entries = True
            try:
                threshold = float(entry["threshold"])
            except (KeyError, TypeError, ValueError):
                return "matching cache entries lack a valid inference threshold"
            if not 0 <= threshold <= min(config.THRESHOLD_SWEEP):
                return "matching cache entries were not inferred at the lowest threshold sweep value"
            models_by_item.setdefault(item, set()).add(model)

    if not found_matching_entries:
        return "no standard-mode detections match the requested settings hash"
    if any(models != required_models for models in models_by_item.values()):
        return "matching cache does not contain all models for every image"
    return None


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
        _print_aggregate_summary(_read_feature_rows(output_path))
        return 0

    cache_path = Path(args.cache)
    cache_error = _cache_validation_error(cache_path, settings_hash)
    if cache_error is not None:
        print(f"ERROR: {cache_error}; run scripts/run_benchmark.py with the selected settings")
        return 1
    cache = ev.load_detection_cache(cache_path, min(config.THRESHOLD_SWEEP), settings_hash)
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

    _print_aggregate_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())