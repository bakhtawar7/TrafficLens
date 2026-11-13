"""Coarse, label-supervised settings sweep on the guarded development subset.

The script reads only images authorized by ``core.dev_guard`` and caches every
per-image inference in ``results/tune_dev_cache.jsonl``. It never uses sliced
inference or the evaluation images.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config, dataset, dev_guard, settings
from core.detector import Detection, build_detector

CONFIDENCE_GRID: tuple[float, ...] = (0.15, 0.25, 0.35, 0.50)
INPUT_SIZE_GRID: tuple[int, ...] = (640, 1280)
YOLO_NMS_IOU_GRID: tuple[float, ...] = (0.5, 0.6, 0.7)
BOOTSTRAP_SAMPLES = 2000
CACHE_VERSION = 1
DEV_CACHE = config.RESULTS_DIR / "tune_dev_cache.jsonl"
DEV_TABLE = config.RESULTS_DIR / "tune_dev_table.csv"
FINAL_SETTINGS = config.FINAL_SETTINGS_JSON

TABLE_COLUMNS: tuple[str, ...] = (
    "model",
    "confidence",
    "input_size",
    "uses_nms",
    "nms_iou",
    "mean_absolute_error",
    "mean_signed_error",
    "mean_inference_ms",
    "mae_ci95_low",
    "mae_ci95_high",
    "images",
)


def image_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def settings_grid(model_key: str) -> list[dict[str, Any]]:
    uses_nms = config.MODEL_REGISTRY[model_key].family == "yolo"
    iou_values: tuple[float | None, ...] = (
        YOLO_NMS_IOU_GRID if uses_nms else (None,)
    )
    return [
        {
            "confidence": confidence,
            "input_size": input_size,
            "uses_nms": uses_nms,
            "nms_iou": nms_iou,
        }
        for confidence, input_size, nms_iou in itertools.product(
            CONFIDENCE_GRID, INPUT_SIZE_GRID, iou_values
        )
    ]


def _cache_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        entry.get("grid_version"),
        entry.get("image_hash"),
        entry.get("model"),
        entry.get("confidence"),
        entry.get("input_size"),
        entry.get("nms_iou"),
    )


def load_dev_cache(path: Path, dev_ids: set[str]) -> dict[tuple[Any, ...], dict[str, Any]]:
    cached: dict[tuple[Any, ...], dict[str, Any]] = {}
    if not path.exists():
        return cached
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid tuning cache JSON at line {line_number}") from error
            if entry.get("item_id") in dev_ids and entry.get("grid_version") == CACHE_VERSION:
                cached[_cache_key(entry)] = entry
    return cached


def _raw_detections(payload: list[dict[str, Any]]) -> list[Detection]:
    return [
        Detection(
            int(entry["class_id"]),
            str(entry["class_name"]),
            float(entry["confidence"]),
            tuple(float(value) for value in entry["xyxy"]),
        )
        for entry in payload
    ]


def _serialize_detections(detections: list[Detection]) -> list[dict[str, Any]]:
    return [
        {
            "class_id": detection.class_id,
            "class_name": detection.class_name,
            "confidence": detection.confidence,
            "xyxy": list(detection.xyxy),
        }
        for detection in detections
    ]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap_interval(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires at least one development image")
    rng = random.Random(seed)
    means = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(samples)
    )
    return (
        means[int(0.025 * samples)],
        means[min(int(0.975 * samples), samples - 1)],
    )


def _setting_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        float(row["confidence"]),
        int(row["input_size"]),
        -1.0 if row["nms_iou"] is None else float(row["nms_iou"]),
    )


def select_setting(
    rows: list[dict[str, Any]],
    defaults: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], float, dict[str, Any]]:
    """Apply minimum MAE, best-CI tie set, normalized default distance, then speed."""
    if not rows:
        raise ValueError("cannot select a setting from an empty result table")
    best = min(rows, key=lambda row: (row["mean_absolute_error"], _setting_sort_key(row)))
    lower, upper = best["mae_ci95"]
    tied = [
        row for row in rows
        if lower <= row["mean_absolute_error"] <= upper
    ]
    if best not in tied:
        tied.append(best)

    varied: list[tuple[str, float]] = [
        ("confidence", max(CONFIDENCE_GRID) - min(CONFIDENCE_GRID)),
        ("input_size", max(INPUT_SIZE_GRID) - min(INPUT_SIZE_GRID)),
    ]
    if defaults.get("uses_nms"):
        varied.append(("nms_iou", max(YOLO_NMS_IOU_GRID) - min(YOLO_NMS_IOU_GRID)))

    def distance(row: dict[str, Any]) -> float:
        return sum(
            abs(float(row[field]) - float(defaults[field])) / scale
            for field, scale in varied
        )

    minimum_distance = min(distance(row) for row in tied)
    closest = [row for row in tied if abs(distance(row) - minimum_distance) < 1e-12]
    chosen = min(
        closest,
        key=lambda row: (row["mean_inference_ms"], _setting_sort_key(row)),
    )
    return chosen, tied, minimum_distance, best


def _read_first_pass(
    labels_path: Path, dev_ids: set[str]
) -> dict[str, dict[str, int]]:
    with labels_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels: dict[str, dict[str, int]] = {}
    for row in rows:
        item_id = (row.get("item_id") or "").strip().lower()
        if item_id not in dev_ids or (row.get("labeller_pass") or "").strip() != "1":
            continue
        if item_id in labels:
            raise ValueError(f"multiple first-pass label rows for {item_id}")
        counts: dict[str, int] = {}
        for category in config.LABEL_CLASS_NAMES:
            value = (row.get(category) or "").strip()
            if not value.isdigit():
                raise ValueError(f"missing or invalid {category} first-pass count for {item_id}")
            counts[category] = int(value)
        labels[item_id] = counts
    missing = sorted(dev_ids - set(labels))
    if missing:
        raise ValueError(
            "development images missing audited first-pass counts: " + ", ".join(missing)
        )
    return labels


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{column: row.get(column) for column in TABLE_COLUMNS},
                "mean_absolute_error": f"{row['mean_absolute_error']:.6f}",
                "mean_signed_error": f"{row['mean_signed_error']:.6f}",
                "mean_inference_ms": f"{row['mean_inference_ms']:.3f}",
                "mae_ci95_low": f"{row['mae_ci95'][0]:.6f}",
                "mae_ci95_high": f"{row['mae_ci95'][1]:.6f}",
                "images": len(row["errors"]),
            })


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = (
        "Model", "Conf", "Img", "NMS", "IoU", "MAE", "Signed", "Infer ms", "MAE 95% CI"
    )
    formatted = [
        (
            row["model"],
            f"{row['confidence']:.2f}",
            str(row["input_size"]),
            "yes" if row["uses_nms"] else "no",
            "—" if row["nms_iou"] is None else f"{row['nms_iou']:.1f}",
            f"{row['mean_absolute_error']:.4f}",
            f"{row['mean_signed_error']:.4f}",
            f"{row['mean_inference_ms']:.1f}",
            f"[{row['mae_ci95'][0]:.4f}, {row['mae_ci95'][1]:.4f}]",
        )
        for row in rows
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in formatted))
        for index, header in enumerate(headers)
    ]
    print(" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in formatted:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tune confidence, input size, and YOLO NMS IoU on development images only."
    )
    parser.add_argument("--cache", type=Path, default=DEV_CACHE)
    parser.add_argument("--table", type=Path, default=DEV_TABLE)
    parser.add_argument("--settings-out", type=Path, default=FINAL_SETTINGS)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    args = parser.parse_args(argv)
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    if args.settings_out.exists():
        print(f"ERROR: refusing to overwrite existing settings file: {args.settings_out}")
        return 1

    dev_ids = {f"tl{number:04d}" for number in dev_guard.development_ids()}
    if len(dev_ids) != config.DEV_SUBSET_SIZE:
        print(
            f"ERROR: configured development subset has {len(dev_ids)} items; "
            f"expected {config.DEV_SUBSET_SIZE}"
        )
        return 1
    try:
        labels = _read_first_pass(config.LABELS_CSV, dev_ids)
        paths = dev_guard.development_image_paths()
        paths_by_id = {
            f"tl{dataset.parse_item_number(path.stem.split('_')[0]):04d}": path
            for path in paths
        }
        if set(paths_by_id) != dev_ids:
            raise ValueError(
                "development image files do not match configured IDs; "
                f"missing={sorted(dev_ids - set(paths_by_id))}, "
                f"extra={sorted(set(paths_by_id) - dev_ids)}"
            )
        images = {
            item_id: dev_guard.read_development_image(path)
            for item_id, path in paths_by_id.items()
        }
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1

    truths = {
        item_id: sum(counts[name] for name in config.TARGET_CLASS_NAMES)
        for item_id, counts in labels.items()
    }
    default_settings = settings.load_settings(config.DEFAULT_SETTINGS_JSON)["models"]
    if set(default_settings) != set(config.PRETRAINED_MODEL_KEYS):
        print("ERROR: default_settings.json does not cover every registered model.")
        return 1
    digests = {item_id: image_hash(paths_by_id[item_id]) for item_id in dev_ids}
    cache = load_dev_cache(args.cache, dev_ids)
    expected_cache_keys = 0
    for model_key in config.PRETRAINED_MODEL_KEYS:
        expected_cache_keys += len(settings_grid(model_key)) * len(dev_ids)
    print(
        f"Development images: {len(dev_ids)}; cached setting/image runs: "
        f"{len(cache)}; planned runs: {expected_cache_keys}"
    )
    print("Mean signed error is prediction minus audited covered total.")
    print("RT-DETR-L uses NMS: no; NMS IoU is not applicable.")

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []
    for model_key in config.PRETRAINED_MODEL_KEYS:
        grid = settings_grid(model_key)
        model_rows: list[dict[str, Any]] = []
        pending = [
            (setting, item_id)
            for setting in grid
            for item_id in sorted(dev_ids)
            if _cache_key({
                "grid_version": CACHE_VERSION,
                "image_hash": digests[item_id],
                "model": model_key,
                **setting,
            }) not in cache
        ]
        detector = None
        if pending:
            detector = build_detector(model_key)
            detector.warmup()
        with args.cache.open("a", encoding="utf-8") as cache_handle:
            for setting_index, setting in enumerate(grid, start=1):
                per_image: list[tuple[str, dict[str, Any]]] = []
                for item_id in sorted(dev_ids):
                    identity = {
                        "grid_version": CACHE_VERSION,
                        "item_id": item_id,
                        "image": paths_by_id[item_id].name,
                        "image_hash": digests[item_id],
                        "model": model_key,
                        **setting,
                    }
                    key = _cache_key(identity)
                    entry = cache.get(key)
                    if entry is None:
                        if detector is None:
                            raise RuntimeError("model was not initialized for a missing cache entry")
                        start = time.perf_counter()
                        detections = detector.detect(
                            images[item_id],
                            setting["confidence"],
                            input_size=setting["input_size"],
                            nms_iou=setting["nms_iou"],
                        )
                        inference_ms = (time.perf_counter() - start) * 1000.0
                        entry = {
                            **identity,
                            "inference_ms": inference_ms,
                            "detections": _serialize_detections(detections),
                        }
                        cache_handle.write(json.dumps(entry) + "\n")
                        cache_handle.flush()
                        cache[key] = entry
                    per_image.append((item_id, entry))
                errors: list[float] = []
                signed_errors: list[float] = []
                times: list[float] = []
                for item_id, entry in per_image:
                    detections = _raw_detections(entry["detections"])
                    prediction = sum(
                        1 for detection in detections
                        if detection.class_name in config.TARGET_CLASS_NAMES
                    )
                    signed = prediction - truths[item_id]
                    errors.append(float(abs(signed)))
                    signed_errors.append(float(signed))
                    times.append(float(entry["inference_ms"]))
                interval = _bootstrap_interval(
                    errors, args.bootstrap_samples, config.SEED
                )
                model_rows.append({
                    "model": model_key,
                    **setting,
                    "mean_absolute_error": sum(errors) / len(errors),
                    "mean_signed_error": sum(signed_errors) / len(signed_errors),
                    "mean_inference_ms": sum(times) / len(times),
                    "mae_ci95": interval,
                    "errors": errors,
                })
                print(
                    f"{model_key}: setting {setting_index}/{len(grid)} cached/evaluated "
                    f"({len(per_image)} development images)"
                )

        default_record = default_settings[model_key]
        chosen, tied, distance, best = select_setting(model_rows, default_record)
        best_ci = best["mae_ci95"]
        chosen["rationale"] = (
            f"Best MAE {best['mean_absolute_error']:.4f} with 95% image-bootstrap CI "
            f"[{best_ci[0]:.4f}, {best_ci[1]:.4f}]; {len(tied)} setting(s) tied by "
            f"that interval; selected MAE {chosen['mean_absolute_error']:.4f} at "
            f"normalized L1 distance {distance:.4f} from defaults, then "
            f"mean inference time {chosen['mean_inference_ms']:.1f} ms."
        )
        result_rows.extend(model_rows)
        print(
            f"Selection {model_key}: best MAE={min(r['mean_absolute_error'] for r in model_rows):.4f}; "
            f"best setting CI95=[{best_ci[0]:.4f}, {best_ci[1]:.4f}]; "
            f"tied={len(tied)}; chosen conf={chosen['confidence']:.2f}, "
            f"input={chosen['input_size']}, NMS IoU={chosen['nms_iou']}; "
            f"default distance={distance:.4f}; time={chosen['mean_inference_ms']:.1f} ms"
        )
    _write_table(args.table, result_rows)
    _print_table(result_rows)

    final_models: dict[str, dict[str, Any]] = {}
    for model_key in config.PRETRAINED_MODEL_KEYS:
        rows = [row for row in result_rows if row["model"] == model_key]
        defaults = default_settings[model_key]
        chosen = select_setting(rows, defaults)[0]
        final_models[model_key] = {
            "confidence": chosen["confidence"],
            "input_size": chosen["input_size"],
            "uses_nms": chosen["uses_nms"],
            "nms_iou": chosen["nms_iou"],
            "sliced": False,
            "rationale": chosen["rationale"],
        }
    final_settings = {
        "_note": (
            "Selected once from the seeded ten-image development subset; frozen before "
            "evaluation. Standard inference was used for tuning. The sliced benchmark arm "
            "reuses these model settings with the default tile settings below."
        ),
        "selection_rule": (
            "Minimum mean absolute covered-count error; tied settings fall within the "
            "minimum-MAE setting's 95% image-level bootstrap interval; then normalized "
            "L1 distance to configs/default_settings.json; then faster mean inference."
        ),
        "development_images": sorted(dev_ids),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_unit": "image",
        "models": final_models,
        "sliced_tiles": {
            "tile_size": config.SLICE_TILE_SIZE,
            "overlap_ratio": config.SLICE_OVERLAP_RATIO,
            "iou_threshold": config.SLICE_IOU_THRESHOLD,
        },
    }
    args.settings_out.parent.mkdir(parents=True, exist_ok=True)
    with args.settings_out.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(final_settings, handle, indent=2)
        handle.write("\n")
    args.settings_out.chmod(0o444)
    print(f"Table: {args.table}")
    print(f"Development cache: {args.cache}")
    print(f"Frozen settings: {args.settings_out} (sha256 {_sha(args.settings_out)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
