"""Cache and evaluate the isolated one-class fine-tuned rickshaw detector."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from core import config, dataset, evaluation, finetuned, settings
from core.detector import Detection, build_detector
from scripts.run_benchmark import image_file_hash, serialise_detections


MODEL_KEY = config.FINETUNED_MODEL_KEY
OUTPUT_SUMMARY = config.RESULTS_DIR / "finetuned_summary.json"
OUTPUT_EVALUATION = config.RESULTS_DIR / "finetuned_evaluation.csv"
OUTPUT_DEVELOPMENT = config.RESULTS_DIR / "finetuned_development.csv"
OUTPUT_BASE_CACHE = config.RESULTS_DIR / "finetuned_baseline_cache.jsonl"
MUTABLE_RESULT_PATHS = {
    config.RAW_DETECTIONS_JSONL,
    config.EVALUATION_SUMMARY_JSON,
    OUTPUT_SUMMARY,
    OUTPUT_EVALUATION,
    OUTPUT_DEVELOPMENT,
    OUTPUT_BASE_CACHE,
    config.RESULTS_DIR / "finetuned_second_pass_evaluation.csv",
    config.RESULTS_DIR / "finetuned_second_pass_development.csv",
}


def _read_cache_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _baseline_entry_digest(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        for line in path.read_bytes().splitlines(keepends=True):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("model") in config.PRETRAINED_MODEL_KEYS:
                digest.update(line)
    return digest.hexdigest()


def _existing_result_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    if not config.RESULTS_DIR.is_dir():
        return hashes
    for path in sorted(config.RESULTS_DIR.rglob("*")):
        if not path.is_file() or path in MUTABLE_RESULT_PATHS:
            continue
        hashes[str(path.relative_to(config.RESULTS_DIR))] = _sha256_file(path)
    return hashes


def _cache_finetuned_images(
    images: list[Path], settings_hash: str, cache_path: Path
) -> None:
    records = _read_cache_records(cache_path)
    cached = {
        (record.get("image_hash"), record.get("model"), record.get("mode"),
         record.get("settings_sha256"))
        for record in records
    }
    detector = build_detector(MODEL_KEY)
    detector.warmup()
    spec = config.MODEL_REGISTRY[MODEL_KEY]
    settings_record = json.loads(config.FINETUNED_SETTINGS_JSON.read_text(encoding="utf-8"))
    with cache_path.open("a", encoding="utf-8") as handle:
        for image_path in images:
            digest = image_file_hash(image_path)
            key = (digest, MODEL_KEY, "standard", settings_hash)
            if key in cached:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"could not decode test image: {image_path}")
            started = time.perf_counter()
            detections = detector.detect(
                image,
                float(settings_record["confidence"]),
                input_size=int(settings_record["input_size"]),
                nms_iou=float(settings_record["nms_iou"]),
            )
            height, width = image.shape[:2]
            record = {
                "image": image_path.name,
                "image_hash": digest,
                "model": MODEL_KEY,
                "model_name": spec.display_name,
                "mode": "standard",
                "threshold": float(settings_record["confidence"]),
                "settings_sha256": settings_hash,
                "weights_sha256": spec.weights_sha256,
                "width": width,
                "height": height,
                "load_seconds": getattr(detector, "load_seconds", None),
                "inference_ms": (time.perf_counter() - started) * 1000.0,
                "detections": serialise_detections(detections),
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            handle.flush()
            cached.add(key)


def _cache_missing_pretrained(
    item_ids: set[int],
    image_by_id: dict[int, Path],
    cache: dict[tuple[int, str, str], list[Detection]],
    settings_hash: str,
) -> None:
    missing: list[tuple[int, str]] = []
    for item in sorted(item_ids):
        image_path = image_by_id.get(item)
        if image_path is None:
            continue
        image_hash = image_file_hash(image_path)
        for model_key in config.PRETRAINED_MODEL_KEYS:
            if (item, model_key, "standard") not in cache:
                missing.append((item, model_key))
    if not missing:
        return

    path = OUTPUT_BASE_CACHE
    known = {
        (record.get("image_hash"), record.get("model"), record.get("mode"),
         record.get("settings_sha256"))
        for record in _read_cache_records(path)
    }
    detector_by_key = {}
    settings_by_key = json.loads(config.FINAL_SETTINGS_JSON.read_text(encoding="utf-8"))["models"]
    with path.open("a", encoding="utf-8") as handle:
        for item, model_key in missing:
            image_path = image_by_id[item]
            digest = image_file_hash(image_path)
            cache_key = (digest, model_key, "standard", settings_hash)
            if cache_key in known:
                continue
            detector = detector_by_key.get(model_key)
            if detector is None:
                detector = build_detector(model_key)
                detector.warmup()
                detector_by_key[model_key] = detector
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"could not decode test image: {image_path}")
            model_settings = settings_by_key[model_key]
            started = time.perf_counter()
            detections = detector.detect(
                image,
                float(model_settings["confidence"]),
                input_size=int(model_settings["input_size"]),
                nms_iou=model_settings.get("nms_iou"),
            )
            record = {
                "image": image_path.name,
                "image_hash": digest,
                "model": model_key,
                "model_name": config.MODEL_REGISTRY[model_key].display_name,
                "mode": "standard",
                "threshold": float(model_settings["confidence"]),
                "settings_sha256": settings_hash,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "load_seconds": getattr(detector, "load_seconds", None),
                "inference_ms": (time.perf_counter() - started) * 1000.0,
                "detections": serialise_detections(detections),
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            handle.flush()
            known.add(cache_key)
    new_cache = evaluation.load_detection_cache(path, settings_sha256=settings_hash)
    cache.update(new_cache)


def _cluster_mean_ci(
    values: list[float], scenes: list[str], seed_offset: int = 0
) -> dict[str, Any]:
    if not values:
        return {"mean": None, "ci95": [None, None], "n_images": 0, "n_scenes": 0}
    ci = evaluation.cluster_bootstrap_ci(
        values, scenes, samples=2000, seed=config.SEED + seed_offset
    )
    return {
        "mean": round(statistics.fmean(values), 4),
        "ci95": list(ci),
        "n_images": len(values),
        "n_scenes": len(set(scenes)),
    }


def _error_metrics(
    rows: list[dict[str, Any]], prediction_field: str, truth_field: str, seed: int
) -> dict[str, Any]:
    errors = [
        int(row[prediction_field]) - int(row[truth_field])
        for row in rows
    ]
    scenes = [str(row["scene"]) for row in rows]
    return {
        "signed_error": _cluster_mean_ci(errors, scenes, seed),
        "absolute_error": _cluster_mean_ci([abs(value) for value in errors], scenes, seed + 1),
    }


def _metrics_for_partition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    metric_fields = {
        "rickshaw_any": ("pred_rickshaw_any", "gt_rickshaw_any"),
        "covered_total": ("pred_covered_total", "gt_covered_total"),
        "all_vehicle_total": ("pred_all_vehicle_total", "gt_all_vehicle_total"),
        **{
            f"category_{name}": (f"pred_{name}", f"gt_{name}")
            for name in ("car", "bus", "truck", "motorcycle")
        },
    }
    for model_key in (*config.PRETRAINED_MODEL_KEYS, MODEL_KEY):
        model_rows = [row for row in rows if row["model"] == model_key]
        output[model_key] = {
            "pilot": True,
            "n_images": len(model_rows),
            "n_scenes": len({row["scene"] for row in model_rows}),
            "metrics": {
                name: _error_metrics(model_rows, *fields, seed=index * 2)
                for index, (name, fields) in enumerate(metric_fields.items(), start=1)
            },
        }
    return output


def _paired_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    targets = {
        "covered_total": ("pred_covered_total", "gt_covered_total"),
        "car": ("pred_car", "gt_car"),
        "bus": ("pred_bus", "gt_bus"),
        "truck": ("pred_truck", "gt_truck"),
        "motorcycle": ("pred_motorcycle", "gt_motorcycle"),
        "rickshaw_any": ("pred_rickshaw_any", "gt_rickshaw_any"),
    }
    for target, (pred_field, truth_field) in targets.items():
        paired = []
        for item in {row["item_id"] for row in rows}:
            fine = next(
                (r for r in rows if r["item_id"] == item and r["model"] == MODEL_KEY), None
            )
            base = next(
                (r for r in rows if r["item_id"] == item and r["model"] == "yolov8n"), None
            )
            if fine is None or base is None:
                continue
            truth = int(fine[truth_field])
            delta = abs(int(fine[pred_field]) - truth) - abs(int(base[pred_field]) - truth)
            paired.append((delta, str(fine["scene"])))
        deltas = [pair[0] for pair in paired]
        scenes = [pair[1] for pair in paired]
        comparisons[target] = {
            "pilot": True,
            "difference_definition": "fine-tuned absolute error minus pretrained YOLOv8n absolute error",
            "mean_difference": _cluster_mean_ci(deltas, scenes, len(comparisons) + 71),
            "images_improved": sum(value < 0 for value in deltas),
            "images_worsened": sum(value > 0 for value in deltas),
            "images_unchanged": sum(value == 0 for value in deltas),
        }
        interval = comparisons[target]["mean_difference"]["ci95"]
        comparisons[target]["regression_flag"] = (
            interval[0] is not None and interval[0] > 0
        )
    return comparisons


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["item_id", "scene", "model"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run() -> dict[str, Any]:
    """Run inference, contamination checks, and separated pilot summaries."""
    existing_results_before = _existing_result_hashes()
    settings_record = json.loads(config.FINETUNED_SETTINGS_JSON.read_text(encoding="utf-8"))
    base_settings_hash = settings.settings_sha256(config.FINAL_SETTINGS_JSON)
    if settings_record["base_settings_sha256"] != base_settings_hash:
        raise ValueError("fine-tuned settings do not match the current frozen base settings")
    fine_settings_hash = settings.settings_sha256(config.FINETUNED_SETTINGS_JSON)
    model_hash = config.MODEL_REGISTRY[MODEL_KEY].weights_sha256
    if not config.FINETUNED_MODEL_WEIGHTS.is_file():
        raise FileNotFoundError(f"fine-tuned weights not found: {config.FINETUNED_MODEL_WEIGHTS}")
    if not model_hash or _sha256_file(config.FINETUNED_MODEL_WEIGHTS) != model_hash:
        raise ValueError("fine-tuned weights SHA-256 does not match the registry")

    sources = {
        dataset.normalise_item_id(row["item_id"]): row
        for row in csv.DictReader(config.SOURCES_CSV.open(encoding="utf-8-sig", newline=""))
        if dataset.normalise_item_id(row.get("item_id", "")) is not None
    }
    images = sorted(
        path for path in config.TESTSET_IMAGES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in config.ALLOWED_EXTENSIONS
    )
    image_by_id = {
        dataset.parse_item_number(path.stem.split("_")[0]): path
        for path in images
        if dataset.parse_item_number(path.stem.split("_")[0]) is not None
    }
    training_images = finetuned.training_subset_images(
        config.PROJECT_ROOT / "runs" / "rickshaw_finetune" / "dataset" / "images" / "train",
        fraction=0.1,
    )
    contaminated = finetuned.find_training_contamination(
        config.TESTSET_IMAGES_DIR, training_images
    )
    contaminated_ids = set(contaminated)
    plan = json.loads(config.TESTSET_PLAN_JSON.read_text(encoding="utf-8"))
    development_ids = {
        dataset.normalise_item_id(value) for value in plan.get("development_subset", [])
    }
    development_ids.discard(None)
    labels = evaluation.load_labels(config.LABELS_CSV)
    labelled_ids = {dataset.format_item_id(item) for item, data in labels.items()
                    if data.get("pass1") is not None}
    evaluation_ids = set(sources) - development_ids
    eligible_dev, eligible_eval = finetuned.partition_labelled_items(
        labelled_ids, development_ids, contaminated_ids
    )
    eligible_eval_ids = sorted(eligible_eval & evaluation_ids)
    eligible_dev_ids = sorted(eligible_dev)

    cache_path = config.RAW_DETECTIONS_JSONL
    baseline_digest_before = _baseline_entry_digest(cache_path)
    _cache_finetuned_images(images, fine_settings_hash, cache_path)

    baseline_cache = evaluation.load_detection_cache(
        cache_path, confidence=config.DEFAULT_CONFIDENCE, settings_sha256=base_settings_hash
    )
    needed_items = {
        dataset.parse_item_number(item) for item in eligible_eval_ids + eligible_dev_ids
    }
    needed_items.discard(None)
    _cache_missing_pretrained(needed_items, image_by_id, baseline_cache, base_settings_hash)
    supplementary_cache = evaluation.load_detection_cache(
        OUTPUT_BASE_CACHE, confidence=config.DEFAULT_CONFIDENCE,
        settings_sha256=base_settings_hash,
    )
    baseline_cache.update(supplementary_cache)

    fine_cache: dict[int, list[Detection]] = {}
    for record in _read_cache_records(cache_path):
        if record.get("model") != MODEL_KEY or record.get("settings_sha256") != fine_settings_hash:
            continue
        item = dataset.parse_item_number(str(record.get("image", "")).split("_")[0])
        if item is None:
            continue
        fine_cache[item] = [
            Detection(
                int(det["class_id"]), str(det["class_name"]), float(det["confidence"]),
                tuple(float(value) for value in det["xyxy"]),
            )
            for det in record.get("detections", [])
            if float(det.get("confidence", 0.0)) >= float(settings_record["confidence"])
        ]

    def partition_rows(item_ids: list[str], pass_key: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item_id in item_ids:
            item = dataset.parse_item_number(item_id)
            if item is None or item not in labels:
                continue
            truth = labels[item].get(pass_key)
            if truth is None:
                continue
            source = sources.get(item_id, {})
            gt = {name: int(truth.get(name, 0)) for name in config.LABEL_CLASS_NAMES}
            for model_key in (*config.PRETRAINED_MODEL_KEYS, MODEL_KEY):
                if model_key == MODEL_KEY:
                    detections = fine_cache.get(item, [])
                else:
                    detections = baseline_cache.get((item, model_key, "standard"), [])
                predicted = {name: 0 for name in config.LABEL_CLASS_NAMES}
                if model_key == MODEL_KEY:
                    predicted["rickshaw"] = sum(d.class_name == "rickshaw" for d in detections)
                else:
                    for detection in detections:
                        if detection.class_name in predicted:
                            predicted[detection.class_name] += 1
                gt_covered = sum(gt[name] for name in ("car", "bus", "truck", "motorcycle"))
                pred_covered = sum(predicted[name] for name in ("car", "bus", "truck", "motorcycle"))
                gt_rickshaw = gt["rickshaw"] + gt["loader_rickshaw"]
                pred_rickshaw = predicted["rickshaw"] + predicted["loader_rickshaw"]
                rows.append({
                    "item_id": item_id,
                    "scene": str(source.get("scene") or item_id),
                    "partition": "development" if item_id in development_ids else "evaluation",
                    "reference_pass": pass_key,
                    "model": model_key,
                    **{f"gt_{name}": gt[name] for name in config.LABEL_CLASS_NAMES},
                    **{f"pred_{name}": predicted[name] for name in config.LABEL_CLASS_NAMES},
                    "gt_rickshaw_any": gt_rickshaw,
                    "pred_rickshaw_any": pred_rickshaw,
                    "gt_covered_total": gt_covered,
                    "pred_covered_total": pred_covered,
                    "gt_all_vehicle_total": gt_covered + gt_rickshaw,
                    "pred_all_vehicle_total": pred_covered + pred_rickshaw,
                    "condition_tags": str(source.get("condition_tags") or ""),
                })
        return rows

    first_eval = partition_rows(eligible_eval_ids, "pass1")
    first_dev = partition_rows(eligible_dev_ids, "pass1")
    second_eval_ids = [
        dataset.format_item_id(item) for item, data in labels.items()
        if data.get("pass1") is not None and data.get("pass2") is not None
        and dataset.format_item_id(item) in evaluation_ids
        and dataset.format_item_id(item) not in contaminated_ids
    ]
    second_dev_ids = [
        dataset.format_item_id(item) for item, data in labels.items()
        if data.get("pass1") is not None and data.get("pass2") is not None
        and dataset.format_item_id(item) in development_ids
        and dataset.format_item_id(item) not in contaminated_ids
    ]
    second_eval = partition_rows(sorted(second_eval_ids), "pass2")
    second_dev = partition_rows(sorted(second_dev_ids), "pass2")
    _write_rows(OUTPUT_EVALUATION, first_eval)
    _write_rows(OUTPUT_DEVELOPMENT, first_dev)
    _write_rows(config.RESULTS_DIR / "finetuned_second_pass_evaluation.csv", second_eval)
    _write_rows(config.RESULTS_DIR / "finetuned_second_pass_development.csv", second_dev)

    baseline_digest_after = _baseline_entry_digest(cache_path)
    if baseline_digest_after != baseline_digest_before:
        raise RuntimeError("pretrained cache entries changed while adding the fine-tuned model")
    existing_results_after = _existing_result_hashes()
    if existing_results_after != existing_results_before:
        raise RuntimeError("an unrelated pre-existing results file changed")

    summary = {
        "label": "pilot",
        "weights_sha256": model_hash,
        "settings_sha256": fine_settings_hash,
        "base_settings_sha256": base_settings_hash,
        "pretrained_cache_entries_sha256": {
            "before": baseline_digest_before,
            "after": baseline_digest_after,
            "unchanged": baseline_digest_before == baseline_digest_after,
        },
        "unrelated_result_files_unchanged": True,
        "training_data_note": settings_record["training_data_note"],
        "contamination": {
            "method": "sorted first 10% of Ultralytics training split; core.dataset.perceptual_hash; Hamming distance < 8",
            "training_image_count": len(training_images),
            "test_image_count": len(images),
            "overlap_count": len(contaminated_ids),
            "overlapping_item_ids": sorted(contaminated_ids),
            "excluded_evaluation_count": len(contaminated_ids & evaluation_ids),
            "excluded_development_count": len(contaminated_ids & development_ids),
        },
        "evaluation_first_pass_n": len(eligible_eval_ids),
        "evaluation_first_pass_scenes": len({
            str(sources[item].get("scene") or item) for item in eligible_eval_ids
        }),
        "development_first_pass_n": len(eligible_dev_ids),
        "development_label": "settings were tuned here, results are optimistic",
        "first_pass_evaluation": _metrics_for_partition(first_eval),
        "second_pass_evaluation": _metrics_for_partition(second_eval),
        "development_first_pass": _metrics_for_partition(first_dev),
        "development_second_pass": _metrics_for_partition(second_dev),
        "no_regression_evaluation": _paired_comparison(first_eval),
        "no_regression_development": _paired_comparison(first_dev),
        "limitations": [
            "Fine tuning used a subset derived from the same user-supplied archive as many test images; overlapping test images are excluded.",
            "The uncontaminated audited evaluation sample is small and is a pilot only.",
            "Training data licence and citation were not provided in the supplied task input.",
            "This detector has one rickshaw class and cannot distinguish loader rickshaws.",
        ],
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    existing_summary = json.loads(config.EVALUATION_SUMMARY_JSON.read_text(encoding="utf-8"))
    existing_summary["finetuned_model"] = {
        "model_key": MODEL_KEY,
        "weights_sha256": model_hash,
        "settings_file": config.FINETUNED_SETTINGS_JSON.name,
        "settings_sha256": fine_settings_hash,
    }
    config.EVALUATION_SUMMARY_JSON.write_text(
        json.dumps(existing_summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    result = run()
    print(
        f"Pilot complete: evaluation n={result['evaluation_first_pass_n']}, "
        f"scenes={result['evaluation_first_pass_scenes']}; "
        f"development n={result['development_first_pass_n']} (reported separately)."
    )
    print(
        f"Contamination: {result['contamination']['overlap_count']} of "
        f"{result['contamination']['test_image_count']} test images excluded."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
