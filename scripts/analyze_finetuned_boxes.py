"""Aggregate annotation-referenced rickshaw-box outcomes on clean evaluation images."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from core import config, dataset
from scripts.run_finetuned_evaluation import _read_cache_records, _sha256_file
from core.settings import settings_sha256

ARCHIVE = Path(r"C:\Users\bakht\Downloads\archive.zip")
OUTPUT = config.RESULTS_DIR / "finetuned_box_analysis.json"
RICKSHAW_LABEL_PARTS = ("rickshaw", "auto", "cng", "three wheeler")
OUTCOME_NAMES = ("rickshaw", "car", "motorcycle", "truck", "bus", "other", "missed")


def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _gt_rickshaw_boxes(xml_bytes: bytes) -> list[tuple[float, float, float, float]]:
    root = ET.fromstring(xml_bytes)
    boxes = []
    for obj in root.findall(".//object"):
        label = " ".join((obj.findtext("name") or "").casefold().replace("_", " ").split())
        if not any(term in label for term in RICKSHAW_LABEL_PARTS):
            continue
        if "loader" in label or "cargo" in label:
            continue
        node = obj.find("bndbox")
        if node is None:
            continue
        try:
            box = tuple(
                float(node.findtext(key) or "nan")
                for key in ("xmin", "ymin", "xmax", "ymax")
            )
        except ValueError:
            continue
        if all(map(lambda value: value == value, box)) and box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
    return boxes


def _outcomes(
    ground_truth: list[tuple[float, float, float, float]],
    detections: list[dict[str, Any]],
) -> dict[str, int]:
    pairs = sorted(
        (
            (_iou(box, tuple(float(value) for value in detection["xyxy"])), gt_index, det_index)
            for gt_index, box in enumerate(ground_truth)
            for det_index, detection in enumerate(detections)
        ),
        reverse=True,
    )
    assigned_gt: set[int] = set()
    assigned_det: set[int] = set()
    outcomes = {name: 0 for name in OUTCOME_NAMES}
    for overlap, gt_index, det_index in pairs:
        if overlap < 0.5:
            break
        if gt_index in assigned_gt or det_index in assigned_det:
            continue
        assigned_gt.add(gt_index)
        assigned_det.add(det_index)
        name = str(detections[det_index]["class_name"]).casefold()
        category = "rickshaw" if name == "rickshaw" else name
        outcomes[category if category in outcomes else "other"] += 1
    outcomes["missed"] = len(ground_truth) - len(assigned_gt)
    return outcomes


def run() -> dict[str, Any]:
    if not ARCHIVE.is_file():
        raise FileNotFoundError(f"source archive not found: {ARCHIVE}")
    model_hash = _sha256_file(config.FINETUNED_MODEL_WEIGHTS)
    if model_hash != config.MODEL_REGISTRY[config.FINETUNED_MODEL_KEY].weights_sha256:
        raise ValueError("fine-tuned weights hash does not match the model registry")
    fine_settings_hash = settings_sha256(config.FINETUNED_SETTINGS_JSON)
    base_settings_hash = settings_sha256(config.FINAL_SETTINGS_JSON)
    sources = {
        row["item_id"]: row
        for row in csv.DictReader(config.SOURCES_CSV.open(encoding="utf-8-sig", newline=""))
    }
    dev = set(json.loads(config.TESTSET_PLAN_JSON.read_text(encoding="utf-8"))["development_subset"])
    contamination = json.loads(
        (config.RESULTS_DIR / "finetuned_summary.json").read_text(encoding="utf-8")
    )["contamination"]
    contaminated = set(contamination["overlapping_item_ids"])
    eligible = sorted(set(sources) - dev - contaminated)

    manifest = list(csv.DictReader(
        (config.PROJECT_ROOT / "runs" / "rickshaw_finetune" / "dataset" / "manifest.csv")
        .open(encoding="utf-8-sig", newline="")
    ))
    image_root = config.PROJECT_ROOT / "runs" / "rickshaw_finetune" / "dataset" / "images"
    prepared_sources: list[tuple[str, dict[str, str], int, int]] = []
    for row in manifest:
        source_stem = PurePosixPath(row["archive_image"]).stem
        image_suffix = PurePosixPath(row["archive_image"]).suffix.lower()
        destination_hash = hashlib.sha256(
            row["archive_image"].encode("utf-8")
        ).hexdigest()[:10]
        prepared_path = (
            image_root / row["split"]
            / f"{source_stem}_{destination_hash}{image_suffix}"
        )
        prepared_image = cv2.imread(str(prepared_path))
        if prepared_image is None:
            continue
        prepared_sources.append((
            dataset.perceptual_hash(prepared_image),
            row,
            prepared_image.shape[1],
            prepared_image.shape[0],
        ))

    detections_by_model: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in _read_cache_records(config.RAW_DETECTIONS_JSONL):
        if record.get("mode") != "standard":
            continue
        model_key = str(record.get("model"))
        expected_hash = fine_settings_hash if model_key == config.FINETUNED_MODEL_KEY else base_settings_hash
        if record.get("settings_sha256") != expected_hash:
            continue
        detections_by_model[(str(record.get("image", "")).split("_")[0], model_key)] = record.get("detections", [])

    result = {
        "label": "annotation referenced, unverified; may be contaminated",
        "iou_threshold": 0.5,
        "excluded_known_contamination": True,
        "eligible_clean_evaluation_images": len(eligible),
        "source_linked_images": 0,
        "annotation_dimensions_verified": 0,
        "skipped_source_or_dimension_mismatch": 0,
        "models": {},
    }
    with zipfile.ZipFile(ARCHIVE) as archive:
        members = {name.lower(): name for name in archive.namelist()}
        accumulators = {
            key: {"outcomes": {name: 0 for name in OUTCOME_NAMES}, "boxes": 0, "scenes": set()}
            for key in ("yolov8n", config.FINETUNED_MODEL_KEY)
        }
        for item_id in eligible:
            source = sources[item_id]
            image_path = config.TESTSET_IMAGES_DIR / source["filename"]
            image = cv2.imread(str(image_path))
            if image is None:
                result["skipped_source_or_dimension_mismatch"] += 1
                continue
            image_hash = dataset.perceptual_hash(image)
            matches = []
            for source_hash, row, width, height in prepared_sources:
                distance = dataset.hash_distance(image_hash, source_hash)
                if distance < dataset.DUPLICATE_HASH_DISTANCE:
                    matches.append((distance, row, width, height))
            if not matches:
                result["skipped_source_or_dimension_mismatch"] += 1
                continue
            best_distance = min(match[0] for match in matches)
            best_matches = [match for match in matches if match[0] == best_distance]
            if best_distance >= dataset.DUPLICATE_HASH_DISTANCE or len(best_matches) != 1:
                result["skipped_source_or_dimension_mismatch"] += 1
                continue
            _, source_row, source_width, source_height = best_matches[0]
            if (source_width, source_height) != (image.shape[1], image.shape[0]):
                result["skipped_source_or_dimension_mismatch"] += 1
                continue
            annotation_path = source_row["archive_annotation"]
            annotation_member = members.get(annotation_path.lower())
            if annotation_member is None:
                result["skipped_source_or_dimension_mismatch"] += 1
                continue
            annotation = ET.fromstring(archive.read(annotation_member))
            size = annotation.find("size")
            if (
                size is None
                or int(size.findtext("width") or 0) != image.shape[1]
                or int(size.findtext("height") or 0) != image.shape[0]
            ):
                result["skipped_source_or_dimension_mismatch"] += 1
                continue
            ground_truth = _gt_rickshaw_boxes(ET.tostring(annotation))
            result["source_linked_images"] += 1
            result["annotation_dimensions_verified"] += 1
            if not ground_truth:
                continue
            scene = str(source.get("scene") or item_id)
            for model_key, accumulator in accumulators.items():
                image_key = image_path.name.split("_")[0]
                detections = detections_by_model.get((image_key, model_key), [])
                if model_key == "yolov8n":
                    detections = [
                        detection for detection in detections
                        if float(detection.get("confidence", 0.0)) >= config.DEFAULT_CONFIDENCE
                        and detection.get("class_name") in config.TARGET_CLASS_NAMES
                    ]
                else:
                    detections = [
                        detection for detection in detections
                        if float(detection.get("confidence", 0.0)) >= 0.25
                    ]
                outcomes = _outcomes(ground_truth, detections)
                for outcome, count in outcomes.items():
                    accumulator["outcomes"][outcome] += count
                accumulator["boxes"] += len(ground_truth)
                accumulator["scenes"].add(scene)
    for model_key, accumulator in accumulators.items():
        box_count = accumulator["boxes"]
        result["models"][model_key] = {
            "n_boxes": box_count,
            "n_scenes": len(accumulator["scenes"]),
            "outcome_counts": accumulator["outcomes"],
            "outcome_shares": {
                name: (count / box_count if box_count else None)
                for name, count in accumulator["outcomes"].items()
            },
        }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
