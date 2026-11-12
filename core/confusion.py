"""Exploratory box-level comparison with unverified source annotations.

This module compares source VOC boxes with freshly inferred detector boxes. It does not
change or select inference settings, and every written artefact carries the
annotation-unverified and exploratory caveats.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import config, curation, dataset

ANNOTATION_STATUS = "annotation referenced, unverified"
EXPLORATORY_STATUS = "exploratory"
TARGET_CATEGORIES = ("rickshaw", "loader_rickshaw", "van", "pickup")
DETECTOR_CLASSES = ("car", "bus", "truck", "motorcycle")
OUTCOMES = (*DETECTOR_CLASSES, "another class", "low confidence", "not seen")
IOU_THRESHOLD = 0.5
DEFAULT_CONFIDENCE = config.DEFAULT_CONFIDENCE
LOW_CONFIDENCE_FLOOR = min(config.THRESHOLD_SWEEP)
AUDIT_BOX_COUNT = 30
AUDIT_SEED = 20261113
BOOTSTRAP_SAMPLES = 2000
REQUIRED_MODELS = tuple(config.MODEL_REGISTRY)


@dataclass(frozen=True)
class AnnotationBox:
    """One selected dataset annotation with source identity and scene."""

    box_id: str
    item_id: str
    image_path: Path
    relative_image: str
    dataset_class: str
    category: str
    xyxy: tuple[float, float, float, float]
    scene: str


@dataclass(frozen=True)
class DetectionBox:
    """A detector box with its reported class and confidence."""

    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


def intersection_over_union(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Return IoU for two xyxy boxes; degenerate boxes have zero overlap."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _mapped_category(source_class: str, mapping: dict[str, str]) -> str | None:
    normalized = dataset.normalise_label(source_class)
    if normalized in {"van", "pickup"}:
        return normalized
    mapped = mapping.get(normalized)
    return mapped if mapped in {"rickshaw", "loader_rickshaw"} else None


def _verified_category(
    verified_class: str, mapping: dict[str, str]
) -> str:
    normalized = dataset.normalise_label(verified_class)
    valid = {*DETECTOR_CLASSES, "rickshaw", "loader_rickshaw", "other", "ignore", "van", "pickup"}
    category = normalized if normalized in {"van", "pickup"} else mapping.get(normalized, normalized)
    if category not in valid:
        raise ValueError(f"unrecognised verified class '{verified_class}'")
    return category


def _box_id(relative_image: str, ordinal: int, box: tuple[float, float, float, float]) -> str:
    payload = json.dumps([relative_image, ordinal, [round(v, 3) for v in box]], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_selected_annotations(
    source_root: Path,
    inventory_csv: Path,
    selection_csv: Path,
    plan_json: Path,
    sources_csv: Path,
    mapping: dict[str, str],
    records: list[curation.AnnotationRecord] | None = None,
) -> tuple[list[AnnotationBox], dict[str, Any]]:
    """Load target-category VOC boxes from selected source images."""
    source_root = Path(source_root).expanduser().resolve()
    with Path(selection_csv).open("r", encoding="utf-8", newline="") as handle:
        selected = {
            dataset.parse_item_number(row.get("item_id", "")): row.get("original_filename", "")
            for row in csv.DictReader(handle)
            if row.get("decision") == "selected"
        }
    selected = {number: name for number, name in selected.items() if number is not None}
    plan = json.loads(Path(plan_json).read_text(encoding="utf-8"))
    scenes_by_item = {
        dataset.parse_item_number(item.get("item_id", "")): str(item.get("scene") or "")
        for item in plan.get("items", [])
    }
    source_scenes = {
        dataset.parse_item_number(key): scene
        for key, scene in curation_scenes(Path(sources_csv)).items()
    }

    inventory_rows: list[dict[str, str]] = []
    with Path(inventory_csv).open("r", encoding="utf-8", newline="") as handle:
        inventory_rows = list(csv.DictReader(handle))
    paths_by_name: dict[str, list[Path]] = defaultdict(list)
    for row in inventory_rows:
        image_path = Path(row.get("image_path", ""))
        if image_path.name:
            paths_by_name[image_path.name.casefold()].append(image_path.resolve())

    source_by_item: dict[int, Path] = {}
    ambiguous = 0
    missing_source = 0
    for number, filename in selected.items():
        matches = paths_by_name.get(Path(filename).name.casefold(), [])
        unique = sorted(set(matches))
        if len(unique) != 1:
            ambiguous += int(len(unique) > 1)
            missing_source += int(not unique)
            continue
        if unique[0].is_file():
            source_by_item[number] = unique[0]
        else:
            missing_source += 1

    item_by_source = {path: number for number, path in source_by_item.items()}
    records = records if records is not None else curation.load_records(source_root)
    ordinal_by_image: Counter[str] = Counter()
    boxes: list[AnnotationBox] = []
    for record in records:
        image_path = Path(record.image_path).resolve()
        item_number = item_by_source.get(image_path)
        if item_number is None:
            continue
        relative = image_path.relative_to(source_root).as_posix()
        ordinal = ordinal_by_image[str(image_path)]
        ordinal_by_image[str(image_path)] += 1
        category = _mapped_category(record.class_name, mapping)
        if category not in TARGET_CATEGORIES:
            continue
        boxes.append(AnnotationBox(
            box_id=_box_id(relative, ordinal, record.box),
            item_id=dataset.format_item_id(item_number),
            image_path=image_path,
            relative_image=relative,
            dataset_class=record.class_name,
            category=category,
            xyxy=record.box,
            scene=source_scenes.get(item_number) or scenes_by_item.get(item_number)
            or f"#{dataset.format_item_id(item_number)}",
        ))

    return boxes, {
        "selected_images": len(selected),
        "source_images_found": len(source_by_item),
        "ambiguous_source_names": ambiguous,
        "missing_source_images": missing_source,
    }


def curation_scenes(path: Path) -> dict[str, str]:
    """Read item id to scene from SOURCES.csv without touching labels."""
    scenes: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            item_id = dataset.normalise_item_id(row.get("item_id", ""))
            scene = (row.get("scene") or "").strip()
            if item_id and scene:
                scenes[item_id] = scene
    return scenes


def match_annotations(
    annotations: list[AnnotationBox],
    detections: list[DetectionBox],
    iou_threshold: float = IOU_THRESHOLD,
    confidence: float = DEFAULT_CONFIDENCE,
    low_confidence_floor: float = LOW_CONFIDENCE_FLOOR,
) -> list[dict[str, str]]:
    """Greedily match boxes one-to-one by descending IoU, ignoring classes."""
    default_indices = [i for i, d in enumerate(detections) if d.confidence >= confidence]
    low_indices = [
        i for i, d in enumerate(detections)
        if low_confidence_floor <= d.confidence < confidence
    ]
    pairs = sorted(
        (
            (intersection_over_union(annotation.xyxy, detections[index].xyxy), ann_index, index)
            for ann_index, annotation in enumerate(annotations)
            for index in default_indices
            if intersection_over_union(annotation.xyxy, detections[index].xyxy) >= iou_threshold
        ),
        key=lambda pair: (-pair[0], pair[1], pair[2]),
    )
    matched_annotations: dict[int, int] = {}
    used_detections: set[int] = set()
    for pair in pairs:
        ann_index, detection_index = pair[1], pair[2]
        if ann_index not in matched_annotations and detection_index not in used_detections:
            matched_annotations[ann_index] = detection_index
            used_detections.add(detection_index)

    results: list[dict[str, str]] = []
    for ann_index, annotation in enumerate(annotations):
        detection_index = matched_annotations.get(ann_index)
        if detection_index is not None:
            detected_class = detections[detection_index].class_name
            outcome = detected_class if detected_class in DETECTOR_CLASSES else "another class"
            matched_class = detected_class
        else:
            has_low_conf = any(
                intersection_over_union(annotation.xyxy, detections[index].xyxy) >= iou_threshold
                for index in low_indices
            )
            outcome = "low confidence" if has_low_conf else "not seen"
            matched_class = ""
        results.append({
            "box_id": annotation.box_id,
            "outcome": outcome,
            "matched_detection_class": matched_class,
        })
    return results


def _bootstrap_rate(
    rows: list[dict[str, str]],
    event: str,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = config.SEED,
) -> tuple[float | None, float | None]:
    if not rows:
        return None, None
    clusters: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        clusters[row["scene"]].append(row)
    if len(clusters) < 2 or samples <= 0:
        return None, None
    names = sorted(clusters)
    rng = random.Random(seed)
    rates: list[float] = []
    for _ in range(samples):
        drawn = [rng.choice(names) for _ in names]
        resampled = [row for scene in drawn for row in clusters[scene]]
        rates.append(sum(row["outcome"] == event for row in resampled) / len(resampled))
    return float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))


def summarize_outcomes(
    outcomes: list[dict[str, str]],
    categories: tuple[str, ...] = TARGET_CATEGORIES,
    models: tuple[str, ...] = REQUIRED_MODELS,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = config.SEED,
) -> list[dict[str, Any]]:
    """Aggregate box outcomes per annotation category/model with scene bootstrap CIs."""
    summary: list[dict[str, Any]] = []
    for category in categories:
        for model in models:
            rows = [
                row for row in outcomes
                if row["category"] == category and row["model"] == model
            ]
            counts = Counter(row["outcome"] for row in rows)
            scenes = {row["scene"] for row in rows}
            total = len(rows)
            miss_count = counts["low confidence"] + counts["not seen"]
            miss_rows = [
                {
                    **row,
                    "outcome": "miss" if row["outcome"] in {"low confidence", "not seen"}
                    else "detected",
                }
                for row in rows
            ]
            miss_low, miss_high = _bootstrap_rate(miss_rows, "miss", samples, seed)
            record: dict[str, Any] = {
                "category": category,
                "model": model,
                "annotated_boxes": total,
                "scene_count": len(scenes),
                "miss_count_at_default": miss_count,
                "miss_rate_at_default": miss_count / total if total else None,
                "miss_rate_ci95_low": miss_low,
                "miss_rate_ci95_high": miss_high,
                "category_exploratory_below_50_boxes": total < 50,
                "annotation_status": ANNOTATION_STATUS,
                "scope_status": EXPLORATORY_STATUS,
            }
            for outcome in OUTCOMES:
                low, high = _bootstrap_rate(rows, outcome, samples, seed + OUTCOMES.index(outcome) + 1)
                record[f"{outcome}_count"] = counts[outcome]
                record[f"{outcome}_rate"] = counts[outcome] / total if total else None
                record[f"{outcome}_ci95_low"] = low
                record[f"{outcome}_ci95_high"] = high
            summary.append(record)
    return summary


def load_verified_classes(path: Path) -> dict[str, str]:
    """Read user-entered true classes from the box-audit sheet."""
    if not Path(path).is_file():
        return {}
    overrides: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            box_id = (row.get("box_id") or "").strip()
            verified = (row.get("verified_class") or "").strip()
            if box_id and verified:
                overrides[box_id] = verified
    return overrides


def apply_verified_classes(
    annotations: list[AnnotationBox],
    verified: dict[str, str],
    mapping: dict[str, str],
) -> list[AnnotationBox]:
    """Replace source categories for verified boxes without mutating source records."""
    result: list[AnnotationBox] = []
    for annotation in annotations:
        source_class = verified.get(annotation.box_id)
        if not source_class:
            result.append(annotation)
            continue
        category = _verified_category(source_class, mapping)
        result.append(AnnotationBox(
            box_id=annotation.box_id,
            item_id=annotation.item_id,
            image_path=annotation.image_path,
            relative_image=annotation.relative_image,
            dataset_class=annotation.dataset_class,
            category=category,
            xyxy=annotation.xyxy,
            scene=annotation.scene,
        ))
    return result


def create_box_audit(
    annotations: list[AnnotationBox],
    output_dir: Path,
    count: int = AUDIT_BOX_COUNT,
    seed: int = AUDIT_SEED,
) -> dict[str, Any]:
    """Create reproducible context crops and a blank/preserved true-class sheet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [box for box in annotations if box.category == "rickshaw"]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected: list[tuple[AnnotationBox, np.ndarray]] = []
    for box in candidates:
        image = dataset.load_image(box.image_path)
        if image is None:
            continue
        selected.append((box, image))
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError(
            f"only {len(selected)} readable rickshaw/CNG boxes available for the {count}-box audit"
        )

    sheet_path = output_dir / "verification_sheet.csv"
    previous = load_verified_classes(sheet_path)
    rows: list[dict[str, str]] = []
    for index, (box, image) in enumerate(selected, 1):
        height, width = image.shape[:2]
        x1, y1, x2, y2 = box.xyxy
        context_x = max(12, int((x2 - x1) * 0.25))
        context_y = max(12, int((y2 - y1) * 0.25))
        left = max(0, math.floor(x1) - context_x)
        top = max(0, math.floor(y1) - context_y)
        right = min(width, math.ceil(x2) + context_x)
        bottom = min(height, math.ceil(y2) + context_y)
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            raise ValueError(f"invalid annotation crop for audit box {box.box_id}")
        crop_name = f"box_{index:02d}.png"
        crop_path = output_dir / crop_name
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        canvas = Image.fromarray(rgb)
        draw = ImageDraw.Draw(canvas)
        banner_height = 54
        labeled = Image.new("RGB", (canvas.width, canvas.height + banner_height), "white")
        labeled.paste(canvas, (0, banner_height))
        draw = ImageDraw.Draw(labeled)
        draw.text((6, 3), f"Dataset class: {box.dataset_class}", fill="black")
        draw.text((6, 19), ANNOTATION_STATUS, fill="black")
        draw.text((6, 35), EXPLORATORY_STATUS, fill="black")
        labeled.save(crop_path)
        rows.append({
            "box_id": box.box_id,
            "crop_file": crop_name,
            "source_image": box.relative_image,
            "dataset_class": box.dataset_class,
            "mapped_category": box.category,
            "x_min": str(round(box.xyxy[0], 3)),
            "y_min": str(round(box.xyxy[1], 3)),
            "x_max": str(round(box.xyxy[2], 3)),
            "y_max": str(round(box.xyxy[3], 3)),
            "verified_class": previous.get(box.box_id, ""),
            "annotation_status": ANNOTATION_STATUS,
            "scope_status": EXPLORATORY_STATUS,
        })
    columns = [
        "box_id", "crop_file", "source_image", "dataset_class", "mapped_category",
        "x_min", "y_min", "x_max", "y_max", "verified_class",
        "annotation_status", "scope_status",
    ]
    with sheet_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return {"selected_boxes": len(rows), "verification_sheet": str(sheet_path)}


def load_rickshaw_audit_pool(
    source_root: Path,
    mapping: dict[str, str],
    records: list[curation.AnnotationRecord] | None = None,
) -> list[AnnotationBox]:
    """Make stable box records for all mapped rickshaw/CNG annotations in the archive."""
    source_root = Path(source_root).expanduser().resolve()
    records = records if records is not None else curation.load_records(source_root)
    ordinal_by_image: Counter[str] = Counter()
    boxes: list[AnnotationBox] = []
    for record in records:
        image_path = Path(record.image_path).resolve()
        if not image_path.is_relative_to(source_root):
            continue
        relative = image_path.relative_to(source_root).as_posix()
        ordinal = ordinal_by_image[str(image_path)]
        ordinal_by_image[str(image_path)] += 1
        if _mapped_category(record.class_name, mapping) != "rickshaw":
            continue
        boxes.append(AnnotationBox(
            box_id=_box_id(relative, ordinal, record.box),
            item_id="",
            image_path=image_path,
            relative_image=relative,
            dataset_class=record.class_name,
            category="rickshaw",
            xyxy=record.box,
            scene=f"source:{relative}",
        ))
    return boxes


def write_outcomes_csv(path: Path, outcomes: list[dict[str, str]]) -> None:
    """Write per-box outcomes for auditability; console output remains aggregate-only."""
    columns = [
        "box_id", "item_id", "scene", "source_image", "dataset_class", "category",
        "model", "outcome", "matched_detection_class", "annotation_status", "scope_status",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(outcomes)


def write_summary_csv(path: Path, summary: list[dict[str, Any]]) -> None:
    """Write category/model aggregate rates and scene-bootstrap intervals."""
    if not summary:
        raise ValueError("no summary rows to write")
    columns = list(summary[0])
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary)


def write_confusion_chart(path: Path, summary: list[dict[str, Any]]) -> None:
    """Write a compact confusion-rate matrix with per-cell counts."""
    row_height = 40
    label_width = 210
    cell_width = 150
    margin_top = 110
    width = label_width + cell_width * len(OUTCOMES) + 24
    height = margin_top + row_height * len(summary) + 35
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((12, 10), "H3 box confusion matrix | annotation referenced, unverified | exploratory",
              fill="black")
    draw.text((12, 32), "Cells show outcome rate and box count; each row reports its box and scene counts.",
              fill="black")
    for column, outcome in enumerate(OUTCOMES):
        x = label_width + column * cell_width
        draw.text((x + 4, 70), outcome, fill="black")
    for row_index, record in enumerate(summary):
        y = margin_top + row_index * row_height
        draw.text((10, y + 6),
                  f"{record['category']} / {record['model']} "
                  f"(n={record['annotated_boxes']}, scenes={record['scene_count']})",
                  fill="black")
        for column, outcome in enumerate(OUTCOMES):
            raw_rate = record[f"{outcome}_rate"]
            rate = 0.0 if raw_rate is None else float(raw_rate)
            count = int(record[f"{outcome}_count"])
            shade = max(0, min(255, int(250 - 170 * rate)))
            x1 = label_width + column * cell_width
            draw.rectangle((x1, y, x1 + cell_width - 2, y + row_height - 2),
                           fill=(shade, shade, 255))
            rate_text = "n/a" if raw_rate is None else f"{rate:.1%}"
            draw.text((x1 + 8, y + 6), f"{rate_text} (n={count})", fill="black")
    image.save(path)


def analyze(
    annotations: list[AnnotationBox],
    detections: dict[tuple[int, str], list[DetectionBox]],
    verified_classes: dict[str, str] | None = None,
    mapping: dict[str, str] | None = None,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = config.SEED,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Compute one-to-one class-agnostic box matches and aggregates."""
    if verified_classes and mapping is not None:
        annotations = apply_verified_classes(annotations, verified_classes, mapping)
    annotations = [box for box in annotations if box.category in TARGET_CATEGORIES]
    by_image: dict[int, list[AnnotationBox]] = defaultdict(list)
    for box in annotations:
        number = dataset.parse_item_number(box.item_id)
        if number is not None:
            by_image[number].append(box)

    outcomes: list[dict[str, str]] = []
    for item_number, image_annotations in by_image.items():
        image_annotations.sort(key=lambda box: box.box_id)
        for model in REQUIRED_MODELS:
            result = match_annotations(image_annotations, detections.get((item_number, model), []))
            for annotation, match in zip(image_annotations, result):
                outcomes.append({
                    "box_id": annotation.box_id,
                    "item_id": annotation.item_id,
                    "scene": annotation.scene,
                    "source_image": annotation.relative_image,
                    "dataset_class": annotation.dataset_class,
                    "category": annotation.category,
                    "model": model,
                    **match,
                    "annotation_status": ANNOTATION_STATUS,
                    "scope_status": EXPLORATORY_STATUS,
                })
    summary = summarize_outcomes(
        outcomes, samples=bootstrap_samples, seed=seed
    )
    return outcomes, summary
