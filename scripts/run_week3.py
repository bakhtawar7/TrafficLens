"""Run the readiness-gated Week 3 analysis pipeline.

The pipeline intentionally does not repair labels or download models/data. It checks
audited first-pass coverage and the test-set checker before running evaluation,
prediction, H3, and the failure gallery in that order.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config, curation, dataset, evaluation, settings  # noqa: E402

MIN_EVALUATION_LABELS = 30
GALLERY_SIZE = 12
BOOTSTRAP_SAMPLES = 2000
STANDARD_COLORS = {
    "yolov8n": (0, 170, 255),
    "yolov8s": (255, 80, 40),
    "rtdetr-l": (60, 220, 60),
}
SLICED_COLOR = (255, 0, 220)
DATASET_COLOR = (255, 255, 0)
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _has_complete_counts(row: dict[str, str]) -> bool:
    for column in config.LABEL_CLASS_NAMES:
        try:
            value = float(row.get(column, ""))
        except (TypeError, ValueError):
            return False
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            return False
    return True


def _write_complete_label_snapshot(source_path: Path, output_path: Path) -> None:
    """Keep incomplete label rows out of evaluator paths that treat blanks as zero."""
    with Path(source_path).open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"LABELS.csv has no header: {source_path}")
        rows = [row for row in reader if _has_complete_counts(row)]
    with Path(output_path).open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def readiness_issues(
    labels_path: Path,
    sources_path: Path,
    plan_path: Path,
) -> list[str]:
    """Return all missing, incomplete, or unaudited first-pass image labels."""
    missing_files = [
        f"Required readiness file is missing: {path}"
        for path in (labels_path, sources_path, plan_path)
        if not Path(path).is_file()
    ]
    if missing_files:
        return missing_files

    labels = _read_csv(labels_path)
    sources = _read_csv(sources_path)
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    dev_ids = {
        dataset.parse_item_number(value)
        for value in plan.get("development_subset", [])
    }
    dev_ids.discard(None)
    image_ids: set[int] = set()
    for row in sources:
        suffix = Path(row.get("filename", "")).suffix.lower()
        item = dataset.parse_item_number(row.get("item_id", ""))
        if item is not None and (not suffix or suffix in _IMAGE_SUFFIXES):
            image_ids.add(item)

    first_pass_candidates: dict[int, dict[str, str]] = {}
    for row in labels:
        item = dataset.parse_item_number(row.get("item_id", ""))
        if item is None:
            continue
        pass_value = (row.get("labeller_pass") or "").strip()
        if pass_value not in {"", "1", "1.0"}:
            continue
        notes = (row.get("notes") or "").lower()
        if evaluation.is_unverified_note(notes):
            continue
        first_pass_candidates[item] = row
    first_pass = {
        item: row for item, row in first_pass_candidates.items()
        if all((row.get(column) or "").strip() for column in config.LABEL_CLASS_NAMES)
    }

    def missing_count_columns(item: int) -> list[str]:
        row = first_pass_candidates.get(item)
        if row is None:
            return list(config.LABEL_CLASS_NAMES)
        return [
            column for column in config.LABEL_CLASS_NAMES
            if not (row.get(column) or "").strip()
        ]

    missing: list[str] = []
    development_missing = sorted(dev_ids - first_pass.keys())
    for item in development_missing:
        columns = missing_count_columns(item)
        missing.append(
            f"{dataset.format_item_id(item)}: missing audited first-pass count(s): "
            f"{', '.join(columns)}"
        )
    evaluation_ids = image_ids - dev_ids
    audited_evaluation = len(evaluation_ids & first_pass.keys())
    if audited_evaluation < MIN_EVALUATION_LABELS:
        missing_evaluation = sorted(evaluation_ids - first_pass.keys())
        missing_detail = "; ".join(
            f"{dataset.format_item_id(item)} ({', '.join(missing_count_columns(item))})"
            for item in missing_evaluation
        )
        missing.append(
            f"Need at least {MIN_EVALUATION_LABELS} audited evaluation images; "
            f"found {audited_evaluation}. Missing evaluation first-pass counts: "
            f"{missing_detail or 'none'}"
        )
    if missing:
        return missing
    return []


def _run_command(command: list[str], description: str) -> bool:
    print(f"== {description} ==")
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n")
        print(f"ERROR: {description} stopped with exit code {completed.returncode}.")
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _item_paths(images_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in sorted(Path(images_dir).iterdir()):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            item = dataset.normalise_item_id(path.stem.split("_")[0])
            if item:
                paths[item] = path
    return paths


def _source_boxes_by_item(
    items: set[str],
    image_paths: dict[str, Path],
) -> tuple[dict[str, list[tuple[str, tuple[float, float, float, float]]]], set[str]]:
    """Return only boxes whose source and evaluated image bytes are identical."""
    selection_path = config.TESTSET_SELECTION_CSV
    inventory_path = config.TESTSET_INVENTORY_CSV
    mapping_path = config.CLASS_MAPPING_JSON
    if not all(path.is_file() for path in (selection_path, inventory_path, mapping_path)):
        return {}, set(items)

    selected = {
        dataset.normalise_item_id(row.get("item_id", "")): row.get("original_filename", "")
        for row in _read_csv(selection_path)
        if row.get("decision") == "selected"
    }
    inventory: dict[str, list[Path]] = {}
    for row in _read_csv(inventory_path):
        image_path = Path(row.get("image_path", ""))
        if image_path.name:
            inventory.setdefault(image_path.name.casefold(), []).append(image_path)

    mapping_payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    source_root = Path(mapping_payload.get("dataset_path", "")).expanduser()
    if not source_root.is_dir():
        return {}, set(items)
    records = curation.load_records(source_root)
    by_path: dict[Path, list[curation.AnnotationRecord]] = {}
    for record in records:
        by_path.setdefault(Path(record.image_path).resolve(), []).append(record)

    output: dict[str, list[tuple[str, tuple[float, float, float, float]]]] = {}
    unavailable: set[str] = set()
    for item in items:
        filename = selected.get(item)
        candidates = sorted(set(inventory.get(Path(filename or "").name.casefold(), [])))
        test_image = image_paths.get(item)
        if len(candidates) != 1 or test_image is None or not candidates[0].is_file():
            unavailable.add(item)
            continue
        source_image = candidates[0].resolve()
        if _sha256(source_image) != _sha256(test_image):
            unavailable.add(item)
            continue
        output[item] = [
            (record.class_name, record.box)
            for record in by_path.get(source_image, [])
        ]
    return output, unavailable


def _load_matching_cache(path: Path, settings_hash: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid detection cache JSON at line {line_number}") from error
            if record.get("settings_sha256") != settings_hash:
                continue
            item = dataset.normalise_item_id(str(record.get("image", "")).split("_")[0])
            model = str(record.get("model", ""))
            mode = str(record.get("mode", ""))
            if item and model in config.MODEL_REGISTRY and mode in {"standard", "sliced"}:
                cache[(item, model, mode)] = record
    return cache


def _filtered_detections(
    entry: dict[str, Any],
    confidence: float,
) -> list[dict[str, Any]]:
    return [
        detection for detection in entry.get("detections", [])
        if float(detection.get("confidence", 0.0)) >= confidence
        and str(detection.get("class_name", "")).lower() in config.TARGET_CLASS_NAMES
    ]


def create_failure_gallery(
    table_path: Path = config.EVALUATION_TABLE_CSV,
    cache_path: Path = config.RAW_DETECTIONS_JSONL,
    labels_path: Path = config.LABELS_CSV,
    output_path: Path | None = None,
    images_dir: Path = config.TESTSET_IMAGES_DIR,
    settings_path: Path = config.FINAL_SETTINGS_JSON,
) -> dict[str, Any]:
    """Build a 12-row gallery with per-arm boxes and audited/model counts."""
    output_path = output_path or config.RESULTS_DIR / "failure_gallery.png"
    rows = _read_csv(table_path)
    rows.sort(
        key=lambda row: (
            -float(row.get("Actual Error") or 0)
            if math.isfinite(float(row.get("Actual Error") or 0)) else 0,
            row.get("Item", ""),
        )
    )
    rows = rows[:GALLERY_SIZE]
    if len(rows) < GALLERY_SIZE:
        raise ValueError(
            f"failure gallery requires {GALLERY_SIZE} evaluation rows; found {len(rows)}"
        )
    covered_label_counts: dict[str, int] = {}
    all_label_counts: dict[str, int] = {}
    for row in _read_csv(labels_path):
        item = dataset.normalise_item_id(row.get("item_id", ""))
        if not item or (row.get("labeller_pass") or "").strip() not in {"", "1", "1.0"}:
            continue
        if evaluation.is_unverified_note(row.get("notes", "")):
            continue
        try:
            covered_label_counts[item] = sum(
                int(row[column]) for column in config.TARGET_CLASS_NAMES
            )
            all_label_counts[item] = sum(
                int(row[column]) for column in config.LABEL_CLASS_NAMES
            )
        except (KeyError, TypeError, ValueError):
            continue

    paths = _item_paths(images_dir)
    settings_record = settings.settings_record(settings_path)
    cache = _load_matching_cache(cache_path, settings_record["settings_sha256"])
    item_ids = {dataset.normalise_item_id(row.get("Item", "")) for row in rows}
    item_ids.discard(None)
    dataset_boxes, boxes_unavailable = _source_boxes_by_item(item_ids, paths)
    per_item: list[dict[str, Any]] = []
    rendered: list[np.ndarray] = []
    tile_width, tile_height = 320, 220
    arms = [("Dataset boxes (source annotations)", None, DATASET_COLOR)]
    arms.extend(
        (config.MODEL_REGISTRY[key].display_name, (key, "standard"), STANDARD_COLORS[key])
        for key in config.MODEL_REGISTRY
    )
    arms.append((f"{config.MODEL_REGISTRY[config.DEFAULT_MODEL_NAME].display_name} sliced",
                 (config.DEFAULT_MODEL_NAME, "sliced"), SLICED_COLOR))

    for row in rows:
        item = dataset.normalise_item_id(row.get("Item", ""))
        if not item or item not in paths:
            raise ValueError(f"evaluation image is unavailable for gallery item {row.get('Item')}")
        source = cv2.imread(str(paths[item]), cv2.IMREAD_COLOR)
        if source is None:
            raise ValueError(f"could not decode gallery image {paths[item].name}")
        image_h, image_w = source.shape[:2]
        scale = min((tile_width - 12) / image_w, (tile_height - 40) / image_h)
        resized = cv2.resize(
            source,
            (max(1, int(image_w * scale)), max(1, int(image_h * scale))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
        tile = np.full((tile_height, tile_width, 3), 245, dtype=np.uint8)
        y_offset, x_offset = 30, (tile_width - resized.shape[1]) // 2
        tile[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized
        per_model: dict[str, int] = {}
        audited_covered = covered_label_counts.get(item, -1)
        audited_all = all_label_counts.get(item, -1)
        for title, arm, color in arms:
            panel = tile.copy()
            cv2.putText(panel, title, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1)
            if arm is None:
                if item in dataset_boxes:
                    for _, box in dataset_boxes[item]:
                        x0, y0, x1, y1 = box
                        p0 = (x_offset + int(x0 * scale), y_offset + int(y0 * scale))
                        p1 = (x_offset + int(x1 * scale), y_offset + int(y1 * scale))
                        cv2.rectangle(panel, p0, p1, color, 2)
                    box_note = "source boxes aligned"
                else:
                    box_note = "boxes unavailable: source mismatch"
                count_text = (
                    f"audited covered/all: "
                    f"{audited_covered if audited_covered >= 0 else 'missing'}/"
                    f"{audited_all if audited_all >= 0 else 'missing'}"
                )
                cv2.putText(panel, box_note, (6, tile_height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1)
                cv2.putText(panel, count_text, (6, tile_height - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 1)
            else:
                model, mode = arm
                entry = cache.get((item, model, mode))
                if entry is None:
                    raise ValueError(f"final-settings {mode} detections missing for {item}/{model}")
                if entry.get("image_hash") != _sha256(paths[item]):
                    raise ValueError(
                        f"stale or mismatched {mode} detection cache image for {item}/{model}"
                    )
                model_settings = settings_record["models"][model]
                detections = _filtered_detections(entry, float(model_settings["confidence"]))
                per_model[f"{model}:{mode}"] = len(detections)
                for detection in detections:
                    x0, y0, x1, y1 = detection["xyxy"]
                    p0 = (x_offset + int(float(x0) * scale), y_offset + int(float(y0) * scale))
                    p1 = (x_offset + int(float(x1) * scale), y_offset + int(float(y1) * scale))
                    cv2.rectangle(panel, p0, p1, color, 2)
                cv2.putText(panel, f"count: {len(detections)}", (6, tile_height - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            rendered.append(panel)
        per_item.append({
            "item_id": item,
            "actual_error": float(row.get("Actual Error") or 0),
            "audited_covered_count": audited_covered if audited_covered >= 0 else None,
            "audited_all_vehicle_count": audited_all if audited_all >= 0 else None,
            "model_counts": per_model,
            "dataset_boxes_aligned": item in dataset_boxes,
        })

    columns = len(arms)
    gallery = np.full(
        (len(rows) * tile_height, columns * tile_width, 3), 255, dtype=np.uint8
    )
    for index, panel in enumerate(rendered):
        row_index, column_index = divmod(index, columns)
        y0, x0 = row_index * tile_height, column_index * tile_width
        gallery[y0:y0 + tile_height, x0:x0 + tile_width] = panel
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), gallery):
        raise OSError(f"could not write failure gallery to {output_path}")
    counts_path = output_path.with_name("failure_gallery_counts.csv")
    with counts_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ("item_id", "actual_error", "audited_covered_count",
                  "audited_all_vehicle_count",
                  "model_counts_json", "dataset_boxes_aligned")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in per_item:
            writer.writerow({
                "item_id": record["item_id"],
                "actual_error": record["actual_error"],
                "audited_covered_count": record["audited_covered_count"],
                "audited_all_vehicle_count": record["audited_all_vehicle_count"],
                "model_counts_json": json.dumps(record["model_counts"], sort_keys=True),
                "dataset_boxes_aligned": record["dataset_boxes_aligned"],
            })
    return {
        "image_count": len(per_item),
        "dataset_box_aligned_images": len(per_item) - len(boxes_unavailable),
        "dataset_box_unavailable_ids": sorted(boxes_unavailable),
        "gallery": str(output_path),
        "counts": str(counts_path),
        "items": per_item,
    }


def _cluster_mean_ci(
    differences: list[float],
    scenes: list[str],
    seed: int = config.SEED,
) -> dict[str, float | int | None]:
    if not differences:
        return {"estimate": None, "low": None, "high": None, "n": 0, "scenes": 0}
    grouped: dict[str, list[float]] = {}
    for value, scene in zip(differences, scenes):
        grouped.setdefault(scene or "unknown", []).append(value)
    keys = sorted(grouped)
    estimate = float(np.mean(differences))
    rng = np.random.default_rng(seed)
    samples = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    for index in range(BOOTSTRAP_SAMPLES):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        draw = [value for key in chosen for value in grouped[str(key)]]
        samples[index] = float(np.mean(draw))
    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "estimate": estimate,
        "low": float(low),
        "high": float(high),
        "n": len(differences),
        "scenes": len(keys),
    }


def _decision_ci(interval: dict[str, Any]) -> str:
    low, high = interval.get("low"), interval.get("high")
    if low is None or high is None or int(interval.get("scenes", 0)) < 2:
        return "inconclusive"
    if float(low) > 0:
        return "supported"
    if float(high) <= 0:
        return "not supported"
    return "inconclusive"


def _format_interval(interval: dict[str, Any]) -> str:
    estimate = interval.get("estimate")
    low, high = interval.get("low"), interval.get("high")
    if estimate is None or low is None or high is None:
        return "not estimable"
    return f"{float(estimate):.3f} (95% scene bootstrap CI {float(low):.3f} to {float(high):.3f})"


def write_week3_summary(
    path: Path,
    evaluation_summary_path: Path,
    predictor_report_path: Path,
    h3_summary_path: Path,
    evaluation_table_path: Path,
    gallery: dict[str, Any],
) -> dict[str, Any]:
    """Create the hypothesis-status report from the completed stage artifacts."""
    eval_summary = json.loads(Path(evaluation_summary_path).read_text(encoding="utf-8"))
    predictor_report = json.loads(Path(predictor_report_path).read_text(encoding="utf-8"))
    h3_rows = _read_csv(h3_summary_path)
    table = _read_csv(evaluation_table_path)
    predictor_hypotheses = predictor_report.get("hypotheses", {})
    h1 = predictor_hypotheses.get("H1", {})
    h2 = predictor_hypotheses.get("H2", {})
    h1_status = (
        "inconclusive" if "reject" not in h1
        else ("not supported" if h1["reject"] else "supported")
    )
    h2_status = (
        "inconclusive" if "reject" not in h2
        else ("not supported" if h2["reject"] else "supported")
    )

    distant_rows = [
        row for row in table
        if "distant" in {
            tag.strip()
            for tag in (row.get("Condition") or "").replace("|", ";").lower().split(";")
        }
    ]
    distant_errors: list[float] = []
    distant_scenes: list[str] = []
    for row in distant_rows:
        try:
            ground_truth = float(row["Ground Truth"])
            standard = float(row[config.MODEL_REGISTRY[config.DEFAULT_MODEL_NAME].display_name])
            sliced = float(row["Sliced Count"])
        except (KeyError, TypeError, ValueError):
            continue
        distant_errors.append(abs(ground_truth - standard) - abs(ground_truth - sliced))
        distant_scenes.append(row.get("Scene", ""))
    h4_interval = _cluster_mean_ci(distant_errors, distant_scenes)
    h4_status = _decision_ci(h4_interval)

    h3_categories = ("rickshaw", "loader_rickshaw")
    h3_extracts = [
        row for row in h3_rows if row.get("category") in h3_categories
    ]
    h3_box_count_by_category = {
        category: max(
            (int(row.get("annotated_boxes") or 0) for row in h3_extracts
             if row.get("category") == category),
            default=0,
        )
        for category in h3_categories
    }
    verified_count = 0
    verification_sheet = config.H3_BOX_AUDIT_DIR / "verification_sheet.csv"
    if verification_sheet.is_file():
        verified_count = sum(
            1 for row in _read_csv(verification_sheet)
            if (row.get("verified_class") or "").strip()
        )
    h3_status = "inconclusive"

    items = eval_summary.get("items", {})
    eval_n = int(items.get("evaluated", 0))
    scene_n = len({row.get("Scene", "") for row in table if row.get("Scene")})
    lines = [
        "# Week 3 results summary",
        "",
        "This report uses audited first-pass labels and the final settings file. "
        "Intervals are scene-level bootstrap intervals. A difference is called an "
        "improvement only when its interval excludes zero.",
        "",
        f"- Evaluation sample: **{eval_n} images**, **{scene_n} scenes**; "
        f"development subset excluded: {items.get('development_excluded', 'unknown')}.",
        f"- Predictor sample: **{predictor_report.get('sample_size', 0)} images**; "
        f"{'exploratory' if predictor_report.get('exploratory') else 'not exploratory by the 60-image rule'}.",
        f"- H3 verified audit boxes supplied: **{verified_count}**; annotation references "
        "remain provisional where not verified.",
        f"- Failure gallery: **{gallery.get('image_count', 0)} images**; "
        f"dataset boxes byte-aligned for **{gallery.get('dataset_box_aligned_images', 0)}**.",
        "",
        "| Hypothesis | Status | Sample size | Evidence and intervals |",
        "|---|---|---:|---|",
    ]
    h1_difference = h1.get("ridge_minus_mean_confidence_spearman", {})
    h1_capture = h1.get("ridge_worst_20_percent_capture", {})
    h1_baseline_capture = h1.get("mean_confidence_worst_20_percent_capture", {})
    lines.append(
        f"| H1 | {h1_status} | {predictor_report.get('sample_size', 0)} images | "
        f"Ridge minus mean-confidence Spearman: {_format_interval(h1_difference)}; "
        f"worst-20% capture recall ridge={h1_capture.get('recall', 'n/a')}, "
        f"baseline={h1_baseline_capture.get('recall', 'n/a')}. |"
    )
    h2_difference = h2.get("cross_minus_within_spearman", {})
    lines.append(
        f"| H2 | {h2_status} | {predictor_report.get('sample_size', 0)} images | "
        f"Cross-family minus within-family Spearman: {_format_interval(h2_difference)}. |"
    )
    h3_evidence = "; ".join(
        f"{row.get('category')} / {row.get('model')}: "
        f"{row.get('miss_count_at_default')}/{row.get('annotated_boxes')} boxes missed, "
        f"{row.get('scene_count')} scenes, 95% CI "
        f"{row.get('miss_rate_ci95_low')} to {row.get('miss_rate_ci95_high')}"
        for row in h3_extracts
    ) or "No category aggregates available."
    lines.append(
        f"| H3 | {h3_status} | {sum(h3_box_count_by_category.values())} unique "
        f"rickshaw/loader boxes; at most "
        f"{max((int(row.get('scene_count') or 0) for row in h3_extracts), default=0)} "
        f"scenes per category | {h3_evidence} Annotation references are unverified except "
        f"for the {verified_count} manually verified audit boxes; no preregistered numeric "
        "threshold defines “far above,” so no confirmatory decision is possible. |"
    )
    lines.append(
        f"| H4 | {h4_status} | {h4_interval['n']} distant images; "
        f"{h4_interval['scenes']} scenes | Paired effect (standard absolute error minus "
        f"sliced absolute error; positive favors sliced): {_format_interval(h4_interval)}. |"
    )
    lines.extend([
        "",
        "## Evaluation context",
        "",
        f"- Final settings: `{eval_summary.get('settings', {}).get('settings_file', 'unknown')}` "
        f"(SHA-256 `{eval_summary.get('settings', {}).get('settings_sha256', 'unknown')}`).",
        f"- Scene bootstrap: {eval_summary.get('settings', {}).get('bootstrap_samples', 'unknown')} "
        "resamples; the evaluator's deterministic scene-level method.",
        "- Per-condition model MAE and signed-error intervals are available in "
        "`results/summary.json`; condition sample sizes and exploratory flags are recorded there.",
        "- Failure gallery counts and box-alignment checks: `results/failure_gallery_counts.csv`.",
        "",
    ])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return {
        "H1": h1_status,
        "H2": h2_status,
        "H3": h3_status,
        "H4": h4_status,
        "h4_paired_interval": h4_interval,
    }


def _run_pipeline(args: argparse.Namespace, audited_labels: Path) -> int:
    final_settings = config.FINAL_SETTINGS_JSON
    if not final_settings.is_file():
        print(f"Readiness not met; pipeline stopped:\n- Final settings file missing: {final_settings}")
        return 1
    benchmark_cmd = [
        sys.executable, str(ROOT / "scripts" / "run_benchmark.py"),
        "--evaluation-only", "--settings", str(final_settings),
    ]
    if not _run_command(benchmark_cmd, "1/5 Final standard+sliced benchmark"):
        return 1

    try:
        evaluation.evaluate(
            labels_csv=audited_labels,
            sources_csv=args.sources,
            plan_json=args.plan,
            settings_json=final_settings,
            bootstrap_samples=BOOTSTRAP_SAMPLES,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: final evaluation failed: {error}")
        return 1

    feature_cmd = [
        sys.executable, str(ROOT / "scripts" / "build_features.py"),
        "--settings", str(final_settings),
    ]
    if not _run_command(feature_cmd, "2/5 Prepare final-settings reliability features"):
        return 1
    predictor_cmd = [
        sys.executable, str(ROOT / "scripts" / "run_predictor.py"),
        "--labels", str(audited_labels), "--sources", str(args.sources),
        "--plan", str(args.plan),
    ]
    if not _run_command(predictor_cmd, "2/5 Predictor and needs-review analysis"):
        return 1

    if not _run_command(
        [sys.executable, str(ROOT / "scripts" / "run_h3_confusion.py")],
        "3/5 H3 analysis (preserving and applying verified box classes)",
    ):
        return 1

    try:
        gallery = create_failure_gallery(
            labels_path=audited_labels, settings_path=final_settings
        )
        print(
            f"== 4/5 Failure gallery ==\nWrote {gallery['image_count']} images to "
            f"{gallery['gallery']}; dataset boxes aligned on "
            f"{gallery['dataset_box_aligned_images']} images."
        )
        report_status = write_week3_summary(
            config.RESULTS_DIR / "week3_summary.md",
            config.EVALUATION_SUMMARY_JSON,
            config.RESULTS_DIR / "predictor_report.json",
            config.H3_CONFUSION_CSV,
            config.EVALUATION_TABLE_CSV,
            gallery,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"ERROR: gallery or Week 3 summary failed: {error}")
        return 1
    print("== 5/5 Week 3 hypothesis summary ==")
    print(f"Wrote {config.RESULTS_DIR / 'week3_summary.md'}: {report_status}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the readiness-gated Week 3 pipeline.")
    parser.add_argument("--labels", type=Path, default=config.LABELS_CSV)
    parser.add_argument("--sources", type=Path, default=config.SOURCES_CSV)
    parser.add_argument("--plan", type=Path, default=config.TESTSET_PLAN_JSON)
    args = parser.parse_args(argv)

    missing = readiness_issues(args.labels, args.sources, args.plan)
    if missing:
        print("Readiness not met; pipeline stopped:")
        for item in missing:
            print(f"- {item}")
        return 1

    checker = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_testset.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if checker.returncode != 0:
        print("Readiness checker exited nonzero; pipeline stopped:")
        print(checker.stdout, end="" if checker.stdout.endswith("\n") else "\n")
        if checker.stderr:
            print(checker.stderr, end="" if checker.stderr.endswith("\n") else "\n")
        return 1
    print(checker.stdout, end="" if checker.stdout.endswith("\n") else "\n")

    try:
        with tempfile.TemporaryDirectory(prefix="trafficlens-week3-") as temporary:
            audited_labels = Path(temporary) / "audited_LABELS.csv"
            _write_complete_label_snapshot(args.labels, audited_labels)
            return _run_pipeline(args, audited_labels)
    except (OSError, ValueError, csv.Error) as error:
        print(f"ERROR: could not prepare complete audited labels: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
