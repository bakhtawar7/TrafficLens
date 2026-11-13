"""Run the readiness-gated Week 3 analysis pipeline or its cache-only pilot.

The pipeline intentionally does not repair labels or download models/data. It checks
audited first-pass coverage and the test-set checker before running evaluation,
prediction, H3, and the failure gallery in that order.

The separate ``--pilot`` path requires ten audited evaluation first passes and uses
only cached detections matching the frozen final settings.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
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

from core import config, counting, curation, dataset, evaluation, reliability, settings  # noqa: E402

MIN_EVALUATION_LABELS = 30
MIN_PILOT_EVALUATION_LABELS = 10
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


def _pilot_labels(path: Path) -> dict[int, dict[str, dict[str, int] | None]]:
    """Load only complete, verified first- and second-pass pilot labels."""
    labels: dict[int, dict[str, dict[str, int] | None]] = {}
    for row in _read_csv(path):
        item = dataset.parse_item_number(row.get("item_id", ""))
        if item is None or evaluation.is_unverified_note(row.get("notes", "")):
            continue
        pass_value = (row.get("labeller_pass") or "").strip()
        slot = "pass2" if pass_value in {"2", "2.0"} else "pass1"
        if slot == "pass1" and pass_value not in {"", "1", "1.0"}:
            continue
        if not _has_complete_counts(row):
            continue
        labels.setdefault(item, {"pass1": None, "pass2": None})[slot] = {
            name: int(row[name]) for name in config.LABEL_CLASS_NAMES
        }
    return labels


def _format_pilot_interval(interval: tuple[float | None, float | None]) -> str:
    lower, upper = interval
    if lower is None or upper is None:
        return "not available"
    return f"[{lower:.2f}, {upper:.2f}]"


def _pilot_model_rows(
    item_ids: set[int],
    labels: dict[int, dict[str, dict[str, int] | None]],
    cache: dict[tuple[int, str, str], list[Any]],
    scenes: dict[int, str],
    mode: str,
    reference: str,
) -> dict[str, dict[str, Any]]:
    """Summarize each model on its own available cached, labelled images."""
    reports: dict[str, dict[str, Any]] = {}
    for model in config.PRETRAINED_MODEL_KEYS:
        pairs: list[tuple[int, int]] = []
        clusters: list[str] = []
        for item in sorted(item_ids):
            truth_counts = labels[item].get(reference)
            detections = cache.get((item, model, mode))
            if truth_counts is None or detections is None:
                continue
            predicted_counts = counting.count_by_class(
                detections, list(config.TARGET_CLASS_NAMES)
            )
            pairs.append((
                evaluation.covered_total(predicted_counts),
                evaluation.covered_total(truth_counts),
            ))
            clusters.append(scenes.get(item, f"#{dataset.format_item_id(item)}"))
        stats = evaluation.error_stats(pairs, clusters, BOOTSTRAP_SAMPLES)
        signed = [prediction - truth for prediction, truth in pairs]
        reports[model] = {
            "stats": stats,
            "signed_ci": evaluation.cluster_bootstrap_ci(
                signed, clusters, BOOTSTRAP_SAMPLES
            ),
            "scenes": len(set(clusters)),
        }
    return reports

def _build_pilot_summary(
    labels_path: Path,
    sources_path: Path,
    plan_path: Path,
    settings_path: Path,
    cache_path: Path,
) -> tuple[str, dict[str, Any]]:
    """Build an isolated pilot report using only existing frozen-settings cache rows."""
    labels = _pilot_labels(labels_path)
    sources = _read_csv(sources_path)
    dev_ids = evaluation.load_development_ids(plan_path)
    image_ids = {
        item
        for row in sources
        if (item := dataset.parse_item_number(row.get("item_id", ""))) is not None
        and (
            not Path(row.get("filename", "")).suffix
            or Path(row.get("filename", "")).suffix.lower() in _IMAGE_SUFFIXES
        )
    }
    eval_ids = image_ids - dev_ids
    labelled_eval = {
        item for item in eval_ids
        if labels.get(item, {}).get("pass1") is not None
    }
    if len(labelled_eval) < MIN_PILOT_EVALUATION_LABELS:
        raise ValueError(
            f"Pilot requires at least {MIN_PILOT_EVALUATION_LABELS} complete audited "
            f"evaluation first-pass labels; found {len(labelled_eval)}."
        )

    frozen = settings.settings_record(settings_path)
    confidence = {
        name: float(record.get("confidence", config.DEFAULT_CONFIDENCE))
        for name, record in frozen["models"].items()
    }
    cache = evaluation.load_detection_cache(
        cache_path, confidence, frozen["settings_sha256"]
    )
    scenes = evaluation.load_scenes(sources_path)
    labelled_dev = {
        item for item in dev_ids & image_ids
        if labels.get(item, {}).get("pass1") is not None
    }
    main_rows = _pilot_model_rows(
        labelled_eval, labels, cache, scenes, "standard", "pass1"
    )
    second_rows = _pilot_model_rows(
        {item for item in labelled_eval if labels[item].get("pass2") is not None},
        labels, cache, scenes, "standard", "pass2",
    )
    dev_rows = _pilot_model_rows(
        labelled_dev, labels, cache, scenes, "standard", "pass1"
    )

    conditions = evaluation.load_conditions(sources_path)
    condition_counts: dict[str, int] = {}
    for item in labelled_eval:
        for tag in (value.strip() for value in conditions.get(item, "").split(";")):
            if tag:
                condition_counts[tag] = condition_counts.get(tag, 0) + 1

    consistency = evaluation.labeller_pass_statistics(
        labels, scenes, excluded=dev_ids, samples=BOOTSTRAP_SAMPLES
    )["covered_total"]

    model_names = {
        model: config.MODEL_REGISTRY[model].display_name
        for model in config.PRETRAINED_MODEL_KEYS
    }
    standard_tables = []
    for model, report in main_rows.items():
        stats = report["stats"]
        standard_tables.append(
            f"| {model_names[model]} | {stats['n']} | {report['scenes']} | "
            f"{stats['mae'] if stats['mae'] is not None else 'n/a'} "
            f"{_format_pilot_interval(stats['ci95'])} | "
            f"{stats['bias'] if stats['bias'] is not None else 'n/a'} "
            f"{_format_pilot_interval(report['signed_ci'])} | "
            f"{stats['undercounted']} | {stats['overcounted']} | {stats['exact']} |"
        )
    second_tables = []
    for model, report in second_rows.items():
        stats = report["stats"]
        second_tables.append(
            f"| {model_names[model]} | {stats['n']} | {stats['mae'] if stats['mae'] is not None else 'n/a'} "
            f"{_format_pilot_interval(stats['ci95'])} | "
            f"{stats['bias'] if stats['bias'] is not None else 'n/a'} "
            f"{_format_pilot_interval(report['signed_ci'])} |"
        )
    dev_tables = []
    for model, report in dev_rows.items():
        stats = report["stats"]
        dev_tables.append(
            f"| {model_names[model]} | {stats['n']} | {report['scenes']} | "
            f"{stats['mae'] if stats['mae'] is not None else 'n/a'} "
            f"{_format_pilot_interval(stats['ci95'])} | "
            f"{stats['bias'] if stats['bias'] is not None else 'n/a'} "
            f"{_format_pilot_interval(report['signed_ci'])} | "
            f"{stats['undercounted']} | {stats['overcounted']} | {stats['exact']} |"
        )

    rickshaw_groups = {
        "none": lambda count: count == 0,
        "one or two": lambda count: 1 <= count <= 2,
        "three or more": lambda count: count >= 3,
    }
    h3_lines: list[str] = []
    for group_name, predicate in rickshaw_groups.items():
        items = [
            item for item in sorted(labelled_eval)
            if predicate(labels[item]["pass1"]["rickshaw"])
        ]
        for model in config.PRETRAINED_MODEL_KEYS:
            errors = []
            for item in items:
                detections = cache.get((item, model, "standard"))
                if detections is None:
                    continue
                prediction = counting.count_by_class(
                    detections, list(config.TARGET_CLASS_NAMES)
                )
                errors.append(
                    evaluation.covered_total(prediction)
                    - evaluation.covered_total(labels[item]["pass1"])
                )
            mean_error = statistics.fmean(errors) if errors else None
            h3_lines.append(
                f"| {group_name} | {model_names[model]} | {len(errors)} | "
                f"{mean_error:.2f} |" if mean_error is not None else
                f"| {group_name} | {model_names[model]} | 0 | n/a |"
            )

    distant_items = {
        item for item in labelled_eval
        if "distant" in {tag.strip() for tag in conditions.get(item, "").split(";")}
    }
    selected_model = config.DEFAULT_MODEL_NAME
    standard_errors: list[float] = []
    sliced_errors: list[float] = []
    distant_clusters: list[str] = []
    for item in sorted(distant_items):
        standard = cache.get((item, selected_model, "standard"))
        sliced = cache.get((item, selected_model, "sliced"))
        if standard is None or sliced is None:
            continue
        truth = evaluation.covered_total(labels[item]["pass1"])
        standard_count = evaluation.covered_total(counting.count_by_class(
            standard, list(config.TARGET_CLASS_NAMES)
        ))
        sliced_count = evaluation.covered_total(counting.count_by_class(
            sliced, list(config.TARGET_CLASS_NAMES)
        ))
        standard_errors.append(abs(standard_count - truth))
        sliced_errors.append(abs(sliced_count - truth))
        distant_clusters.append(scenes.get(item, f"#{dataset.format_item_id(item)}"))
    if len(distant_items) < 3:
        h4_text = "Not tested: fewer than 3 distant images."
    else:
        differences = [
            standard - sliced
            for standard, sliced in zip(standard_errors, sliced_errors)
        ]
        h4_ci = evaluation.cluster_bootstrap_ci(
            differences, distant_clusters, BOOTSTRAP_SAMPLES
        )
        h4_text = (
            f"Descriptive only: {selected_model} standard mean absolute error "
            f"{statistics.fmean(standard_errors):.2f}; sliced "
            f"{statistics.fmean(sliced_errors):.2f}; paired standard-minus-sliced "
            f"difference {statistics.fmean(differences):.2f} "
            f"{_format_pilot_interval(h4_ci)} (n={len(differences)} cached pairs from "
            f"{len(distant_items)} labelled distant images)."
        ) if differences else (
            f"Descriptive only: no paired cached results available among "
            f"{len(distant_items)} labelled distant images."
        )

    common_items = [
        item for item in sorted(labelled_eval)
        if all((item, model, "standard") in cache for model in config.PRETRAINED_MODEL_KEYS)
    ]
    disagreement: dict[int, float] = {}
    true_errors: dict[int, float] = {}
    for item in common_items:
        by_model = {
            model_names[model]: cache[(item, model, "standard")]
            for model in config.PRETRAINED_MODEL_KEYS
        }
        family_by_name = {
            model_names[model]: config.MODEL_REGISTRY[model].family
            for model in config.PRETRAINED_MODEL_KEYS
        }
        agreement = reliability.build_reliability_report(
            by_model, family_by_name, config.DEFAULT_CONFIDENCE
        )
        cross = agreement.cross_family
        if cross is None:
            continue
        disagreement[item] = cross.count_disagreement
        totals = [
            evaluation.covered_total(counting.count_by_class(
                cache[(item, model, "standard")], list(config.TARGET_CLASS_NAMES)
            ))
            for model in config.PRETRAINED_MODEL_KEYS
        ]
        truth = evaluation.covered_total(labels[item]["pass1"])
        true_errors[item] = abs(statistics.median(totals) - truth)
    if disagreement:
        flagged_count = max(1, math.ceil(len(disagreement) / 4))
        flagged = set(sorted(disagreement, key=lambda item: (-disagreement[item], item))[:flagged_count])
        unflagged = set(disagreement) - flagged
        flagged_mean = statistics.fmean(true_errors[item] for item in flagged)
        unflagged_mean = (
            statistics.fmean(true_errors[item] for item in unflagged)
            if unflagged else None
        )
        flag_text = (
            f"Unvalidated; descriptive association on a pilot sample; not evidence that "
            f"disagreement predicts error. Highest-disagreement quarter: {len(flagged)} "
            f"flagged of {len(disagreement)} common cached images; mean absolute "
            f"median-model error {flagged_mean:.2f} flagged (n={len(flagged)}) versus "
            f"{unflagged_mean:.2f} unflagged (n={len(unflagged)})."
            if unflagged_mean is not None else
            f"Unvalidated; descriptive association on a pilot sample; not evidence that "
            f"disagreement predicts error. Highest-disagreement quarter: {len(flagged)} "
            f"flagged of {len(disagreement)} common cached images; mean absolute "
            f"median-model error {flagged_mean:.2f} flagged (n={len(flagged)}); no "
            "unflagged comparison is available."
        )
    else:
        flag_text = (
            "Unvalidated; descriptive association on a pilot sample; not evidence that "
            "disagreement predicts error. No common cached images are available."
        )

    conditions_table = "\n".join(
        f"| {tag} | {count} |" for tag, count in sorted(condition_counts.items())
    ) or "| none | 0 |"
    h1_h2 = (
        f"Not tested: {len(labelled_eval)} audited evaluation images, minimum 30 required."
    )
    consistency_text = (
        f"n={consistency['n']}, mean absolute covered difference "
        f"{consistency['mean_absolute_difference'] if consistency['mean_absolute_difference'] is not None else 'n/a'} "
        f"{_format_pilot_interval(consistency['mean_absolute_difference_ci95'])}."
    )
    second_pass_text = "\n".join(second_tables) or "| No second-pass rows | 0 | n/a | n/a | n/a |"
    development_text = "\n".join(dev_tables) or (
        "| No development rows with cached detections | 0 | 0 | n/a | n/a | 0 | 0 | 0 |"
    )
    markdown = f"""# Week 3 Pilot Evaluation

Pilot analysis using first-pass counts as the pre-registered reference. Frozen settings SHA-256: `{frozen['settings_sha256']}`. Existing cached detections only; no inference or tuning was run, and the predictor was not run. There are {len(labelled_eval)} audited evaluation first-pass images; standard-cache coverage varies by model, as shown below. {len(labelled_eval) - len(common_items)} evaluation images lack a complete standard cache across all three models.

## Main evaluation results

All rows are pilot results; `n` and scene counts are per model. Mean absolute and signed errors include scene-level bootstrap 95% intervals.

| Model | n images | n scenes | Mean absolute error [95% interval] | Mean signed error [95% interval] | Undercounted | Overcounted | Exact |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(standard_tables)}

### Second-pass sensitivity analysis

Second-pass counts are used as the reference only for items that have a second pass. Pilot results; sample sizes are per model.

| Model | n images | Mean absolute error [95% interval] | Mean signed error [95% interval] |
|---|---:|---:|---:|
{second_pass_text}

### Condition counts

Descriptive counts of labelled evaluation images by condition tag; images may carry multiple tags.

| Condition tag | Images |
|---|---:|
{conditions_table}

## Development set: settings were tuned here, results are optimistic

Separate descriptive results; development images are not pooled with evaluation results.

| Model | n images | n scenes | Mean absolute error [95% interval] | Mean signed error [95% interval] | Undercounted | Overcounted | Exact |
|---|---:|---:|---:|---:|---:|---:|---:|
{development_text}

## Labeller consistency

{consistency_text} Model errors below this level cannot be distinguished from labelling noise.

## Hypotheses

- H1: {h1_h2}
- H2: {h1_h2}
- H3: Descriptive only. Signed covered error (model covered total minus first-pass covered total) grouped by first-pass rickshaw count:

| Rickshaw group | Model | n images | Mean signed covered error |
|---|---|---:|---:|
{chr(10).join(h3_lines)}

This is not a confusion analysis; the models have no rickshaw class.
- H4: {h4_text}

## Needs-review flag

{flag_text}
"""
    aggregate = {
        "labelled_evaluation": len(labelled_eval),
        "common_cached_evaluation": len(common_items),
        "main": main_rows,
        "second_pass": second_rows,
        "development": dev_rows,
        "flagged_n": len(flagged) if disagreement else 0,
        "flagged_mean_error": flagged_mean if disagreement else None,
        "unflagged_n": len(unflagged) if disagreement else 0,
        "unflagged_mean_error": unflagged_mean if disagreement else None,
        "h4_n": len(standard_errors),
        "h4_status": h4_text,
    }
    return markdown, aggregate


def _run_pilot(args: argparse.Namespace) -> int:
    if not config.FINAL_SETTINGS_JSON.is_file():
        print(f"Pilot stopped: frozen settings file missing: {config.FINAL_SETTINGS_JSON}")
        return 1
    try:
        report, aggregate = _build_pilot_summary(
            args.labels, args.sources, args.plan, config.FINAL_SETTINGS_JSON,
            config.RAW_DETECTIONS_JSONL,
        )
        output = config.RESULTS_DIR / "week3_summary.md"
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", newline="", dir=output.parent,
                prefix=".week3_summary.", suffix=".tmp", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(report)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, output)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"ERROR: pilot evaluation failed: {error}")
        return 1
    print(f"Wrote pilot report: {output}")
    print(
        f"Pilot sample: {aggregate['labelled_evaluation']} audited evaluation images; "
        f"{aggregate['common_cached_evaluation']} have all three standard model caches."
    )
    for model, report in aggregate["main"].items():
        stats = report["stats"]
        print(
            f"{config.MODEL_REGISTRY[model].display_name}: n={stats['n']}, "
            f"MAE={stats['mae']}, signed error={stats['bias']}."
        )
    if aggregate["flagged_n"]:
        print(
            f"Descriptive flag comparison: flagged n={aggregate['flagged_n']}, "
            f"mean error={aggregate['flagged_mean_error']}; unflagged "
            f"n={aggregate['unflagged_n']}, mean error={aggregate['unflagged_mean_error']}."
        )
    if aggregate["h4_n"]:
        print(f"H4 paired cached sample: n={aggregate['h4_n']}.")
    return 0


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
            if item and model in config.PRETRAINED_MODEL_KEYS and mode in {"standard", "sliced"}:
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
        for key in config.PRETRAINED_MODEL_KEYS
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


def select_failure_cases(
    absolute_errors: dict[str, float],
    disagreements: dict[str, float],
    consistency: float,
    max_error_cases: int = 12,
    max_disagreement_cases: int = 5,
) -> dict[str, list[str]]:
    """Apply the registered consistency threshold and top-disagreement limits."""
    if not math.isfinite(consistency) or consistency < 0:
        raise ValueError("labeller consistency must be a finite non-negative value")
    error_items = sorted(
        (
            item for item, error in absolute_errors.items()
            if math.isfinite(error) and error > consistency
        ),
        key=lambda item: (-absolute_errors[item], item),
    )
    disagreement_items = sorted(
        (
            item for item, value in disagreements.items()
            if math.isfinite(value)
        ),
        key=lambda item: (-disagreements[item], item),
    )
    return {
        "error": error_items[:max_error_cases],
        "disagreement": disagreement_items[:max_disagreement_cases],
    }


_FAILURE_CAUSES = (
    "occlusion",
    "distant or small",
    "night or glare",
    "local vehicle type",
    "annotation or labelling issue",
    "dense overlap",
    "other",
)


def create_pilot_failure_cases(
    labels_path: Path = config.LABELS_CSV,
    sources_path: Path = config.SOURCES_CSV,
    plan_path: Path = config.TESTSET_PLAN_JSON,
    settings_path: Path = config.FINAL_SETTINGS_JSON,
    cache_path: Path = config.RAW_DETECTIONS_JSONL,
    images_dir: Path = config.TESTSET_IMAGES_DIR,
    output_dir: Path = config.RESULTS_DIR / "failures",
    report_path: Path = ROOT / "docs" / "failure_cases.md",
) -> dict[str, Any]:
    """Write selected pilot failure figures and a human-review report."""
    labels = _pilot_labels(labels_path)
    source_rows = _read_csv(sources_path)
    sources = {
        item: row for row in source_rows
        if (item := dataset.parse_item_number(row.get("item_id", ""))) is not None
    }
    image_ids = {
        item for item, row in sources.items()
        if not Path(row.get("filename", "")).suffix
        or Path(row.get("filename", "")).suffix.lower() in _IMAGE_SUFFIXES
    }
    development = evaluation.load_development_ids(plan_path)
    evaluation_ids = image_ids - development
    labelled_ids = {
        item for item in evaluation_ids
        if labels.get(item, {}).get("pass1") is not None
    }
    if len(labelled_ids) < MIN_PILOT_EVALUATION_LABELS:
        raise ValueError(
            "Failure analysis requires at least 10 audited evaluation first-pass labels; "
            f"found {len(labelled_ids)}."
        )

    frozen = settings.settings_record(settings_path)
    confidence = {
        model: float(record.get("confidence", config.DEFAULT_CONFIDENCE))
        for model, record in frozen["models"].items()
    }
    cached_entries = _load_matching_cache(cache_path, frozen["settings_sha256"])
    image_paths = _item_paths(images_dir)
    valid_entries: dict[tuple[int, str], list[dict[str, Any]]] = {}
    model_counts: dict[str, dict[int, dict[str, int]]] = {}
    model_mae: dict[str, float] = {}
    model_sample_size: dict[str, int] = {}
    for model in config.PRETRAINED_MODEL_KEYS:
        item_counts: dict[int, dict[str, int]] = {}
        for item in labelled_ids:
            path = image_paths.get(dataset.format_item_id(item))
            entry = cached_entries.get((dataset.format_item_id(item), model, "standard"))
            if (
                path is None or entry is None
                or entry.get("image_hash") != _sha256(path)
            ):
                continue
            detections = _filtered_detections(entry, confidence[model])
            valid_entries[(item, model)] = detections
            item_counts[item] = {
                name: sum(
                    1 for detection in detections
                    if detection.get("class_name", "").lower() == name
                )
                for name in config.TARGET_CLASS_NAMES
            }
        model_counts[model] = item_counts
        model_sample_size[model] = len(item_counts)
        if item_counts:
            model_mae[model] = statistics.fmean(
                abs(
                    sum(item_counts[item].values())
                    - evaluation.covered_total(labels[item]["pass1"])
                )
                for item in item_counts
            )
        else:
            model_mae[model] = float("inf")
    eligible_models = [
        model for model, count in model_sample_size.items()
        if count >= MIN_PILOT_EVALUATION_LABELS
    ]
    if not eligible_models:
        raise ValueError(
            "No model has matching frozen-settings detections for at least 10 "
            "labelled evaluation images."
        )
    best_model = min(
        eligible_models, key=lambda model: (model_mae[model], model)
    )
    first_pass_labels = {
        item: {"pass1": labels[item]["pass1"], "pass2": None}
        for item in labelled_ids
    }
    consistency = evaluation.labeller_pass_statistics(
        labels,
        evaluation.load_scenes(sources_path),
        excluded=development,
    )["covered_total"]["mean_absolute_difference"]
    if consistency is None:
        raise ValueError(
            "Failure analysis requires at least one evaluation item with two audited passes."
        )
    errors = {
        dataset.format_item_id(item): abs(
            sum(model_counts[best_model][item].values())
            - evaluation.covered_total(labels[item]["pass1"])
        )
        for item in labelled_ids if item in model_counts[best_model]
    }
    disagreements: dict[str, float] = {}
    for item in labelled_ids:
        totals = [
            sum(model_counts[model][item].values())
            for model in config.PRETRAINED_MODEL_KEYS
            if item in model_counts[model]
        ]
        if len(totals) >= 2:
            disagreements[dataset.format_item_id(item)] = max(totals) - min(totals)
    selected = select_failure_cases(errors, disagreements, consistency)
    error_ids = selected["error"]
    disagreement_ids = selected["disagreement"]
    union = list(dict.fromkeys(error_ids + disagreement_ids))
    if not union:
        raise ValueError(
            "No labelled evaluation items met the failure selection criteria."
        )

    selected_keys = set(union)
    dataset_boxes, boxes_unavailable = _source_boxes_by_item(
        selected_keys, image_paths
    )
    reviewed_causes = {
        "tl0007": (
            "dense overlap",
            "Several foreground vehicles overlap or are partly occluded in busy traffic.",
        ),
        "tl0038": (
            "distant or small",
            "The road scene is wide and multiple vehicles appear small in the distance.",
        ),
        "tl0022": (
            "distant or small",
            "Vehicles are spread across a broad roadway and some are small in the frame.",
        ),
        "tl0070": (
            "dense overlap",
            "Multiple nearby road users occupy the same busy area and partly overlap.",
        ),
        "tl0008": (
            "distant or small",
            "Most vehicles are small or far down the roadway.",
        ),
        "tl0029": (
            "distant or small",
            "The elevated view places many vehicles at a small apparent size.",
        ),
        "tl0044": (
            "distant or small",
            "Several vehicles are small against the broad road and distant background.",
        ),
        "tl0046": (
            "night or glare",
            "The dim dusk lighting reduces contrast around some vehicles.",
        ),
        "tl0043": (
            "dense overlap",
            "Crowded traffic includes vehicles close together and partially occluded.",
        ),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_records: list[dict[str, Any]] = []
    palette = {
        "dataset": DATASET_COLOR,
        "yolov8n": STANDARD_COLORS["yolov8n"],
        "yolov8s": STANDARD_COLORS["yolov8s"],
        "rtdetr-l": STANDARD_COLORS["rtdetr-l"],
    }
    label_names = list(config.LABEL_CLASS_NAMES)
    for item_id in union:
        item = dataset.parse_item_number(item_id)
        path = image_paths.get(item_id)
        if path is None:
            raise ValueError(f"selected failure image is missing: {item_id}")
        source = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if source is None:
            raise ValueError(f"could not decode selected failure image: {path}")
        height, width = source.shape[:2]
        scale = min(1.0, 1440 / width, 900 / height)
        if scale < 1:
            source = cv2.resize(
                source,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        canvas_width = max(1200, source.shape[1])
        image_x = (canvas_width - source.shape[1]) // 2
        canvas = np.full((source.shape[0] + 235, canvas_width, 3), 255, dtype=np.uint8)
        canvas[:source.shape[0], image_x:image_x + source.shape[1]] = source

        if item_id in dataset_boxes:
            for _category, box in dataset_boxes[item_id]:
                x0, y0, x1, y1 = box
                p0 = (image_x + int(x0 * scale), int(y0 * scale))
                p1 = (image_x + int(x1 * scale), int(y1 * scale))
                cv2.rectangle(canvas, p0, p1, palette["dataset"], 2)

        labels_for_item = labels[item]["pass1"]
        counts_for_item: dict[str, dict[str, int]] = {}
        footer = [
            (
                f"{item_id} | audited pass 1: "
                + ", ".join(f"{name}={labels_for_item[name]}" for name in label_names),
                (20, 20, 20),
            ),
            (
                "Dataset boxes: source annotations aligned"
                if item_id in dataset_boxes
                else "Dataset boxes unavailable: source-image bytes do not match selected copy",
                palette["dataset"] if item_id in dataset_boxes else (90, 90, 90),
            ),
        ]
        for model in config.PRETRAINED_MODEL_KEYS:
            detections = valid_entries.get((item, model))
            if detections is None:
                footer.append((
                    f"{config.MODEL_REGISTRY[model].display_name}: cached detections unavailable",
                    (90, 90, 90),
                ))
                continue
            for detection in detections:
                x0, y0, x1, y1 = detection["xyxy"]
                p0 = (image_x + int(float(x0) * scale), int(float(y0) * scale))
                p1 = (image_x + int(float(x1) * scale), int(float(y1) * scale))
                cv2.rectangle(canvas, p0, p1, palette[model], 2)
            counts_for_item[model] = model_counts[model][item]
            footer.append((
                f"{config.MODEL_REGISTRY[model].display_name}: "
                + ", ".join(
                    f"{name}={counts_for_item[model][name]}"
                    for name in config.TARGET_CLASS_NAMES
                ),
                palette[model],
            ))
        for index, (text, color) in enumerate(footer):
            y = source.shape[0] + 25 + index * 29
            cv2.putText(
                canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.53,
                color, 1, cv2.LINE_AA,
            )
        output_path = output_dir / f"{item_id}.png"
        if not cv2.imwrite(str(output_path), canvas):
            raise OSError(f"could not write failure figure: {output_path}")
        tags = [
            tag.strip() for tag in (sources[item].get("condition_tags") or "").split(";")
            if tag.strip()
        ]
        cause, explanation = reviewed_causes.get(
            item_id, ("other", "No specific visual cause was selected.")
        )
        image_records.append({
            "item_id": item_id,
            "tags": tags,
            "labels": labels_for_item,
            "models": counts_for_item,
            "missing_models": [
                model for model in config.PRETRAINED_MODEL_KEYS
                if model not in counts_for_item
            ],
            "error_selected": item_id in error_ids,
            "disagreement_selected": item_id in disagreement_ids,
            "cause": cause,
            "explanation": explanation,
        })

    def label_vector(counts: dict[str, int]) -> str:
        return ", ".join(f"{name}={counts[name]}" for name in label_names)

    def model_vector(model: str, counts: dict[str, int] | None) -> str:
        if counts is None:
            return f"{config.MODEL_REGISTRY[model].display_name}: cache unavailable"
        return (
            f"{config.MODEL_REGISTRY[model].display_name}: "
            + ", ".join(
                f"{name}={counts[name]}" for name in config.TARGET_CLASS_NAMES
            )
        )

    table_rows: list[str] = []
    for record in image_records:
        selection_reasons = []
        if record["error_selected"]:
            selection_reasons.append("best-model error")
        if record["disagreement_selected"]:
            selection_reasons.append("model disagreement")
        model_text = "<br>".join(
            model_vector(model, record["models"].get(model))
            for model in config.PRETRAINED_MODEL_KEYS
        )
        table_rows.append(
            f"| {record['item_id']} | {', '.join(record['tags']) or 'none'} | "
            f"{', '.join(selection_reasons)} | {label_vector(record['labels'])} | "
            f"{model_text} | {record['cause']} — unconfirmed | "
            f"{record['explanation']} Possible labeller miscount; blind recheck requested. |"
        )
    cause_counts = {
        cause: sum(1 for record in image_records if record["cause"] == cause)
        for cause in _FAILURE_CAUSES
    }
    cause_lines = [
        f"| {cause} | {cause_counts[cause]} |"
        for cause in _FAILURE_CAUSES
    ]
    overlap_count = len(set(error_ids) & set(disagreement_ids))
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# Pilot failure cases

Selection used the frozen final settings and existing cached detections only; these images were not used to tune anything. The best model was {config.MODEL_REGISTRY[best_model].display_name}, the lowest-MAE model among those with at least 10 image-hash-matched cached evaluation results. The labeller-consistency threshold was {consistency:.4f} covered vehicles. The first selection contains {len(error_ids)} of up to 12 images whose best-model absolute covered error strictly exceeds that threshold. The separate disagreement selection contains {len(disagreement_ids)} of up to 5 images, ranked by the range between available models' covered totals. {overlap_count} images overlap; {len(union)} unique images are shown.

Causes are judgments from viewing the images and are **unconfirmed** until the labeller confirms them. Possible labeller miscounts are flagged for blind review; a model discrepancy alone does not establish that the label is wrong.

## Important overlay limitations

The available cache has incomplete RT-DETR-L results for selected images; figures mark missing entries rather than generating detections. The source annotation files do not match the selected image copies byte-for-byte, so source boxes are not drawn because their alignment cannot be verified. Each figure states this explicitly. Audited labels and all available model counts are displayed for reference.

## Selected images

| Image id | Condition tags | Selection | Audited pass-1 counts | Model counts (covered classes) | Likely cause | Explanation / review |
|---|---|---|---|---|---|---|
{chr(10).join(table_rows)}

## Cause frequency

With this few selected images, this pattern is descriptive only.

| Cause | Images |
|---|---:|
{chr(10).join(cause_lines)}
"""
    report_path.write_text(report, encoding="utf-8")
    return {
        "best_model": best_model,
        "best_model_sample_size": model_sample_size[best_model],
        "error_selected_count": len(error_ids),
        "disagreement_selected_count": len(disagreement_ids),
        "overlap_count": overlap_count,
        "image_count": len(union),
        "source_boxes_aligned_count": len(dataset_boxes),
        "selected_images_missing_any_model": sum(
            bool(record["missing_models"]) for record in image_records
        ),
        "output_dir": str(output_dir),
        "report_path": str(report_path),
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
    parser.add_argument(
        "--pilot", action="store_true",
        help="write a pilot report from existing frozen-settings cache entries only",
    )
    args = parser.parse_args(argv)

    if args.pilot:
        return _run_pilot(args)

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
