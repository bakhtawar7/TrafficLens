"""Write ids whose best standard model count differs substantially from pass 1.

The best model is the registry model with the lowest mean absolute covered-total
error over the fully labelled evaluation set, using the frozen settings cache.
The output contains ids only so images can be reviewed without model counts.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config, dataset, evaluation, settings  # noqa: E402


def qualifies_for_adjudication(first_count: int, model_count: int) -> bool:
    """Require both the registered absolute and relative discrepancy thresholds."""
    difference = abs(model_count - first_count)
    return difference >= 5 and difference >= 0.4 * first_count


def _evaluation_first_passes(
    labels_csv: Path, sources_csv: Path, plan_json: Path
) -> tuple[list[int], dict[int, dict[str, int]]]:
    excluded = evaluation.load_development_ids(plan_json)
    source_ids: set[int] = set()
    with sources_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            item = evaluation.normalise_item_id(row.get("item_id", ""))
            if item is not None and item not in excluded:
                source_ids.add(item)

    labels = evaluation.load_labels(labels_csv)
    complete_pass1: set[int] = set()
    with labels_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            item = evaluation.normalise_item_id(row.get("item_id", ""))
            try:
                pass_number = int(str(row.get("labeller_pass", "")).strip() or "1")
            except ValueError:
                continue
            if item is None or pass_number not in (None, 1):
                continue
            if evaluation.is_unverified_note(row.get("notes", "")):
                continue
            try:
                complete = all(
                    int(str(row.get(name, "")).strip()) >= 0
                    for name in config.LABEL_CLASS_NAMES
                )
            except ValueError:
                complete = False
            if complete:
                complete_pass1.add(item)
    missing = [
        item for item in sorted(source_ids)
        if labels.get(item, {}).get("pass1") is None or item not in complete_pass1
    ]
    if missing:
        missing_ids = ", ".join(dataset.format_item_id(item) for item in missing)
        raise ValueError(f"first-pass counts are missing for evaluation images: {missing_ids}")
    return sorted(source_ids), {
        item: labels[item]["pass1"] for item in sorted(source_ids)
    }


def create_adjudication_list(
    labels_csv: Path = config.LABELS_CSV,
    sources_csv: Path = config.SOURCES_CSV,
    plan_json: Path = config.TESTSET_PLAN_JSON,
    detections_jsonl: Path = config.RAW_DETECTIONS_JSONL,
    settings_json: Path = config.FINAL_SETTINGS_JSON,
    output_csv: Path = config.RESULTS_DIR / "label_qa" / "adjudication_list.csv",
) -> list[str]:
    """Validate readiness, select discrepancy ids, and atomically write an id-only CSV."""
    eval_ids, first_pass = _evaluation_first_passes(
        Path(labels_csv), Path(sources_csv), Path(plan_json)
    )
    settings_record = settings.settings_record(settings_json)
    confidence = {
        key: float(record.get("confidence", config.DEFAULT_CONFIDENCE))
        for key, record in settings_record["models"].items()
    }
    cache = evaluation.load_detection_cache(
        Path(detections_jsonl),
        confidence,
        settings_record["settings_sha256"],
    )
    model_keys = list(config.MODEL_REGISTRY)
    missing_cache = [
        (item, model)
        for item in eval_ids
        for model in model_keys
        if (item, model, "standard") not in cache
    ]
    if missing_cache:
        raise ValueError(
            "frozen standard detections are missing for evaluation images; "
            "run the benchmark before creating the adjudication list"
        )

    model_mae = {
        model: sum(
            abs(
                len(cache[(item, model, "standard")])
                - evaluation.covered_total(first_pass[item])
            )
            for item in eval_ids
        ) / len(eval_ids)
        for model in model_keys
    }
    best_model = min(model_keys, key=lambda model: model_mae[model])
    selected = [
        dataset.format_item_id(item)
        for item in eval_ids
        if qualifies_for_adjudication(
            evaluation.covered_total(first_pass[item]),
            len(cache[(item, best_model, "standard")]),
        )
    ]

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="adjudication_", suffix=".tmp", dir=output_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["item_id"])
            writer.writerows((item,) for item in selected)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=config.LABELS_CSV)
    parser.add_argument("--sources", type=Path, default=config.SOURCES_CSV)
    parser.add_argument("--plan", type=Path, default=config.TESTSET_PLAN_JSON)
    parser.add_argument("--detections", type=Path, default=config.RAW_DETECTIONS_JSONL)
    parser.add_argument("--settings", type=Path, default=config.FINAL_SETTINGS_JSON)
    parser.add_argument(
        "--output",
        type=Path,
        default=config.RESULTS_DIR / "label_qa" / "adjudication_list.csv",
    )
    args = parser.parse_args(argv)
    try:
        create_adjudication_list(
            args.labels, args.sources, args.plan, args.detections, args.settings, args.output
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    print(f"Adjudication ids written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
