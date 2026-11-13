"""Run the provisional H3 annotation-referenced confusion analysis.

The only console output is aggregate counts and rates. Individual box outcomes
are written to the marked results artefacts for reproducibility.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config, confusion, curation, dataset, settings  # noqa: E402
from core.detector import RtdetrDetector, YoloDetector  # noqa: E402


def _read_mapping(path: Path) -> tuple[dict[str, str], Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = curation.load_class_mapping(path)
    if not mapping:
        raise ValueError(f"class mapping is empty or invalid: {path}")
    source_root = Path(payload.get("dataset_path", "")).expanduser()
    if not source_root.is_dir():
        raise FileNotFoundError(f"source annotation folder not found: {source_root}")
    return mapping, source_root.resolve()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _infer_selected_sources(
    annotations: list[confusion.AnnotationBox],
) -> tuple[dict[tuple[int, str], list[confusion.DetectionBox]], dict[str, Any]]:
    """Run local standard inference at the sweep floor on source images with target boxes."""
    settings_record = settings.settings_record(config.DEFAULT_SETTINGS_JSON)
    settings_by_model = settings_record["models"]
    missing_weights = [
        config.PROJECT_ROOT / spec.weights_file
        for spec in (config.MODEL_REGISTRY[key] for key in config.PRETRAINED_MODEL_KEYS)
        if not (config.PROJECT_ROOT / spec.weights_file).is_file()
    ]
    if missing_weights:
        raise FileNotFoundError(
            "local model weights are required; missing "
            + ", ".join(str(path) for path in missing_weights)
            + "; no download attempted"
        )
    image_by_item: dict[int, Path] = {}
    for box in annotations:
        number = dataset.parse_item_number(box.item_id)
        if number is not None:
            image_by_item[number] = box.image_path

    cache: dict[tuple[int, str], list[confusion.DetectionBox]] = {}
    for model_key in config.PRETRAINED_MODEL_KEYS:
        spec = config.MODEL_REGISTRY[model_key]
        weight_path = config.PROJECT_ROOT / spec.weights_file
        if spec.family == "yolo":
            detector = YoloDetector(str(weight_path), spec.display_name)
        else:
            detector = RtdetrDetector(str(weight_path), spec.display_name)
        detector.warmup()
        model_settings = settings_by_model[model_key]
        input_size = int(model_settings["input_size"])
        nms_iou = (
            float(model_settings["nms_iou"])
            if model_settings.get("uses_nms") and model_settings.get("nms_iou") is not None
            else None
        )
        for item_number, image_path in sorted(image_by_item.items()):
            image = dataset.load_image(image_path)
            if image is None:
                raise ValueError(f"could not decode a selected source image: {image_path.name}")
            detections = detector.detect(
                image,
                confusion.LOW_CONFIDENCE_FLOOR,
                input_size=input_size,
                nms_iou=nms_iou,
            )
            cache[(item_number, model_key)] = [
                confusion.DetectionBox(
                    class_name=detection.class_name.strip().lower(),
                    confidence=float(detection.confidence),
                    xyxy=tuple(float(value) for value in detection.xyxy),
                )
                for detection in detections
            ]
    return cache, {
        "source": "fresh local standard-mode inference; no cache reused",
        "settings_file": settings_record["settings_file"],
        "settings_sha256": settings_record["settings_sha256"],
        "inference_threshold": confusion.LOW_CONFIDENCE_FLOOR,
        "reported_default_threshold": confusion.DEFAULT_CONFIDENCE,
        "images": len(image_by_item),
        "models": list(config.PRETRAINED_MODEL_KEYS),
        "weights": "all required local model files checked before loading; no download attempted",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Provisional H3 box confusion analysis; never use to select settings."
    )
    parser.add_argument("--source-root", default=None,
                        help="VOC archive root (defaults to configs/class_mapping.json)")
    parser.add_argument("--mapping", default=str(config.CLASS_MAPPING_JSON))
    parser.add_argument("--inventory", default=str(config.TESTSET_INVENTORY_CSV))
    parser.add_argument("--selection", default=str(config.TESTSET_SELECTION_CSV))
    parser.add_argument("--plan", default=str(config.TESTSET_PLAN_JSON))
    parser.add_argument("--sources", default=str(config.SOURCES_CSV))
    parser.add_argument("--output-dir", default=str(config.RESULTS_DIR))
    parser.add_argument("--audit-dir", default=str(config.H3_BOX_AUDIT_DIR))
    parser.add_argument("--bootstrap-samples", type=int, default=confusion.BOOTSTRAP_SAMPLES)
    args = parser.parse_args(argv)

    try:
        mapping, mapped_source_root = _read_mapping(Path(args.mapping))
        source_root = Path(args.source_root).expanduser().resolve() if args.source_root else mapped_source_root
        records = curation.load_records(source_root)
        annotations, source_summary = confusion.load_selected_annotations(
            source_root,
            Path(args.inventory),
            Path(args.selection),
            Path(args.plan),
            Path(args.sources),
            mapping,
            records,
        )
        if not annotations:
            raise ValueError("no selected annotations in the requested H3 categories were found")

        audit_pool = confusion.load_rickshaw_audit_pool(source_root, mapping, records)
        audit_summary = confusion.create_box_audit(
            audit_pool, Path(args.audit_dir)
        )
        verified_path = Path(args.audit_dir) / "verification_sheet.csv"
        verified = confusion.load_verified_classes(verified_path)
        cache, inference_summary = _infer_selected_sources(annotations)
        outcomes, summary = confusion.analyze(
            annotations,
            cache,
            verified_classes=verified,
            mapping=mapping,
            bootstrap_samples=args.bootstrap_samples,
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / config.H3_CONFUSION_CSV.name
        outcomes_path = output_dir / config.H3_BOX_OUTCOMES_CSV.name
        chart_path = output_dir / config.H3_CONFUSION_CHART.name
        metadata_path = output_dir / config.H3_CONFUSION_METADATA.name
        confusion.write_summary_csv(summary_path, summary)
        confusion.write_outcomes_csv(outcomes_path, outcomes)
        confusion.write_confusion_chart(chart_path, summary)
        _write_json(metadata_path, {
            "annotation_status": confusion.ANNOTATION_STATUS,
            "scope_status": confusion.EXPLORATORY_STATUS,
            "interpretation": "Provisional only. Source dataset boxes are unverified and must not be used to select inference settings.",
            "settings_selection": "none; no settings changed or selected from these results",
            "scope": "selected source images with target-category annotations; fresh standard-mode local inference",
            "source": source_summary,
            "inference": inference_summary,
            "source_annotation_boxes_in_scope": len(annotations),
            "boxes_analyzed": len(annotations),
            "box_model_outcomes": len(outcomes),
            "categories_below_50_boxes_are_exploratory": True,
            "box_audit": audit_summary,
        })

        print(
            f"{confusion.ANNOTATION_STATUS}; {confusion.EXPLORATORY_STATUS}. "
            f"Images analyzed: {inference_summary['images']}; boxes analyzed: {len(annotations)}; "
            f"model-box outcomes: {len(outcomes)}."
        )
        print("Inference: local weights; no download; standard mode; settings not selected from H3.")
        print("Aggregate by category/model (boxes, scenes, default-threshold miss rate):")
        for row in summary:
            miss_rate = row["miss_rate_at_default"]
            miss_rate_text = "n/a" if miss_rate is None else f"{miss_rate:.3f}"
            print(
                f"  {row['category']} / {row['model']}: boxes={row['annotated_boxes']}, "
                f"scenes={row['scene_count']}, miss_rate={miss_rate_text}, "
                f"exploratory_below_50={row['category_exploratory_below_50_boxes']}"
            )
        print(f"Aggregate CSV: {summary_path}")
        print(f"Confusion matrix: {chart_path}")
        print(f"Box audit crops/sheet: {args.audit_dir}")
        return 0
    except (OSError, ValueError, json.JSONDecodeError, curation.CurationError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
