"""Readiness-gate tests for the Week 3 pipeline orchestrator."""
from __future__ import annotations

import csv
import json

import cv2
import numpy as np
import pytest

from core import config
from scripts import run_week3


def test_week3_refuses_empty_labels_before_checker_or_pipeline(
    tmp_path, monkeypatch, capsys
):
    labels = tmp_path / "LABELS.csv"
    with labels.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(
            ["item_id", *config.LABEL_CLASS_NAMES, "labeller_pass", "notes"]
        )
    sources = tmp_path / "SOURCES.csv"
    with sources.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "filename"])
        writer.writeheader()
        writer.writerow({"item_id": "tl0001", "filename": "tl0001.jpg"})
        writer.writerow({"item_id": "tl0002", "filename": "tl0002.jpg"})
    plan = tmp_path / "testset_plan.json"
    plan.write_text(
        json.dumps({"development_subset": ["tl0001"], "items": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(run_week3.subprocess, "run", pytest.fail)

    result = run_week3.main([
        "--labels", str(labels), "--sources", str(sources), "--plan", str(plan)
    ])

    output = capsys.readouterr().out
    assert result == 1
    assert "Readiness not met; pipeline stopped:" in output
    assert "tl0001: missing audited first-pass count(s)" in output
    assert "tl0002 (car, bus, truck, motorcycle, rickshaw, loader_rickshaw, other)" in output
    assert "Need at least 30 audited evaluation images; found 0." in output


def test_readiness_requires_all_development_and_at_least_30_evaluation_labels(tmp_path):
    labels = tmp_path / "LABELS.csv"
    sources = tmp_path / "SOURCES.csv"
    plan = tmp_path / "testset_plan.json"
    columns = ["item_id", *config.LABEL_CLASS_NAMES, "labeller_pass", "notes"]
    with sources.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "filename"])
        writer.writeheader()
        for item in range(1, 34):
            writer.writerow({"item_id": f"tl{item:04d}", "filename": f"tl{item:04d}.jpg"})
    with labels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in [1, *range(2, 32)]:
            writer.writerow({
                "item_id": f"tl{item:04d}",
                **{category: 0 for category in config.LABEL_CLASS_NAMES},
                "labeller_pass": 1,
                "notes": "",
            })
    plan.write_text(
        json.dumps({"development_subset": ["tl0001"], "items": []}),
        encoding="utf-8",
    )

    assert run_week3.readiness_issues(labels, sources, plan) == []


def test_pipeline_label_snapshot_does_not_turn_blank_counts_into_zeros(tmp_path):
    source = tmp_path / "LABELS.csv"
    snapshot = tmp_path / "audited_LABELS.csv"
    columns = ["item_id", *config.LABEL_CLASS_NAMES, "labeller_pass", "notes"]
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow({
            "item_id": "tl0001",
            **{category: 0 for category in config.LABEL_CLASS_NAMES},
            "labeller_pass": 1,
        })
        writer.writerow({
            "item_id": "tl0002",
            **{category: 0 for category in config.LABEL_CLASS_NAMES[:-1]},
            "other": "",
            "labeller_pass": 1,
        })

    run_week3._write_complete_label_snapshot(source, snapshot)

    with snapshot.open("r", encoding="utf-8", newline="") as handle:
        assert [row["item_id"] for row in csv.DictReader(handle)] == ["tl0001"]


def test_synthetic_failure_gallery_reports_all_audited_and_model_counts(
    tmp_path, monkeypatch
):
    images = tmp_path / "images"
    images.mkdir()
    table_path = tmp_path / "evaluation.csv"
    labels_path = tmp_path / "labels.csv"
    cache_path = tmp_path / "cache.jsonl"
    settings_path = tmp_path / "final_settings.json"
    settings_path.write_text(json.dumps({
        "models": {
            key: {"confidence": 0.25, "input_size": 640, "uses_nms": True, "nms_iou": 0.7}
            for key in config.MODEL_REGISTRY
        },
    }), encoding="utf-8")
    monkeypatch.setattr(config, "TESTSET_SELECTION_CSV", tmp_path / "absent_selection.csv")
    monkeypatch.setattr(config, "TESTSET_INVENTORY_CSV", tmp_path / "absent_inventory.csv")
    monkeypatch.setattr(config, "CLASS_MAPPING_JSON", tmp_path / "absent_mapping.json")

    evaluation_fields = [
        "Item", "Condition", "Ground Truth", "All Vehicles Ground Truth",
        "YOLOv8n", "YOLOv8s", "RT-DETR-L", "Sliced Count", "Actual Error", "Scene",
    ]
    label_fields = [
        "item_id", *config.LABEL_CLASS_NAMES, "labeller_pass", "notes",
    ]
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=evaluation_fields)
        writer.writeheader()
        for item in range(1, 13):
            writer.writerow({
                "Item": f"tl{item:04d}", "Condition": "distant",
                "Ground Truth": 1, "All Vehicles Ground Truth": 1,
                "YOLOv8n": 1, "YOLOv8s": 1, "RT-DETR-L": 1,
                "Sliced Count": 1, "Actual Error": 12 - item,
                "Scene": f"scene-{item // 2}",
            })
    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=label_fields)
        writer.writeheader()
        for item in range(1, 13):
            writer.writerow({
                "item_id": f"tl{item:04d}", "car": 1,
                **{category: 0 for category in config.LABEL_CLASS_NAMES if category != "car"},
                "labeller_pass": 1, "notes": "",
            })

    from core import settings

    settings_hash = settings.settings_sha256(settings_path)
    with cache_path.open("w", encoding="utf-8") as handle:
        for item in range(1, 13):
            filename = f"tl{item:04d}_synthetic.jpg"
            image = np.zeros((24, 32, 3), dtype=np.uint8)
            cv2.rectangle(image, (3, 3), (12, 12), (80, 120, 200), -1)
            image_path = images / filename
            assert cv2.imwrite(str(image_path), image)
            digest = run_week3._sha256(image_path)
            for model in config.MODEL_REGISTRY:
                modes = ("standard", "sliced") if model == config.DEFAULT_MODEL_NAME else ("standard",)
                for mode in modes:
                    handle.write(json.dumps({
                        "image": filename,
                        "image_hash": digest,
                        "settings_sha256": settings_hash,
                        "model": model,
                        "mode": mode,
                        "detections": [{
                            "class_name": "car",
                            "confidence": 0.9,
                            "xyxy": [3, 3, 12, 12],
                        }],
                    }) + "\n")

    gallery_path = tmp_path / "failure_gallery.png"
    result = run_week3.create_failure_gallery(
        table_path=table_path,
        cache_path=cache_path,
        labels_path=labels_path,
        output_path=gallery_path,
        images_dir=images,
        settings_path=settings_path,
    )

    assert result["image_count"] == 12
    assert result["dataset_box_aligned_images"] == 0
    assert gallery_path.is_file()
    with (tmp_path / "failure_gallery_counts.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        count_rows = list(csv.DictReader(handle))
    assert len(count_rows) == 12
    assert count_rows[0]["audited_covered_count"] == "1"
    assert count_rows[0]["audited_all_vehicle_count"] == "1"
    assert set(json.loads(count_rows[0]["model_counts_json"]).values()) == {1}
