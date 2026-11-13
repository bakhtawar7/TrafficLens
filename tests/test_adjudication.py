from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from core import config, evaluation as ev, settings
from scripts.make_adjudication_list import create_adjudication_list, qualifies_for_adjudication


def _counts(**values: int) -> dict[str, int]:
    return {name: values.get(name, 0) for name in config.LABEL_CLASS_NAMES}


def test_labeller_pass_statistics_include_category_and_covered_differences():
    labels = {
        2: {"pass1": _counts(car=4, bus=1), "pass2": _counts(car=2, bus=1)},
        3: {"pass1": _counts(car=1, motorcycle=2), "pass2": _counts(car=2, motorcycle=2)},
        5: {"pass1": _counts(bus=3), "pass2": _counts()},
        6: {"pass1": _counts(truck=2), "pass2": _counts(truck=2)},
    }

    result = ev.labeller_pass_statistics(
        labels,
        scenes={2: "scene-a", 3: "scene-a", 5: "scene-b", 6: "scene-b"},
        samples=400,
    )

    assert result["items_both_passes"] == 4
    assert result["items"][0]["categories"]["car"] == -2
    assert result["items"][0]["covered_total_difference"] == -2
    assert result["per_category"]["car"]["mean_difference"] == pytest.approx(-0.25)
    assert result["per_category"]["car"]["mean_absolute_difference"] == pytest.approx(0.75)
    assert result["per_category"]["bus"]["mean_absolute_difference"] == pytest.approx(0.75)
    assert result["covered_total"]["mean_difference"] == pytest.approx(-1.0)
    assert result["covered_total"]["mean_absolute_difference"] == pytest.approx(1.5)
    assert result["covered_total"]["exact_match_fraction"] == pytest.approx(0.25)
    assert result["covered_total"]["mean_absolute_difference_ci95"] == ev.cluster_bootstrap_ci(
        [2, 1, 3, 0], ["scene-a", "scene-a", "scene-b", "scene-b"], samples=400
    )


def test_adjudication_rule_requires_both_absolute_and_relative_thresholds():
    assert qualifies_for_adjudication(10, 15)
    assert qualifies_for_adjudication(0, 5)
    assert not qualifies_for_adjudication(15, 20)
    assert not qualifies_for_adjudication(10, 14)


def test_adjudication_loader_preserves_original_and_corrected_values(tmp_path):
    path = tmp_path / "LABELS.csv"
    fields = [
        "item_id", *config.LABEL_CLASS_NAMES,
        *(f"original_{name}" for name in config.LABEL_CLASS_NAMES),
        "labeller_pass", "adjudicated",
    ]
    row = {
        "item_id": "tl0002",
        **_counts(car=3),
        **{f"original_{name}": value for name, value in _counts(car=5).items()},
        "labeller_pass": "1",
        "adjudicated": "yes",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)

    labels = ev.load_labels(path)
    adjudications = ev.load_adjudications(path)
    restored = ev.restore_original_labels(labels, adjudications)

    assert labels[2]["pass1"]["car"] == 3
    assert adjudications[2]["original"]["car"] == 5
    assert adjudications[2]["corrected"]["car"] == 3
    assert restored[2]["pass1"]["car"] == 5


def test_adjudication_list_refuses_incomplete_evaluation_labels(tmp_path):
    labels = tmp_path / "LABELS.csv"
    sources = tmp_path / "SOURCES.csv"
    plan = tmp_path / "plan.json"
    output = tmp_path / "label_qa" / "adjudication_list.csv"
    with labels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", *config.LABEL_CLASS_NAMES, "labeller_pass"])
        writer.writeheader()
        writer.writerow({"item_id": "tl0002", "car": 1, "labeller_pass": 1})
    with sources.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id"])
        writer.writeheader()
        writer.writerows({"item_id": item} for item in ("tl0002", "tl0003"))
    plan.write_text(json.dumps({"development_subset": ["tl0001"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="tl0003"):
        create_adjudication_list(
            labels_csv=labels,
            sources_csv=sources,
            plan_json=plan,
            detections_jsonl=tmp_path / "missing.jsonl",
            settings_json=tmp_path / "missing-settings.json",
            output_csv=output,
        )
    assert not output.exists()


def test_adjudication_list_writes_only_qualifying_ids(tmp_path):
    labels = tmp_path / "LABELS.csv"
    sources = tmp_path / "SOURCES.csv"
    plan = tmp_path / "plan.json"
    settings_path = tmp_path / "settings.json"
    detections = tmp_path / "detections.jsonl"
    output = tmp_path / "label_qa" / "adjudication_list.csv"

    fields = ["item_id", *config.LABEL_CLASS_NAMES, "labeller_pass", "notes"]
    with labels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item, car_count in (("tl0002", 10), ("tl0003", 20)):
            writer.writerow({
                "item_id": item, **_counts(car=car_count),
                "labeller_pass": "1", "notes": "",
            })
    with sources.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id"])
        writer.writeheader()
        writer.writerows({"item_id": item} for item in ("tl0002", "tl0003"))
    plan.write_text(json.dumps({"development_subset": ["tl0001"]}), encoding="utf-8")
    settings_path.write_text(json.dumps({
        "models": {key: {"confidence": 0.25} for key in config.MODEL_REGISTRY},
    }), encoding="utf-8")
    settings_hash = settings.settings_sha256(settings_path)

    def detections_for(count: int) -> list[dict]:
        return [
            {"class_id": 2, "class_name": "car", "confidence": 0.9,
             "xyxy": [100.0 * index, 0.0, 100.0 * index + 10, 10.0]}
            for index in range(count)
        ]

    counts = {
        ("tl0002", "yolov8n"): 15, ("tl0002", "yolov8s"): 10, ("tl0002", "rtdetr-l"): 20,
        ("tl0003", "yolov8n"): 20, ("tl0003", "yolov8s"): 30, ("tl0003", "rtdetr-l"): 30,
    }
    with detections.open("w", encoding="utf-8") as handle:
        for (item, model), count in counts.items():
            handle.write(json.dumps({
                "image": f"{item}_sample.jpg",
                "model": model,
                "mode": "standard",
                "settings_sha256": settings_hash,
                "detections": detections_for(count),
            }) + "\n")

    selected = create_adjudication_list(
        labels_csv=labels,
        sources_csv=sources,
        plan_json=plan,
        detections_jsonl=detections,
        settings_json=settings_path,
        output_csv=output,
    )

    assert selected == ["tl0002"]
    with output.open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == [{"item_id": "tl0002"}]
