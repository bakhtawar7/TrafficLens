"""Tests for core.evaluation on a small synthetic table.

Everything runs offline from temporary files: a hand written LABELS.csv,
SOURCES.csv, testset_plan.json, and a benchmark-style raw_detections.jsonl. The
fixtures are sized well under thirty labelled items on purpose, so the "fewer
than 30 labelled" warning path is exercised by the full ``evaluate`` run. Box
coordinates are chosen so class-aware box agreement is exactly 1.0 (identical
sets) or exactly 0.0 (disjoint sets), making the reported measure predictable.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from core import config, evaluation as ev
from core.detector import Detection

CAR = (2, "car")


def _det(cls=CAR, box=(0.0, 0.0, 10.0, 10.0), conf=0.9) -> dict:
    return {
        "class_id": cls[0],
        "class_name": cls[1],
        "confidence": conf,
        "xyxy": list(box),
    }


def _row(item_id: str, counts: dict, labeller_pass: int = 1) -> dict:
    row = {"item_id": item_id, "labeller_pass": str(labeller_pass), "notes": ""}
    for name in config.LABEL_CLASS_NAMES:
        row[name] = str(counts.get(name, 0))
    return row


def _car_boxes(n: int, offset: float = 0.0) -> list[dict]:
    """n identical-class car detections at well separated x positions."""
    return [_det(box=(offset + 100.0 * i, 0.0, offset + 100.0 * i + 10.0, 10.0))
            for i in range(n)]


def _entry(image: str, model: str, mode: str, dets: list[dict]) -> dict:
    return {"image": image, "image_hash": "h", "model": model, "model_name": model,
            "mode": mode, "threshold": 0.15, "load_seconds": 1.0, "inference_ms": 5.0,
            "width": 640, "height": 480, "detections": dets}


def _write_labels(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", *config.LABEL_CLASS_NAMES,
                                                     "labeller_pass", "notes"])
        writer.writeheader()
        writer.writerows(rows)


def _write_sources(path: Path, pairs: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "condition_tags"])
        writer.writeheader()
        for item_id, tags in pairs:
            writer.writerow({"item_id": item_id, "condition_tags": tags})


def _write_plan(path: Path, dev_ids: list[str]) -> None:
    path.write_text(json.dumps({"development_subset": dev_ids}), encoding="utf-8")


def _write_detections(path: Path, entries: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


@pytest.fixture
def workspace(tmp_path: Path) -> dict:
    """A two-item evaluation set plus one development and one missing item."""
    labels = tmp_path / "LABELS.csv"
    sources = tmp_path / "SOURCES.csv"
    plan = tmp_path / "testset_plan.json"
    dets = tmp_path / "raw_detections.jsonl"

    # item 1 is in the development subset and must never appear;
    # item 2 has identical boxes across models (agreement 1.0, covered error 0);
    # item 3 has disjoint boxes across models (agreement 0.0);
    # item 4 is labelled but has no cached detections (goes to missing).
    _write_labels(labels, [
        _row("tl001", {"car": 4}),
        _row("tl002", {"car": 4, "rickshaw": 3}),
        _row("tl002", {"car": 3, "rickshaw": 3}, labeller_pass=2),
        _row("tl003", {"car": 2, "motorcycle": 0, "rickshaw": 1}),
        _row("tl003", {"car": 2}, labeller_pass=2),
        _row("tl004", {"car": 5}),
    ])
    _write_sources(sources, [("tl001", "day;urban"), ("tl002", "day;urban"),
                             ("tl003", "night"), ("tl004", "rain")])
    _write_plan(plan, ["tl001"])

    same = _car_boxes(4)
    entries = []
    # item 2: every standard model reports the identical four cars.
    for model in config.MODEL_REGISTRY:
        entries.append(_entry("tl0002_dhaka_ai.jpg", model, "standard", same))
    entries.append(_entry("tl0002_dhaka_ai.jpg", "yolov8n", "sliced", same))
    # item 3: every standard model reports two cars, at disjoint offsets.
    boxes_by_model = {"yolov8n": _car_boxes(2, 0.0),
                      "yolov8s": _car_boxes(2, 1000.0),
                      "rtdetr-l": _car_boxes(2, 2000.0)}
    for model, boxes in boxes_by_model.items():
        entries.append(_entry("tl0003_dhaka_ai.jpg", model, "standard", boxes))
    entries.append(_entry("tl0003_dhaka_ai.jpg", "yolov8n", "sliced", boxes_by_model["yolov8n"]))
    _write_detections(dets, entries)

    return {"labels": labels, "sources": sources, "plan": plan, "dets": dets,
            "table": tmp_path / "evaluation_table.csv",
            "summary": tmp_path / "summary.json"}


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------
def test_normalise_item_id_collapses_zero_padding():
    assert ev.normalise_item_id("tl001") == 1
    assert ev.normalise_item_id("TL0001") == 1
    assert ev.normalise_item_id("  tl12 ") == 12
    assert ev.normalise_item_id("not-an-id") is None


def test_covered_and_all_vehicle_totals():
    counts = {"car": 2, "bus": 1, "truck": 0, "motorcycle": 1,
              "rickshaw": 3, "loader_rickshaw": 1, "other": 5}
    assert ev.covered_total(counts) == 4
    # "other" is not a vehicle class and is excluded from both targets.
    assert ev.all_vehicles_total(counts) == 8


def test_load_labels_keeps_rows_with_counts_and_splits_passes(tmp_path):
    p = tmp_path / "L.csv"
    _write_labels(p, [
        _row("tl002", {"car": 4}),
        _row("tl002", {"car": 3}, labeller_pass=2),
        {"item_id": "tl009", **{n: "" for n in config.LABEL_CLASS_NAMES},
         "labeller_pass": "1", "notes": "blank counts ignored"},
    ])
    labels = ev.load_labels(p)
    assert labels[2]["pass1"]["car"] == 4
    assert labels[2]["pass2"]["car"] == 3
    assert 9 not in labels  # a fully blank row is not a first pass count


def test_load_detection_cache_filters_confidence_and_class(tmp_path):
    p = tmp_path / "r.jsonl"
    _write_detections(p, [
        _entry("tl0002_x.jpg", "yolov8n", "standard",
               [_det(box=(0, 0, 10, 10), conf=0.9),
                _det(box=(20, 0, 30, 10), conf=0.10),        # below threshold
                _det(cls=(0, "person"), conf=0.99)]),          # not a vehicle
    ])
    cache = ev.load_detection_cache(p, confidence=0.25)
    kept = cache[(2, "yolov8n", "standard")]
    assert len(kept) == 1
    assert isinstance(kept[0], Detection)


def test_bootstrap_ci_is_deterministic_and_empty_safe():
    lo1, hi1 = ev.bootstrap_ci([1.0, 2.0, 3.0, 4.0], samples=500)
    lo2, hi2 = ev.bootstrap_ci([1.0, 2.0, 3.0, 4.0], samples=500)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 is not None and lo1 <= hi1
    assert ev.bootstrap_ci([]) == (None, None)


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------
def test_build_rows_excludes_dev_and_flags_missing(workspace):
    labels = ev.load_labels(workspace["labels"])
    conditions = ev.load_conditions(workspace["sources"])
    excluded = ev.load_development_ids(workspace["plan"])
    cache = ev.load_detection_cache(workspace["dets"])
    rows, missing = ev.build_rows(labels, conditions, cache, excluded)

    items = [r["Item"] for r in rows]
    assert items == ["tl0002", "tl0003"]     # dev (1) excluded, missing (4) not a row
    assert missing == [4]


def test_row_columns_and_values(workspace):
    labels = ev.load_labels(workspace["labels"])
    conditions = ev.load_conditions(workspace["sources"])
    excluded = ev.load_development_ids(workspace["plan"])
    cache = ev.load_detection_cache(workspace["dets"])
    rows, _ = ev.build_rows(labels, conditions, cache, excluded)
    by_item = {r["Item"]: r for r in rows}

    assert list(rows[0].keys()) == list(ev.TABLE_COLUMNS)

    r2 = by_item["tl0002"]
    assert r2["Ground Truth"] == 4
    assert r2["YOLOv8n"] == r2["YOLOv8s"] == r2["RT-DETR-L"] == 4
    assert r2["Sliced Count"] == 4
    assert r2["Count Range"] == "4-4"
    assert r2["Agreement"] == pytest.approx(1.0)
    assert r2["Actual Error"] == 0
    assert r2["Condition"] == "day;urban"

    r3 = by_item["tl0003"]
    assert r3["Ground Truth"] == 2
    assert r3["Count Range"] == "2-2"
    assert r3["Agreement"] == pytest.approx(0.0)
    assert r3["Actual Error"] == 0


# ---------------------------------------------------------------------------
# Summary statistics and end-to-end
# ---------------------------------------------------------------------------
def test_summary_model_mae_targets_and_labeller(workspace):
    summary = ev.evaluate(
        labels_csv=workspace["labels"], sources_csv=workspace["sources"],
        plan_json=workspace["plan"], detections_jsonl=workspace["dets"],
        table_out=workspace["table"], summary_out=workspace["summary"],
        write=False,
    )
    mae = summary["model_mae"]["YOLOv8n"]
    assert mae["covered"]["mae"] == 0.0
    assert mae["covered"]["n"] == 2
    # item 2 all-vehicles gt = 4+3 = 7 (count 4); item 3 gt = 2+1 = 3 (count 2).
    assert mae["all_vehicles"]["mae"] == pytest.approx(2.0)

    lab = summary["labeller_passes"]
    assert lab["items_both_passes"] == 2
    # covered pass1/pass2 differences: item2 |4-3|=1, item3 |2-2|=0.
    assert lab["covered_mean_abs_difference"] == pytest.approx(0.5)
    assert lab["identical_fraction"] == pytest.approx(0.5)


def test_summary_warns_below_thirty_and_records_counts(workspace):
    summary = ev.evaluate(
        labels_csv=workspace["labels"], sources_csv=workspace["sources"],
        plan_json=workspace["plan"], detections_jsonl=workspace["dets"],
        table_out=workspace["table"], summary_out=workspace["summary"],
        write=False,
    )
    assert summary["warning"] is not None and "fewer than 30" in summary["warning"]
    counts = summary["items"]
    assert counts["development_excluded"] == 1
    assert counts["evaluated"] == 2
    assert counts["missing_detections"] == ["tl0004"]


def test_evaluate_writes_table_and_summary(workspace):
    summary = ev.evaluate(
        labels_csv=workspace["labels"], sources_csv=workspace["sources"],
        plan_json=workspace["plan"], detections_jsonl=workspace["dets"],
        table_out=workspace["table"], summary_out=workspace["summary"],
        write=True,
    )
    assert workspace["table"].exists() and workspace["summary"].exists()
    with workspace["table"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0].keys()) == list(ev.TABLE_COLUMNS)
    assert len(rows) == 2
    on_disk = json.loads(workspace["summary"].read_text(encoding="utf-8"))
    assert on_disk["items"]["evaluated"] == summary["items"]["evaluated"]


def test_evaluate_rejects_unknown_sliced_model(workspace):
    with pytest.raises(ValueError):
        ev.evaluate(labels_csv=workspace["labels"], sources_csv=workspace["sources"],
                    plan_json=workspace["plan"], detections_jsonl=workspace["dets"],
                    sliced_model_key="not-a-model", write=False)
