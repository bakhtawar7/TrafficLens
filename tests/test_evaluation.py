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
import hashlib
import json
from pathlib import Path

import pytest

from core import config, dataset, evaluation as ev
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


def _write_sources_scene(path: Path, triples: list[tuple[str, str, str]]) -> None:
    """Write a SOURCES.csv carrying item_id, condition_tags and a scene column."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "condition_tags", "scene"])
        writer.writeheader()
        for item_id, tags, scene in triples:
            writer.writerow({"item_id": item_id, "condition_tags": tags, "scene": scene})


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


def test_load_detection_cache_requires_settings_hash_and_model_confidence(tmp_path):
    p = tmp_path / "r.jsonl"
    first = _entry("tl0002_x.jpg", "yolov8n", "standard",
                   [_det(box=(0, 0, 10, 10), conf=0.9)])
    second = _entry("tl0002_x.jpg", "yolov8s", "standard",
                    [_det(box=(0, 0, 10, 10), conf=0.9)])
    first["settings_sha256"] = "selected"
    second["settings_sha256"] = "other"
    _write_detections(p, [first, second])

    cache = ev.load_detection_cache(
        p,
        {"yolov8n": 0.95, "yolov8s": 0.25},
        settings_sha256="selected",
    )
    assert cache[(2, "yolov8n", "standard")] == []
    assert (2, "yolov8s", "standard") not in cache


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


def test_evaluate_reports_metrics_with_and_without_adjudicated_rows(workspace):
    with workspace["labels"].open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        original_fields = list(reader.fieldnames)
    extra_fields = ["adjudicated", *(f"original_{name}" for name in config.LABEL_CLASS_NAMES)]
    for row in rows:
        row.update({field: "" for field in extra_fields})
        if ev.normalise_item_id(row["item_id"]) == 2 and row["labeller_pass"] == "1":
            row["adjudicated"] = "yes"
            for name in config.LABEL_CLASS_NAMES:
                row[f"original_{name}"] = row[name]
            row["original_car"] = "5"
    with workspace["labels"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*original_fields, *extra_fields])
        writer.writeheader()
        writer.writerows(rows)

    summary = ev.evaluate(
        labels_csv=workspace["labels"], sources_csv=workspace["sources"],
        plan_json=workspace["plan"], detections_jsonl=workspace["dets"],
        table_out=workspace["table"], summary_out=workspace["summary"],
        write=False,
    )

    adjudication = summary["adjudication"]
    assert adjudication["adjudicated_rows"] == 1
    assert adjudication["preserved_values"][0]["original"]["car"] == 5
    assert adjudication["preserved_values"][0]["corrected"]["car"] == 4
    assert adjudication["with_adjudicated"]["model_mae"]["YOLOv8n"]["covered"]["mae"] == 0
    assert adjudication["without_adjudicated"]["model_mae"]["YOLOv8n"]["covered"]["mae"] == pytest.approx(0.5)


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


# ---------------------------------------------------------------------------
# Refusal of unverified draft rows
# ---------------------------------------------------------------------------

DRAFT_NOTE = dataset.UNVERIFIED_DRAFT_NOTE


def _draft(row: dict) -> dict:
    """Mark a label row as the unverified machine draft."""
    row["notes"] = DRAFT_NOTE
    return row


def test_the_canonical_draft_note_stays_inside_the_refusal_net():
    """The sentence both producers write must keep being refused by the evaluator.

    ``core.dataset.UNVERIFIED_DRAFT_NOTE`` is the only place that wording is spelled out, so
    this is the seam that breaks loudly if someone rewords the note, or rewords the markers,
    and would otherwise let an unverified draft count be scored in silence.
    """
    assert ev.is_unverified_note(dataset.UNVERIFIED_DRAFT_NOTE)
    lowered = dataset.UNVERIFIED_DRAFT_NOTE.lower()
    assert all(marker in lowered for marker in ev.UNVERIFIED_NOTE_MARKERS)


def test_is_unverified_note_matches_both_markers_case_insensitively():
    assert ev.is_unverified_note(DRAFT_NOTE)
    assert ev.is_unverified_note("Counts were AUTO DERIVED from the archive")
    assert ev.is_unverified_note("requires Labeller Verification")
    assert not ev.is_unverified_note("counted by hand, second pass")
    assert not ev.is_unverified_note("")
    assert not ev.is_unverified_note(None)


def test_load_labels_refuses_rows_marked_unverified(tmp_path):
    path = tmp_path / "LABELS.csv"
    _write_labels(path, [_row("tl001", {"car": 4}), _draft(_row("tl002", {"car": 3}))])
    assert set(ev.load_labels(path)) == {1}
    assert set(ev.load_labels(path, refuse_unverified=False)) == {1, 2}


def test_unverified_label_rows_reports_canonical_ids(tmp_path):
    path = tmp_path / "LABELS.csv"
    _write_labels(path, [_draft(_row("tl001", {"car": 4})), _row("tl002", {"car": 3})])
    assert ev.unverified_label_rows(path) == ["tl0001"]


def test_evaluate_refuses_to_score_draft_only_labels(workspace):
    """When every row is an unverified draft, nothing is scored and the refusal is stated."""
    _write_labels(workspace["labels"], [
        _draft(_row("tl002", {"car": 4, "rickshaw": 3})),
        _draft(_row("tl003", {"car": 2, "rickshaw": 1})),
    ])
    summary = ev.evaluate(
        labels_csv=workspace["labels"], sources_csv=workspace["sources"],
        plan_json=workspace["plan"], detections_jsonl=workspace["dets"],
        table_out=workspace["table"], summary_out=workspace["summary"],
        write=True,
    )
    refused = summary["refusals"]["unverified_label_rows"]
    assert refused["count"] == 2
    assert refused["items"] == ["tl0002", "tl0003"]
    assert summary["items"]["evaluated"] == 0
    assert "no error statistic computed" in summary["warning"]
    assert summary["model_mae"]["YOLOv8n"]["covered"]["mae"] is None
    with workspace["table"].open(encoding="utf-8", newline="") as handle:
        assert handle.readline().strip() == ",".join(ev.TABLE_COLUMNS)


def test_mixed_labels_score_only_the_verified_rows(workspace):
    """A verified row is still scored while its unverified neighbour is refused."""
    _write_labels(workspace["labels"], [
        _row("tl002", {"car": 4, "rickshaw": 3}),
        _draft(_row("tl003", {"car": 2, "rickshaw": 1})),
    ])
    summary = ev.evaluate(
        labels_csv=workspace["labels"], sources_csv=workspace["sources"],
        plan_json=workspace["plan"], detections_jsonl=workspace["dets"],
        table_out=workspace["table"], summary_out=workspace["summary"],
        write=False,
    )
    assert summary["refusals"]["unverified_label_rows"]["count"] == 1
    assert summary["items"]["evaluated"] == 1
    assert summary["items"]["present_in_table"] == ["tl0002"]


# ---------------------------------------------------------------------------
# Signed error, bias, relative error, cluster bootstrap, exploratory conditions
# ---------------------------------------------------------------------------
def test_error_stats_signed_negative_positive_exact_and_zero():
    # (prediction, ground truth): undercount, overcount, exact, zero-true exact, zero-true overcount.
    pairs = [(3, 5), (7, 5), (4, 4), (0, 0), (2, 0)]
    clusters = ["a", "b", "c", "d", "e"]  # distinct so the CI path is exercised lightly
    stats = ev.error_stats(pairs, clusters, samples=200)
    assert stats["n"] == 5
    assert stats["bias"] == pytest.approx(0.4)         # mean(-2, 2, 0, 0, 2)
    assert stats["mae"] == pytest.approx(1.2)          # mean(2, 2, 0, 0, 2)
    assert stats["undercounted"] == 1
    assert stats["overcounted"] == 2
    assert stats["exact"] == 2
    assert stats["zero_true_items"] == 2
    assert stats["relative_error_items"] == 3
    assert stats["mean_relative_error"] == pytest.approx(round((0.4 + 0.4 + 0.0) / 3, 4))


def test_error_stats_all_zero_true_has_no_relative_mean():
    stats = ev.error_stats([(0, 0), (2, 0)], ["a", "b"], samples=100)
    assert stats["mean_relative_error"] is None        # every true count was zero
    assert stats["relative_error_items"] == 0
    assert stats["zero_true_items"] == 2
    assert stats["exact"] == 1 and stats["overcounted"] == 1
    assert stats["bias"] == pytest.approx(1.0)


def test_error_stats_empty_is_a_zero_profile():
    stats = ev.error_stats([], [])
    assert stats["n"] == 0 and stats["mae"] is None and stats["bias"] is None
    assert stats["ci95"] == (None, None)
    assert stats["mean_relative_error"] is None and stats["zero_true_items"] == 0
    assert stats["undercounted"] == stats["overcounted"] == stats["exact"] == 0


def test_cluster_bootstrap_is_deterministic_and_handles_singletons():
    values = [1.0, 2.0, 3.0, 4.0]
    clusters = ["a", "a", "b", "b"]
    first = ev.cluster_bootstrap_ci(values, clusters, samples=500)
    second = ev.cluster_bootstrap_ci(values, clusters, samples=500)
    assert first == second and first[0] is not None
    assert ev.cluster_bootstrap_ci([], []) == (None, None)
    # a single cluster always resamples to the same mean -> zero-width interval
    lower, upper = ev.cluster_bootstrap_ci([2.0, 4.0], ["x", "x"], samples=200)
    assert lower == upper


def test_cluster_bootstrap_is_at_least_as_wide_as_image_level_on_correlated_scenes():
    # Within a scene every item shares a value; scene means alternate 0 and 10, so the
    # items are strongly clustered and the image-level bootstrap is over-confident.
    values, clusters = [], []
    for index in range(10):
        scene = f"scene{index}"
        value = 0.0 if index % 2 == 0 else 10.0
        for _ in range(3):
            values.append(value)
            clusters.append(scene)
    image_lo, image_hi = ev.bootstrap_ci(values, samples=2000)
    cluster_lo, cluster_hi = ev.cluster_bootstrap_ci(values, clusters, samples=2000)
    assert (cluster_hi - cluster_lo) >= (image_hi - image_lo)
    assert (cluster_hi - cluster_lo) > 0


def _scene_workspace(tmp_path: Path) -> dict:
    """Four items across two scenes, verified labels, one zero-truth and one overcount."""
    labels = tmp_path / "LABELS.csv"
    sources = tmp_path / "SOURCES.csv"
    plan = tmp_path / "testset_plan.json"
    dets = tmp_path / "raw_detections.jsonl"
    _write_labels(labels, [
        _row("tl002", {"car": 4, "rickshaw": 3}),
        _row("tl003", {"car": 2, "rickshaw": 1}),
        _row("tl005", {"car": 0}),
        _row("tl006", {"car": 1}),
    ])
    _write_sources_scene(sources, [
        ("tl002", "day", "sceneA"),
        ("tl003", "day", "sceneA"),
        ("tl005", "night", "sceneB"),
        ("tl006", "night", "sceneB"),
    ])
    _write_plan(plan, [])
    counts_by_item = {"tl0002": 4, "tl0003": 2, "tl0005": 0, "tl0006": 2}
    entries = []
    for image, count in counts_by_item.items():
        boxes = _car_boxes(count)
        for model in config.MODEL_REGISTRY:
            entries.append(_entry(f"{image}_dhaka_ai.jpg", model, "standard", boxes))
        entries.append(_entry(f"{image}_dhaka_ai.jpg", "yolov8n", "sliced", boxes))
    _write_detections(dets, entries)
    return {"labels": labels, "sources": sources, "plan": plan, "dets": dets,
            "table": tmp_path / "evaluation_table.csv", "summary": tmp_path / "summary.json"}


def test_build_rows_signed_errors_and_scene(tmp_path):
    ws = _scene_workspace(tmp_path)
    labels = ev.load_labels(ws["labels"])
    conditions = ev.load_conditions(ws["sources"])
    scenes = ev.load_scenes(ws["sources"])
    excluded = ev.load_development_ids(ws["plan"])
    cache = ev.load_detection_cache(ws["dets"])
    rows, _ = ev.build_rows(labels, conditions, cache, excluded, scenes=scenes)
    by_item = {r["Item"]: r for r in rows}

    assert scenes == {2: "sceneA", 3: "sceneA", 5: "sceneB", 6: "sceneB"}
    r2 = by_item["tl0002"]
    assert r2["Scene"] == "sceneA"
    assert r2["Ground Truth"] == 4 and r2["All Vehicles Ground Truth"] == 7
    assert r2["YOLOv8n Covered Signed Error"] == 0        # 4 - 4
    assert r2["YOLOv8n All Vehicles Signed Error"] == -3  # 4 - 7, an all-vehicles undercount
    r6 = by_item["tl0006"]
    assert r6["YOLOv8n Covered Signed Error"] == 1        # 2 - 1, an overcount


def test_evaluate_new_columns_appended_and_summary_enriched(tmp_path):
    ws = _scene_workspace(tmp_path)
    summary = ev.evaluate(
        labels_csv=ws["labels"], sources_csv=ws["sources"], plan_json=ws["plan"],
        detections_jsonl=ws["dets"], table_out=ws["table"], summary_out=ws["summary"],
        bootstrap_samples=200, write=True,
    )
    with ws["table"].open(encoding="utf-8", newline="") as handle:
        header = list(csv.DictReader(handle).fieldnames)
    # Existing ten columns keep their names and positions; new ones are appended.
    assert header[:10] == ["Item", "Condition", "Ground Truth", "YOLOv8n", "YOLOv8s",
                           "RT-DETR-L", "Sliced Count", "Count Range", "Agreement", "Actual Error"]
    assert header == list(ev.TABLE_COLUMNS)

    covered = summary["model_mae"]["YOLOv8n"]["covered"]
    for key in ("bias", "undercounted", "overcounted", "exact",
                "mean_relative_error", "relative_error_items", "zero_true_items"):
        assert key in covered
    # covered pairs (4,4),(2,2),(0,0),(2,1): one overcount, three exact, one zero-truth.
    assert covered["n"] == 4
    assert covered["overcounted"] == 1 and covered["exact"] == 3 and covered["undercounted"] == 0
    assert covered["zero_true_items"] == 1
    assert covered["bias"] == pytest.approx(0.25)
    assert covered["ci95"][0] is not None

    assert summary["settings"]["bootstrap_method"] == "scene-level cluster"
    assert summary["settings"]["exploratory_min_images"] == 15


def test_condition_detail_marks_exploratory_and_counts_scenes(tmp_path):
    ws = _scene_workspace(tmp_path)
    summary = ev.evaluate(
        labels_csv=ws["labels"], sources_csv=ws["sources"], plan_json=ws["plan"],
        detections_jsonl=ws["dets"], table_out=ws["table"], summary_out=ws["summary"],
        bootstrap_samples=200, write=True,
    )
    detail = summary["condition_detail"]
    assert detail["day"] == {"n_images": 2, "scenes": 1, "exploratory": True}
    assert detail["night"] == {"n_images": 2, "scenes": 1, "exploratory": True}
    with ws["table"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_item = {r["Item"]: r for r in rows}
    assert by_item["tl0002"]["Exploratory Conditions"] == "day"
    assert by_item["tl0006"]["Exploratory Conditions"] == "night"


def test_annotate_exploratory_threshold_and_scene_counts():
    rows = []
    # 'day' appears on 15 items (not exploratory); 'night' on 3 (exploratory); 'glare' on 14.
    for index in range(15):
        rows.append({"Item": f"tl{index:04d}", "Condition": "day", "Scene": f"day-scene-{index % 5}"})
    for index in range(3):
        rows.append({"Item": f"n{index:04d}", "Condition": "night", "Scene": "night-scene"})
    for index in range(14):
        rows.append({"Item": f"g{index:04d}", "Condition": "glare", "Scene": f"glare-{index}"})

    counts = ev.annotate_exploratory(rows)
    assert counts["day"] == 15 and counts["night"] == 3 and counts["glare"] == 14
    assert ev.exploratory_tags(counts) == {"night", "glare"}
    by_item = {r["Item"]: r for r in rows}
    assert by_item["tl0000"]["Exploratory Conditions"] == ""        # day is not exploratory
    assert by_item["n0000"]["Exploratory Conditions"] == "night"    # night is exploratory

    scene_counts = ev.condition_scene_counts(rows)
    assert scene_counts["day"] == 5      # five distinct day scenes
    assert scene_counts["night"] == 1    # all night images share one scene
    assert scene_counts["glare"] == 14


# ---------------------------------------------------------------------------
# Development isolation and settings provenance
# ---------------------------------------------------------------------------
def test_development_images_never_appear_in_reported_statistics(workspace):
    """The required isolation test.

    The development item is given every artefact an evaluated item has — labels
    in both passes, condition tags, and cached detections for all models and
    modes — and must still appear in no table row and no reported statistic.
    """
    with workspace["labels"].open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", *config.LABEL_CLASS_NAMES,
                                                     "labeller_pass", "notes"])
        writer.writerow(_row("tl001", {"car": 4}, labeller_pass=2))
    dev_entries = [_entry("tl0001_dhaka_ai.jpg", model, "standard", _car_boxes(4))
                   for model in config.MODEL_REGISTRY]
    dev_entries.append(_entry("tl0001_dhaka_ai.jpg", "yolov8n", "sliced", _car_boxes(4)))
    with workspace["dets"].open("a", encoding="utf-8") as handle:
        for entry in dev_entries:
            handle.write(json.dumps(entry) + "\n")

    summary = ev.evaluate(
        labels_csv=workspace["labels"], sources_csv=workspace["sources"],
        plan_json=workspace["plan"], detections_jsonl=workspace["dets"],
        table_out=workspace["table"], summary_out=workspace["summary"],
        write=True,
    )
    with workspace["table"].open(encoding="utf-8", newline="") as handle:
        items = [row["Item"] for row in csv.DictReader(handle)]
    assert items == ["tl0002", "tl0003"]

    counts = summary["items"]
    assert counts["development_excluded"] == 1
    assert counts["evaluated"] == 2
    assert counts["present_in_table"] == ["tl0002", "tl0003"]
    assert counts["missing_detections"] == ["tl0004"]
    for block in summary["model_mae"].values():
        assert block["covered"]["n"] == 2
        assert block["all_vehicles"]["n"] == 2
    # tl0001 shares the 'day' tag with tl0002; a leak would double the day count.
    assert summary["condition_detail"]["day"]["n_images"] == 1
    assert summary["condition_detail"]["night"]["n_images"] == 1
    for per_model in summary["condition_mae"].values():
        for block in per_model.values():
            assert block["n"] == 1
    # The development item's second pass must not enter the labeller statistics.
    assert summary["labeller_passes"]["items_both_passes"] == 2
    assert "tl0001" not in json.dumps(summary)


def test_summary_records_the_settings_file_fingerprint(workspace, tmp_path):
    settings_path = tmp_path / "run_settings.json"
    payload = {
        "models": {
            key: {"confidence": 0.25, "input_size": 1280,
                  "uses_nms": key != "rtdetr-l",
                  "nms_iou": None if key == "rtdetr-l" else 0.6,
                  "sliced": False}
            for key in config.MODEL_REGISTRY
        },
        "sliced_tiles": {"tile_size": 640, "overlap_ratio": 0.2, "iou_threshold": 0.5},
    }
    settings_path.write_text(json.dumps(payload), encoding="utf-8")
    summary = ev.evaluate(
        labels_csv=workspace["labels"], sources_csv=workspace["sources"],
        plan_json=workspace["plan"], detections_jsonl=workspace["dets"],
        table_out=workspace["table"], summary_out=workspace["summary"],
        settings_json=settings_path, write=True,
    )
    recorded = summary["settings"]
    assert recorded["settings_file"] == "run_settings.json"
    assert recorded["settings_sha256"] == hashlib.sha256(settings_path.read_bytes()).hexdigest()
    assert recorded["models"]["yolov8n"]["input_size"] == 1280
    assert recorded["models"]["yolov8n"]["nms_iou"] == 0.6
    assert recorded["models"]["rtdetr-l"]["uses_nms"] is False
    assert recorded["models"]["rtdetr-l"]["nms_iou"] is None
    on_disk = json.loads(workspace["summary"].read_text(encoding="utf-8"))
    assert on_disk["settings"]["settings_sha256"] == recorded["settings_sha256"]


def test_summary_defaults_to_the_registered_settings_file(workspace):
    summary = ev.evaluate(
        labels_csv=workspace["labels"], sources_csv=workspace["sources"],
        plan_json=workspace["plan"], detections_jsonl=workspace["dets"],
        table_out=workspace["table"], summary_out=workspace["summary"],
        write=False,
    )
    recorded = summary["settings"]
    assert recorded["settings_file"] == "default_settings.json"
    expected = hashlib.sha256(config.DEFAULT_SETTINGS_JSON.read_bytes()).hexdigest()
    assert recorded["settings_sha256"] == expected
    assert recorded["models"]["yolov8n"]["input_size"] == 640
    assert recorded["models"]["rtdetr-l"]["nms_iou"] is None


def test_evaluate_missing_settings_file_raises(workspace, tmp_path):
    with pytest.raises(FileNotFoundError):
        ev.evaluate(
            labels_csv=workspace["labels"], sources_csv=workspace["sources"],
            plan_json=workspace["plan"], detections_jsonl=workspace["dets"],
            settings_json=tmp_path / "absent.json", write=False,
        )
