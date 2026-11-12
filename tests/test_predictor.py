"""Synthetic-only tests for scene-grouped error prediction and review flags."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from core import config, features, predictor
from scripts import run_predictor


def _synthetic_rows(signal: bool, scenes: int = 24, per_scene: int = 4) -> list[predictor.PredictorRow]:
    rng = np.random.default_rng(20261113)
    rows: list[predictor.PredictorRow] = []
    for scene_number in range(scenes):
        for image_number in range(per_scene):
            item_index = scene_number * per_scene + image_number
            driver = rng.uniform(0, 10)
            noise = rng.normal(0, 0.35 if signal else 2.5)
            target = max(0.0, driver + noise if signal else rng.uniform(0, 10))
            values: dict[str, float | None] = {column: 0.0 for column in features.TABLE_COLUMNS[2:]}
            values["mean_brightness"] = driver
            values["yolov8n_mean_confidence"] = rng.uniform(0, 1)
            values["yolov8s_mean_confidence"] = rng.uniform(0, 1)
            values["rtdetr-l_mean_confidence"] = rng.uniform(0, 1)
            values["cross_family_count_disagreement"] = driver / 10
            values["within_family_count_disagreement"] = (driver / 10) * 0.4
            rows.append(predictor.PredictorRow(
                item_id=f"tl{item_index + 1:04d}",
                scene=f"scene-{scene_number:02d}",
                target=target,
                values=values,
            ))
    return rows


def test_scene_grouped_folds_never_split_a_scene():
    rows = _synthetic_rows(signal=True, scenes=11, per_scene=3)
    scenes = [row.scene for row in rows]
    for train, validation in predictor.scene_group_folds(scenes, n_splits=5):
        assert {scenes[index] for index in train}.isdisjoint(
            {scenes[index] for index in validation}
        )


def test_signal_dependent_errors_beat_both_baselines():
    rows = _synthetic_rows(signal=True)
    report = predictor.evaluate_rows(rows, bootstrap_samples=120)
    for baseline in ("mean_confidence", "training_mean"):
        improvement = report["paired_mae_improvement_vs_baselines"]["ridge"][baseline]
        assert improvement["estimate"] > 0
        assert improvement["low"] > 0
    assert report["paired_mae_improvement_vs_baselines"]["gradient_boosted"]["training_mean"]["estimate"] > 0


def test_random_noise_does_not_show_clear_skill_and_permutation_is_chance():
    rows = _synthetic_rows(signal=False, scenes=20, per_scene=4)
    report = predictor.evaluate_rows(rows, bootstrap_samples=160)
    improvement = report["paired_mae_improvement_vs_baselines"]["ridge"]["mean_confidence"]
    assert improvement["low"] <= 0 <= improvement["high"]

    permutation = predictor.permutation_skill_test(rows, permutations=19)
    assert permutation["one_sided_p_value"] >= 0.05


def test_h2_requires_positive_cross_family_advantage_with_interval():
    rows = _synthetic_rows(signal=True, scenes=20, per_scene=4)
    rng = np.random.default_rng(42)
    rows = [
        predictor.PredictorRow(
            item_id=row.item_id,
            scene=row.scene,
            target=row.target,
            values={
                **row.values,
                "cross_family_count_disagreement": row.values["mean_brightness"],
                "within_family_count_disagreement": float(rng.uniform()),
            },
        )
        for row in rows
    ]
    report = predictor.evaluate_rows(rows, bootstrap_samples=100)
    h2 = report["hypotheses"]["H2"]
    assert h2["cross_minus_within_spearman"]["estimate"] > 0
    assert h2["cross_minus_within_spearman"]["low"] > 0
    assert h2["reject"] is False


def test_primary_needs_review_flag_uses_the_highest_predicted_quarter():
    scores = np.arange(20, dtype=float)
    actual = np.arange(20, dtype=float)
    result = predictor._flag_metrics(
        actual, scores, [f"tl{i:04d}" for i in range(20)], 0.25
    )
    assert result["flagged_count"] == 5
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["mean_true_error_flagged"] > result["mean_true_error_unflagged"]


def test_audited_label_readiness_lists_exact_missing_ids(tmp_path: Path):
    sources = tmp_path / "SOURCES.csv"
    labels = tmp_path / "LABELS.csv"
    plan = tmp_path / "testset_plan.json"
    with sources.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "scene"])
        writer.writeheader()
        writer.writerows([
            {"item_id": "tl0001", "scene": "scene-a"},
            {"item_id": "tl0002", "scene": "scene-a"},
            {"item_id": "tl0003", "scene": "scene-b"},
        ])
    with labels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["item_id", *config.LABEL_CLASS_NAMES, "labeller_pass", "notes"]
        )
        writer.writeheader()
        writer.writerow({
            "item_id": "tl0001", **{name: 0 for name in config.LABEL_CLASS_NAMES},
            "labeller_pass": 1, "notes": "",
        })
    plan.write_text(json.dumps({"development_subset": ["tl0002"]}), encoding="utf-8")

    readiness = predictor.audited_label_readiness(sources, labels, plan)
    assert readiness == {
        "evaluation_total": 2,
        "evaluation_audited": 1,
        "evaluation_audited_ids": ["tl0001"],
        "evaluation_missing_ids": ["tl0003"],
        "development_total": 1,
        "development_missing_ids": ["tl0002"],
    }


def test_runner_refuses_small_labelled_set_and_lists_missing_ids(tmp_path: Path, capsys):
    sources = tmp_path / "SOURCES.csv"
    labels = tmp_path / "LABELS.csv"
    plan = tmp_path / "plan.json"
    features_path = tmp_path / "features.csv"
    evaluation_path = tmp_path / "evaluation.csv"
    for path in (features_path, evaluation_path):
        path.write_text("placeholder\n", encoding="utf-8")
    with sources.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "scene"])
        writer.writeheader()
        writer.writerows([
            {"item_id": "tl0001", "scene": "scene-a"},
            {"item_id": "tl0002", "scene": "scene-b"},
        ])
    with labels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["item_id", *config.LABEL_CLASS_NAMES, "labeller_pass", "notes"]
        )
        writer.writeheader()
    plan.write_text(json.dumps({"development_subset": ["tl0002"]}), encoding="utf-8")

    result = run_predictor.main([
        "--features", str(features_path),
        "--evaluation", str(evaluation_path),
        "--sources", str(sources),
        "--labels", str(labels),
        "--plan", str(plan),
        "--output", str(tmp_path / "report.json"),
    ])
    output = capsys.readouterr().out
    assert result == 1
    assert "at least 30 evaluation images" in output
    assert "Missing evaluation ids: tl0001" in output
    assert "Missing development ids: tl0002" in output


def test_runner_completes_on_synthetic_evaluation_only(tmp_path: Path, capsys):
    sources = tmp_path / "SOURCES.csv"
    labels = tmp_path / "LABELS.csv"
    plan = tmp_path / "plan.json"
    features_path = tmp_path / "features.csv"
    evaluation_path = tmp_path / "evaluation.csv"
    output_path = tmp_path / "predictor_report.json"
    ids = [f"tl{index:04d}" for index in range(1, 34)]
    development_ids = set(ids[:3])

    with sources.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "scene"])
        writer.writeheader()
        writer.writerows({
            "item_id": item_id,
            "scene": f"scene-{(index - 1) // 3:02d}",
        } for index, item_id in enumerate(ids, 1))
    with labels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["item_id", *config.LABEL_CLASS_NAMES, "labeller_pass", "notes"]
        )
        writer.writeheader()
        for item_id in ids:
            writer.writerow({
                "item_id": item_id, **{name: 0 for name in config.LABEL_CLASS_NAMES},
                "labeller_pass": 1, "notes": "",
            })
    plan.write_text(json.dumps({"development_subset": sorted(development_ids)}), encoding="utf-8")
    with features_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(features.TABLE_COLUMNS))
        writer.writeheader()
        for index, item_id in enumerate(ids):
            writer.writerow({
                "item_id": item_id,
                "settings_sha256": "synthetic",
                "mean_brightness": index,
                "yolov8n_mean_confidence": 0.5,
                "yolov8s_mean_confidence": 0.5,
                "rtdetr-l_mean_confidence": 0.5,
                "cross_family_count_disagreement": index / len(ids),
                "within_family_count_disagreement": 0.1,
            })
    with evaluation_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Item", "Actual Error", "Scene"])
        writer.writeheader()
        for index, item_id in enumerate(ids):
            if item_id in development_ids:
                continue
            writer.writerow({
                "Item": item_id,
                "Actual Error": index,
                "Scene": f"scene-{index // 3:02d}",
            })

    result = run_predictor.main([
        "--features", str(features_path),
        "--evaluation", str(evaluation_path),
        "--sources", str(sources),
        "--labels", str(labels),
        "--plan", str(plan),
        "--output", str(output_path),
        "--bootstrap-samples", "0",
        "--permutations", "0",
    ])
    output = capsys.readouterr().out
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert result == 0
    assert saved["sample_size"] == 30
    assert saved["exploratory"] is True
    assert "Sample size: 30 evaluation images (EXPLORATORY" in output


def test_join_uses_evaluator_error_and_excludes_development(tmp_path: Path):
    feature_path = tmp_path / "features.csv"
    eval_path = tmp_path / "evaluation.csv"
    source_path = tmp_path / "SOURCES.csv"
    with feature_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(features.TABLE_COLUMNS))
        writer.writeheader()
        writer.writerow({"item_id": "tl0001", "settings_sha256": "synthetic",
                         "mean_brightness": "2"})
        writer.writerow({"item_id": "tl0002", "settings_sha256": "synthetic",
                         "mean_brightness": "3"})
    with eval_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Item", "Actual Error", "Scene"])
        writer.writeheader()
        writer.writerow({"Item": "tl0001", "Actual Error": 4, "Scene": "one"})
        writer.writerow({"Item": "tl0002", "Actual Error": 8, "Scene": "two"})
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "scene"])
        writer.writeheader()
        writer.writerow({"item_id": "tl0001", "scene": "one"})
        writer.writerow({"item_id": "tl0002", "scene": "two"})

    rows = predictor.load_predictor_rows(feature_path, eval_path, source_path, {2})
    assert [(row.item_id, row.target, row.scene) for row in rows] == [("tl0001", 4.0, "one")]
