"""Synthetic tests for the isolated fine-tuned rickshaw path."""
from __future__ import annotations

import cv2
import numpy as np

from core import config, evaluation, finetuned
from scripts import run_finetuned_evaluation as runner


def test_finetuned_mapping_rule_and_registered_checkpoint():
    assert finetuned.map_finetuned_class("rickshaw") == "rickshaw"
    assert finetuned.map_finetuned_class("auto rickshaw") == "rickshaw"
    assert finetuned.map_finetuned_class("CNG") == "rickshaw"
    assert finetuned.map_finetuned_class("three wheeler cargo") == "loader_rickshaw"
    assert finetuned.map_finetuned_class("person") == "other"
    assert finetuned.load_finetuned_class_mapping() == {0: ("rickshaw", "rickshaw")}
    spec = config.MODEL_REGISTRY[config.FINETUNED_MODEL_KEY]
    assert spec.finetuned is True
    assert spec.family == "yolo"
    assert spec.weights_sha256 == "44a6a3e383f074c79a56d9160acbf3f627b056a94a54f843f9ee7f0cef942c24"


def test_contamination_check_uses_project_perceptual_hash(tmp_path):
    training_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    training_dir.mkdir()
    test_dir.mkdir()
    rng = np.random.default_rng(42)
    original = rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)
    unrelated = rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)
    assert cv2.imwrite(str(training_dir / "source.jpg"), original)
    assert cv2.imwrite(str(test_dir / "tl0001.jpg"), original)
    assert cv2.imwrite(str(test_dir / "tl0002.jpg"), unrelated)

    overlaps = finetuned.find_training_contamination(
        test_dir, [training_dir / "source.jpg"]
    )

    assert set(overlaps) == {"tl0001"}
    assert overlaps["tl0001"]["nearest_hash_distance"] == 0


def test_development_and_evaluation_items_are_separate_and_exclude_overlap():
    dev, evaluation_ids = finetuned.partition_labelled_items(
        {"tl0001", "tl0002", "tl0003"},
        {"tl0001", "tl0002"},
        {"tl0002"},
    )
    assert dev == {"tl0001"}
    assert evaluation_ids == {"tl0003"}
    assert not (dev & evaluation_ids)


def test_finetuned_evaluator_writes_separate_outputs_and_preserves_base_columns(tmp_path):
    original_columns = evaluation.TABLE_COLUMNS
    output = tmp_path / "finetuned.csv"
    runner._write_rows(output, [{"item_id": "tl0001", "model": config.FINETUNED_MODEL_KEY}])

    assert output != config.EVALUATION_TABLE_CSV
    assert evaluation.TABLE_COLUMNS is original_columns
    assert tuple(evaluation.TABLE_COLUMNS[:10]) == (
        "Item", "Condition", "Ground Truth", "YOLOv8n", "YOLOv8s",
        "RT-DETR-L", "Sliced Count", "Count Range", "Agreement", "Actual Error",
    )
