"""Tests for core.settings and the pre-registered configs/default_settings.json.

The defaults file is pinned against the config constants and the ultralytics
predict defaults it documents, so a silent drift between what the code runs
and what the file claims is caught here. Hashing is checked against hashlib
on temporary files, never against a stored digest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core import config, settings


def test_default_settings_cover_every_registry_model():
    data = settings.load_settings()
    assert set(data["models"]) == set(config.MODEL_REGISTRY)


def test_default_settings_match_the_documented_defaults():
    data = settings.load_settings()
    for key, record in data["models"].items():
        assert record["confidence"] == config.DEFAULT_CONFIDENCE
        assert record["input_size"] == 640          # ultralytics predict default
        assert record["sliced"] is False
        if config.MODEL_REGISTRY[key].family == "yolo":
            assert record["uses_nms"] is True
            assert record["nms_iou"] == 0.7         # ultralytics NMS default
        else:
            assert record["uses_nms"] is False      # RT-DETR is NMS-free
            assert record["nms_iou"] is None
    assert data["sliced_tiles"] == {
        "tile_size": config.SLICE_TILE_SIZE,
        "overlap_ratio": config.SLICE_OVERLAP_RATIO,
        "iou_threshold": config.SLICE_IOU_THRESHOLD,
    }


def test_load_settings_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        settings.load_settings(tmp_path / "absent.json")


def test_settings_sha256_matches_hashlib_and_tracks_edits(tmp_path):
    path = tmp_path / "settings.json"
    payload = {"models": {"yolov8n": {"input_size": 640}}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert settings.settings_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    payload["models"]["yolov8n"]["input_size"] = 1280
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert settings.settings_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_settings_sha256_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        settings.settings_sha256(tmp_path / "absent.json")


def test_settings_record_carries_name_hash_and_model_fields(tmp_path):
    path = tmp_path / "custom_settings.json"
    payload = {
        "models": {"yolov8n": {"confidence": 0.25, "input_size": 1280,
                               "uses_nms": True, "nms_iou": 0.5, "sliced": False}},
        "sliced_tiles": {"tile_size": 640, "overlap_ratio": 0.2, "iou_threshold": 0.5},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    record = settings.settings_record(path)
    assert record["settings_file"] == "custom_settings.json"
    assert record["settings_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["models"] == payload["models"]
    assert record["sliced_tiles"] == payload["sliced_tiles"]


def test_default_path_resolves_to_the_registered_file():
    assert settings.resolve_settings_path() == config.DEFAULT_SETTINGS_JSON
    assert settings.resolve_settings_path(None) == config.DEFAULT_SETTINGS_JSON
    explicit = Path("configs") / "other.json"
    assert settings.resolve_settings_path(explicit) == explicit