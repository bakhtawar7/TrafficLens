"""Tests for the settings provenance wired into scripts/run_benchmark.py.

The benchmark runs offline here: build_detector and SlicedDetector are
replaced with a stub, and the images are synthetic jpegs under tmp_path. What
is under test is the plumbing — the --settings flag, the default file, and the
hash/input-size/NMS fields written into every cached entry.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from core import config
from scripts import run_benchmark as rb


class _StubDetector:
    name = "stub"
    load_seconds = 0.01

    def warmup(self) -> None:
        pass

    def detect(self, image, confidence_threshold: float) -> list:
        return []


@pytest.fixture
def benchmark_workspace(tmp_path: Path, monkeypatch) -> dict:
    images = tmp_path / "images"
    images.mkdir()
    ok, encoded = cv2.imencode(".jpg", np.full((16, 16, 3), 90, dtype=np.uint8))
    assert ok
    encoded.tofile(str(images / "tl0001_dhaka_ai.jpg"))
    monkeypatch.setattr(rb, "build_detector", lambda key: _StubDetector())
    monkeypatch.setattr(rb, "SlicedDetector", lambda detector: detector)
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({
        "models": {"yolov8n": {"confidence": 0.25, "input_size": 640,
                               "uses_nms": True, "nms_iou": 0.7, "sliced": False}},
    }), encoding="utf-8")
    return {"images": images, "settings": settings_path,
            "output": tmp_path / "raw_detections.jsonl"}


def _entries(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_entries_record_the_settings_fingerprint(benchmark_workspace):
    ws = benchmark_workspace
    code = rb.main(["--images-dir", str(ws["images"]), "--models", "yolov8n",
                    "--output", str(ws["output"]), "--settings", str(ws["settings"])])
    assert code == 0
    entries = _entries(ws["output"])
    assert {entry["mode"] for entry in entries} == {"standard", "sliced"}
    expected_hash = hashlib.sha256(ws["settings"].read_bytes()).hexdigest()
    for entry in entries:
        assert entry["settings_file"] == "settings.json"
        assert entry["settings_sha256"] == expected_hash
        assert entry["input_size"] == 640
        assert entry["uses_nms"] is True
        assert entry["nms_iou"] == 0.7


def test_default_settings_used_when_flag_omitted(benchmark_workspace):
    ws = benchmark_workspace
    code = rb.main(["--images-dir", str(ws["images"]), "--models", "yolov8n",
                    "--output", str(ws["output"])])
    assert code == 0
    expected_hash = hashlib.sha256(config.DEFAULT_SETTINGS_JSON.read_bytes()).hexdigest()
    for entry in _entries(ws["output"]):
        assert entry["settings_file"] == "default_settings.json"
        assert entry["settings_sha256"] == expected_hash
        assert entry["input_size"] == 640


def test_missing_settings_file_stops_before_any_model_work(benchmark_workspace, monkeypatch):
    ws = benchmark_workspace

    def _never_called(key):
        raise AssertionError("a model was loaded despite the missing settings file")

    monkeypatch.setattr(rb, "build_detector", _never_called)
    code = rb.main(["--images-dir", str(ws["images"]), "--models", "yolov8n",
                    "--output", str(ws["output"]),
                    "--settings", str(ws["settings"].parent / "absent.json")])
    assert code == 1
    assert not ws["output"].exists()