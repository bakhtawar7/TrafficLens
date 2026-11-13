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

    def detect(self, image, confidence_threshold: float, **kwargs) -> list:
        return []


@pytest.fixture
def benchmark_workspace(tmp_path: Path, monkeypatch) -> dict:
    images = tmp_path / "images"
    images.mkdir()
    ok, encoded = cv2.imencode(".jpg", np.full((16, 16, 3), 90, dtype=np.uint8))
    assert ok
    encoded.tofile(str(images / "tl0001_dhaka_ai.jpg"))
    monkeypatch.setattr(rb, "build_detector", lambda key: _StubDetector())
    monkeypatch.setattr(rb, "SlicedDetector", lambda detector, **kwargs: detector)
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


def test_cache_keys_are_scoped_to_the_settings_hash(tmp_path: Path):
    cache = tmp_path / "cache.jsonl"
    cache.write_text(
        "\n".join(json.dumps({
            "image_hash": "image",
            "model": "yolov8n",
            "mode": "standard",
            "settings_sha256": fingerprint,
        }) for fingerprint in ("old", "new")),
        encoding="utf-8",
    )
    assert rb.load_cached_keys(cache, "new") == {
        ("image", "yolov8n", "standard", "new"),
    }


def test_evaluation_only_image_list_excludes_development_subset(tmp_path: Path):
    from core import dev_guard

    dev_id = f"tl{min(dev_guard.development_ids()):04d}"
    dev_image = tmp_path / f"{dev_id}_sample.jpg"
    eval_image = tmp_path / "tl9999_sample.jpg"
    dev_image.touch()
    eval_image.touch()
    assert rb.list_images(tmp_path, dev_only=False, evaluation_only=True) == [eval_image]

def test_benchmark_package_contains_no_image_files():
    """No image bytes may be committed inside benchmark/."""
    benchmark_dir = Path(__file__).resolve().parent.parent / "benchmark"
    if not benchmark_dir.is_dir():
        pytest.skip("benchmark/ directory does not exist yet")
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif"}
    found = [
        str(p.relative_to(benchmark_dir))
        for p in benchmark_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in image_exts
    ]
    assert found == [], f"Image files found in benchmark/: {found}"
