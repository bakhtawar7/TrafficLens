"""Tests for core.dev_guard, the development-subset leakage guard.

Everything is hermetic: the plan and the images are synthetic files under
tmp_path, so no real test set image is ever opened. The central contract is
that the tuning read path refuses evaluation images *before* reading bytes.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from core import dev_guard


def _write_plan(path: Path, dev_ids: list[str]) -> None:
    path.write_text(json.dumps({"development_subset": dev_ids}), encoding="utf-8")


def _write_image(path: Path) -> None:
    ok, encoded = cv2.imencode(".jpg", np.full((16, 16, 3), 90, dtype=np.uint8))
    assert ok
    encoded.tofile(str(path))


@pytest.fixture
def guard_workspace(tmp_path: Path) -> dict:
    """A seeded plan naming items 5 and 6, plus dev and evaluation image files."""
    plan = tmp_path / "testset_plan.json"
    _write_plan(plan, ["tl0005", "tl0006"])
    images = tmp_path / "images"
    images.mkdir()
    for name in ("tl0005_dhaka_ai.jpg", "tl0006_dhaka_ai.jpg",
                 "tl0007_dhaka_ai.jpg", "tl0008_dhaka_ai.jpg"):
        _write_image(images / name)
    (images / "notes.txt").write_text("not an image", encoding="utf-8")
    return {"plan": plan, "images": images,
            "dev": images / "tl0005_dhaka_ai.jpg",
            "eval": images / "tl0007_dhaka_ai.jpg"}


def test_development_ids_returns_only_the_seeded_subset(guard_workspace):
    assert dev_guard.development_ids(guard_workspace["plan"]) == {5, 6}


def test_development_ids_missing_plan_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        dev_guard.development_ids(tmp_path / "absent_plan.json")


def test_development_ids_unparseable_id_raises(tmp_path):
    plan = tmp_path / "plan.json"
    _write_plan(plan, ["tl0005", "not-an-item"])
    with pytest.raises(ValueError):
        dev_guard.development_ids(plan)


def test_assert_development_accepts_legacy_spellings(guard_workspace):
    plan = guard_workspace["plan"]
    assert dev_guard.assert_development("tl0005", plan) == 5
    assert dev_guard.assert_development("tl005", plan) == 5
    assert dev_guard.assert_development("TL5", plan) == 5


def test_assert_development_refuses_evaluation_and_unknown_ids(guard_workspace):
    plan = guard_workspace["plan"]
    with pytest.raises(dev_guard.DevelopmentOnlyError):
        dev_guard.assert_development("tl0007", plan)
    with pytest.raises(dev_guard.DevelopmentOnlyError):
        dev_guard.assert_development("junk", plan)


def test_tuning_read_path_refuses_evaluation_image(guard_workspace, monkeypatch):
    """The required leakage test: asking the tuning path for an evaluation
    image raises, and the refusal happens before any image bytes are read."""
    plan = guard_workspace["plan"]

    def _never_called(path):
        raise AssertionError(f"guard opened image bytes before refusing: {path}")

    monkeypatch.setattr(dev_guard.dataset, "load_image", _never_called)
    with pytest.raises(dev_guard.DevelopmentOnlyError):
        dev_guard.read_development_image(guard_workspace["eval"], plan)


def test_read_development_image_returns_the_dev_image(guard_workspace):
    image = dev_guard.read_development_image(guard_workspace["dev"], guard_workspace["plan"])
    assert isinstance(image, np.ndarray)
    assert image.shape == (16, 16, 3)


def test_read_development_image_missing_file_raises(tmp_path, guard_workspace):
    absent = guard_workspace["images"] / "tl0006_dhaka_ai.jpg"
    absent.unlink()
    with pytest.raises(FileNotFoundError):
        dev_guard.read_development_image(absent, guard_workspace["plan"])


def test_read_development_image_undecodable_raises(guard_workspace):
    broken = guard_workspace["images"] / "tl0006_dhaka_ai.jpg"
    broken.write_bytes(b"this is not a jpeg")
    with pytest.raises(ValueError):
        dev_guard.read_development_image(broken, guard_workspace["plan"])


def test_read_development_image_unknown_name_raises(guard_workspace):
    with pytest.raises(dev_guard.DevelopmentOnlyError):
        dev_guard.read_development_image(
            guard_workspace["images"] / "random_photo.jpg", guard_workspace["plan"])


def test_development_image_paths_filters_to_the_subset(guard_workspace):
    paths = dev_guard.development_image_paths(guard_workspace["images"], guard_workspace["plan"])
    assert [p.name for p in paths] == ["tl0005_dhaka_ai.jpg", "tl0006_dhaka_ai.jpg"]


def test_development_image_paths_missing_folder_raises(tmp_path, guard_workspace):
    with pytest.raises(FileNotFoundError):
        dev_guard.development_image_paths(tmp_path / "absent", guard_workspace["plan"])