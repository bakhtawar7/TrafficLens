"""Tests for stratified sampling and the test set build helpers.

Sampling tests use pure synthetic candidate objects so quotas, scene caps, honest
reporting, and determinism can be checked without any dataset. The count and overlay tests
use one tiny image and its annotation in ``tmp_path``. No test touches a model detector.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from core import curation, dataset


def _candidate(index: int, tags: tuple[str, ...], scene: str, rickshaw: bool) -> curation.SampleCandidate:
    return curation.SampleCandidate(
        image_path=f"img_{index:04d}.jpg", scene=scene, split="train",
        tags=tags, rickshaw_or_cng=rickshaw,
    )


def _feasible_pool(count: int = 72, scenes: int = 24) -> list[curation.SampleCandidate]:
    tags = list(dataset.CONDITION_TAGS)
    return [
        _candidate(i, (tags[i % 6],), f"s{i % scenes}", i % 2 == 0)
        for i in range(count)
    ]


def test_stratified_sample_meets_quotas_and_scene_cap() -> None:
    result = curation.stratified_sample(
        _feasible_pool(), seed=7, target=48,
        min_per_tag=8, min_rickshaw=15, max_per_scene=3,
    )
    assert len(result.chosen) == 48
    assert result.unmet_tags == {}
    assert result.unmet_rickshaw == 0
    assert result.rickshaw_count >= 15
    assert all(count <= 3 for count in result.scene_counts.values())


def test_stratified_sample_is_deterministic() -> None:
    pool = _feasible_pool()
    first = curation.stratified_sample(
        pool, seed=99, target=40, min_per_tag=8, min_rickshaw=15, max_per_scene=3
    )
    second = curation.stratified_sample(
        pool, seed=99, target=40, min_per_tag=8, min_rickshaw=15, max_per_scene=3
    )
    assert [c.image_path for c in first.chosen] == [c.image_path for c in second.chosen]


def test_stratified_sample_reports_unmet_instead_of_padding() -> None:
    pool = [_candidate(i, ("night",), f"n{i}", False) for i in range(5)]
    pool += [
        _candidate(100 + i, ("day",), f"d{i % 15}", False)
        for i in range(40)
    ]
    result = curation.stratified_sample(
        pool, seed=3, target=20, min_per_tag=8, min_rickshaw=0, max_per_scene=3
    )
    assert result.tag_coverage["night"] == 5  # only five exist, none invented
    assert result.unmet_tags.get("night") == 3
    assert len(result.chosen) == 20  # still filled from the abundant day pool


def test_load_tag_cutoffs(tmp_path: Path) -> None:
    good = tmp_path / "c.json"
    good.write_text('{"cutoffs": {"brightness": {"q1": 1, "median": 2, "q3": 3}}}', encoding="utf-8")
    loaded = curation.load_tag_cutoffs(good)
    assert loaded["brightness"]["q3"] == 3

    try:
        curation.load_tag_cutoffs(tmp_path / "missing.json")
    except curation.UnknownAnnotationFormatError:
        pass
    else:
        raise AssertionError("missing cut-off file should raise")


def _tiny_scene(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    path = root / "one.jpg"
    cv2.imwrite(str(path), image)
    path.with_suffix(".xml").write_text(
        "<annotation><filename>one.jpg</filename>"
        "<size><width>40</width><height>40</height><depth>3</depth></size>"
        "<object><name>car</name><bndbox>"
        "<xmin>5</xmin><ymin>5</ymin><xmax>25</xmax><ymax>25</ymax>"
        "</bndbox></object></annotation>",
        encoding="utf-8",
    )
    return path


def test_annotation_counts(tmp_path: Path) -> None:
    path = _tiny_scene(tmp_path / "ds")
    counts = curation.annotation_counts(path)
    assert counts["car"] == 1
    assert counts["total"] == 1


def test_write_overlay_leaves_original_intact(tmp_path: Path) -> None:
    path = _tiny_scene(tmp_path / "ds")
    before = path.read_bytes()
    out = tmp_path / "overlays" / "one.jpg"
    assert curation.write_overlay(path, out) is True
    assert out.exists()
    assert path.read_bytes() == before  # source image not modified


def test_draws_are_deterministic_and_sized(tmp_path: Path) -> None:
    ids = [f"tl{i:04d}" for i in range(1, 31)]
    dev, second = curation_dev_draw(ids)
    assert len(dev) == 10
    assert len(second) == math.ceil(len(ids) * 0.2)
    assert set(dev) <= set(ids) and set(second) <= set(ids)
    dev2, second2 = curation_dev_draw(ids)
    assert dev == dev2 and second == second2


def curation_dev_draw(ids: list[str]) -> tuple[list[str], list[str]]:
    """Call the script's draws function by loading it from the scripts folder."""
    return _load_build_module().draws(1234, ids)


def _load_build_module():
    import importlib.util

    script = Path(__file__).resolve().parent.parent / "scripts" / "build_testset.py"
    spec = importlib.util.spec_from_file_location("build_testset", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scene_key_is_finer_than_area(tmp_path: Path) -> None:
    module = _load_build_module()
    root = tmp_path
    unknown_a = root / "train" / "Final Train Dataset" / "0001.jpg"
    unknown_b = root / "train" / "Final Train Dataset" / "0060.jpg"
    place = root / "test" / "test" / "Asraf_50_jpg.rf.abc.jpg"

    scene_a, split_a = module.scene_key(unknown_a, root)
    scene_b, _ = module.scene_key(unknown_b, root)
    scene_p, split_p = module.scene_key(place, root)

    assert split_a == "train"
    assert scene_a != scene_b  # numeric train frames split into distinct bursts
    assert scene_a.startswith("train:") and "blk0000" in scene_a
    assert scene_p.startswith("test:asraf")
    assert split_p == "test"


def test_draw_is_complete(tmp_path: Path) -> None:
    module = _load_build_module()
    good = curation.stratified_sample(
        _feasible_pool(), seed=1, target=48, min_per_tag=8, min_rickshaw=15, max_per_scene=3
    )
    assert module.draw_is_complete(good, 48) is True

    # A per scene cap of 1 over 24 scenes allows at most 24, so a target of 40 is short.
    short = curation.stratified_sample(
        _feasible_pool(), seed=1, target=40, min_per_tag=8, min_rickshaw=15, max_per_scene=1
    )
    assert module.draw_is_complete(short, 40) is False
