"""Tests for the append-only test set expansion.

These exercise the pure selection helpers in ``scripts/expand_testset.py`` with synthetic
candidate pools only, mirroring ``tests/test_build_testset.py``: no archive, no model, and
no filesystem writes are involved. They prove the three guarantees the pre-registered rule
depends on: determinism, preservation of the existing selection, and the per-scene cap rule.
"""
from __future__ import annotations

import math

from core import dataset
from scripts import expand_testset as ex


def _cand(index: int, tags: tuple[str, ...], scene: str, rickshaw: bool):
    return ex.curation.SampleCandidate(
        image_path=f"img_{index:04d}.jpg", scene=scene, split="train",
        tags=tags, rickshaw_or_cng=rickshaw,
    )


def _pool(count: int = 48, scenes: int = 8, per_scene: int = 6):
    tags = list(dataset.CONDITION_TAGS)
    return [
        _cand(i, (tags[i % 6],), f"s{i % scenes}", i % 2 == 0)
        for i in range(count)
    ]


# --- determinism -----------------------------------------------------------
def test_select_with_base_is_deterministic() -> None:
    pool = _pool()
    base = {c.image_path for c in pool[:10]}
    r1, a1 = ex.select_with_base(pool, seed=7, base_paths=base, target=30, min_per_tag=2, min_rickshaw=5, max_per_scene=4)
    r2, a2 = ex.select_with_base(pool, seed=7, base_paths=base, target=30, min_per_tag=2, min_rickshaw=5, max_per_scene=4)
    assert [c.image_path for c in r1.chosen] == [c.image_path for c in r2.chosen]
    assert [c.image_path for c in a1] == [c.image_path for c in a2]


# --- preservation ----------------------------------------------------------
def test_select_with_base_never_drops_a_base_item() -> None:
    pool = _pool()
    base = {c.image_path for c in pool[:12]}
    result, additions = ex.select_with_base(pool, seed=3, base_paths=base, target=40, min_per_tag=2, min_rickshaw=0, max_per_scene=5)
    chosen_paths = {c.image_path for c in result.chosen}
    assert base <= chosen_paths  # every existing item survives
    assert not (set(c.image_path for c in additions) & base)  # additions are strictly new
    assert len(result.chosen) == len(base) + len(additions)


def test_select_with_base_respects_scene_cap_and_ceiling() -> None:
    pool = _pool()
    result, _ = ex.select_with_base(pool, seed=5, base_paths=set(), target=100, min_per_tag=1, min_rickshaw=0, max_per_scene=3)
    assert all(count <= 3 for count in result.scene_counts.values())
    assert len(result.chosen) == 8 * 3  # the ceiling: 8 scenes at three each


# --- cap rule --------------------------------------------------------------
def test_choose_cap_picks_the_smallest_cap_that_meets_the_rule() -> None:
    # 8 scenes of 6: cap 3 -> 24 eval, cap 4 -> 32 eval. min_eval 30 forces cap 4.
    pool = _pool()
    decision = ex.choose_cap(pool, seed=9, base_paths=set(), dev_paths=set(), min_eval=30, min_per_tag=2, min_rick_images=0, target_total=100)
    assert decision["chosen_cap"] == 4
    assert decision["trials"][3]["eval_images"] < 30
    assert decision["trials"][4]["meets_rule"] is True


def test_choose_cap_falls_back_to_five_when_no_cap_meets_the_rule() -> None:
    # No cap in {3,4,5} supplies 45 evaluation images from 8 scenes, so the rule falls back.
    pool = _pool()
    decision = ex.choose_cap(pool, seed=9, base_paths=set(), dev_paths=set(), min_eval=45, min_per_tag=2, min_rick_images=0, target_total=100)
    assert decision["chosen_cap"] == 5
    assert all(not decision["trials"][c]["meets_rule"] for c in ex.CAPS)
    assert "maximum feasible" in decision["reason"]


def test_choose_cap_never_exceeds_five() -> None:
    pool = _pool()
    decision = ex.choose_cap(pool, seed=1, base_paths=set(), dev_paths=set(), min_eval=1, min_per_tag=1, min_rick_images=0, target_total=100)
    assert decision["chosen_cap"] <= max(ex.CAPS)


# --- second pass -----------------------------------------------------------
def test_second_pass_preserves_existing_and_reaches_fraction() -> None:
    eval_ids = [f"tl{i:04d}" for i in range(1, 21)]  # 20 evaluation images
    existing = ["tl0007", "tl0008"]
    marks = ex.second_pass_update(existing, eval_ids, seed=2, fraction=0.2)
    assert set(existing) <= set(marks)
    assert len(marks) == math.ceil(len(eval_ids) * 0.2)
    assert marks == ex.second_pass_update(existing, eval_ids, seed=2, fraction=0.2)  # deterministic


def test_second_pass_never_removes_a_mark_even_when_over_target() -> None:
    eval_ids = [f"tl{i:04d}" for i in range(1, 11)]  # 10 images -> target 2
    existing = ["tl0001", "tl0002", "tl0003", "tl0004"]
    marks = ex.second_pass_update(existing, eval_ids, seed=4, fraction=0.2)
    assert set(existing) <= set(marks)  # over the target is fine; nothing is dropped


def test_second_pass_extras_are_restricted_to_add_candidates() -> None:
    # Only the newly added ids may receive a fresh mark; older unmarked eval ids are skipped.
    existing = ["tl0001", "tl0002"]
    new_ids = [f"tl{i:04d}" for i in range(101, 111)]  # 10 newly added ids
    old_unmarked = [f"tl{i:04d}" for i in range(3, 21)]  # 18 older unmarked eval ids
    eval_ids = existing + old_unmarked + new_ids  # 30 evaluation images total
    marks = ex.second_pass_update(existing, eval_ids, seed=6, fraction=0.2, add_candidates=new_ids)
    assert len(marks) == math.ceil(len(eval_ids) * 0.2)  # 6
    assert set(existing) <= set(marks)  # existing marks preserved
    extras = [i for i in marks if i not in existing]
    assert set(extras) <= set(new_ids)  # every fresh mark lands on a new row
    assert not (set(extras) & set(old_unmarked))
    assert marks == ex.second_pass_update(existing, eval_ids, seed=6, fraction=0.2, add_candidates=new_ids)


# --- diagnosis helper ------------------------------------------------------
def test_condition_counts_over_candidates() -> None:
    counts = ex._condition_counts([_cand(0, ("day", "dense"), "s0", False), _cand(1, ("day",), "s0", False)])
    assert counts["day"] == 2 and counts["dense"] == 1 and counts["night"] == 0
