"""Expand the existing test set by appending images under a per-scene cap.

This is an append-only companion to ``scripts/build_testset.py``. The original 45 items,
their ids, their copied images, and their 10 development images must stay exactly as they
are; only NEW images are added. Selection uses the same ground-truth-and-pixel features as
the original build (scene, quartile condition tags, rickshaw/CNG presence) and never a model
detection. The archive folder is read only; nothing there is modified.

Pre-registered rule (the caller supplies the numbers, nothing is tuned on the test set):
choose the smallest per-scene cap in ``{3, 4, 5}`` that yields at least ``min_eval``
evaluation images with at least ``min_per_tag`` images per condition tag. If no cap reaches
that, fall back to cap 5 and take the maximum feasible, reporting every shortfall honestly.
A cap above 5 is never used.

Determinism: :func:`select_with_base` mirrors :func:`core.curation.stratified_sample`
exactly, but starts from a forced base set, so the same pool, seed, and base always produce
the same additions. :func:`main` verifies that the existing 45 items are reproduced by the
deterministic base draw before writing anything; a mismatch aborts without touching the
test set.

Dry run (print the Step 1 diagnosis and the Step 2 cap decision, write nothing)::

    python scripts/expand_testset.py --source "C:/path/to/archive" --dry-run

Apply (copy only the new images and append the artefact rows)::

    python scripts/expand_testset.py --source "C:/path/to/archive" --apply
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from core import config, curation, dataset  # noqa: E402
import build_testset as bt  # noqa: E402  (scene_key, build_pool reuse)

CAPS: tuple[int, ...] = (3, 4, 5)


# ---------------------------------------------------------------------------
# Pure selection helpers (no filesystem, unit testable)
# ---------------------------------------------------------------------------
def _condition_counts(candidates) -> dict[str, int]:
    """Per condition tag image counts over a list of candidates (dev included)."""
    tally = Counter(tag for cand in candidates for tag in cand.tags)
    return {tag: tally.get(tag, 0) for tag in dataset.CONDITION_TAGS}


def select_with_base(
    pool, *, seed: int, base_paths: set[str], target: int,
    min_per_tag: int, min_rickshaw: int, max_per_scene: int,
):
    """Extend a forced base selection toward ``target`` under a per-scene cap.

    Mirrors :func:`core.curation.stratified_sample` but treats ``base_paths`` as already
    chosen, so the returned selection always contains the base and only ever ADDS images.
    Returns ``(SampleResult, additions)`` where ``additions`` are the newly chosen
    candidates in the order they were added.
    """
    ordered = sorted(pool, key=lambda c: c.image_path)
    rng = random.Random(seed)
    rank = {c.image_path: rng.random() for c in ordered}

    base = [c for c in ordered if c.image_path in base_paths]
    chosen: list[curation.SampleCandidate] = list(base)
    chosen_paths: set[str] = set(base_paths)
    scene_counts: dict[str, int] = Counter(c.scene for c in base)
    tag_counts = {tag: 0 for tag in dataset.CONDITION_TAGS}
    for cand in base:
        for tag in cand.tags:
            tag_counts[tag] += 1
    rickshaw_count = sum(1 for c in base if c.rickshaw_or_cng)
    additions: list[curation.SampleCandidate] = []

    def add(candidate: curation.SampleCandidate) -> None:
        nonlocal rickshaw_count
        chosen.append(candidate)
        chosen_paths.add(candidate.image_path)
        scene_counts[candidate.scene] = scene_counts.get(candidate.scene, 0) + 1
        for tag in candidate.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        if candidate.rickshaw_or_cng:
            rickshaw_count += 1
        if candidate.image_path not in base_paths:
            additions.append(candidate)

    def eligible() -> list[curation.SampleCandidate]:
        return [
            c for c in ordered
            if c.image_path not in chosen_paths and scene_counts.get(c.scene, 0) < max_per_scene
        ]

    def open_tags() -> list[str]:
        return [tag for tag in dataset.CONDITION_TAGS if tag_counts[tag] < min_per_tag]

    def helps(candidate, want_tags, want_rickshaw) -> int:
        utility = sum(1 for tag in want_tags if tag in candidate.tags)
        if want_rickshaw and candidate.rickshaw_or_cng:
            utility += 1
        return utility

    while len(chosen) < target:  # phase 1: close tag and rickshaw quotas
        want_tags = open_tags()
        want_rick = rickshaw_count < min_rickshaw
        if not want_tags and not want_rick:
            break
        cands = eligible()
        if not cands:
            break
        utilities = [helps(c, want_tags, want_rick) for c in cands]
        best = max(utilities)
        if best <= 0:
            break
        pick = max((c for c, u in zip(cands, utilities) if u == best), key=lambda c: rank[c.image_path])
        add(pick)

    while len(chosen) < target:  # phase 2: fill by seeded rank
        cands = eligible()
        if not cands:
            break
        add(max(cands, key=lambda c: rank[c.image_path]))

    unmet_tags = {
        tag: min_per_tag - tag_counts[tag]
        for tag in dataset.CONDITION_TAGS if tag_counts[tag] < min_per_tag
    }
    result = curation.SampleResult(
        chosen=chosen, tag_coverage=dict(tag_counts), rickshaw_count=rickshaw_count,
        scene_counts=dict(scene_counts), unmet_tags=unmet_tags,
        unmet_rickshaw=max(0, min_rickshaw - rickshaw_count), pool_size=len(ordered), target=target,
    )
    return result, additions


def choose_cap(
    pool, *, seed: int, base_paths: set[str], dev_paths: set[str],
    min_eval: int, min_per_tag: int, min_rick_images: int, target_total: int,
) -> dict:
    """Apply the pre-registered rule and return the chosen cap plus its diagnostics."""
    trials: dict[int, dict] = {}
    chosen_cap = CAPS[-1]
    reason = ""
    for cap in CAPS:
        result, additions = select_with_base(
            pool, seed=seed, base_paths=base_paths, target=target_total,
            min_per_tag=min_per_tag, min_rickshaw=min_rick_images, max_per_scene=cap,
        )
        eval_candidates = [c for c in result.chosen if c.image_path not in dev_paths]
        counts = _condition_counts(eval_candidates)
        meets = len(eval_candidates) >= min_eval and all(counts[t] >= min_per_tag for t in dataset.CONDITION_TAGS)
        trials[cap] = {
            "total": len(result.chosen),
            "additional": len(additions),
            "eval_images": len(eval_candidates),
            "tag_counts": counts,
            "rick_images": sum(1 for c in eval_candidates if c.rickshaw_or_cng),
            "meets_rule": meets,
        }
        if meets and chosen_cap == CAPS[-1] and not reason:
            chosen_cap = cap
            reason = f"smallest cap in {CAPS} meeting >= {min_eval} eval and >= {min_per_tag} per tag"
    if not reason:
        chosen_cap = CAPS[-1]
        reason = f"no cap in {CAPS} reached {min_eval} evaluation images; used cap {CAPS[-1]} and took the maximum feasible"
    return {"chosen_cap": chosen_cap, "reason": reason, "trials": trials}


def second_pass_update(
    existing_marks: list[str], eval_ids: list[str], *, seed: int, fraction: float,
    add_candidates: list[str] | None = None,
) -> list[str]:
    """Reach ``ceil(fraction * eval)`` second-pass marks, never removing an existing mark.

    Existing marks are preserved verbatim. The extra marks needed to reach the fraction are
    drawn by a seeded rank from ``add_candidates`` when supplied (so only the new rows are
    touched), otherwise from every unmarked evaluation id.
    """
    required = math.ceil(len(eval_ids) * fraction)
    kept = sorted(set(existing_marks) & set(eval_ids))
    already = set(kept)
    need = max(0, required - len(already))
    source = add_candidates if add_candidates is not None else eval_ids
    candidates = [i for i in sorted(source) if i not in already]
    rng = random.Random(seed)
    rank = {item_id: rng.random() for item_id in candidates}
    extra = sorted(candidates, key=lambda i: rank[i], reverse=True)[:need]
    return sorted(set(kept) | set(extra))


# ---------------------------------------------------------------------------
# Archive loading and base reconstruction
# ---------------------------------------------------------------------------
def load_pool(source: Path, cutoffs_path: Path) -> tuple[list, curation.CleaningReport]:
    """Run the cleaning, feature, and pool stages over the archive (read only)."""
    root = Path(source).expanduser().resolve()
    cutoffs = curation.load_tag_cutoffs(Path(cutoffs_path))
    report = curation.clean_dataset(root, write_log=False)
    features = curation.compute_features(report.kept_paths)
    pool = bt.build_pool(features, cutoffs, root)
    return pool, report


def reconstruct_base(pool, plan: dict) -> tuple[dict[str, str], set[str], set[str], list[str]]:
    """Reproduce the existing 45 ids and verify they match ``configs/testset_plan.json``.

    Returns ``(path_to_item_id, base_paths, dev_paths, existing_second_pass)``. Raises
    ``ValueError`` if the reconstructed base disagrees with the recorded plan, which means
    the deterministic draw cannot be trusted to preserve the current selection.
    """
    base_result = curation.stratified_sample(
        pool, seed=config.SEED, target=config.TESTSET_TARGET,
        min_per_tag=config.TESTSET_MIN_PER_TAG, min_rickshaw=config.TESTSET_MIN_RICKSHAW,
        max_per_scene=config.TESTSET_MAX_PER_SCENE,
    )
    ordered = sorted(base_result.chosen, key=lambda c: c.image_path)
    path_to_id = {c.image_path: dataset.format_item_id(i) for i, c in enumerate(ordered, start=1)}

    plan_items = {it["item_id"]: it for it in plan["items"]}
    for cand in ordered:
        item_id = path_to_id[cand.image_path]
        recorded = plan_items.get(item_id)
        if recorded is None:
            raise ValueError(f"reconstructed base item {item_id} is absent from the plan")
        if recorded["scene"] != cand.scene or recorded["rickshaw_or_cng"] != cand.rickshaw_or_cng:
            raise ValueError(f"reconstructed base item {item_id} scene/rickshaw mismatch vs plan")
        if dataset.merge_tags(cand.tags) != recorded["condition_tags"]:
            raise ValueError(f"reconstructed base item {item_id} condition tags mismatch vs plan")

    dev_ids = set(plan["development_subset"])
    base_paths = set(path_to_id)
    dev_paths = {p for p, i in path_to_id.items() if i in dev_ids}
    return path_to_id, base_paths, dev_paths, list(plan["second_pass"])


def rickshaw_boxes(paths) -> int:
    """Sum rickshaw and loader-rickshaw annotated boxes over the given source images."""
    total = 0
    for path in paths:
        counts = curation.annotation_counts(Path(path))
        total += counts["rickshaw"] + counts["loader_rickshaw"]
    return total


# ---------------------------------------------------------------------------
# Writing (append only; existing rows preserved verbatim)
# ---------------------------------------------------------------------------
def _read_preserving(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def new_row(item_id: str, candidate, *, source_path: Path, note: str) -> dict:
    """Build one SOURCES row for a newly added item, matching the existing schema."""
    dest_name = f"{item_id}_{config.DATASET_NAME}{source_path.suffix.lower()}"
    return {
        "item_id": item_id,
        "filename": dest_name,
        "type": "image",
        "dataset": config.DATASET_NAME,
        "split": candidate.split,
        "scene": candidate.scene,
        "source": config.DATASET_SOURCE,
        "licence": config.DATASET_LICENCE,
        "url": "",
        "condition_tags": dataset.merge_tags(candidate.tags),
        "notes": note,
    }, dest_name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Append images to the test set under a per-scene cap.")
    parser.add_argument("--source", default=None, help="Raw annotated archive to sample from.")
    parser.add_argument("--cutoffs", default=str(config.TAG_CUTOFFS_JSON), help="Quartile cut-off JSON.")
    parser.add_argument("--seed", type=int, default=config.SEED, help="Master seed (default config).")
    parser.add_argument("--min-eval", type=int, default=100, help="Rule: evaluation images required.")
    parser.add_argument("--min-per-tag", type=int, default=15, help="Rule: images per condition tag required.")
    parser.add_argument("--min-rick-images", type=int, default=50, help="Rickshaw/CNG image target that biases selection.")
    parser.add_argument("--target-total", type=int, default=150, help="Desired total selected images (eval + dev).")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Print the diagnosis, write nothing.")
    mode.add_argument("--apply", action="store_true", help="Copy new images and append artefact rows.")
    args = parser.parse_args(argv)

    source = Path(args.source) if args.source else None
    if source is None:
        cutoff_payload = json.loads(Path(args.cutoffs).read_text(encoding="utf-8"))
        source = Path(cutoff_payload.get("source", ""))
    if not source or not source.is_dir():
        print(f"ERROR: archive not found: {source}")
        return 1

    plan = json.loads(config.TESTSET_PLAN_JSON.read_text(encoding="utf-8"))
    pool, report = load_pool(source, Path(args.cutoffs))
    try:
        path_to_id, base_paths, dev_paths, existing_second = reconstruct_base(pool, plan)
    except ValueError as error:
        print(f"ERROR: existing selection is not reproducible; refusing to change anything.\n  {error}")
        return 1

    decision = choose_cap(
        pool, seed=args.seed, base_paths=base_paths, dev_paths=dev_paths,
        min_eval=args.min_eval, min_per_tag=args.min_per_tag,
        min_rick_images=args.min_rick_images, target_total=args.target_total,
    )
    cap = decision["chosen_cap"]
    result, additions = select_with_base(
        pool, seed=args.seed, base_paths=base_paths, target=args.target_total,
        min_per_tag=args.min_per_tag, min_rickshaw=args.min_rick_images, max_per_scene=cap,
    )
    eval_paths = [c.image_path for c in result.chosen if c.image_path not in dev_paths]

    _print_diagnosis(report, pool, decision, cap, eval_paths)

    if not additions:
        print("\nNo new images are selectable; the test set is already at the feasible maximum.")
        return 0
    if args.dry_run:
        print("\nDry run only. Add --apply to copy the new images and append artefact rows.")
        return 0
    return _apply(additions, path_to_id, plan, existing_second, result, eval_paths, cap, args)


def _print_diagnosis(report, pool, decision, cap, eval_paths) -> None:
    scenes = Counter(c.scene for c in pool)
    d = report.dropped_by_reason
    dropped_ann = d.get("no_annotation", 0) + d.get("annotation_unreadable", 0) + d.get("all_annotations_ignored", 0)
    dropped_unread = d.get("unreadable_image", 0)
    print("=== STEP 1: read-only diagnosis ===")
    print(f"loaded: {report.loaded}")
    print(f"after dropping unannotated + unreadable: {report.loaded - dropped_ann - dropped_unread} "
          f"(unannotated/ignored {dropped_ann}, unreadable {dropped_unread})")
    print(f"after near-duplicate removal: {report.remaining} (removed {report.duplicates_removed})")
    print(f"after scene grouping (candidates): {len(pool)}")
    print(f"distinct scenes: {len(scenes)} | largest scene sizes: {sorted(scenes.values(), reverse=True)[:6]}")
    for c in CAPS:
        t = decision["trials"][c]
        print(f"  cap {c}: additional {t['additional']}, total {t['total']}, eval {t['eval_images']}, "
              f"rick_images {t['rick_images']}, per_tag {t['tag_counts']}, meets_rule {t['meets_rule']}")
    print(f"\n=== STEP 2: cap decision ===\nchosen cap: {cap}\nwhy: {decision['reason']}")


def _apply(additions, path_to_id, plan, existing_second, result, eval_paths, cap, args) -> int:
    sources = _read_preserving(config.SOURCES_CSV)
    counts = _read_preserving(config.ANNOTATION_COUNTS_CSV)
    audit = _read_preserving(config.AUDIT_SHEET_CSV)
    note = sources["notes"].iloc[0] if len(sources) else bt.ANNOTATION_NOTE

    existing_ids = set(sources["item_id"])
    start = max(dataset.parse_item_number(i) for i in existing_ids) + 1
    new_sources, new_counts, new_audit = [], [], []
    copied = 0
    new_item_ids = []
    for offset, cand in enumerate(sorted(additions, key=lambda c: c.image_path)):
        item_id = dataset.format_item_id(start + offset)
        source_path = Path(cand.image_path)
        dest_name = f"{item_id}_{config.DATASET_NAME}{source_path.suffix.lower()}"
        shutil.copy2(source_path, config.TESTSET_IMAGES_DIR / dest_name)
        copied += 1
        curation.write_overlay(source_path, config.AUDIT_OVERLAYS_DIR / f"{item_id}.jpg")
        cat_counts = curation.annotation_counts(source_path)
        new_sources.append({**new_row(item_id, cand, source_path=source_path, note=note)[0]})
        new_counts.append({
            "item_id": item_id, "filename": dest_name,
            **{category: cat_counts[category] for category in dataset.LABEL_CATEGORIES},
            "total": cat_counts["total"],
        })
        new_audit.append({
            "item_id": item_id, "filename": dest_name, "scene": cand.scene,
            "condition_tags": dataset.merge_tags(cand.tags),
            **{f"my_{category}": "" for category in dataset.LABEL_CATEGORIES},
            "second_pass": "no", "development_subset": "no", "notes": "",
        })
        new_item_ids.append(item_id)

    # Second pass: keep existing marks; add only new ids to reach ceil(fraction * eval).
    all_eval_ids = sorted([i for i in sources["item_id"] if i not in set(plan["development_subset"])] + new_item_ids)
    marks = second_pass_update(
        existing_second, all_eval_ids, seed=args.seed, fraction=config.SECOND_PASS_FRACTION,
        add_candidates=new_item_ids,
    )
    mark_set = set(marks)
    for row in new_audit:
        row["second_pass"] = "yes" if row["item_id"] in mark_set else "no"

    pd.concat([sources, pd.DataFrame(new_sources, columns=sources.columns.tolist())], ignore_index=True) \
        .to_csv(config.SOURCES_CSV, index=False)
    pd.concat([counts, pd.DataFrame(new_counts, columns=counts.columns.tolist())], ignore_index=True) \
        .to_csv(config.ANNOTATION_COUNTS_CSV, index=False)
    pd.concat([audit, pd.DataFrame(new_audit, columns=audit.columns.tolist())], ignore_index=True) \
        .to_csv(config.AUDIT_SHEET_CSV, index=False)

    _write_plan(plan, result, path_to_id, additions, start, new_item_ids, marks, cap)
    print(f"\nCopied {copied} new image(s); SOURCES now {len(sources) + len(new_sources)} rows; "
          f"second-pass marks {len(marks)}. LABELS.csv untouched.")
    return 0


def _write_plan(plan, result, path_to_id, additions, start, new_item_ids, marks, cap) -> None:
    items = list(plan["items"])
    path_by_new_id = {dataset.format_item_id(start + offset): c for offset, c in enumerate(sorted(additions, key=lambda c: c.image_path))}
    for item_id, cand in path_by_new_id.items():
        items.append({
            "condition_tags": dataset.merge_tags(cand.tags), "item_id": item_id,
            "rickshaw_or_cng": cand.rickshaw_or_cng, "scene": cand.scene, "split": cand.split,
        })
    payload = {
        "seed": config.SEED,
        "params": {**plan["params"], "max_per_scene": cap},
        "pool_size": result.pool_size,
        "selected": len(result.chosen),
        "tag_coverage": _condition_counts(result.chosen),
        "rickshaw_count": result.rickshaw_count,
        "unmet_tags": result.unmet_tags,
        "unmet_rickshaw": result.unmet_rickshaw,
        "development_subset": sorted(plan["development_subset"]),
        "second_pass": sorted(marks),
        "items": sorted(items, key=lambda it: it["item_id"]),
    }
    config.TESTSET_PLAN_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
