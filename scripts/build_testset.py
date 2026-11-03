"""Sample a stratified test set from the cleaned, quartile tagged pool.

This runs after ``tag_dataset.py``: it reads the persisted quartile cut-offs so the same
fixed reference used to propose tags also drives selection, then makes a seeded stratified
draw (configurable target, default 45, at least 8 per condition tag, at least 15 with a
rickshaw or CNG class, at most 3 per apparent scene) and writes the test set and its audit
artefacts. 60 is unreachable at three per real scene for this archive, so the honest
maximum is the default; the abort gate refuses to write a padded set.

Every count is derived from the ground truth annotation and the pixels, never from a model
detector. Running the script twice with the same seed reproduces the same selection and the
same byte for byte identical outputs. The source dataset is only read.

Dry run (print the plan, change nothing)::

    python scripts/build_testset.py --source "C:/path/to/archive"

Apply (copy images and write every artefact)::

    python scripts/build_testset.py --source "C:/path/to/archive" --apply
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config, curation, dataset  # noqa: E402

SOURCES_COLUMNS: tuple[str, ...] = (
    "item_id", "filename", "type", "dataset", "split", "scene",
    "source", "licence", "url", "condition_tags", "notes",
)
ANNOTATION_NOTE = dataset.UNVERIFIED_DRAFT_NOTE


def _frame_number(stem: str) -> int:
    """Return the trailing integer in a file stem, or 0 when there is none."""
    numbers = re.findall(r"\d+", stem)
    return int(numbers[-1]) if numbers else 0


def scene_key(path: Path, root: Path) -> tuple[str, str]:
    """Return a finer ``(scene, split)`` pair for one image.

    The coarse ``infer_area`` token collapses this archive to only six scenes, so a strict
    per-scene cap cannot reach 60. This groups by the true top level split plus the place
    name when the filename carries one (and its date), and otherwise by a consecutive frame
    block inside the folder, which turns the large unlabelled train set into many bursts.
    """
    relative = path.relative_to(root)
    split = relative.parts[0] if len(relative.parts) > 1 else "root"
    folder = relative.parts[-2] if len(relative.parts) > 2 else split
    area, _origin = dataset.infer_area(path.name, folder)
    if area != "unknown":
        date = dataset.extract_date(path.name)
        scene = f"{split}:{area}:{date}" if date else f"{split}:{area}"
        return scene, split
    block = _frame_number(path.stem) // max(1, config.TESTSET_SCENE_BURST)
    return f"{split}:{folder}:blk{block:04d}", split


def build_pool(
    features: list[curation.ImageFeature], cutoffs: dict, root: Path
) -> list[curation.SampleCandidate]:
    """Turn measured features into sampling candidates with scene, split, and tags."""
    records: list[curation.SampleCandidate] = []
    for feature in features:
        path = Path(feature.image_path)
        scene, split = scene_key(path, root)
        tags = tuple(curation.assign_tags(feature, cutoffs))
        records.append(
            curation.SampleCandidate(
                image_path=str(path),
                scene=scene,
                split=split,
                tags=tags,
                rickshaw_or_cng=feature.rickshaw_or_cng,
            )
        )
    return records


def draw_is_complete(result: curation.SampleResult, target: int) -> bool:
    """True only when the target is met and no tag or rickshaw quota is short."""
    return len(result.chosen) >= target and not result.unmet_tags and result.unmet_rickshaw == 0


def print_coverage(result: curation.SampleResult, min_per_tag: int, min_rickshaw: int) -> None:
    """Print the overlap-aware coverage table and honestly flag any unmet quota."""
    print(f"Selected {len(result.chosen)} of a pool of {result.pool_size} (target {result.target}).")
    print("\n  Condition coverage (tags overlap, so a row can count toward several):")
    for tag in dataset.CONDITION_TAGS:
        count = result.tag_coverage.get(tag, 0)
        gap = result.unmet_tags.get(tag, 0)
        status = "met" if count >= min_per_tag else f"SHORT by {gap} (not padded)"
        print(f"    {tag:<8}: {count:>3}  (min {min_per_tag})  {status}")
    rick_status = "met" if result.rickshaw_count >= min_rickshaw else (
        f"SHORT by {result.unmet_rickshaw} (not padded)"
    )
    print(f"    rickshaw/CNG: {result.rickshaw_count:>3}  (min {min_rickshaw})  {rick_status}")

    if len(result.chosen) < result.target:
        print("\n  NOTE: the target was not reached because the per scene cap left no "
              "eligible images; nothing was padded to fill it.")


def _clean_images(directory: Path) -> int:
    """Remove previously copied test media so a smaller run never leaves stale files."""
    removed = 0
    media = {".jpg", ".jpeg", ".png", ".webp"}
    if directory.exists():
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in media:
                path.unlink()
                removed += 1
    return removed


def _clear_overlays(directory: Path) -> None:
    """Delete prior overlay renders before regenerating."""
    if directory.exists():
        for path in directory.glob("*.jpg"):
            path.unlink()
    directory.mkdir(parents=True, exist_ok=True)


def draws(seed: int, item_ids: list[str]) -> tuple[list[str], list[str]]:
    """Return the development subset and the seeded second pass draw, deterministically."""
    rng = random.Random(seed)
    dev_order = sorted(item_ids)
    rng.shuffle(dev_order)
    dev = dev_order[: min(config.DEV_SUBSET_SIZE, len(dev_order))]
    second_order = sorted(item_ids)
    rng.shuffle(second_order)
    need = math.ceil(len(second_order) * config.SECOND_PASS_FRACTION)
    second_pass = second_order[:need]
    return dev, second_pass


def write_artifacts(
    chosen: list[tuple[str, curation.SampleCandidate]],
    dev: list[str],
    second_pass: list[str],
) -> None:
    """Copy images and write SOURCES, annotation counts, overlays, and the audit sheet."""
    config.TESTSET_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    removed = _clean_images(config.TESTSET_IMAGES_DIR)
    _clear_overlays(config.AUDIT_OVERLAYS_DIR)

    sources_rows: list[dict[str, object]] = []
    counts_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    second_set = set(second_pass)
    copied = 0

    for item_id, candidate in chosen:
        source = Path(candidate.image_path)
        destination_name = f"{item_id}_{config.DATASET_NAME}{source.suffix.lower()}"
        shutil.copy2(source, config.TESTSET_IMAGES_DIR / destination_name)
        copied += 1

        counts = curation.annotation_counts(source)
        curation.write_overlay(source, config.AUDIT_OVERLAYS_DIR / f"{item_id}.jpg")

        tags = dataset.merge_tags(candidate.tags)
        sources_rows.append({
            "item_id": item_id,
            "filename": destination_name,
            "type": "image",
            "dataset": config.DATASET_NAME,
            "split": candidate.split,
            "scene": candidate.scene,
            "source": config.DATASET_SOURCE,
            "licence": config.DATASET_LICENCE,
            "url": "",
            "condition_tags": tags,
            "notes": ANNOTATION_NOTE,
        })
        counts_rows.append({
            "item_id": item_id, "filename": destination_name,
            **{category: counts[category] for category in dataset.LABEL_CATEGORIES},
            "total": counts["total"],
        })
        audit_rows.append({
            "item_id": item_id,
            "filename": destination_name,
            "scene": candidate.scene,
            "condition_tags": tags,
            **{f"my_{category}": "" for category in dataset.LABEL_CATEGORIES},
            "second_pass": "yes" if item_id in second_set else "no",
            "development_subset": "yes" if item_id in dev else "no",
            "notes": "",
        })

    pd.DataFrame(sources_rows, columns=list(SOURCES_COLUMNS)).to_csv(config.SOURCES_CSV, index=False)
    pd.DataFrame(counts_rows).to_csv(config.ANNOTATION_COUNTS_CSV, index=False)
    pd.DataFrame(audit_rows).to_csv(config.AUDIT_SHEET_CSV, index=False)

    # LABELS.csv is deliberately never written here: the instruction forbids it, and a
    # derived count must not be presented as verified ground truth. check_testset treats a
    # missing or unrelated LABELS file leniently, so the build stays valid on its own.
    print(f"Cleared {removed} previous test image(s); copied {copied} image(s) to {config.TESTSET_IMAGES_DIR}")
    print(f"Wrote {config.SOURCES_CSV}, {config.ANNOTATION_COUNTS_CSV}, {config.AUDIT_SHEET_CSV}")
    print(f"Wrote {len(chosen)} overlay(s) to {config.AUDIT_OVERLAYS_DIR}")
    print(f"Left {config.LABELS_CSV} untouched (per instruction not to write it).")


def write_plan(result: curation.SampleResult, chosen: list[tuple[str, curation.SampleCandidate]],
               dev: list[str], second_pass: list[str], seed: int) -> None:
    """Record the run in config so the subsets and parameters are reproducible and auditable."""
    payload = {
        "seed": seed,
        "params": {
            "target": config.TESTSET_TARGET,
            "min_per_tag": config.TESTSET_MIN_PER_TAG,
            "min_rickshaw": config.TESTSET_MIN_RICKSHAW,
            "max_per_scene": config.TESTSET_MAX_PER_SCENE,
            "dev_subset_size": config.DEV_SUBSET_SIZE,
            "second_pass_fraction": config.SECOND_PASS_FRACTION,
        },
        "pool_size": result.pool_size,
        "selected": len(chosen),
        "tag_coverage": result.tag_coverage,
        "rickshaw_count": result.rickshaw_count,
        "unmet_tags": result.unmet_tags,
        "unmet_rickshaw": result.unmet_rickshaw,
        "development_subset": sorted(dev),
        "second_pass": sorted(second_pass),
        "items": [
            {"item_id": item_id, "scene": c.scene, "split": c.split,
             "condition_tags": dataset.merge_tags(c.tags), "rickshaw_or_cng": c.rickshaw_or_cng}
            for item_id, c in chosen
        ],
    }
    config.TESTSET_PLAN_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.TESTSET_PLAN_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote sampling plan (seed {seed}, {len(dev)} development, {len(second_pass)} second pass) to {config.TESTSET_PLAN_JSON}")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, sample, report coverage, and optionally write every artefact."""
    parser = argparse.ArgumentParser(description="Build the stratified TrafficLens test set.")
    parser.add_argument("--source", required=True, help="Raw dataset folder to sample from.")
    parser.add_argument("--cutoffs", default=str(config.TAG_CUTOFFS_JSON), help="Quartile cut-off JSON.")
    parser.add_argument("--seed", type=int, default=config.SEED, help="Master seed.")
    parser.add_argument("--target", type=int, default=config.TESTSET_TARGET, help="Images to select.")
    parser.add_argument("--apply", action="store_true", help="Copy images and write artefacts.")
    parser.add_argument(
        "--force", action="store_true",
        help="Write artefacts even when the draw cannot meet the target or a quota.",
    )
    args = parser.parse_args(argv)

    root = Path(args.source).expanduser().resolve()
    cutoffs = curation.load_tag_cutoffs(Path(args.cutoffs))
    cleaning = curation.clean_dataset(root, write_log=False)
    features = curation.compute_features(cleaning.kept_paths)
    if not features:
        print("ERROR: nothing readable and annotated to sample from.")
        return 1

    pool = build_pool(features, cutoffs, root)
    result = curation.stratified_sample(
        pool,
        seed=args.seed,
        target=args.target,
        min_per_tag=config.TESTSET_MIN_PER_TAG,
        min_rickshaw=config.TESTSET_MIN_RICKSHAW,
        max_per_scene=config.TESTSET_MAX_PER_SCENE,
    )

    chosen = [(f"tl{i:04d}", c) for i, c in enumerate(
        sorted(result.chosen, key=lambda c: c.image_path), start=1
    )]
    item_ids = [item_id for item_id, _ in chosen]
    dev, second_pass = draws(args.seed, item_ids)

    print_coverage(result, config.TESTSET_MIN_PER_TAG, config.TESTSET_MIN_RICKSHAW)

    if not args.apply:
        print("\nDry run only. Add --apply to copy images and write the artefacts.")
        return 0

    if not draw_is_complete(result, args.target) and not args.force:
        print(
            "\nERROR: the draw cannot meet the target or a stated quota, so nothing was "
            "written. The existing test set is left untouched. Re-run with --force to write "
            "the partial result anyway."
        )
        return 1

    write_artifacts(chosen, dev, second_pass)
    write_plan(result, chosen, dev, second_pass, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
