"""Fill in condition tags for images that the frozen cut-offs left tagless.

Some images measure in the middle of every screening feature, so the quartile cut-offs in
``configs/tag_cutoffs.json`` assign them no condition band and they were recorded with an
empty ``condition_tags`` cell. ``scripts/check_testset.py`` treats a tagless item as an
error, so these items blocked the data session gate.

This script closes that gap without changing a single cut-off and without inventing a tag
the pixels do not support. For every SOURCES row that currently carries no tag it:

1. matches the copied test image back to its annotated archive source by file content,
2. recomputes the same :class:`core.curation.ImageFeature` the build measured,
3. re-applies the existing cut-offs through :func:`core.curation.assign_tags`, and
4. writes whatever tags that rule produces (which for a mid-band image is none).

Provenance is recorded in two new SOURCES columns. ``tag_source`` is ``"auto_proposed"``
for the rows processed here and ``"cutoff_rule"`` for rows whose tags were already set by
the cut-off rule at build time. ``tag_confirmed`` is written blank for every row; a human
reviewer sets it after inspecting ``results/tag_review/``, where each processed image is
rendered with its measured features and the cut-off bands that decided them.

The cut-offs, the class mapping, the seed, and the archive are only read, never modified,
and no model detection contributes to any value. Running the script twice reproduces the
same tags, the same provenance, and byte identical review images.

Usage::

    python scripts/tag_proposal.py            # preview, writes nothing
    python scripts/tag_proposal.py --apply    # write SOURCES tags and review images
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config, curation, dataset  # noqa: E402

#: ``tag_source`` value for rows whose tags were set by the cut-off rule when built.
CUTOFF_RULE_SOURCE = "cutoff_rule"
#: ``tag_source`` value for rows this script (re)proposed from the frozen cut-offs.
AUTO_PROPOSED_SOURCE = "auto_proposed"

SOURCES_TAG_COLUMNS: tuple[str, ...] = ("tag_source", "tag_confirmed")


# ---------------------------------------------------------------------------
# Pure helpers (no filesystem, unit testable)
# ---------------------------------------------------------------------------
def decide_tag_source(existing_tags: object) -> str:
    """Return the provenance label for one row from its stored ``condition_tags`` cell.

    A row that already carries tags was assigned by the cut-off rule at build time, so it
    is labelled ``cutoff_rule``. A row that carries none is a candidate for a fresh,
    still-unconfirmed proposal, so it is labelled ``auto_proposed``.
    """
    return CUTOFF_RULE_SOURCE if _split_existing(existing_tags) else AUTO_PROPOSED_SOURCE


def _split_existing(raw: object) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    parts = str(raw).replace("|", ";").split(";")
    return [part.strip().lower() for part in parts if part.strip()]


def review_lines(
    item_id: str, feature: curation.ImageFeature,
    cutoffs: dict[str, dict[str, float]], tags: list[str],
) -> list[str]:
    """Describe one image's measured features against the cut-off bands that judged it."""
    brightness = cutoffs["brightness"]
    saturation = cutoffs["saturation"]
    vehicles = cutoffs["vehicle_count"]
    box = cutoffs["median_box_fraction"]
    verdict = ";".join(tags) if tags else "NONE (unconfirmed)"
    return [
        f"id {item_id}",
        f"brightness {feature.brightness:.2f}  night<={brightness['q1']:.2f} day>={brightness['q3']:.2f}",
        f"saturation {feature.saturation:.4f}  glare>={saturation['q3']:.4f}",
        f"vehicle_count {feature.vehicle_count}  sparse<={vehicles['q1']:.1f} dense>={vehicles['q3']:.1f}",
        f"box_fraction {feature.median_box_fraction:.6f}  distant<={box['q1']:.6f}",
        f"rickshaw/CNG {'yes' if feature.rickshaw_or_cng else 'no'}",
        f"proposed: {verdict}",
    ]


def compose_review(image: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw the review text over a copy of ``image``; the original array is untouched."""
    canvas = image.copy()
    height, width = canvas.shape[:2]
    pad = 8
    line_height = 22
    panel_height = pad * 2 + line_height * len(lines)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, min(panel_height, height)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    for index, text in enumerate(lines):
        origin = (pad, pad + line_height * (index + 1) - 6)
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def file_hash(path: Path) -> str:
    """SHA-256 of a file's bytes, used to match a copied image back to its archive source."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def match_archive_sources(archive_root: Path, copied: dict[str, Path]) -> dict[str, Path]:
    """Map each copied image to its archive source by exact file content.

    ``copied`` is ``{item_id: copied_path}``. The archive is walked in sorted order, so an
    exact byte match is resolved to a single, reproducible source path. Items with no match
    (for example when the archive is absent) are simply omitted.
    """
    wanted: dict[str, str] = {}
    for item_id, path in copied.items():
        if Path(path).is_file():
            wanted.setdefault(file_hash(path), item_id)

    resolved: dict[str, Path] = {}
    root = Path(archive_root)
    if not root.is_dir():
        return resolved
    for image in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in config.ALLOWED_EXTENSIONS):
        item_id = wanted.get(file_hash(image))
        if item_id is not None and item_id not in resolved:
            resolved[item_id] = image
            if len(resolved) == len(wanted):
                break
    return resolved


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _sources_without_tags(sources: pd.DataFrame) -> list[str]:
    return [
        str(row["item_id"]) for _, row in sources.iterrows()
        if not _split_existing(row.get("condition_tags"))
    ]


def build_updates(sources: pd.DataFrame, cutoffs: dict, archive_root: Path) -> dict[str, dict]:
    """Compute the new ``condition_tags`` and provenance for every row.

    Returns ``{item_id: {"condition_tags", "tag_source", "tag_confirmed", "feature",
    "review_lines"}}``. Only rows that lack tags get a recomputed feature and review panel;
    rows already carrying tags are preserved verbatim and marked ``cutoff_rule``.
    """
    tagless_ids = _sources_without_tags(sources)
    copied = {
        item_id: config.TESTSET_IMAGES_DIR / str(sources.loc[sources["item_id"] == item_id, "filename"].iloc[0])
        for item_id in tagless_ids
    }
    archive_by_id = match_archive_sources(archive_root, copied)

    updates: dict[str, dict] = {}
    for _, row in sources.iterrows():
        item_id = str(row["item_id"])
        existing = row.get("condition_tags")
        if _split_existing(existing):
            updates[item_id] = {
                "condition_tags": dataset.merge_tags(_split_existing(existing)),
                "tag_source": CUTOFF_RULE_SOURCE, "tag_confirmed": "",
            }
            continue
        source = archive_by_id.get(item_id)
        feature = curation.compute_feature(source) if source is not None else None
        tags = curation.assign_tags(feature, cutoffs) if feature is not None else []
        updates[item_id] = {
            "condition_tags": dataset.merge_tags(tags),
            "tag_source": AUTO_PROPOSED_SOURCE, "tag_confirmed": "",
            "feature": feature,
            "copied_path": copied.get(item_id),
            "review_lines": review_lines(item_id, feature, cutoffs, tags) if feature is not None else [
                f"id {item_id}", "archive source not found; no feature to propose",
            ],
        }
    return updates


def write_review_images(updates: dict[str, dict], review_dir: Path) -> int:
    """Render one feature panel per processed image and return how many were written."""
    written = 0
    for item_id, update in sorted(updates.items()):
        if update.get("tag_source") != AUTO_PROPOSED_SOURCE:
            continue
        feature = update.get("feature")
        candidates = [update.get("copied_path")]
        if feature is not None:
            candidates.append(Path(feature.image_path))
        image = None
        for candidate in candidates:
            if candidate is not None:
                image = dataset.load_image(Path(candidate))
                if image is not None:
                    break
        if image is None:
            continue
        canvas = compose_review(image, update["review_lines"])
        out = Path(review_dir) / f"{item_id}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        written += 1
    return written


def apply_updates(sources: pd.DataFrame, updates: dict[str, dict]) -> pd.DataFrame:
    """Return a copy of ``sources`` with the new tags and two provenance columns appended."""
    result = sources.copy()
    for column in ("tag_source", "tag_confirmed"):
        if column not in result.columns:
            result[column] = ""
    for index, row in result.iterrows():
        update = updates.get(str(row["item_id"]))
        if update is None:
            continue
        result.at[index, "condition_tags"] = update["condition_tags"]
        result.at[index, "tag_source"] = update["tag_source"]
        result.at[index, "tag_confirmed"] = update["tag_confirmed"]
    return result


def _print_summary(updates: dict[str, dict]) -> None:
    proposed = [i for i, u in updates.items() if u["tag_source"] == AUTO_PROPOSED_SOURCE]
    still_empty = [i for i in proposed if not u_has_tags(updates[i])]
    print("=== condition tag proposal ===")
    print(f"cutoff_rule rows preserved : {len(updates) - len(proposed)}")
    print(f"auto_proposed rows processed: {len(proposed)}")
    for item_id in sorted(proposed):
        update = updates[item_id]
        tags = update["condition_tags"] or "(none)"
        note = "" if update["condition_tags"] else "  <- mid-band, unfilled without changing cutoffs"
        print(f"  {item_id}: tags={tags}{note}")
    print(f"\n{len(still_empty)} of these still carry no tag because they sit in the middle of every "
          "feature; they are marked auto_proposed and left for review.")


def u_has_tags(update: dict) -> bool:
    return bool(_split_existing(update.get("condition_tags")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Propose condition tags from the frozen cut-offs.")
    parser.add_argument("--cutoffs", default=str(config.TAG_CUTOFFS_JSON), help="Quartile cut-off JSON.")
    parser.add_argument("--source", default=None, help="Annotated archive root (default: read from the cut-off file).")
    parser.add_argument("--apply", action="store_true", help="Write SOURCES and review images (default: preview).")
    args = parser.parse_args(argv)

    cutoff_path = Path(args.cutoffs)
    cutoffs = curation.load_tag_cutoffs(cutoff_path)
    archive = Path(args.source) if args.source else Path(json.loads(cutoff_path.read_text(encoding="utf-8")).get("source", ""))

    sources = pd.read_csv(config.SOURCES_CSV, dtype=str, keep_default_na=False)
    updates = build_updates(sources, cutoffs, archive)
    _print_summary(updates)

    if not args.apply:
        print("\nDry run only. Add --apply to write SOURCES and the tag_review images.")
        return 0

    apply_updates(sources, updates).to_csv(config.SOURCES_CSV, index=False)
    written = write_review_images(updates, config.RESULTS_DIR / "tag_review")
    print(f"\nWrote {config.SOURCES_CSV} and {written} review image(s) to {config.RESULTS_DIR / 'tag_review'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
