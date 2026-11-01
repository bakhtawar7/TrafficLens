"""Fold the completed review checklist back into the test set.

``REVIEW.csv`` is the operator's answer sheet. This script reads it, keeps the draft pass 1
rows, writes the reviewed numbers as authoritative pass 2 rows as the labelling protocol
requires, applies corrected condition tags to ``SOURCES.csv``, and renames files that no
longer match their tags. Items marked ``keep = no`` are dropped only when
``--drop-excluded`` is passed, so a half filled sheet never destroys data.

Run from the project root after the checklist is complete::

    python scripts/apply_review.py --dry-run
    python scripts/apply_review.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config, dataset  # noqa: E402

REQUIRED_ANSWERS: tuple[str, ...] = ("tags_are_correct", "counts_are_correct", "privacy_ok")


def _text(value: object) -> str:
    """Return an empty string for blanks and for the NaN markers pandas writes."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def validate(frame: pd.DataFrame, allow_blank: bool) -> list[str]:
    """Return every problem that makes the answer sheet unsafe to apply."""
    problems: list[str] = []
    missing_columns = [column for column in (*REQUIRED_ANSWERS, *dataset.LABEL_CATEGORIES) if column not in frame]
    if missing_columns:
        problems.append(f"REVIEW.csv is missing columns: {', '.join(missing_columns)}")
        return problems

    for _, row in frame.iterrows():
        item_id = str(row.get("item_id", "?"))
        for column in REQUIRED_ANSWERS:
            if not _text(row.get(column)) and not allow_blank:
                problems.append(f"{item_id}: '{column}' is not answered")
        _valid, invalid = dataset.split_tags(row.get("corrected_condition_tags"))
        if invalid:
            problems.append(f"{item_id}: invalid condition tag(s) {', '.join(invalid)}")
        for category in dataset.LABEL_CATEGORIES:
            value = _text(row.get(category))
            if not value:
                problems.append(f"{item_id}: '{category}' count is blank")
            elif not value.lstrip("-").isdigit():
                problems.append(f"{item_id}: '{category}' count is not a whole number")
            elif int(value) < 0:
                problems.append(f"{item_id}: '{category}' count is negative")
    return problems


def reviewed_row(item_id: str, review: pd.Series, note: str) -> dict[str, object]:
    """Build the authoritative pass 2 label row from one answer sheet row."""
    row: dict[str, object] = {"item_id": item_id}
    for category in dataset.LABEL_CATEGORIES:
        row[category] = int(float(review[category]))
    row["labeller_pass"] = 2
    row["notes"] = note
    return row


def changed_categories(draft: pd.DataFrame, review: pd.Series, item_id: str) -> list[str]:
    """List the categories where the reviewer disagree with the automated count."""
    match = draft[draft["item_id"] == item_id]
    if match.empty:
        return []
    original = match.iloc[0]
    return [
        category
        for category in dataset.LABEL_CATEGORIES
        if int(float(original[category])) != int(float(review[category]))
    ]


def apply_tags(sources: pd.DataFrame, review: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    """Update condition tags and report the files that need renaming.

    Returns:
        The updated sources frame plus ``(item_id, old_filename, new_filename)`` triples.
    """
    renames: list[tuple[str, str, str]] = []
    for index, row in sources.iterrows():
        answer = review[review["item_id"] == row["item_id"]]
        if answer.empty:
            continue
        tags, _invalid = dataset.split_tags(answer.iloc[0].get("corrected_condition_tags"))
        if not tags:
            continue
        merged = dataset.merge_tags(tags)
        if merged != str(row["condition_tags"]):
            filename = str(row["filename"])
            stem = Path(filename).stem
            parts = stem.split("_", 2)
            if len(parts) == 3:
                new_name = f"{parts[0]}_{parts[1]}_{dataset.tags_to_slug(tags)}{Path(filename).suffix}"
                renames.append((str(row["item_id"]), filename, new_name))
            sources.at[index, "condition_tags"] = merged
    return sources, renames


def main(argv: list[str] | None = None) -> int:
    """Validate the answer sheet and write the reviewed test set back to disk."""
    parser = argparse.ArgumentParser(description="Apply the completed review checklist.")
    parser.add_argument("--review", default=str(config.TESTSET_REVIEW_CHECKLIST_CSV), help="Answer sheet CSV.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change and stop.")
    parser.add_argument("--allow-blank", action="store_true", help="Apply even where a question is unanswered.")
    parser.add_argument("--drop-excluded", action="store_true", help="Delete images marked keep = no.")
    args = parser.parse_args(argv)

    review_path = Path(args.review)
    if not review_path.exists():
        print(f"ERROR: {review_path} not found. Run scripts/make_review_sheet.py first.")
        return 1
    if not config.LABELS_CSV.exists() or not config.SOURCES_CSV.exists():
        print("ERROR: LABELS.csv or SOURCES.csv is missing. Run scripts/select_testset.py --apply first.")
        return 1

    review = pd.read_csv(review_path)
    if review.empty:
        print("ERROR: the review checklist has no rows.")
        return 1

    problems = validate(review, args.allow_blank)
    if problems:
        print(f"Found {len(problems)} problem(s) in the answer sheet:")
        for message in problems[:40]:
            print(f"  - {message}")
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        return 1

    draft = pd.read_csv(config.LABELS_CSV)
    sources = pd.read_csv(config.SOURCES_CSV)

    excluded = [str(row["item_id"]) for _, row in review.iterrows() if _text(row.get("keep")).lower() == "no"]
    updated_rows: list[dict[str, object]] = []
    corrections = 0
    for _, row in review.iterrows():
        item_id = str(row["item_id"])
        if item_id in excluded:
            continue
        changed = changed_categories(draft, row, item_id)
        corrections += len(changed)
        note = "reviewed, authoritative"
        if changed:
            note += "; corrected " + ", ".join(changed)
        if _text(row.get("privacy_ok")).lower().startswith("n"):
            note += "; privacy answer needs attention"
        initials = _text(row.get("reviewer_initials"))
        if initials:
            note += f"; reviewer {initials}"
        extra = _text(row.get("notes"))
        if extra:
            note += f" | {extra}"
        updated_rows.append(reviewed_row(item_id, row, note))

    sources, renames = apply_tags(sources, review[review["item_id"].isin(sources["item_id"])])

    print(f"Review rows: {len(review)}. Excluded: {len(excluded)}. Category corrections: {corrections}. Renames: {len(renames)}.")
    for item_id, old, new in renames:
        print(f"  {item_id}: {old} -> {new}")

    if args.dry_run:
        print("Dry run. Nothing was written.")
        return 0

    final_sources = sources[~sources["item_id"].astype(str).isin(excluded)].copy()

    pass_one = draft[draft["labeller_pass"].astype(str) == "1"]
    merged_labels = pd.concat(
        [pass_one, pd.DataFrame(updated_rows, columns=list(draft.columns))],
        ignore_index=True,
    )

    label_columns = list(draft.columns)
    if set(merged_labels.columns) != set(label_columns):
        print("ERROR: LABELS.csv and the answer sheet do not share the same columns.")
        return 1

    merged_labels = merged_labels[label_columns].sort_values(["item_id", "labeller_pass"])

    for item_id, old, new in renames:
        old_path = config.TESTSET_IMAGES_DIR / old
        if old_path.exists():
            old_path.rename(config.TESTSET_IMAGES_DIR / new)
        final_sources.loc[final_sources["item_id"].astype(str) == item_id, "filename"] = new

    if args.drop_excluded:
        for item_id in excluded:
            match = sources[sources["item_id"].astype(str) == item_id]
            for path in match["filename"].astype(str).map(lambda name: config.TESTSET_IMAGES_DIR / name):
                if path.exists():
                    path.unlink()
                    print(f"  deleted {path.name}")

    if not config.LABELS_CSV.with_name("LABELS.pass1.bak").exists():
        config.LABELS_CSV.with_name("LABELS.pass1.bak").write_bytes(config.LABELS_CSV.read_bytes())

    merged_labels.to_csv(config.LABELS_CSV, index=False)
    final_sources.to_csv(config.SOURCES_CSV, index=False)
    print(f"Wrote {len(merged_labels)} LABELS rows (pass 1 draft kept, pass 2 authoritative) to {config.LABELS_CSV}")
    print(f"Wrote {len(final_sources)} SOURCES rows to {config.SOURCES_CSV}")
    print(f"Backup of the draft labels is at {config.LABELS_CSV.with_name('LABELS.pass1.bak')}")
    print("Run python scripts/check_testset.py to confirm the set is still consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
