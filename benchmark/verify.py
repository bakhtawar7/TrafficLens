"""Verify the benchmark package integrity without requiring the original dataset.

Checks:
- No image files are present in the benchmark directory (only metadata).
- benchmark/checksums.json matches the images currently in data/testset/images/.
- benchmark/image_ids.csv covers every item in data/testset/SOURCES.csv.
- benchmark/audited_counts.csv is internally consistent.
- The class mapping JSON is present and parseable.

Usage::

    python benchmark/verify.py

Exits 0 on success; exits 1 and prints all failures on any mismatch.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "data" / "testset" / "images"
SOURCES_CSV = ROOT / "data" / "testset" / "SOURCES.csv"
CLASS_MAPPING_JSON = ROOT / "configs" / "class_mapping.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_no_image_files() -> list[str]:
    """No image bytes may be committed to the benchmark directory."""
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif"}
    errors = []
    for path in BENCHMARK_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in image_exts:
            errors.append(f"Image file found in benchmark/: {path.relative_to(ROOT)}")
    return errors


def check_checksums() -> list[str]:
    """Stored SHA-256s match the images currently in data/testset/images/."""
    checksums_path = BENCHMARK_DIR / "checksums.json"
    if not checksums_path.is_file():
        return ["benchmark/checksums.json not found"]
    stored: dict[str, str] = json.loads(checksums_path.read_text(encoding="utf-8"))
    errors = []
    present_images = {
        f.name: f
        for f in IMAGE_DIR.iterdir()
        if f.is_file() and f.name != ".gitkeep"
    }
    for filename, expected_hash in stored.items():
        if filename not in present_images:
            errors.append(f"checksums.json references {filename} but it is not in data/testset/images/")
        else:
            actual = _sha256(present_images[filename])
            if actual != expected_hash:
                errors.append(
                    f"SHA-256 mismatch for {filename}: "
                    f"stored={expected_hash[:16]}… actual={actual[:16]}…"
                )
    for filename in present_images:
        if filename not in stored:
            errors.append(f"{filename} is in data/testset/images/ but not in checksums.json")
    return errors


def check_image_ids() -> list[str]:
    """benchmark/image_ids.csv must cover all items in SOURCES.csv."""
    ids_path = BENCHMARK_DIR / "image_ids.csv"
    if not ids_path.is_file():
        return ["benchmark/image_ids.csv not found"]
    if not SOURCES_CSV.is_file():
        return ["data/testset/SOURCES.csv not found — cannot verify image_ids coverage"]
    benchmark_ids = {
        row["item_id"]
        for row in csv.DictReader(ids_path.open(encoding="utf-8", newline=""))
    }
    source_ids = {
        row["item_id"]
        for row in csv.DictReader(SOURCES_CSV.open(encoding="utf-8-sig", newline=""))
    }
    errors = []
    for item_id in source_ids - benchmark_ids:
        errors.append(f"{item_id} is in SOURCES.csv but not in benchmark/image_ids.csv")
    for item_id in benchmark_ids - source_ids:
        errors.append(f"{item_id} is in benchmark/image_ids.csv but not in SOURCES.csv")
    return errors


def check_audited_counts() -> list[str]:
    """benchmark/audited_counts.csv must have valid non-negative counts."""
    counts_path = BENCHMARK_DIR / "audited_counts.csv"
    if not counts_path.is_file():
        return ["benchmark/audited_counts.csv not found"]
    errors = []
    count_cols = ("car", "bus", "truck", "motorcycle", "rickshaw", "loader_rickshaw", "other")
    for row_num, row in enumerate(
        csv.DictReader(counts_path.open(encoding="utf-8", newline="")), start=2
    ):
        for col in count_cols:
            if col not in row:
                errors.append(f"Row {row_num}: missing column {col}")
                continue
            if row[col] == "":
                continue  # blank = not yet labelled
            try:
                value = int(row[col])
                if value < 0:
                    errors.append(f"Row {row_num} ({row.get('item_id')}): {col}={value} is negative")
            except ValueError:
                errors.append(f"Row {row_num} ({row.get('item_id')}): {col}={row[col]!r} is not an integer")
        if row.get("labeller_pass") not in {"1", "2", ""}:
            errors.append(
                f"Row {row_num} ({row.get('item_id')}): labeller_pass={row.get('labeller_pass')!r} "
                "must be 1 or 2"
            )
    return errors


def check_class_mapping() -> list[str]:
    """The class mapping JSON must be present and parseable."""
    if not CLASS_MAPPING_JSON.is_file():
        return ["configs/class_mapping.json not found"]
    try:
        mapping = json.loads(CLASS_MAPPING_JSON.read_text(encoding="utf-8"))
        if "classes" not in mapping:
            return ["configs/class_mapping.json missing 'classes' key"]
    except json.JSONDecodeError as error:
        return [f"configs/class_mapping.json is not valid JSON: {error}"]
    return []


def main() -> None:
    all_errors: list[str] = []
    checks = [
        ("No image files in benchmark/", check_no_image_files),
        ("SHA-256 checksums", check_checksums),
        ("image_ids.csv coverage", check_image_ids),
        ("audited_counts.csv validity", check_audited_counts),
        ("class_mapping.json present", check_class_mapping),
    ]
    for name, check_fn in checks:
        errors = check_fn()
        status = "PASS" if not errors else f"FAIL ({len(errors)} error(s))"
        print(f"  {status}  {name}")
        all_errors.extend(errors)

    if all_errors:
        print(f"\n{len(all_errors)} error(s) found:")
        for error in all_errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print("\nAll checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
