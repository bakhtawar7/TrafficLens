"""Helpers for the isolated, one-class fine-tuned rickshaw detector."""
from __future__ import annotations

import csv
from pathlib import Path

import cv2

from . import config, dataset

PROJECT_CATEGORIES = frozenset(config.LABEL_CLASS_NAMES) | {"ignore"}
RICKSHAW_TERMS = ("rickshaw", "auto", "cng", "three wheeler")


def map_finetuned_class(class_name: str) -> str:
    """Map one detector label to the project's category vocabulary."""
    normalized = " ".join(class_name.casefold().replace("_", " ").replace("-", " ").split())
    if any(term in normalized for term in RICKSHAW_TERMS):
        if "loader" in normalized or "cargo" in normalized:
            return "loader_rickshaw"
        return "rickshaw"
    category_aliases = {
        "motorbike": "motorcycle",
        "motor-bike": "motorcycle",
        "loader rickshaw": "loader_rickshaw",
    }
    mapped = category_aliases.get(normalized, normalized)
    return mapped if mapped in PROJECT_CATEGORIES else "other"


def load_finetuned_class_mapping(
    path: Path = config.FINETUNED_CLASS_MAPPING_JSON,
) -> dict[int, tuple[str, str]]:
    """Load and validate class-name-to-project-category mappings."""
    import json

    record = json.loads(path.read_text(encoding="utf-8"))
    mapping = record.get("class_mapping")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"fine-tuned class mapping is empty or invalid: {path}")
    names = {0: "rickshaw"}
    result: dict[int, tuple[str, str]] = {}
    for raw_id, class_name in names.items():
        category = mapping.get(class_name)
        expected = map_finetuned_class(class_name)
        if category != expected:
            raise ValueError(
                f"mapping for {class_name!r} must be {expected!r}, got {category!r}"
            )
        result[raw_id] = (class_name, category)
    return result


def training_subset_images(
    train_images_dir: Path,
    fraction: float = 0.1,
) -> list[Path]:
    """Return the deterministic sorted prefix selected by Ultralytics fraction."""
    if not 0 < fraction <= 1:
        raise ValueError("training fraction must be in (0, 1]")
    images = sorted(
        (
            path for path in train_images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in config.ALLOWED_EXTENSIONS
        ),
        key=lambda path: str(path),
    )
    count = max(1, round(len(images) * fraction))
    return images[:count]


def find_training_contamination(
    test_images_dir: Path,
    training_images: list[Path],
    threshold: int = dataset.DUPLICATE_HASH_DISTANCE,
) -> dict[str, dict[str, object]]:
    """Return test-image overlaps by filename or project perceptual hash."""
    training_hashes: dict[str, str] = {}
    training_names: set[str] = set()
    for image_path in training_images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"could not decode training image for contamination check: {image_path}")
        training_names.add(image_path.name.casefold())
        training_hashes[image_path.name.casefold()] = dataset.perceptual_hash(image)

    records: dict[str, dict[str, object]] = {}
    for image_path in sorted(test_images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in config.ALLOWED_EXTENSIONS:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"could not decode test image for contamination check: {image_path}")
        item_id = dataset.normalise_item_id(image_path.stem.split("_")[0])
        if item_id is None:
            continue
        item_id = dataset.format_item_id(dataset.parse_item_number(item_id) or 0)
        phash = dataset.perceptual_hash(image)
        filename_overlap = image_path.name.casefold() in training_names
        nearest = min(
            (
                dataset.hash_distance(phash, candidate_hash)
                for candidate_hash in training_hashes.values()
            ),
            default=None,
        )
        if filename_overlap or (nearest is not None and nearest < threshold):
            records[item_id] = {
                "filename_overlap": filename_overlap,
                "nearest_hash_distance": nearest,
            }
    return records


def read_image_scenes(path: Path = config.SOURCES_CSV) -> dict[str, str]:
    """Load canonical item-to-scene values for result grouping."""
    scenes: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            item = dataset.normalise_item_id(row.get("item_id", ""))
            if item is not None:
                scenes[item] = str(row.get("scene") or item)
    return scenes


def partition_labelled_items(
    labelled_ids: set[str],
    development_ids: set[str],
    contaminated_ids: set[str],
) -> tuple[set[str], set[str]]:
    """Split eligible labelled IDs without allowing development/evaluation pooling."""
    clean = labelled_ids - contaminated_ids
    development = clean & development_ids
    evaluation_ids = clean - development_ids
    if development & evaluation_ids:
        raise AssertionError("development and evaluation partitions overlap")
    return development, evaluation_ids
