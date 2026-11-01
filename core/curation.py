"""Annotation loading into one flat table for TrafficLens.

This module reads a raw annotated dataset and returns one row per annotated object,
each carrying the image path, the dataset class name, and the bounding box. It performs
no copying, filtering, or image modification: it only reads annotation files and, where
possible, pairs each one with the image that shares its stem.

The archive in use is PASCAL VOC (one ``.xml`` per image). Rather than reimplement the
VOC parser, this loader reuses ``core.dataset.parse_voc_annotation`` so there is a single
source of truth for the format. If a future archive uses a different format it is added
here as another branch. When none of the supported formats is recognised the loader
raises :class:`UnknownAnnotationFormatError` instead of guessing.
"""
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import config, dataset

#: Image extensions searched when pairing an annotation with its picture.
IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")


class CurationError(Exception):
    """Base error for the curation loader."""


class DatasetNotFoundError(CurationError):
    """Raised when the supplied dataset location is not an existing folder."""


class UnknownAnnotationFormatError(CurationError):
    """Raised when no supported annotation format is recognised in the dataset."""


@dataclass(frozen=True)
class AnnotationRecord:
    """One annotated object, flattened for a table.

    Attributes:
        image_path: Best effort path to the image the box belongs to.
        class_name: Raw class name as written in the annotation.
        box: ``(x_min, y_min, x_max, y_max)`` in pixel coordinates.
    """

    image_path: str
    class_name: str
    box: tuple[float, float, float, float]


def _resolve_image(annotation_path: Path, declared_filename: str) -> str:
    """Locate the image that pairs with an annotation file.

    Prefers a sibling file sharing the annotation stem. Falls back to the filename the
    annotation declares, then to the annotation path, so a table row always has a value.
    """
    for suffix in IMAGE_SUFFIXES:
        candidate = annotation_path.with_suffix(suffix)
        if candidate.exists():
            return str(candidate)
    if declared_filename:
        declared = annotation_path.parent / declared_filename
        if declared.exists():
            return str(declared)
        return str(declared)
    return str(annotation_path)


def _voc_records(root: Path) -> tuple[list[AnnotationRecord], int, int]:
    """Parse every PASCAL VOC XML under ``root`` into flat records.

    Returns:
        ``(records, files_parsed, files_failed)`` so callers can tell an empty but valid
        archive from an unreadable one.
    """
    records: list[AnnotationRecord] = []
    parsed = 0
    failed = 0
    for xml_path in sorted(root.rglob("*.xml")):
        annotation, error = dataset.parse_voc_annotation(xml_path)
        if annotation is None:
            failed += 1
            continue
        parsed += 1
        image_path = _resolve_image(xml_path, annotation.filename)
        for obj in annotation.objects:
            records.append(
                AnnotationRecord(image_path=image_path, class_name=obj.label, box=obj.xyxy)
            )
    return records, parsed, failed


def load_records(root: Path) -> list[AnnotationRecord]:
    """Load a dataset folder into a flat list of :class:`AnnotationRecord`.

    Args:
        root: Folder holding the raw dataset. It is only read, never written.

    Returns:
        One record per annotated object.

    Raises:
        DatasetNotFoundError: ``root`` is not an existing directory.
        UnknownAnnotationFormatError: no supported annotation format was recognised.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise DatasetNotFoundError(f"No dataset folder at: {root}")

    if any(root.rglob("*.xml")):
        records, parsed, failed = _voc_records(root)
        if parsed == 0:
            raise UnknownAnnotationFormatError(
                f"Found XML files but none parsed as PASCAL VOC ({failed} unreadable) under {root}."
            )
        return records

    raise UnknownAnnotationFormatError(
        f"No supported annotation format found under {root}. "
        "Currently recognised: PASCAL VOC XML (one .xml beside each image)."
    )


def load_table(root: Path):  # noqa: ANN201 - returns a pandas DataFrame built lazily
    """Return the dataset as a single table with columns image_path, class_name, box.

    A pandas DataFrame keeps the rows easy to count and inspect in scripts while the
    record loader stays framework free for tests.
    """
    import pandas as pd

    records = load_records(root)
    return pd.DataFrame(
        [
            {
                "image_path": r.image_path,
                "class_name": r.class_name,
                "box": list(r.box),
            }
            for r in records
        ],
        columns=["image_path", "class_name", "box"],
    )


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

DROP_UNREADABLE = "unreadable_image"
DROP_NO_ANNOTATION = "no_annotation"
DROP_ANNOTATION_UNREADABLE = "annotation_unreadable"
DROP_ALL_IGNORED = "all_annotations_ignored"
DROP_DUPLICATE = "near_duplicate"


@dataclass
class CleaningReport:
    """Outcome of a cleaning pass, in image counts.

    Attributes:
        loaded: Images discovered under the dataset folder.
        dropped_by_reason: Count of dropped images for each reason.
        duplicates_removed: Subset of drops that were near duplicates.
        remaining: Images kept after every drop.
        dropped_rows: One ``{image_path, reason, detail}`` per dropped image.
        kept_paths: Paths of the surviving images.
    """

    loaded: int
    dropped_by_reason: dict[str, int]
    duplicates_removed: int
    remaining: int
    dropped_rows: list[dict[str, str]] = field(default_factory=list)
    kept_paths: list[str] = field(default_factory=list)

    @property
    def dropped_total(self) -> int:
        """Total images dropped across every reason."""
        return sum(self.dropped_by_reason.values())


def load_class_mapping(mapping_path: Path) -> dict[str, str]:
    """Read the reviewed class mapping into ``{normalised_label: category}``.

    Missing or unreadable files return an empty mapping so cleaning falls back to the
    in-code ``dataset.LABEL_TO_CATEGORY`` rather than failing.
    """
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        return {}
    try:
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    classes = payload.get("classes", {}) if isinstance(payload, dict) else {}
    return {
        dataset.normalise_label(name): str(info.get("proposed", "other"))
        for name, info in classes.items()
        if isinstance(info, dict)
    }


def _classify(label: str, mapping: dict[str, str]) -> str:
    """Map a raw label to a category, preferring the reviewed mapping then the code."""
    key = dataset.normalise_label(label)
    if key in mapping:
        return mapping[key]
    return dataset.map_label(label)[0]


def _annotation_status(
    image_path: Path, mapping: dict[str, str]
) -> tuple[str, int]:
    """Classify an image's annotation as usable or a specific drop reason.

    Returns:
        ``(status, kept_objects)`` where ``status`` is ``"ok"`` or one of the
        ``DROP_*`` reasons. Pairing uses the same sibling ``.xml`` convention as the rest
        of the project.
    """
    xml_path = image_path.with_suffix(".xml")
    if not xml_path.exists():
        return DROP_NO_ANNOTATION, 0
    annotation, _ = dataset.parse_voc_annotation(xml_path)
    if annotation is None:
        return DROP_ANNOTATION_UNREADABLE, 0
    if not annotation.objects:
        return DROP_NO_ANNOTATION, 0
    kept = sum(1 for obj in annotation.objects if _classify(obj.label, mapping) != "ignore")
    if kept == 0:
        return DROP_ALL_IGNORED, 0
    return "ok", kept


def _find_duplicate(phash: str, kept: list[tuple[str, str]], threshold: int) -> str | None:
    """Return the path of the first kept image within ``threshold`` bits, else ``None``."""
    for kept_path, kept_hash in kept:
        if dataset.hash_distance(phash, kept_hash) < threshold:
            return kept_path
    return None


def write_curation_log(rows: list[dict[str, str]], log_path: Path) -> None:
    """Write dropped-image rows to a CSV log, creating the parent folder."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "reason", "detail"])
        writer.writeheader()
        writer.writerows(rows)


def clean_dataset(
    root: Path,
    *,
    hamming_threshold: int = dataset.DUPLICATE_HASH_DISTANCE,
    mapping_path: Path | None = None,
    log_path: Path | None = None,
    write_log: bool = True,
) -> CleaningReport:
    """Drop unusable images and near duplicates from a dataset, without modifying it.

    Order of operations: read every image, drop unreadable files, drop images whose
    annotation is missing, unreadable, or entirely mapped to ``ignore``, then remove near
    duplicate frames with perceptual hashing. Each dropped image is recorded with its
    reason. The source folder is only read, never written.

    Args:
        root: Dataset folder to clean.
        hamming_threshold: Bit distance below which two frames count as duplicates.
        mapping_path: Reviewed class mapping JSON. Defaults to ``config.CLASS_MAPPING_JSON``.
        log_path: Destination CSV. Defaults to ``config.CURATION_LOG_CSV``.
        write_log: When False, skip writing the CSV.

    Returns:
        A :class:`CleaningReport` of counts and per image drop rows.

    Raises:
        DatasetNotFoundError: ``root`` is not an existing directory.
    """
    from collections import Counter

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise DatasetNotFoundError(f"No dataset folder at: {root}")

    mapping = load_class_mapping(mapping_path or config.CLASS_MAPPING_JSON)
    images = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    dropped: list[dict[str, str]] = []
    survivors: list[tuple[str, str]] = []
    for image in images:
        array = dataset.load_image(image)
        if array is None:
            dropped.append({"image_path": str(image), "reason": DROP_UNREADABLE, "detail": ""})
            continue
        status, _ = _annotation_status(image, mapping)
        if status != "ok":
            dropped.append({"image_path": str(image), "reason": status, "detail": ""})
            continue
        survivors.append((str(image), dataset.perceptual_hash(array)))

    kept: list[tuple[str, str]] = []
    for path, phash in survivors:
        match = _find_duplicate(phash, kept, hamming_threshold)
        if match is not None:
            dropped.append({"image_path": path, "reason": DROP_DUPLICATE, "detail": match})
        else:
            kept.append((path, phash))

    dropped_by_reason = dict(Counter(row["reason"] for row in dropped))
    report = CleaningReport(
        loaded=len(images),
        dropped_by_reason=dropped_by_reason,
        duplicates_removed=dropped_by_reason.get(DROP_DUPLICATE, 0),
        remaining=len(kept),
        dropped_rows=dropped,
        kept_paths=[path for path, _ in kept],
    )

    if write_log:
        write_curation_log(dropped, log_path or config.CURATION_LOG_CSV)
    return report


# ---------------------------------------------------------------------------
# Feature measurement and quartile based tagging
# ---------------------------------------------------------------------------

#: Numeric features whose spread defines the proposed tag cut-offs.
FEATURE_NAMES: tuple[str, ...] = (
    "brightness",
    "saturation",
    "vehicle_count",
    "median_box_fraction",
)

#: Human readable rule behind each proposed tag, stored with the cut-offs.
TAG_RULES: dict[str, str] = {
    "day": "brightness >= cutoffs.brightness.q3",
    "night": "brightness <= cutoffs.brightness.q1",
    "glare": "saturation >= cutoffs.saturation.q3",
    "dense": "vehicle_count >= cutoffs.vehicle_count.q3",
    "sparse": "vehicle_count <= cutoffs.vehicle_count.q1",
    "distant": "median_box_fraction <= cutoffs.median_box_fraction.q1",
}


@dataclass(frozen=True)
class ImageFeature:
    """Per image measurements used to propose condition tags.

    Every value is derived from the pixels or the ground truth annotation. No model
    detection contributes to any field.
    """

    image_path: str
    brightness: float
    saturation: float
    vehicle_count: int
    median_box_fraction: float
    rickshaw_or_cng: bool


def saturation_fraction(image: np.ndarray) -> float:
    """Return the share of pixels whose HSV saturation is at least ``SATURATION_MIN``."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[..., 1] >= dataset.SATURATION_MIN))


def _median_box_fraction(objects, width: int, height: int) -> float:
    """Median annotated box area expressed as a fraction of the whole image."""
    if not objects or width <= 0 or height <= 0:
        return 0.0
    areas = [obj.area for obj in objects]
    return float(np.median(areas)) / float(width * height)


def _rickshaw_or_cng(objects) -> bool:
    """True when any annotation maps to the rickshaw category or names a CNG class."""
    for obj in objects:
        category, _ = dataset.map_label(obj.label)
        if category == "rickshaw" or "cng" in dataset.normalise_label(obj.label):
            return True
    return False


def compute_feature(image_path: Path, image: np.ndarray | None = None) -> ImageFeature:
    """Measure one readable image and its paired annotation.

    Args:
        image_path: Image to measure. Its sibling ``.xml`` supplies the annotation.
        image: Pre-decoded array, reused when already loaded.

    Returns:
        An :class:`ImageFeature` with brightness, saturation, count, box fraction, and
        rickshaw or CNG presence.

    Raises:
        ValueError: ``image`` is not supplied and the file cannot be decoded.
    """
    image_path = Path(image_path)
    if image is None:
        image = dataset.load_image(image_path)
        if image is None:
            raise ValueError(f"Cannot decode image for feature measurement: {image_path}")
    metrics = dataset.measure_image(image)
    annotation, _ = dataset.parse_voc_annotation(image_path.with_suffix(".xml"))
    objects = annotation.objects if annotation is not None else ()
    counts = dataset.count_by_category(objects)["counts"]
    kept = dataset.surviving_objects(objects)
    return ImageFeature(
        image_path=str(image_path),
        brightness=float(metrics.mean_luma),
        saturation=saturation_fraction(image),
        vehicle_count=dataset.total_vehicles(counts),
        median_box_fraction=_median_box_fraction(kept, metrics.width, metrics.height),
        rickshaw_or_cng=_rickshaw_or_cng(objects),
    )


def compute_features(paths) -> list[ImageFeature]:
    """Measure every readable image in ``paths``, skipping any that fail to decode."""
    features: list[ImageFeature] = []
    for path in paths:
        path = Path(path)
        image = dataset.load_image(path)
        if image is None:
            continue
        features.append(compute_feature(path, image))
    return features


def _series(features: list[ImageFeature], name: str) -> np.ndarray:
    return np.asarray([getattr(f, name) for f in features], dtype=float)


def summarize_distribution(values) -> dict[str, float]:
    """Summarise a numeric series with its count, spread, and quartiles."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"count": 0, "min": 0.0, "q1": 0.0, "median": 0.0, "q3": 0.0,
                "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "q1": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "q3": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def compute_cutoffs(features: list[ImageFeature]) -> dict[str, dict[str, float]]:
    """Return the first, second, and third quartile of each numeric feature.

    These quartiles are the proposed tag cut-offs and are fixed on the full cleaned pool
    before any sampling takes place.
    """
    cutoffs: dict[str, dict[str, float]] = {}
    for name in FEATURE_NAMES:
        summary = summarize_distribution(_series(features, name))
        cutoffs[name] = {"q1": summary["q1"], "median": summary["median"], "q3": summary["q3"]}
    return cutoffs


def assign_tags(feature: ImageFeature, cutoffs: dict[str, dict[str, float]]) -> list[str]:
    """Return the proposed condition tags for one image under the given cut-offs."""
    tags: list[str] = []
    if feature.brightness >= cutoffs["brightness"]["q3"]:
        tags.append("day")
    if feature.brightness <= cutoffs["brightness"]["q1"]:
        tags.append("night")
    if feature.saturation >= cutoffs["saturation"]["q3"]:
        tags.append("glare")
    if feature.vehicle_count >= cutoffs["vehicle_count"]["q3"]:
        tags.append("dense")
    if feature.vehicle_count <= cutoffs["vehicle_count"]["q1"]:
        tags.append("sparse")
    if feature.median_box_fraction <= cutoffs["median_box_fraction"]["q1"]:
        tags.append("distant")
    return tags


def tag_counts(features: list[ImageFeature], cutoffs: dict[str, dict[str, float]]) -> dict[str, int]:
    """Count how many images fall under each proposed tag."""
    counts = {tag: 0 for tag in dataset.CONDITION_TAGS}
    for feature in features:
        for tag in assign_tags(feature, cutoffs):
            counts[tag] += 1
    return counts


def feature_distributions(features: list[ImageFeature]) -> dict[str, dict[str, float]]:
    """Summarise the distribution of every numeric feature plus rickshaw presence."""
    distributions = {name: summarize_distribution(_series(features, name)) for name in FEATURE_NAMES}
    distributions["rickshaw_or_cng_rate"] = {
        "rate": float(np.mean([f.rickshaw_or_cng for f in features])) if features else 0.0,
        "images": int(sum(1 for f in features if f.rickshaw_or_cng)),
    }
    return distributions


def write_tag_cutoffs(
    cutoffs: dict[str, dict[str, float]],
    *,
    source: str,
    images_used: int,
    path: Path,
) -> None:
    """Persist the proposed quartile cut-offs, creating the parent folder."""
    payload = {
        "_note": (
            "Quartile cut-offs computed on the cleaned pool and fixed before any "
            "sampling. Tags are proposed from ground truth and pixels only, never from "
            "model detections."
        ),
        "source": source,
        "images_used": images_used,
        "quantiles": {"q1": 25.0, "q3": 75.0},
        "cutoffs": cutoffs,
        "tag_rules": TAG_RULES,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Seeded stratified sampling
# ---------------------------------------------------------------------------


def load_tag_cutoffs(path: Path) -> dict[str, dict[str, float]]:
    """Read the persisted quartile cut-offs so sampling reuses the fixed reference.

    Raises:
        UnknownAnnotationFormatError: The file is missing or has no ``cutoffs`` object,
            which means ``tag_dataset.py`` has not been run yet.
    """
    path = Path(path)
    if not path.exists():
        raise UnknownAnnotationFormatError(
            f"No tag cut-offs at {path}. Run scripts/tag_dataset.py first so cut-offs "
            "exist before any sampling."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    cutoffs = payload.get("cutoffs") if isinstance(payload, dict) else None
    if not isinstance(cutoffs, dict) or not cutoffs:
        raise UnknownAnnotationFormatError(f"Tag cut-off file is missing a cutoffs object: {path}")
    return cutoffs


@dataclass(frozen=True)
class SampleCandidate:
    """One pool image expressed for sampling."""

    image_path: str
    scene: str
    split: str
    tags: tuple[str, ...]
    rickshaw_or_cng: bool


@dataclass
class SampleResult:
    """Outcome of a stratified draw, with honest coverage and any unreachable quota."""

    chosen: list[SampleCandidate]
    tag_coverage: dict[str, int]
    rickshaw_count: int
    scene_counts: dict[str, int]
    unmet_tags: dict[str, int]
    unmet_rickshaw: int
    pool_size: int
    target: int


def stratified_sample(
    candidates: list[SampleCandidate],
    *,
    seed: int,
    target: int,
    min_per_tag: int,
    min_rickshaw: int,
    max_per_scene: int,
) -> SampleResult:
    """Select a stratified subset with a reproducible seeded draw.

    Reserves at least ``min_per_tag`` images for every condition tag and at least
    ``min_rickshaw`` images carrying a rickshaw or CNG class, while taking no more than
    ``max_per_scene`` images from any one apparent scene. Quotas are treated as minima,
    never padded: a quota that cannot be met under the scene cap is reported, not faked.
    Ties are broken by a seeded value assigned over the pool sorted by path, so the same
    inputs and seed always produce the same selection.
    """
    pool = sorted(candidates, key=lambda c: c.image_path)
    rng = random.Random(seed)
    rank = {c.image_path: rng.random() for c in pool}

    chosen: list[SampleCandidate] = []
    chosen_paths: set[str] = set()
    scene_counts: dict[str, int] = {}
    tag_counts = {tag: 0 for tag in dataset.CONDITION_TAGS}
    rickshaw_count = 0

    def add(candidate: SampleCandidate) -> None:
        nonlocal rickshaw_count
        chosen.append(candidate)
        chosen_paths.add(candidate.image_path)
        scene_counts[candidate.scene] = scene_counts.get(candidate.scene, 0) + 1
        for tag in candidate.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        if candidate.rickshaw_or_cng:
            rickshaw_count += 1

    def eligible() -> list[SampleCandidate]:
        return [
            c
            for c in pool
            if c.image_path not in chosen_paths and scene_counts.get(c.scene, 0) < max_per_scene
        ]

    def open_tags() -> list[str]:
        return [tag for tag in dataset.CONDITION_TAGS if tag_counts[tag] < min_per_tag]

    def helps(candidate: SampleCandidate, want_tags: list[str], want_rickshaw: bool) -> int:
        utility = sum(1 for tag in want_tags if tag in candidate.tags)
        if want_rickshaw and candidate.rickshaw_or_cng:
            utility += 1
        return utility

    # Phase 1: close the tag and rickshaw quotas, prioritising the scarcest needs.
    while len(chosen) < target:
        want_tags = open_tags()
        want_rickshaw = rickshaw_count < min_rickshaw
        if not want_tags and not want_rickshaw:
            break
        cands = eligible()
        if not cands:
            break
        utilities = [helps(c, want_tags, want_rickshaw) for c in cands]
        best = max(utilities)
        if best <= 0:
            break  # remaining quotas are unreachable under the scene cap
        pick = max(
            (c for c, u in zip(cands, utilities) if u == best),
            key=lambda c: rank[c.image_path],
        )
        add(pick)

    # Phase 2: fill toward the target with the highest seeded ranks.
    while len(chosen) < target:
        cands = eligible()
        if not cands:
            break
        add(max(cands, key=lambda c: rank[c.image_path]))

    unmet_tags = {
        tag: min_per_tag - tag_counts[tag]
        for tag in dataset.CONDITION_TAGS
        if tag_counts[tag] < min_per_tag
    }
    return SampleResult(
        chosen=chosen,
        tag_coverage=dict(tag_counts),
        rickshaw_count=rickshaw_count,
        scene_counts=dict(scene_counts),
        unmet_tags=unmet_tags,
        unmet_rickshaw=max(0, min_rickshaw - rickshaw_count),
        pool_size=len(pool),
        target=target,
    )


def annotation_counts(image_path: Path) -> dict[str, int]:
    """Return the annotation derived per category counts and total for one image.

    These counts come from the ground truth XML only, never from a model detector.
    """
    annotation, _ = dataset.parse_voc_annotation(Path(image_path).with_suffix(".xml"))
    objects = annotation.objects if annotation is not None else ()
    counts = dataset.count_by_category(objects)["counts"]
    result = {category: int(counts.get(category, 0)) for category in dataset.LABEL_CATEGORIES}
    result["total"] = dataset.total_vehicles(counts)
    return result


def write_overlay(image_path: Path, out_path: Path) -> bool:
    """Draw the dataset's own boxes onto a copy of the image and save it.

    The original file is left untouched; only a copy is annotated. ``out_path`` parents
    are created as needed.
    """
    image = dataset.load_image(image_path)
    if image is None:
        return False
    annotation, _ = dataset.parse_voc_annotation(Path(image_path).with_suffix(".xml"))
    objects = annotation.objects if annotation is not None else ()
    canvas = image.copy()
    for obj in objects:
        x_min, y_min, x_max, y_max = (int(round(value)) for value in obj.xyxy)
        category, _ = dataset.map_label(obj.label)
        rgb = config.CLASS_COLORS_RGB.get(category)
        color = (rgb[2], rgb[1], rgb[0]) if rgb else (128, 128, 128)
        cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(
            canvas, category, (x_min, max(y_min - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return True
