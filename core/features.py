"""Label free reliability features for one image, from inference time information only.

Every feature here is computed from the cached raw detections (standard mode,
all registry models) and the image pixels. No label file, annotation count, or
dataset annotation is ever read, so the table can be built before any human
labelling exists and can later be joined against labels without having leaked
them into the features.

Feature groups (each defined in ``docs/metrics.md``):

* confidence: mean and minimum detection confidence and the share of detections
  in the mid band ``[0.25, 0.50]``, per model;
* threshold sensitivity: max minus min of the model's total count across
  ``config.THRESHOLD_SWEEP``, derived by confidence filtering;
* agreement: the H2 box agreement and count disagreement for every model pair
  plus the within-family and cross-family aggregates, reusing
  ``core.reliability.build_reliability_report`` unchanged;
* scene: mean brightness, contrast, saturated pixel share, and blur (variance
  of the Laplacian), the same brightness and blur definitions as
  ``core.dataset.measure_image``;
* geometry: median box area as a fraction of the image area, the share of boxes
  below ``SMALL_BOX_AREA_FRACTION``, and detection density per megapixel,
  pooled over every present model's boxes at the default confidence.

Undefined values are ``None`` (written as empty CSV cells), never a fabricated
zero: an empty detection set has no mean confidence, but its threshold
sensitivity and detection density are honestly zero.
"""
from __future__ import annotations

import csv
import itertools
import statistics
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import config, counting, dataset, reliability
from .detector import Detection

#: Inclusive confidence band whose share is reported per model. Detections here
#: are the ones most likely to flip when the threshold moves.
MID_CONFIDENCE_BAND: tuple[float, float] = (0.25, 0.50)

#: A box is small when its area is below this fraction of the image area
#: (0.1%, about a 30x30 box in a 1000x1000 frame). Small boxes mark distant
#: vehicles, which detectors miss most often.
SMALL_BOX_AREA_FRACTION: float = 0.001

_MODEL_KEYS: tuple[str, ...] = tuple(config.MODEL_REGISTRY)
_DISPLAY_BY_KEY: dict[str, str] = {key: spec.display_name for key, spec in config.MODEL_REGISTRY.items()}
_FAMILY_BY_DISPLAY: dict[str, str] = {spec.display_name: spec.family for spec in config.MODEL_REGISTRY.values()}
_PAIRS: tuple[tuple[str, str], ...] = tuple(
    itertools.combinations([_DISPLAY_BY_KEY[key] for key in _MODEL_KEYS], 2)
)


def _feature_columns() -> tuple[str, ...]:
    """The ordered feature column names, stable across runs."""
    columns: list[str] = []
    for key in _MODEL_KEYS:
        columns.extend([
            f"{key}_mean_confidence",
            f"{key}_min_confidence",
            f"{key}_mid_band_share",
            f"{key}_threshold_sensitivity",
        ])
    for first, second in _PAIRS:
        columns.append(f"{first} vs {second} Box Agreement")
        columns.append(f"{first} vs {second} Count Disagreement")
    columns.extend([
        "within_family_box_agreement",
        "within_family_count_disagreement",
        "cross_family_box_agreement",
        "cross_family_count_disagreement",
        "mean_brightness",
        "contrast",
        "saturated_share",
        "blur",
        "median_box_area_fraction",
        "small_box_share",
        "detection_density",
    ])
    return tuple(columns)


#: Full CSV header: identity, settings provenance, then every feature.
TABLE_COLUMNS: tuple[str, ...] = ("item_id", "settings_sha256", *_feature_columns())


def confidence_features(detections: list[Detection]) -> dict[str, float | None]:
    """Mean, minimum, and mid-band share of one model's detections.

    The band bounds are inclusive. An empty set yields ``None`` for all three:
    there is no confidence to summarise and no denominator for the share.
    """
    if not detections:
        return {"mean_confidence": None, "min_confidence": None, "mid_band_share": None}
    confidences = [d.confidence for d in detections]
    low, high = MID_CONFIDENCE_BAND
    in_band = sum(1 for value in confidences if low <= value <= high)
    return {
        "mean_confidence": statistics.fmean(confidences),
        "min_confidence": min(confidences),
        "mid_band_share": in_band / len(confidences),
    }


def threshold_sensitivity(
    detections: list[Detection],
    thresholds: tuple[float, ...] = config.THRESHOLD_SWEEP,
) -> float:
    """Max minus min of the total vehicle count across the sweep thresholds.

    ``detections`` must come from a run at or below ``min(thresholds)``; higher
    thresholds are derived by confidence filtering, exactly as in
    ``core.reliability``. Classes outside the covered set never count. An empty
    set counts zero at every threshold, so the sensitivity is 0.0.
    """
    kept = counting.filter_by_classes(detections, list(config.TARGET_CLASS_NAMES))
    totals = [
        sum(1 for detection in kept if detection.confidence >= threshold)
        for threshold in thresholds
    ]
    return float(max(totals) - min(totals))


def scene_features(image: np.ndarray | None) -> dict[str, float | None]:
    """Brightness, contrast, saturation, and blur from the pixels alone.

    Brightness is the mean grayscale value and blur the variance of the
    Laplacian, the same definitions as ``core.dataset.measure_image`` uses for
    ``mean_luma`` and ``sharpness``. Contrast is the grayscale standard
    deviation. A pixel is saturated when any BGR channel is clipped at 255.
    A missing or undecodable image yields ``None`` for all four.
    """
    if image is None:
        return {"mean_brightness": None, "contrast": None, "saturated_share": None, "blur": None}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "mean_brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "saturated_share": float(np.mean(np.any(image >= 255, axis=2))),
        "blur": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
    }


def geometry_features(
    detections: list[Detection], width: int, height: int
) -> dict[str, float | None]:
    """Box size and density features over pooled detections from every model.

    Each box contributes once per model that reported it; the features describe
    detector output, not unique vehicles. ``median_box_area_fraction`` is the
    median box area divided by the image area, ``small_box_share`` the fraction
    of boxes strictly below ``SMALL_BOX_AREA_FRACTION``, and
    ``detection_density`` the number of boxes per megapixel. With no boxes the
    median and share are ``None`` while the density is honestly 0.0.
    """
    image_area = float(width * height)
    if not detections:
        return {"median_box_area_fraction": None, "small_box_share": None, "detection_density": 0.0}
    fractions = [
        max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) / image_area
        for box in (detection.xyxy for detection in detections)
    ]
    return {
        "median_box_area_fraction": statistics.median(fractions),
        "small_box_share": sum(1 for value in fractions if value < SMALL_BOX_AREA_FRACTION) / len(fractions),
        "detection_density": len(fractions) * 1_000_000.0 / image_area,
    }


def image_features(
    detections_by_model: dict[str, list[Detection]],
    image: np.ndarray | None,
    confidence: float = config.DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    """Every feature column for one image, keyed like :data:`TABLE_COLUMNS`.

    ``detections_by_model`` maps registry keys to the cached standard-mode
    detection lists produced at the lowest sweep threshold; a missing key means
    that model has no cached entry and its columns stay ``None``. Confidence,
    agreement, and geometry use the detections filtered at ``confidence``;
    threshold sensitivity uses the full low-threshold list.
    """
    row: dict[str, Any] = {}
    at_default: dict[str, list[Detection]] = {}
    for key in _MODEL_KEYS:
        detections = detections_by_model.get(key)
        if detections is None:
            row.update({
                f"{key}_mean_confidence": None,
                f"{key}_min_confidence": None,
                f"{key}_mid_band_share": None,
                f"{key}_threshold_sensitivity": None,
            })
            continue
        kept = counting.filter_by_classes(detections, list(config.TARGET_CLASS_NAMES))
        filtered = [d for d in kept if d.confidence >= confidence]
        at_default[key] = filtered
        row.update({f"{key}_{name}": value for name, value in confidence_features(filtered).items()})
        row[f"{key}_threshold_sensitivity"] = threshold_sensitivity(kept)

    report = reliability.build_reliability_report(
        {_DISPLAY_BY_KEY[key]: detections for key, detections in at_default.items()},
        {_DISPLAY_BY_KEY[key]: _FAMILY_BY_DISPLAY[_DISPLAY_BY_KEY[key]] for key in at_default},
        confidence,
    )
    pair_lookup = {pair.models: pair.agreement for pair in report.pairs}
    for first, second in _PAIRS:
        agreement = pair_lookup.get((first, second))
        row[f"{first} vs {second} Box Agreement"] = agreement.box_agreement if agreement else None
        row[f"{first} vs {second} Count Disagreement"] = agreement.count_disagreement if agreement else None
    for label, aggregate in (("within_family", report.within_family), ("cross_family", report.cross_family)):
        row[f"{label}_box_agreement"] = aggregate.box_agreement if aggregate else None
        row[f"{label}_count_disagreement"] = aggregate.count_disagreement if aggregate else None

    row.update(scene_features(image))
    if image is None:
        row.update({"median_box_area_fraction": None, "small_box_share": None, "detection_density": None})
    else:
        height, width = image.shape[:2]
        pooled = [detection for detections in at_default.values() for detection in detections]
        row.update(geometry_features(pooled, width, height))
    return row


def build_feature_table(
    cache: dict[tuple[int, str, str], list[Detection]],
    image_paths: dict[int, Path],
    settings_sha256: str,
    confidence: float = config.DEFAULT_CONFIDENCE,
) -> list[dict[str, Any]]:
    """One row per item that has any standard-mode cache entry, in item order.

    ``cache`` is the detection cache loaded at the lowest sweep threshold (see
    ``core.evaluation.load_detection_cache``); ``image_paths`` maps item numbers
    to their test set image files. Items whose image file is missing still get
    detection features, with the pixel and geometry columns left ``None``.
    """
    items = sorted({item for item, _model, mode in cache if mode == "standard"})
    rows: list[dict[str, Any]] = []
    for item in items:
        detections_by_model = {
            key: cache[(item, key, "standard")]
            for key in _MODEL_KEYS
            if (item, key, "standard") in cache
        }
        path = image_paths.get(item)
        image = dataset.load_image(path) if path is not None else None
        row = {"item_id": dataset.format_item_id(item), "settings_sha256": settings_sha256}
        row.update(image_features(detections_by_model, image, confidence))
        rows.append(row)
    return rows


def summarise_features(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per feature column: mean, min, max over present values, and missing count.

    Only aggregates are produced; no per-image value ever leaves this function.
    """
    summary: dict[str, dict[str, Any]] = {}
    for column in TABLE_COLUMNS[2:]:
        values = [row[column] for row in rows if row.get(column) is not None]
        summary[column] = {
            "mean": statistics.fmean(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "missing": len(rows) - len(values),
        }
    return summary


def needs_recompute(path: Path, settings_sha256: str) -> bool:
    """True when the feature table must be rebuilt for this settings hash.

    A missing, empty, unreadable, or differently stamped table is recomputed;
    a table whose every row carries the current hash is reused as is.
    """
    if not path.exists():
        return True
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != TABLE_COLUMNS:
                return True
            rows = list(reader)
    except (OSError, csv.Error):
        return True
    if not rows:
        return True
    return any(row.get("settings_sha256") != settings_sha256 for row in rows)