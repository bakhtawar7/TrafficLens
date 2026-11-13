"""Zone geometry and optional, user-supplied occupancy configurations."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

from . import config
from .detector import Detection


class ZoneConfigurationError(ValueError):
    """A supplied zone-analysis configuration is invalid."""


def parse_polygon_coordinates(text: str, image_width: int, image_height: int) -> Polygon:
    """Parse one ``x,y`` pixel coordinate per line into a valid image polygon."""
    coordinates: list[tuple[float, float]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) != 2:
            raise ValueError(f"Line {line_number}: enter coordinates as x,y.")
        try:
            x, y = (float(part.strip()) for part in parts)
        except ValueError as error:
            raise ValueError(f"Line {line_number}: x and y must be numbers.") from error
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"Line {line_number}: coordinates must be finite.")
        if not 0 <= x <= image_width or not 0 <= y <= image_height:
            raise ValueError(
                f"Line {line_number}: coordinates must lie within the image "
                f"(0..{image_width}, 0..{image_height})."
            )
        coordinates.append((x, y))
    if len(coordinates) < 3:
        raise ValueError("Enter at least three polygon vertices.")
    polygon = Polygon(coordinates)
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError("The zone must be a non-self-intersecting polygon with area.")
    return polygon


def summarize_zone(
    detections: list[Detection],
    polygon: Polygon,
    selected_classes: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Count box centres in a zone and compute exact clipped box-union occupancy."""
    counts = {name: 0 for name in selected_classes}
    clipped_boxes = []
    for detection in detections:
        if detection.class_name not in counts:
            continue
        x_min, y_min, x_max, y_max = detection.xyxy
        if not all(math.isfinite(value) for value in detection.xyxy):
            continue
        if x_max <= x_min or y_max <= y_min:
            continue
        if polygon.covers(Point((x_min + x_max) / 2, (y_min + y_max) / 2)):
            counts[detection.class_name] += 1
        clipped = box(x_min, y_min, x_max, y_max).intersection(polygon)
        if not clipped.is_empty and clipped.area > 0:
            clipped_boxes.append(clipped)
    covered_area = unary_union(clipped_boxes).area if clipped_boxes else 0.0
    return {
        "counts_by_class": counts,
        "vehicle_count": sum(counts.values()),
        "occupancy": min(covered_area / polygon.area, 1.0),
    }


def load_pcu_weights(
    path: Path | None = None,
) -> tuple[dict[str, float] | None, str | None, str | None]:
    """Load complete, cited category weights; never substitute default PCU values."""
    path = path or config.PCU_WEIGHTS_JSON
    if not path.exists():
        return None, "PCU needs a cited source before weighted values can be shown.", None
    data = _read_object(path, "PCU weights")
    raw_weights = data.get("category_weights")
    if not isinstance(raw_weights, dict):
        raise ZoneConfigurationError("PCU config must contain a category_weights object.")
    if any(
        name not in raw_weights or raw_weights[name] is None
        for name in config.TARGET_CLASS_NAMES
    ):
        return None, "PCU needs a cited source and a weight for every vehicle category.", None
    citation = data.get("source_citation")
    retrieval_date = data.get("retrieval_date")
    if (
        not isinstance(citation, str)
        or not citation.strip()
        or not isinstance(retrieval_date, str)
        or not retrieval_date.strip()
    ):
        return (
            None,
            "PCU needs a cited source and retrieval date before weighted values can be shown.",
            None,
        )
    weights: dict[str, float] = {}
    for name in config.TARGET_CLASS_NAMES:
        value = raw_weights[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ZoneConfigurationError(
                f"PCU weight for {name} must be a finite, non-negative number or null."
            )
        weights[name] = float(value)
    return weights, None, f"{citation.strip()} (retrieved {retrieval_date.strip()})"


def load_occupancy_threshold(
    path: Path | None = None,
) -> tuple[float | None, str | None, str | None]:
    """Load the user-provided occupancy threshold and its justification."""
    path = path or config.CONGESTION_THRESHOLDS_JSON
    if not path.exists():
        return None, None, None
    data = _read_object(path, "congestion thresholds")
    threshold = data.get("occupancy_threshold")
    justification = data.get("justification")
    if threshold is None:
        return None, None, None
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ZoneConfigurationError("occupancy_threshold must be a number from 0 to 1.")
    if not isinstance(justification, str) or not justification.strip():
        raise ZoneConfigurationError("A non-empty justification is required for the occupancy threshold.")
    return float(threshold), justification.strip(), None


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ZoneConfigurationError(f"Could not read {label} config at {path}: {error}") from error
    if not isinstance(value, dict):
        raise ZoneConfigurationError(f"{label} config must be a JSON object.")
    return value
