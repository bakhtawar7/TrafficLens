"""Exact polygon-zone statistics and optional user-supplied configuration."""
from __future__ import annotations

import json

import pytest
from shapely.geometry import Polygon

from core import config, zones
from core.detector import Detection


def _detection(
    class_name: str,
    box: tuple[float, float, float, float],
) -> Detection:
    return Detection(2, class_name, 0.9, box)


def test_polygon_parser_accepts_image_edge_vertices_and_rejects_bad_polygons() -> None:
    polygon = zones.parse_polygon_coordinates("0,0\n10,0\n10,10\n0,10", 10, 10)
    assert polygon.area == pytest.approx(100)

    with pytest.raises(ValueError, match="within the image"):
        zones.parse_polygon_coordinates("0,0\n11,0\n10,10", 10, 10)
    with pytest.raises(ValueError, match="non-self-intersecting"):
        zones.parse_polygon_coordinates("0,0\n10,10\n0,10\n10,0", 10, 10)


def test_zone_counts_centres_on_boundary_and_unions_overlapping_boxes() -> None:
    polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    detections = [
        _detection("car", (-2, 2, 2, 4)),  # centre lies on the polygon boundary
        _detection("bus", (0, 0, 6, 6)),
        _detection("truck", (4, 4, 10, 10)),
        _detection("motorcycle", (-4, 2, 0, 4)),  # centre outside, box touches edge
        _detection("person", (0, 0, 10, 10)),
    ]

    result = zones.summarize_zone(detections, polygon, config.TARGET_CLASS_NAMES)

    assert result["counts_by_class"] == {
        "car": 1, "motorcycle": 0, "bus": 1, "truck": 1,
    }
    assert result["vehicle_count"] == 3
    # 36 + 36 - 4 shared square; the car box is covered by the bus box.
    assert result["occupancy"] == pytest.approx(0.68)


def test_empty_zone_has_zero_counts_and_occupancy() -> None:
    result = zones.summarize_zone(
        [], Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        config.TARGET_CLASS_NAMES,
    )
    assert result["vehicle_count"] == 0
    assert set(result["counts_by_class"].values()) == {0}
    assert result["occupancy"] == 0


def test_missing_or_null_pcu_weights_only_allow_raw_counts(tmp_path, monkeypatch) -> None:
    path = tmp_path / "pcu_weights.json"
    monkeypatch.setattr(config, "PCU_WEIGHTS_JSON", path)
    weights, message, source = zones.load_pcu_weights()
    assert weights is None
    assert "PCU needs a cited source" in message
    assert source is None

    path.write_text(json.dumps({
        "category_weights": {name: 1.0 for name in config.TARGET_CLASS_NAMES}
        | {"bus": None},
        "source_citation": "Example citation",
        "retrieval_date": "2026-11-13",
    }), encoding="utf-8")
    weights, message, source = zones.load_pcu_weights()
    assert weights is None
    assert "every vehicle category" in message
    assert source is None


def test_pcu_weights_require_citation_and_retrieval_date(tmp_path) -> None:
    path = tmp_path / "pcu_weights.json"
    path.write_text(json.dumps({
        "category_weights": {name: 1.0 for name in config.TARGET_CLASS_NAMES},
        "source_citation": "",
        "retrieval_date": "",
    }), encoding="utf-8")
    weights, message, source = zones.load_pcu_weights(path)
    assert weights is None
    assert "cited source and retrieval date" in message
    assert source is None


def test_valid_pcu_config_returns_only_user_values_and_source(tmp_path) -> None:
    path = tmp_path / "pcu_weights.json"
    path.write_text(json.dumps({
        "category_weights": {name: 0.75 for name in config.TARGET_CLASS_NAMES},
        "source_citation": "User citation",
        "retrieval_date": "2026-11-13",
    }), encoding="utf-8")
    weights, message, source = zones.load_pcu_weights(path)
    assert weights == {name: 0.75 for name in config.TARGET_CLASS_NAMES}
    assert message is None
    assert source == "User citation (retrieved 2026-11-13)"


def test_missing_congestion_config_leaves_occupancy_unclassified(tmp_path) -> None:
    threshold, justification, message = zones.load_occupancy_threshold(
        tmp_path / "congestion_thresholds.json"
    )
    assert (threshold, justification, message) == (None, None, None)


def test_congestion_threshold_requires_justification(tmp_path) -> None:
    path = tmp_path / "congestion_thresholds.json"
    path.write_text(json.dumps({
        "occupancy_threshold": 0.5,
        "justification": "  Source-backed rationale  ",
    }), encoding="utf-8")
    assert zones.load_occupancy_threshold(path) == (
        0.5, "Source-backed rationale", None,
    )

    path.write_text(json.dumps({"occupancy_threshold": 0.5}), encoding="utf-8")
    with pytest.raises(zones.ZoneConfigurationError, match="justification"):
        zones.load_occupancy_threshold(path)
