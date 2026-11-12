from __future__ import annotations

from scripts import tune_on_dev as tune


def _row(confidence: float, input_size: int, nms_iou: float | None,
         mae: float, interval: tuple[float, float], time_ms: float) -> dict:
    return {
        "confidence": confidence,
        "input_size": input_size,
        "nms_iou": nms_iou,
        "uses_nms": nms_iou is not None,
        "mean_absolute_error": mae,
        "mae_ci95": interval,
        "mean_inference_ms": time_ms,
    }


def test_grid_covers_requested_parameters_and_rtdetr_has_no_nms() -> None:
    yolo = tune.settings_grid("yolov8n")
    detr = tune.settings_grid("rtdetr-l")
    assert len(yolo) == 24
    assert {row["confidence"] for row in yolo} == {0.15, 0.25, 0.35, 0.50}
    assert {row["input_size"] for row in yolo} == {640, 1280}
    assert {row["nms_iou"] for row in yolo} == {0.5, 0.6, 0.7}
    assert len(detr) == 8
    assert all(not row["uses_nms"] and row["nms_iou"] is None for row in detr)


def test_selection_uses_best_interval_then_default_distance_then_speed() -> None:
    rows = [
        _row(0.15, 1280, 0.5, 1.0, (0.8, 1.4), 20.0),
        _row(0.15, 640, 0.7, 1.2, (1.0, 1.5), 18.0),
        _row(0.35, 640, 0.7, 1.2, (1.0, 1.5), 12.0),
    ]
    defaults = {
        "confidence": 0.25,
        "input_size": 640,
        "uses_nms": True,
        "nms_iou": 0.7,
    }
    chosen, tied, distance, best = tune.select_setting(rows, defaults)
    assert best is rows[0]
    assert len(tied) == 3
    assert chosen is rows[2]
    assert abs(distance - 2 / 7) < 1e-12
