"""Evaluation tab placeholder. Benchmark evaluation arrives in Week 4."""
from __future__ import annotations

from . import planned_panel


def render() -> None:
    """Render the Week 4 placeholder panel for benchmark evaluation."""
    planned_panel(
        week=4,
        summary="Measure counting quality against the labelled local road scene benchmark.",
        items=[
            "Compare predicted counts to ground truth for every test set item.",
            "Report mean absolute error and error distribution by condition tag.",
            "Break down results for night, glare, dense, sparse, and distant scenes.",
            "Quantify how often local vehicle types are missed or mislabelled.",
        ],
    )
