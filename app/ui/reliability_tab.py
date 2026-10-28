"""Reliability tab placeholder. Reliability scoring arrives in Week 3."""
from __future__ import annotations

from . import planned_panel


def render() -> None:
    """Render the Week 3 placeholder panel for the learned error predictor."""
    planned_panel(
        week=3,
        summary="Estimate how far a count can be trusted using a learned error predictor.",
        items=[
            "Predict the likely counting error for each input from detection features.",
            "Compare predicted error against a mean confidence baseline.",
            "Flag inputs that need manual review when predicted error is high.",
            "Show a reliability score next to every reported count.",
        ],
    )
