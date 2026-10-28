"""Video tab placeholder. Video counting arrives in Week 3."""
from __future__ import annotations

from . import planned_panel


def render() -> None:
    """Render the Week 3 placeholder panel for vehicle counting in video."""
    planned_panel(
        week=3,
        summary="Count distinct vehicles across a video clip using ByteTrack tracking.",
        items=[
            "Upload a road scene clip and run detection per frame.",
            "Track each vehicle across frames so one vehicle is counted once.",
            "Report total unique vehicles and per class totals for the clip.",
            "Play the tracked video back with per class counters.",
        ],
    )
