"""History tab placeholder. SQLite history arrives in Week 4."""
from __future__ import annotations

from . import planned_panel


def render() -> None:
    """Render the Week 4 placeholder panel for stored run history."""
    planned_panel(
        week=4,
        summary="Keep a searchable record of every counting run in a local SQLite store.",
        items=[
            "Persist each run with its inputs, counts, model, and reliability score.",
            "Browse past runs and reopen an earlier result.",
            "Filter history by date, model, class, or needs review flag.",
            "Export a selected history range as a single CSV summary.",
        ],
    )
