"""Tab renderers for TrafficLens.

Week 1 delivers the Image tab in full. The other tabs render an honest "Planned for
Week N" panel through :func:`planned_panel` so the shell is complete without fake data.
"""
from __future__ import annotations

import streamlit as st


def planned_panel(week: int, summary: str, items: list[str]) -> None:
    """Render a clearly labelled placeholder panel for a feature not built yet.

    Args:
        week: The delivery week the feature is planned for.
        summary: One sentence describing what the tab will become.
        items: Concrete capabilities that will appear there.
    """
    st.warning(f"Planned for Week {week}. This tab is a placeholder and shows no results yet.")
    st.markdown(f"**What this tab will do:** {summary}")
    st.markdown("**Planned capabilities:**")
    for item in items:
        st.markdown(f"* {item}")
