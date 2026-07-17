"""
components/pipeline.py
Renders the horizontal agent pipeline with live status indicators.
"""

import streamlit as st


_NODES = [
    ("🧠", "Agent 1", "Intent"),
    ("💬", "Agent 2", "Clarify"),
    ("🔍", "Agent 3", "Discovery"),
    ("📦", "Agent 4", "Retrieve"),
    ("⚙️", "Agent 5", "Preprocess"),
    ("✅", "Results", ""),
]

_STAGE_MAP = {
    "idle":       [None,        None,        None,        None,        None,        None],
    "running_a1": ["running",   None,        None,        None,        None,        None],
    "running_a2": ["completed", "running",   None,        None,        None,        None],
    "clarifying": ["completed", "running",   None,        None,        None,        None],
    "running_a3": ["completed", "completed", "running",   None,        None,        None],
    "running_a4": ["completed", "completed", "completed", "running",   None,        None],
    "running_a5": ["completed", "completed", "completed", "completed", "running",   None],
    "done":       ["completed", "completed", "completed", "completed", "waiting",   "completed"],
    "error":      ["failed",    None,        None,        None,        None,        None],
}


def render_pipeline() -> None:
    """Render the horizontal pipeline bar."""
    stage = st.session_state.get("pipeline_stage", "idle")
    statuses = _STAGE_MAP.get(stage, [None] * 5)

    cols = st.columns(len(_NODES) * 2 - 1)
    col_indices = [0, 2, 4, 6, 8, 10]

    for i, (icon, name, sub) in enumerate(_NODES):
        status = statuses[i] if i < len(statuses) else None
        css_class = status or "waiting"

        label_map = {
            "waiting":   "Waiting",
            "running":   "⏳ Running…",
            "completed": "✓ Done",
            "failed":    "✗ Failed",
        }
        status_label = label_map.get(css_class, "Waiting")

        timing = st.session_state.timings.get(f"agent{i+1}", None) if i < len(_NODES) - 1 else None
        timing_str = f"<div style='font-size:10px;color:#94a3b8;margin-top:2px;'>{timing:.1f}s</div>" if timing else ""

        cols[col_indices[i]].markdown(
            f"""
            <div class='pipeline-node {css_class}'>
                <div style='font-size:20px;'>{icon}</div>
                <div style='margin-top:4px;'>{name}</div>
                {f"<div style='font-size:10px;opacity:0.7;'>{sub}</div>" if sub else ""}
                <div style='font-size:11px;margin-top:4px;'>{status_label}</div>
                {timing_str}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Arrow between nodes
        if i < len(_NODES) - 1:
            cols[col_indices[i] + 1].markdown(
                "<div style='text-align:center;padding-top:28px;font-size:20px;color:#cbd5e1;'>→</div>",
                unsafe_allow_html=True,
            )
