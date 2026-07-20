"""
components/pipeline.py
Renders the horizontal agent pipeline with live status indicators.
Light-theme cards matching the dashboard's white/light-grey background.
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
    "idle":                  [None,        None,        None,        None,        None,        None],
    "running_a1":            ["running",   None,        None,        None,        None,        None],
    "running_a2":            ["completed", "running",   None,        None,        None,        None],
    "clarifying":            ["completed", "running",   None,        None,        None,        None],
    "confirm_understanding": ["completed", "running",   None,        None,        None,        None],
    "running_a3":            ["completed", "completed", "running",   None,        None,        None],
    "agent3_ranking":        ["completed", "completed", "running",   None,        None,        None],
    "running_a4":            ["completed", "completed", "completed", "running",   None,        None],
    "running_a5":            ["completed", "completed", "completed", "completed", "running",   None],
    "done":                  ["completed", "completed", "completed", "completed", "completed", "completed"],
    "error":                 ["failed",    None,        None,        None,        None,        None],
}

_STATUS_CONFIG = {
    "waiting": {
        "label": "Waiting",
        "bg": "#ffffff",
        "border": "#e2e8f0",
        "icon_opacity": "0.45",
        "name_color": "#94a3b8",
        "sub_color": "#94a3b8",
        "label_color": "#94a3b8",
        "shadow": "0 1px 4px rgba(15,23,42,0.06)",
    },
    "running": {
        "label": "⏳ Running…",
        "bg": "#fffbeb",
        "border": "#fcd34d",
        "icon_opacity": "1",
        "name_color": "#92400e",
        "sub_color": "#b45309",
        "label_color": "#b45309",
        "shadow": "0 4px 14px rgba(245,158,11,0.18)",
    },
    "completed": {
        "label": "✓ Done",
        "bg": "#f0fdf4",
        "border": "#22c55e",
        "icon_opacity": "1",
        "name_color": "#166534",
        "sub_color": "#15803d",
        "label_color": "#15803d",
        "shadow": "0 2px 8px rgba(34,197,94,0.12)",
    },
    "failed": {
        "label": "✗ Failed",
        "bg": "#fff1f2",
        "border": "#fca5a5",
        "icon_opacity": "1",
        "name_color": "#991b1b",
        "sub_color": "#b91c1c",
        "label_color": "#b91c1c",
        "shadow": "0 4px 14px rgba(239,68,68,0.15)",
    },
}


def _card_html(icon: str, name: str, sub: str, status_key: str, timing) -> str:
    cfg = _STATUS_CONFIG[status_key]

    sub_html = (
        f'<div style="font-size:13px;color:{cfg["sub_color"]};margin-top:2px;">{sub}</div>'
        if sub else ""
    )
    timing_html = (
        f'<div style="font-size:12px;color:#94a3b8;margin-top:6px;">{timing:.1f}s</div>'
        if timing else ""
    )

    # Built as one continuous string (no leading indentation on any line) so
    # Streamlit's markdown parser never mistakes this for an indented code block.
    return (
        f'<div style="background:{cfg["bg"]};border:1.5px solid {cfg["border"]};'
        f'border-radius:16px;padding:20px 16px 16px;text-align:center;'
        f'box-shadow:{cfg["shadow"]};">'
        f'<div style="font-size:28px;line-height:1;opacity:{cfg["icon_opacity"]};">{icon}</div>'
        f'<div style="font-size:16px;font-weight:700;color:{cfg["name_color"]};margin-top:10px;">{name}</div>'
        f'{sub_html}'
        f'<div style="font-size:14px;font-weight:700;color:{cfg["label_color"]};margin-top:10px;">{cfg["label"]}</div>'
        f'{timing_html}'
        f'</div>'
    )


def render_pipeline() -> None:
    stage    = st.session_state.get("pipeline_stage", "idle")
    statuses = _STAGE_MAP.get(stage, [None] * len(_NODES))

    cols        = st.columns(len(_NODES) * 2 - 1)
    col_indices = [0, 2, 4, 6, 8, 10]

    for i, (icon, name, sub) in enumerate(_NODES):
        status_key = (statuses[i] if i < len(statuses) else None) or "waiting"

        timing = (
            st.session_state.get("timings", {}).get(f"agent{i + 1}")
            if i < len(_NODES) - 1 else None
        )

        cols[col_indices[i]].markdown(
            _card_html(icon, name, sub, status_key, timing),
            unsafe_allow_html=True,
        )

        if i < len(_NODES) - 1:
            arrow_color = "#22c55e" if status_key == "completed" else "#cbd5e1"
            cols[col_indices[i] + 1].markdown(
                f'<div style="text-align:center;padding-top:44px;font-size:20px;color:{arrow_color};">→</div>',
                unsafe_allow_html=True,
            )
