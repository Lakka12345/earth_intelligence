"""
components/pipeline.py
Renders the horizontal agent pipeline with live status indicators.
UI-only improvements: spacing, typography, alignment, icons, readability.
Execution flow, agent names, and backend logic are unchanged.
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
    "clarifying":     ["completed", "running",   None,        None,        None,        None],
    "confirm_understanding": ["completed", "running", None,   None,        None,        None],
    "running_a3": ["completed", "completed", "running",   None,        None,        None],
    "agent3_ranking": ["completed", "completed", "running", None,       None,        None],
    "running_a4": ["completed", "completed", "completed", "running",   None,        None],
    "running_a5": ["completed", "completed", "completed", "completed", "running",   None],
    "done":       ["completed", "completed", "completed", "completed", "completed", "completed"],
    "error":      ["failed",    None,        None,        None,        None,        None],
}

# Visual config per status key
_STATUS_CONFIG = {
    "waiting":   {"label": "Waiting",      "dot": "#475569", "bg": "#1e293b", "border": "#334155", "text": "#94a3b8"},
    "running":   {"label": "⏳ Running…",  "dot": "#f59e0b", "bg": "#1c1a09", "border": "#92400e", "text": "#fcd34d"},
    "completed": {"label": "✓ Done",       "dot": "#22c55e", "bg": "#052e16", "border": "#166534", "text": "#4ade80"},
    "failed":    {"label": "✗ Failed",     "dot": "#ef4444", "bg": "#2d0000", "border": "#7f1d1d", "text": "#f87171"},
}


def render_pipeline() -> None:
    """
    Render the horizontal pipeline bar.
    Only visual presentation is modified here — execution logic lives in dashboard_app.py.
    """
    stage = st.session_state.get("pipeline_stage", "idle")
    statuses = _STAGE_MAP.get(stage, [None] * 5)

    cols = st.columns(len(_NODES) * 2 - 1)
    col_indices = [0, 2, 4, 6, 8, 10]

    for i, (icon, name, sub) in enumerate(_NODES):
        status_key = (statuses[i] if i < len(statuses) else None) or "waiting"
        cfg = _STATUS_CONFIG.get(status_key, _STATUS_CONFIG["waiting"])

        timing = st.session_state.timings.get(f"agent{i+1}", None) if i < len(_NODES) - 1 else None
        timing_html = (
            f"<div style='font-size:10px;color:#64748b;margin-top:4px;'>{timing:.1f}s</div>"
            if timing else ""
        )

        sub_html = (
            f"<div style='font-size:10px;color:#64748b;margin-top:1px;letter-spacing:0.08em;text-transform:uppercase;'>{sub}</div>"
            if sub else ""
        )

        # Animated pulse ring for running nodes
        pulse_style = (
            "box-shadow:0 0 0 3px rgba(245,158,11,0.35);"
            if status_key == "running" else ""
        )

        cols[col_indices[i]].markdown(
            f"""
            <div style='
                background:{cfg["bg"]};
                border:1.5px solid {cfg["border"]};
                border-radius:14px;
                padding:16px 10px 12px;
                text-align:center;
                {pulse_style}
                transition:border-color 0.3s;
            '>
                <div style='font-size:24px;line-height:1;'>{icon}</div>
                <div style='
                    font-size:13px;
                    font-weight:700;
                    color:#e2e8f0;
                    margin-top:8px;
                    letter-spacing:0.02em;
                '>{name}</div>
                {sub_html}
                <div style='
                    display:inline-block;
                    margin-top:8px;
                    font-size:11px;
                    font-weight:600;
                    color:{cfg["text"]};
                    background:rgba(0,0,0,0.25);
                    padding:2px 10px;
                    border-radius:20px;
                '>{cfg["label"]}</div>
                {timing_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Arrow connector between nodes
        if i < len(_NODES) - 1:
            arrow_color = "#22c55e" if status_key == "completed" else "#334155"
            cols[col_indices[i] + 1].markdown(
                f"<div style='text-align:center;padding-top:32px;font-size:18px;color:{arrow_color};'>→</div>",
                unsafe_allow_html=True,
            )
