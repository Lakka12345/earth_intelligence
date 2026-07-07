"""
components/logs.py
Renders the live execution log panel.
"""

import streamlit as st


def render_logs() -> None:
    """Render the execution log panel with all current log entries."""
    logs = st.session_state.get("logs", [])

    st.markdown("<div class='section-header'>📋 Execution Log</div>", unsafe_allow_html=True)

    if not logs:
        st.markdown(
            "<div style='color:#94a3b8;font-size:13px;padding:8px 0;'>No activity yet. Run an analysis to begin.</div>",
            unsafe_allow_html=True,
        )
        return

    # Render newest-last so the log reads chronologically.
    log_html = ""
    for entry in logs:
        log_html += (
            f"<div class='log-entry'>"
            f"<span class='log-ts'>{entry['ts']}</span>"
            f"{entry['msg']}"
            f"</div>"
        )

    st.markdown(
        f"<div style='max-height:340px;overflow-y:auto;padding:4px 0;'>{log_html}</div>",
        unsafe_allow_html=True,
    )
