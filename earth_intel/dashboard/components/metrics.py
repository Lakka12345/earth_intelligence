"""
components/metrics.py
Renders the system metrics row (processing time, counts, success rate).
"""

import streamlit as st


def render_metrics() -> None:
    """Display pipeline metrics derived from session state."""
    timings   = st.session_state.get("timings", {})
    a3_result = st.session_state.get("agent3_result", None)

    total_time  = round(sum(timings.values()), 1) if timings else 0.0
    agents_done = len([v for v in timings.values() if v > 0])

    datasets_found      = 0
    datasets_filtered   = 0
    datasets_recommended = 0

    if a3_result is not None:
        datasets_found       = a3_result.total_candidates_found
        datasets_recommended = (
            len(a3_result.ranked_sources) +
            len(getattr(a3_result, "auth_required_sources", []))
        )
        datasets_filtered = (
            len(getattr(a3_result, "rejected_sources", [])) +
            len(getattr(a3_result, "needs_evaluation_sources", []))
        )

    success_rate = (
        f"{round(datasets_recommended / datasets_found * 100)}%"
        if datasets_found > 0
        else "—"
    )

    metrics = [
        ("⏱ Processing Time",      f"{total_time}s"),
        ("🤖 Agents Completed",    str(agents_done)),
        ("📦 Datasets Found",      str(datasets_found)),
        ("🚫 Filtered / Deferred", str(datasets_filtered)),
        ("✅ Recommended",         str(datasets_recommended)),
        ("📈 Acceptance Rate",     success_rate),
    ]

    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.markdown(
            f"""
            <div class='metric-card'>
                <div class='label'>{label}</div>
                <div class='value'>{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
