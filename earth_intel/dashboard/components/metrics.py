"""
components/metrics.py
=====================
Renders the system metrics row (processing time, counts, success rate).

UI improvements applied:
  - Consistent dark-card design matching Security Reports palette
  - Colour-coded value accents (green for success, amber for filtered, neutral for counts)
  - Subtle top-border accent strip per card for visual hierarchy
  - Improved typography: larger value, smaller muted label
  - No backend logic changed — all values derived identically from session state

To add a new metric: append a tuple to the `metrics` list below.
"""

import streamlit as st


# ── Per-metric accent colours (top border strip + value text) ─────────────
# Keys match the metric label for easy lookup; fallback is neutral slate.
_ACCENT = {
    "⏱ Processing Time":      "#7dd3fc",   # sky-blue  — time feel
    "🤖 Agents Completed":    "#a78bfa",   # violet    — AI/agent feel
    "📦 Datasets Found":      "#94a3b8",   # slate     — neutral count
    "🚫 Filtered / Deferred": "#f87171",   # red-ish   — exclusion
    "✅ Recommended":         "#4ade80",   # green     — success
    "📈 Acceptance Rate":     "#34d399",   # emerald   — positive rate
}


def _metric_card(label: str, value: str) -> str:
    """
    Return HTML for a single metric card.
    Consistent with the dark-panel style used in Security Reports:
      background #0f172a, border #1e293b, border-radius 12px.
    """
    accent = _ACCENT.get(label, "#94a3b8")
    return f"""
    <div style='
        background: #0f172a;
        border: 1px solid #1e293b;
        border-top: 3px solid {accent};
        border-radius: 12px;
        padding: 16px 18px 14px;
        text-align: center;
        height: 100%;
    '>
        <div style='
            font-size: 11px;
            color: #64748b;
            font-weight: 600;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            margin-bottom: 8px;
        '>{label}</div>
        <div style='
            font-size: 26px;
            font-weight: 800;
            color: {accent};
            line-height: 1;
        '>{value}</div>
    </div>
    """


def render_metrics() -> None:
    """
    Display pipeline metrics derived from session state.
    All calculation logic is unchanged from the original implementation.
    """
    timings   = st.session_state.get("timings", {})
    a3_result = st.session_state.get("agent3_result", None)

    # ── Metric calculations (identical to original) ───────────────────────
    total_time  = round(sum(timings.values()), 1) if timings else 0.0
    agents_done = len([v for v in timings.values() if v > 0])

    datasets_found       = 0
    datasets_filtered    = 0
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

    # ── Metric definitions ────────────────────────────────────────────────
    # To add a new metric: append ("label", "value") here.
    metrics = [
        ("⏱ Processing Time",      f"{total_time}s"),
        ("🤖 Agents Completed",    str(agents_done)),
        ("📦 Datasets Found",      str(datasets_found)),
        ("🚫 Filtered / Deferred", str(datasets_filtered)),
        ("✅ Recommended",         str(datasets_recommended)),
        ("📈 Acceptance Rate",     success_rate),
    ]

    # ── Render ────────────────────────────────────────────────────────────
    cols = st.columns(len(metrics), gap="small")
    for col, (label, value) in zip(cols, metrics):
        col.markdown(_metric_card(label, value), unsafe_allow_html=True)
