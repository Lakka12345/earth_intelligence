"""
components/dataset_cards.py
Renders Agent 3 dataset cards, the comparison table, and the detail panel.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────── #
# Helpers
# ─────────────────────────────────────────────────────────────────────────── #

def _score_bar(score: float) -> str:
    pct = int(score * 100)
    colour = "#22c55e" if pct >= 75 else "#f59e0b" if pct >= 50 else "#ef4444"
    return (
        f"<div class='score-bar-bg'>"
        f"<div class='score-bar-fill' style='width:{pct}%;background:{colour};'></div>"
        f"</div>"
    )


def _auth_badge(required: bool) -> str:
    if required:
        return "<span class='auth-yes'>🔑 Auth Required</span>"
    return "<span class='auth-no'>✓ Open Access</span>"


def _status_chip(status: str) -> str:
    colours = {
        "Accepted":                  ("#dcfce7", "#15803d"),
        "Authentication Required":   ("#fef3c7", "#92400e"),
        "Needs Further Evaluation":  ("#dbeafe", "#1d4ed8"),
        "Rejected":                  ("#fee2e2", "#b91c1c"),
    }
    bg, fg = colours.get(status, ("#f1f5f9", "#64748b"))
    return (
        f"<span style='background:{bg};color:{fg};padding:2px 8px;"
        f"border-radius:8px;font-size:11px;font-weight:600;'>{status}</span>"
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Agent 1 output panel
# ─────────────────────────────────────────────────────────────────────────── #

def render_agent1_panel(result) -> None:
    """Render Agent 1 output as structured cards — never raw JSON."""
    st.markdown("### 🧠 Agent 1 — Scientific Intent")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🎯 Research Goal**")
        st.info(result.inferred_user_research_goal)

        st.markdown("**📐 Spatial Context**")
        sc = result.spatial_context
        st.markdown(
            f"- **Location:** {sc.location}\n"
            f"- **Extent:** {sc.geographic_extent}\n"
            f"- **Resolution needed:** {sc.spatial_resolution_requirements}\n"
            f"- **Boundary type:** {sc.study_boundary_type}"
        )

        st.markdown("**📅 Temporal Context**")
        tc = result.temporal_context
        st.markdown(
            f"- **Date range:** {tc.date_range}\n"
            f"- **Temporal resolution:** {tc.temporal_resolution}\n"
            f"- **Analysis type:** {tc.temporal_analysis_type}\n"
            f"- **Historical baseline:** {tc.historical_baseline}"
        )

    with col2:
        st.markdown("**🔬 Scientific Variables**")
        for sv in result.scientific_variables:
            with st.expander(f"📊 {sv.variable}  [{sv.priority}]"):
                st.markdown(f"**Meaning:** {sv.scientific_meaning}")
                st.markdown(f"**Relevance:** {sv.relevance}")

        st.markdown("**🎯 Objectives**")
        for obj in result.scientific_objectives:
            st.markdown(
                f"<div style='background:#f8fafc;border-left:3px solid #3b82f6;"
                f"padding:8px 12px;border-radius:0 8px 8px 0;margin-bottom:6px;"
                f"font-size:13px;'>"
                f"<b>[{obj.priority}]</b> {obj.objective}</div>",
                unsafe_allow_html=True,
            )

    # Missing information
    missing = getattr(result, "missing_information", None)
    if missing:
        st.markdown("**⚠️ Missing Information**")
        for item in missing:
            st.warning(f"• {item}")


# ─────────────────────────────────────────────────────────────────────────── #
# Agent 3 dataset cards
# ─────────────────────────────────────────────────────────────────────────── #

def _render_single_card(source, bucket_label: str) -> None:
    """Render one ScoredSource as an expandable card."""
    c    = source.candidate
    sc   = source.score_card
    pct  = int(source.final_score * 100)
    name = c.name
    rank = getattr(source, "rank", "—")

    header = (
        f"**#{rank}  {name}**  "
        f"— Score: {pct}%  "
        f"| {_status_chip(source.status.value) if hasattr(source.status, 'value') else bucket_label}"
    )

    with st.expander(header, expanded=False):
        left, right = st.columns([3, 2])

        with left:
            st.markdown(
                f"{_auth_badge(c.authentication_required if hasattr(c, 'authentication_required') else c.requires_login)}"
                f"&nbsp;&nbsp;**Provider:** {c.name}",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='margin:8px 0;'>{_score_bar(source.final_score)}</div>",
                unsafe_allow_html=True,
            )

            col_a, col_b = st.columns(2)
            col_a.markdown(f"**📍 Coverage:** {c.spatial_coverage}")
            col_b.markdown(f"**📅 Temporal:** {c.temporal_coverage}")
            col_a.markdown(f"**📏 Spatial Res:** {c.spatial_resolution}")
            col_b.markdown(f"**⏱ Temporal Res:** {c.temporal_resolution}")

            if c.variables_available:
                st.markdown(
                    "**Variables:** " +
                    "  ".join(
                        f"`{v}`" for v in c.variables_available[:8]
                    )
                )

            if c.description:
                st.markdown(f"*{c.description[:250]}{'…' if len(c.description) > 250 else ''}*")

            if c.available_formats:
                st.markdown(
                    "**Formats:** " +
                    "  ".join(f"`{f.value if hasattr(f, 'value') else f}`" for f in c.available_formats)
                )

            st.markdown(f"**🔗 URL:** [{c.url}]({c.url})")
            doc_url = getattr(c, "documentation_url", None)
            if doc_url:
                st.markdown(f"**📖 Docs:** [{doc_url}]({doc_url})")

        with right:
            st.markdown("**Score Breakdown**")
            score_fields = [
                ("Authority",     sc.authority.score),
                ("Freshness",     sc.freshness.score),
                ("Relevance",     sc.relevance.score),
                ("Resolution",    sc.resolution.score),
                ("Completeness",  sc.completeness.score),
                ("Consistency",   sc.consistency.score),
                ("Metadata",      sc.metadata_quality.score),
                ("Reliability",   sc.historical_reliability.score),
                ("Sci. Accept.",  sc.scientific_acceptance.score),
                ("Realtime",      sc.real_time_availability.score),
                ("Geo Match",     sc.geographic_match.score),
                ("Temp Match",    sc.temporal_match.score),
                ("Platform",      sc.platform_match.score),
            ]
            for label, score in score_fields:
                p = int(score * 100)
                colour = "#22c55e" if p >= 75 else "#f59e0b" if p >= 50 else "#ef4444"
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:3px;'>"
                    f"<span style='font-size:11px;color:#64748b;width:90px;'>{label}</span>"
                    f"<div style='flex:1;background:#e2e8f0;border-radius:4px;height:5px;'>"
                    f"<div style='width:{p}%;background:{colour};height:5px;border-radius:4px;'></div>"
                    f"</div>"
                    f"<span style='font-size:11px;color:#334155;width:30px;text-align:right;'>{p}%</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("")
            st.markdown(f"**Confidence:** {int(source.confidence_score * 100)}%")

        if source.selection_justification:
            st.markdown("**📝 Justification**")
            st.markdown(
                f"<div style='background:#f8fafc;border-radius:8px;padding:10px 14px;"
                f"font-size:12px;color:#475569;'>{source.selection_justification}</div>",
                unsafe_allow_html=True,
            )

        if source.rejection_reason:
            st.markdown(
                f"<div style='background:#fff1f2;border-left:3px solid #f43f5e;"
                f"padding:8px 12px;border-radius:0 8px 8px 0;font-size:12px;"
                f"color:#be123c;margin-top:8px;'>"
                f"⚠️ {source.rejection_reason}</div>",
                unsafe_allow_html=True,
            )

        # Select for detail panel
        if st.button(f"🔎 View Full Details", key=f"detail_{c.source_id}"):
            st.session_state.selected_dataset_id = c.source_id
            st.rerun()


def render_agent3_panel(result) -> None:
    """Render the full Agent 3 output: bucket tabs + comparison table."""
    st.markdown("### 🔍 Agent 3 — Dataset Discovery")

    accepted    = result.ranked_sources
    auth_req    = getattr(result, "auth_required_sources", [])
    needs_eval  = getattr(result, "needs_evaluation_sources", [])
    rejected    = getattr(result, "rejected_sources", [])

    # Discovery notes
    if result.discovery_notes:
        with st.expander("ℹ️ Discovery Notes", expanded=False):
            for note in result.discovery_notes:
                st.markdown(f"• {note}")

    # Tabs per bucket
    tab_labels = [
        f"✅ Accepted ({len(accepted)})",
        f"🔑 Auth Required ({len(auth_req)})",
        f"🔄 Needs Evaluation ({len(needs_eval)})",
        f"❌ Rejected ({len(rejected)})",
        "📊 Comparison Table",
    ]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        if accepted:
            for src in accepted:
                _render_single_card(src, "Accepted")
        else:
            st.info("No accepted sources.")

    with tabs[1]:
        if auth_req:
            for src in auth_req:
                _render_single_card(src, "Auth Required")
        else:
            st.info("No authentication-required sources.")

    with tabs[2]:
        if needs_eval:
            for src in needs_eval:
                _render_single_card(src, "Needs Evaluation")
        else:
            st.info("No sources in this category.")

    with tabs[3]:
        if rejected:
            for src in rejected:
                _render_single_card(src, "Rejected")
        else:
            st.success("No sources were rejected.")

    with tabs[4]:
        render_comparison_table(result)


# ─────────────────────────────────────────────────────────────────────────── #
# Comparison table
# ─────────────────────────────────────────────────────────────────────────── #

def render_comparison_table(result) -> None:
    """Interactive sortable/searchable comparison table of all sources."""
    all_sources = (
        result.ranked_sources +
        getattr(result, "auth_required_sources", []) +
        getattr(result, "needs_evaluation_sources", []) +
        result.rejected_sources
    )

    if not all_sources:
        st.info("No sources to display.")
        return

    rows = []
    for s in all_sources:
        c = s.candidate
        auth = c.requires_login or c.requires_payment if hasattr(c, "requires_login") else c.authentication_required
        rows.append({
            "Dataset":        c.name,
            "Provider":       c.provider if hasattr(c, "provider") else c.name,
            "Status":         s.status.value if hasattr(s.status, "value") else str(s.status),
            "Score":          round(s.final_score * 100, 1),
            "Confidence":     round(s.confidence_score * 100, 1),
            "Variables":      ", ".join(c.variables_available[:4]),
            "Spatial Res":    c.spatial_resolution,
            "Coverage":       c.spatial_coverage[:40],
            "Auth Required":  "Yes" if auth else "No",
            "Formats":        ", ".join(f.value if hasattr(f, "value") else f for f in c.available_formats[:3]),
        })

    df = pd.DataFrame(rows)

    search = st.text_input("🔍 Filter datasets", placeholder="Search by name, provider, variable…", key="table_search")
    if search:
        mask = df.apply(lambda col: col.astype(str).str.contains(search, case=False, na=False)).any(axis=1)
        df = df[mask]

    sort_col = st.selectbox("Sort by", options=["Score", "Confidence", "Dataset", "Status"], index=0, key="table_sort")
    df = df.sort_values(sort_col, ascending=(sort_col == "Dataset"))

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score":      st.column_config.ProgressColumn("Score (%)", min_value=0, max_value=100),
            "Confidence": st.column_config.ProgressColumn("Confidence (%)", min_value=0, max_value=100),
        },
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Dataset detail panel
# ─────────────────────────────────────────────────────────────────────────── #

def render_dataset_detail(result) -> None:
    """Full detail panel for the currently selected dataset."""
    selected_id = st.session_state.get("selected_dataset_id")
    if not selected_id:
        return

    all_sources = (
        result.ranked_sources +
        getattr(result, "auth_required_sources", []) +
        getattr(result, "needs_evaluation_sources", []) +
        result.rejected_sources
    )

    source = next(
        (s for s in all_sources if s.candidate.source_id == selected_id),
        None,
    )
    if not source:
        return

    c = source.candidate

    st.markdown("---")
    st.markdown(f"### 📋 Dataset Details — {c.name}")

    if st.button("✖ Close", key="close_detail"):
        st.session_state.selected_dataset_id = None
        st.rerun()

    col1, col2, col3 = st.columns(3)
    col1.metric("Score",      f"{int(source.final_score * 100)}%")
    col2.metric("Confidence", f"{int(source.confidence_score * 100)}%")
    col3.metric("Rank",       f"#{source.rank}" if source.rank else "—")

    st.markdown("**Description**")
    st.markdown(c.description or "*No description available.*")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Scientific Variables**")
        for v in c.variables_available:
            st.markdown(f"  • `{v}`")
        st.markdown(f"**Spatial Extent:** {c.spatial_coverage}")
        st.markdown(f"**Temporal Extent:** {c.temporal_coverage}")
        st.markdown(f"**Spatial Resolution:** {c.spatial_resolution}")
        st.markdown(f"**Temporal Resolution:** {c.temporal_resolution}")

    with col_b:
        st.markdown(f"**Formats:** {', '.join(f.value if hasattr(f, 'value') else f for f in c.available_formats) or 'Unknown'}")
        auth = c.requires_login or c.requires_payment if hasattr(c, "requires_login") else c.authentication_required
        st.markdown(f"**Auth Required:** {'🔑 Yes' if auth else '✓ No'}")
        st.markdown(f"**API Type:** {c.api_type.value if hasattr(c.api_type, 'value') else c.api_type}")
        st.markdown(f"**Access Type:** {c.access_type.value if hasattr(c.access_type, 'value') else c.access_type}")
        if c.metadata_url:
            st.markdown(f"**Metadata URL:** [{c.metadata_url}]({c.metadata_url})")
        doc_url = getattr(c, "documentation_url", None)
        if doc_url:
            st.markdown(f"**Documentation:** [{doc_url}]({doc_url})")

    # Download panel
    st.markdown("**⬇️ Access Options**")
    d1, d2, d3, d4 = st.columns(4)
    d1.link_button("🌐 Open URL",   c.url)
    doc_url = getattr(c, "documentation_url", None)
    if doc_url:
        d2.link_button("📖 Docs", doc_url)
    if c.metadata_url:
        d3.link_button("🗂 Metadata", c.metadata_url)
    d4.button("📋 Copy URL", on_click=lambda: None, key="copy_url_btn")
    st.code(c.url, language=None)
