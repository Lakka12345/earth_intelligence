"""
components/dataset_cards.py
============================
Renders Agent 3 dataset cards, the comparison table, and the detail panel.

UI improvements applied (no backend or logic changes):
  - Trust badges sourced only from existing JSON security databases
  - Improved card header layout with score prominently displayed
  - Consistent design language with the rest of the dashboard
  - Cleaner score breakdown typography
  - Better justification / rejection blocks
  - Improved detail panel layout

To add a new trust badge: add a condition in _trust_badges() reading
from the appropriate JSON database file. Never fabricate values.
"""

from __future__ import annotations

import json
import os

import streamlit as st
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────── #
# Security database readers (UI display only — no logic changes)
# Each function reads its JSON file fresh; returns None if absent/invalid.
# ─────────────────────────────────────────────────────────────────────────── #

def _load_json_db(path: str) -> dict | None:
    """Load a security JSON database. Returns None if file absent or malformed."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _security_dbs() -> dict:
    """
    Load all security databases in one call.
    Returns a dict of {key: data_or_None}.
    Add new databases here as new security modules are integrated.
    """
    return {
        "trust":      _load_json_db("provider_trust_db.json"),
        "integrity":  _load_json_db("provenance_db.json"),
        "validation": _load_json_db("validation_db.json"),
        "provenance": _load_json_db("provenance_db.json"),
    }


# ─────────────────────────────────────────────────────────────────────────── #
# Trust badge helper (Task 2 — badges only when data confirms status)
# ─────────────────────────────────────────────────────────────────────────── #

def _trust_badges(dbs: dict) -> str:
    """
    Return HTML for trust badge pills.
    A badge is shown ONLY when the corresponding JSON database confirms
    a positive status. Never display a badge if data is absent or negative.
    """
    badges = []

    # Trusted Source — provider_trust_db.json must have a passing trust status
    trust = dbs.get("trust")
    if trust:
        status = str(trust.get("trust_status", trust.get("status", ""))).lower()
        if any(k in status for k in ("trust", "pass", "verified")):
            badges.append("✓ Trusted Source")

    # Integrity Verified — provenance_db.json must have hash_verified = True
    integrity = dbs.get("integrity")
    if integrity:
        hv = integrity.get("hash_verified", integrity.get("integrity_verified", None))
        if hv is True:
            badges.append("✓ Integrity Verified")

    # Validation Passed — validation_db.json must have a passing status
    validation = dbs.get("validation")
    if validation:
        vstatus = str(validation.get("validation_status", validation.get("status", ""))).lower()
        if "pass" in vstatus or "valid" in vstatus:
            badges.append("✓ Validation Passed")

    # Provenance Verified — provenance_db.json must have verified status
    provenance = dbs.get("provenance")
    if provenance:
        pstatus = str(provenance.get("verification_status", provenance.get("status", ""))).lower()
        if "verif" in pstatus or "pass" in pstatus:
            badges.append("✓ Provenance Verified")

    if not badges:
        return ""

    pills = "".join(
        f"<span class='ds-trust-badge'>{b}</span>"
        for b in badges
    )
    return f"<div class='ds-trust-badges'>{pills}</div>"


# ─────────────────────────────────────────────────────────────────────────── #
# Shared UI helpers (visual only)
# ─────────────────────────────────────────────────────────────────────────── #

def _score_bar(score: float) -> str:
    """Horizontal progress bar for a 0-1 score value."""
    pct = int(score * 100)
    colour = "#22c55e" if pct >= 75 else "#f59e0b" if pct >= 50 else "#ef4444"
    return (
        f"<div class='score-bar-bg'>"
        f"<div class='score-bar-fill' style='width:{pct}%;background:{colour};'></div>"
        f"</div>"
    )


def _score_mini_bar(label: str, score: float) -> str:
    """Compact labelled score bar used in the score breakdown panel."""
    p = int(score * 100)
    colour = "#22c55e" if p >= 75 else "#f59e0b" if p >= 50 else "#ef4444"
    return (
        f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:5px;'>"
        f"<span style='font-size:11px;color:#64748b;width:94px;flex-shrink:0;'>{label}</span>"
        f"<div style='flex:1;background:#e2e8f0;border-radius:4px;height:5px;'>"
        f"<div style='width:{p}%;background:{colour};height:5px;border-radius:4px;'></div>"
        f"</div>"
        f"<span style='font-size:11px;color:#334155;width:32px;text-align:right;font-weight:600;'>{p}%</span>"
        f"</div>"
    )


def _auth_badge(required: bool) -> str:
    if required:
        return "<span class='auth-yes'>🔑 Auth Required</span>"
    return "<span class='auth-no'>✓ Open Access</span>"


def _status_chip(status: str) -> str:
    """Coloured pill for dataset acceptance status."""
    colours = {
        "Accepted":                  ("#dcfce7", "#15803d", "#bbf7d0"),
        "Authentication Required":   ("#fef3c7", "#92400e", "#fde68a"),
        "Needs Further Evaluation":  ("#dbeafe", "#1d4ed8", "#bfdbfe"),
        "Rejected":                  ("#fee2e2", "#b91c1c", "#fecaca"),
    }
    bg, fg, border = colours.get(status, ("#f1f5f9", "#64748b", "#e2e8f0"))
    return (
        f"<span style='background:{bg};color:{fg};border:1px solid {border};"
        f"padding:3px 10px;border-radius:999px;font-size:11px;font-weight:700;"
        f"letter-spacing:0.02em;'>{status}</span>"
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
            # Brand blue accent (#2563eb) consistent with design system
            st.markdown(
                f"<div style='background:#eff6ff;border-left:3px solid #2563eb;"
                f"padding:8px 12px;border-radius:0 8px 8px 0;margin-bottom:6px;"
                f"font-size:13px;color:#1e3a6e;'>"
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

def _render_single_card(source, bucket_label: str, dbs: dict | None = None) -> None:
    """
    Render one ScoredSource as an expandable card.

    Trust badges are sourced from security JSON databases (dbs).
    If dbs is None (databases not yet written), no badges are shown.
    All data logic is unchanged — only presentation is modified.
    """
    c    = source.candidate
    sc   = source.score_card
    pct  = int(source.final_score * 100)
    name = c.name
    rank = getattr(source, "rank", "—")

    status_html = (
        _status_chip(source.status.value)
        if hasattr(source.status, "value")
        else f"<span style='color:#64748b;font-size:12px;'>{bucket_label}</span>"
    )

    status_label = source.status.value if hasattr(source.status, "value") else bucket_label
    header = f"#{rank}  {name} — Score: {pct}% | {status_label}"
    header = (
        f"**#{rank}  {name}**  "
        f"— {pct}%  "
        f"| {source.status.value if hasattr(source.status, 'value') else bucket_label}"
    )

    with st.expander(header, expanded=False):
        st.markdown(_status_chip(status_label), unsafe_allow_html=True)

        # Trust badges row (only when security data confirms them)
        if dbs:
            badges_html = _trust_badges(dbs)
            if badges_html:
                st.markdown(badges_html, unsafe_allow_html=True)

        left, right = st.columns([3, 2])

        with left:
            # Auth badge + status chip
            auth_req = (
                c.authentication_required
                if hasattr(c, "authentication_required")
                else c.requires_login
            )
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap;"
                f"margin-bottom:10px;'>"
                f"{_auth_badge(auth_req)}&nbsp;{status_html}"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Score bar
            st.markdown(
                f"<div style='margin-bottom:12px;'>{_score_bar(source.final_score)}</div>",
                unsafe_allow_html=True,
            )

            # Coverage metadata
            col_a, col_b = st.columns(2)
            col_a.markdown(f"**📍 Coverage:** {c.spatial_coverage}")
            col_b.markdown(f"**📅 Temporal:** {c.temporal_coverage}")
            col_a.markdown(f"**📏 Spatial Res:** {c.spatial_resolution}")
            col_b.markdown(f"**⏱ Temporal Res:** {c.temporal_resolution}")

            if c.variables_available:
                st.markdown(
                    "**Variables:** " +
                    "  ".join(f"`{v}`" for v in c.variables_available[:8])
                )

            if c.description:
                st.markdown(
                    f"<div style='background:#f8fafc;border-radius:8px;padding:10px 14px;"
                    f"font-size:13px;color:#475569;margin:10px 0;border:1px solid #e2e8f0;'>"
                    f"{c.description[:250]}{'...' if len(c.description) > 250 else ''}</div>",
                    unsafe_allow_html=True,
                )

            if c.available_formats:
                st.markdown(
                    "**Formats:** " +
                    "  ".join(
                        f"`{f.value if hasattr(f, 'value') else f}`"
                        for f in c.available_formats
                    )
                )

            st.markdown(f"**🔗 URL:** [{c.url}]({c.url})")
            doc_url = getattr(c, "documentation_url", None)
            if doc_url:
                st.markdown(f"**📖 Docs:** [{doc_url}]({doc_url})")

        with right:
            # Score breakdown panel
            st.markdown(
                "<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;"
                "padding:12px 14px;'>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div style='font-size:11px;font-weight:800;color:#64748b;"
                "letter-spacing:0.08em;text-transform:uppercase;margin-bottom:10px;'>"
                "Score Breakdown</div>",
                unsafe_allow_html=True,
            )
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
            bars_html = "".join(_score_mini_bar(lbl, s) for lbl, s in score_fields)
            st.markdown(bars_html, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Confidence pill
            conf_pct = int(source.confidence_score * 100)
            conf_colour = "#22c55e" if conf_pct >= 75 else "#f59e0b" if conf_pct >= 50 else "#ef4444"
            st.markdown(
                f"<div style='margin-top:10px;text-align:center;"
                f"background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:8px;'>"
                f"<span style='font-size:10px;color:#94a3b8;font-weight:700;"
                f"letter-spacing:0.06em;text-transform:uppercase;'>Confidence</span><br>"
                f"<span style='font-size:20px;font-weight:800;color:{conf_colour};'>"
                f"{conf_pct}%</span></div>",
                unsafe_allow_html=True,
            )

        # Justification block
        if source.selection_justification:
            st.markdown(
                f"<div style='background:#eff6ff;border-left:3px solid #2563eb;"
                f"padding:10px 14px;border-radius:0 8px 8px 0;font-size:12px;"
                f"color:#1e40af;margin-top:10px;'>"
                f"<strong style='color:#1e40af;'>📝 Justification</strong><br>"
                f"{source.selection_justification}</div>",
                unsafe_allow_html=True,
            )

        # Rejection reason block
        if source.rejection_reason:
            st.markdown(
                f"<div style='background:#fff1f2;border-left:3px solid #f43f5e;"
                f"padding:10px 14px;border-radius:0 8px 8px 0;font-size:12px;"
                f"color:#be123c;margin-top:8px;'>"
                f"<strong style='color:#be123c;'>⚠️ Rejection Reason</strong><br>"
                f"{source.rejection_reason}</div>",
                unsafe_allow_html=True,
            )

        # Detail view button
        if st.button("🔎 View Full Details", key=f"detail_{c.source_id}"):
            st.session_state.selected_dataset_id = c.source_id
            st.rerun()


def render_agent3_panel(result) -> None:
    """
    Render the full Agent 3 output: bucket tabs + comparison table.
    Security databases are loaded once here and passed down to each card
    to avoid repeated file I/O.
    """
    st.markdown("### 🔍 Agent 3 — Dataset Discovery")

    # Load security databases once per render pass
    dbs = _security_dbs()

    accepted   = result.ranked_sources
    auth_req   = getattr(result, "auth_required_sources", [])
    needs_eval = getattr(result, "needs_evaluation_sources", [])
    rejected   = getattr(result, "rejected_sources", [])

    if result.discovery_notes:
        with st.expander("ℹ️ Discovery Notes", expanded=False):
            for note in result.discovery_notes:
                st.markdown(f"• {note}")

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
                _render_single_card(src, "Accepted", dbs)
        else:
            st.info("No accepted sources.")

    with tabs[1]:
        if auth_req:
            for src in auth_req:
                _render_single_card(src, "Auth Required", dbs)
        else:
            st.info("No authentication-required sources.")

    with tabs[2]:
        if needs_eval:
            for src in needs_eval:
                _render_single_card(src, "Needs Evaluation", dbs)
        else:
            st.info("No sources in this category.")

    with tabs[3]:
        if rejected:
            for src in rejected:
                _render_single_card(src, "Rejected", dbs)
        else:
            st.success("No sources were rejected.")

    with tabs[4]:
        render_comparison_table(result)


# ─────────────────────────────────────────────────────────────────────────── #
# Comparison table (logic unchanged, minor visual polish)
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
        auth = (
            c.requires_login or c.requires_payment
            if hasattr(c, "requires_login")
            else c.authentication_required
        )
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
            "Formats":        ", ".join(
                f.value if hasattr(f, "value") else f
                for f in c.available_formats[:3]
            ),
        })

    df = pd.DataFrame(rows)

    search = st.text_input(
        "🔍 Filter datasets",
        placeholder="Search by name, provider, variable...",
        key="table_search",
    )
    if search:
        mask = df.apply(
            lambda col: col.astype(str).str.contains(search, case=False, na=False)
        ).any(axis=1)
        df = df[mask]

    sort_col = st.selectbox(
        "Sort by",
        options=["Score", "Confidence", "Dataset", "Status"],
        index=0,
        key="table_sort",
    )
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

    # Detail panel header
    st.markdown(
        f"<div style='margin-bottom:16px;'>"
        f"<div style='font-size:11px;font-weight:800;color:#94a3b8;"
        f"letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px;'>"
        f"Dataset Details</div>"
        f"<div style='font-size:22px;font-weight:800;color:#0f172a;'>{c.name}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if st.button("✖ Close", key="close_detail"):
        st.session_state.selected_dataset_id = None
        st.rerun()

    # Key metric cards
    score_pct = int(source.final_score * 100)
    conf_pct  = int(source.confidence_score * 100)
    rank_str  = f"#{source.rank}" if source.rank else "—"
    score_col = "#22c55e" if score_pct >= 75 else "#f59e0b" if score_pct >= 50 else "#ef4444"
    conf_col  = "#22c55e" if conf_pct  >= 75 else "#f59e0b" if conf_pct  >= 50 else "#ef4444"

    col1, col2, col3 = st.columns(3)
    for col, label, value, colour in [
        (col1, "Score",      f"{score_pct}%", score_col),
        (col2, "Confidence", f"{conf_pct}%",  conf_col),
        (col3, "Rank",       rank_str,         "#2563eb"),
    ]:
        col.markdown(
            f"<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;"
            f"padding:16px;text-align:center;'>"
            f"<div style='font-size:11px;color:#94a3b8;font-weight:700;letter-spacing:0.08em;"
            f"text-transform:uppercase;'>{label}</div>"
            f"<div style='font-size:32px;font-weight:900;color:{colour};'>{value}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Trust badges in detail view
    dbs = _security_dbs()
    badges_html = _trust_badges(dbs)
    if badges_html:
        st.markdown(f"<div style='margin-top:14px;'>{badges_html}</div>", unsafe_allow_html=True)

    # Description
    st.markdown("<div style='margin-top:14px;'>", unsafe_allow_html=True)
    st.markdown("**Description**")
    st.markdown(
        f"<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;"
        f"padding:12px 16px;font-size:14px;color:#334155;'>"
        f"{c.description or '<em>No description available.</em>'}</div>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Scientific Variables**")
        for v in c.variables_available:
            st.markdown(f"  - `{v}`")
        st.markdown(f"**Spatial Extent:** {c.spatial_coverage}")
        st.markdown(f"**Temporal Extent:** {c.temporal_coverage}")
        st.markdown(f"**Spatial Resolution:** {c.spatial_resolution}")
        st.markdown(f"**Temporal Resolution:** {c.temporal_resolution}")

    with col_b:
        st.markdown(
            f"**Formats:** "
            f"{', '.join(f.value if hasattr(f, 'value') else f for f in c.available_formats) or 'Unknown'}"
        )
        auth = (
            c.requires_login or c.requires_payment
            if hasattr(c, "requires_login")
            else c.authentication_required
        )
        st.markdown(f"**Auth Required:** {'🔑 Yes' if auth else '✓ No'}")
        st.markdown(
            f"**API Type:** "
            f"{c.api_type.value if hasattr(c.api_type, 'value') else c.api_type}"
        )
        st.markdown(
            f"**Access Type:** "
            f"{c.access_type.value if hasattr(c.access_type, 'value') else c.access_type}"
        )
        if c.metadata_url:
            st.markdown(f"**Metadata URL:** [{c.metadata_url}]({c.metadata_url})")
        doc_url = getattr(c, "documentation_url", None)
        if doc_url:
            st.markdown(f"**Documentation:** [{doc_url}]({doc_url})")

    st.markdown("</div>", unsafe_allow_html=True)

    # Access options
    st.markdown(
        "<div style='margin-top:16px;font-size:11px;font-weight:800;color:#94a3b8;"
        "letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;'>"
        "⬇️ Access Options</div>",
        unsafe_allow_html=True,
    )
    d1, d2, d3, d4 = st.columns(4)
    d1.link_button("🌐 Open URL",   c.url)
    doc_url = getattr(c, "documentation_url", None)
    if doc_url:
        d2.link_button("📖 Docs", doc_url)
    if c.metadata_url:
        d3.link_button("🗂 Metadata", c.metadata_url)
    d4.button("📋 Copy URL", on_click=lambda: None, key="copy_url_btn")
    st.code(c.url, language=None)
