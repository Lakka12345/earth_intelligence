"""
dashboard/components/agent4_results.py
=======================================
Renders the Agent 4 retrieval results panel in the Streamlit dashboard.
Reads from Agent4Output — no imports from Agent 4 internals beyond the schema.
"""

from __future__ import annotations

import streamlit as st
from models.agent4_schemas import AccessDecisionType, Agent4Output, format_bytes


# Decision type → human label + colour
_DECISION_LABELS = {
    AccessDecisionType.free_access:              ("✅ Public",              "normal"),
    AccessDecisionType.agent_self_registered:    ("🔑 Self-registered",     "normal"),
    AccessDecisionType.user_provided_credentials:("🔐 Credentials used",   "normal"),
    AccessDecisionType.payment_redirect:         ("💳 Payment required",    "off"),
    AccessDecisionType.skipped_declined:         ("⏭ Skipped",             "off"),
    AccessDecisionType.skipped_unresolved:       ("❓ Unresolved",          "off"),
}

_DOWNLOADABLE = {
    AccessDecisionType.free_access,
    AccessDecisionType.agent_self_registered,
    AccessDecisionType.user_provided_credentials,
}

_COVERAGE_COLOURS = {
    "Retrieved": "🟢",
    "Pending":   "🟡",
    "Skipped":   "⚪",
    "Failed":    "🔴",
    "Missing":   "🔴",
}


def render_agent4_panel(result: Agent4Output) -> None:
    """Main entry point — called from dashboard_app._render_dashboard()."""
    st.markdown("### 📦 Agent 4 — Data Retrieval")

    # ── Top-line KPIs ─────────────────────────────────────────────────────
    succeeded = [m for m in result.manifest if m.success]
    failed    = [m for m in result.manifest if not m.success]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Coverage", f"{result.coverage_percent:.1f}%")
    col2.metric("Downloads succeeded", len(succeeded))
    col3.metric("Downloads failed",    len(failed))
    col4.metric("Data retrieved",      format_bytes(result.actual_downloaded_bytes or 0))

    # ── Coverage table ─────────────────────────────────────────────────────
    if result.coverage_table:
        st.markdown("#### Variable Coverage")
        rows = []
        for row in result.coverage_table:
            icon = _COVERAGE_COLOURS.get(row.get("Coverage Status", ""), "⚪")
            rows.append({
                "Status":    f"{icon} {row.get('Coverage Status', '—')}",
                "Variable":  row.get("Variable", "—"),
                "Source":    row.get("Retrieved From", "—"),
                "Access":    row.get("Access Type", "—"),
                "Download":  row.get("Download Status", "—"),
                "Notes":     row.get("Reason if unavailable", ""),
            })
        st.dataframe(rows, use_container_width=True)

    # ── Missing variables warning ──────────────────────────────────────────
    if result.uncovered_variables:
        st.warning(
            f"**{len(result.uncovered_variables)} variable(s) not retrieved:** "
            + ", ".join(result.uncovered_variables)
        )
    else:
        st.success("All requested variables were retrieved.")

    # ── Source decisions detail ────────────────────────────────────────────
    if result.source_decisions:
        with st.expander("📋 Source Decisions", expanded=False):
            for decision in result.source_decisions:
                label, _ = _DECISION_LABELS.get(
                    decision.decision, (decision.decision.value, "normal")
                )
                vars_str = (
                    ", ".join(decision.variables_expected)
                    if decision.variables_expected else "—"
                )
                st.markdown(
                    f"**{decision.source_id}** &nbsp;·&nbsp; {label}  \n"
                    f"Variables: `{vars_str}`"
                    + (f"  \n*{decision.notes}*" if decision.notes else "")
                )
                st.divider()

    # ── Download manifest ──────────────────────────────────────────────────
    if result.manifest:
        with st.expander("📁 Download Manifest", expanded=False):
            for entry in result.manifest:
                icon = "✅" if entry.success else "❌"
                size = format_bytes(entry.size_bytes) if entry.size_bytes else "unknown size"
                st.markdown(
                    f"{icon} **{entry.source_id}** — {size}"
                    + (f"  \n`{entry.local_path}`" if getattr(entry, "local_path", None) else "")
                    + (f"  \n⚠ {entry.error}" if entry.error else "")
                )

    # ── Security reports ───────────────────────────────────────────────────
    sec = result.security_reports or {}
    if any(v is not None for v in sec.values()):
        with st.expander("🔒 Security Reports", expanded=False):
            _render_security_gate("Integrity",              sec.get("integrity"))
            _render_security_gate("Provenance",             sec.get("provenance"))
            _render_security_gate("Cross-Agent Verification", sec.get("cross_agent_verification"))
            _render_security_gate("Dataset Validation",     sec.get("dataset_validation"))

    # ── Download location / Agent 5 handoff ───────────────────────────────
    if result.download_location:
        st.info(f"📂 Files saved to: `{result.download_location}`")
    if result.send_to_agent5:
        st.info("➡ Data queued for Agent 5 preprocessing.")

    # ── Notes ──────────────────────────────────────────────────────────────
    if result.notes:
        with st.expander("📝 Notes", expanded=False):
            for note in result.notes:
                st.markdown(f"- {note}")


def _render_security_gate(label: str, gate: dict | None) -> None:
    if gate is None:
        return
    passed = gate.get("integrity_passed") or gate.get("provenance_verified") \
             or gate.get("verification_passed") or gate.get("validation_passed")
    score_key = next(
        (k for k in ("integrity_score", "provenance_score", "consistency_score", "validation_score")
         if k in gate), None
    )
    score_str = f" — {gate[score_key]:.1f}" if score_key else ""
    icon = "✅" if passed else "⚠️"
    st.markdown(f"**{icon} {label}**{score_str}")
    errors = gate.get("errors") or gate.get("inconsistencies") or []
    for e in errors[:3]:
        st.caption(f"• {e}")
