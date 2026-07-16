"""
dashboard_app.py
================
Streamlit entry point for the Earth Intelligence Platform dashboard.

Integrates the original multi-agent dashboard (Agents 1-3) with the new
Voice / Chat interface that connects to the FastAPI backend
(/api/stt, /api/tts, /api/chat).

Run alongside the FastAPI backend:
    # Terminal 1 — FastAPI + voice endpoints
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

    # Terminal 2 — Streamlit dashboard
    streamlit run dashboard_app.py

Or launch both from a process manager (honcho, supervisord, etc.).
"""

import sys
import os

# Allow imports from the project root (agents/, models/, security/, etc.)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# ── Streamlit page config (must be first st call) ─────────────────────────
st.set_page_config(
    page_title="Earth Intelligence Platform",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Internal imports ───────────────────────────────────────────────────────
from dashboard.utils.session_manager import init_session_state, append_log
from dashboard.utils.styles import inject_styles

from dashboard.components.auth import render_login_page, render_profile_header, render_welcome_modal
from dashboard.components.sidebar import render_sidebar
from dashboard.components.pipeline import render_pipeline
from dashboard.components.logs import render_logs
from dashboard.components.metrics import render_metrics
from dashboard.components.clarification import render_clarification_panel
from dashboard.components.dataset_cards import (
    render_agent1_panel,
    render_agent3_panel,
    render_dataset_detail,
)
from dashboard.components.map import render_map
from dashboard.components.pages import (
    save_completed_analysis,
    render_previous_analyses,
    render_security_reports,
    render_download_center,
    render_settings_page,
    render_agent_outputs_page,
    render_pipeline_page,
    render_help_page,
)
from dashboard.components.voice import render_voice_page

# ── Agent imports (guarded so the dashboard loads even without agents) ─────
try:
    from agents.agent1 import run_agent1
    from agents.agent2 import run_agent2
    from agents.agent3_discovery import run_agent3
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════ #
#  Bootstrap
# ══════════════════════════════════════════════════════════════════════════ #

init_session_state()
inject_styles()

# ── Auth gate ──────────────────────────────────────────────────────────────
if not st.session_state.get("is_authenticated", False):
    render_login_page()
    st.stop()

render_profile_header()
render_welcome_modal()

# ── Sidebar (nav + query input) ────────────────────────────────────────────
query, run_clicked = render_sidebar()
active_page = st.session_state.get("active_page", "Dashboard")


# ══════════════════════════════════════════════════════════════════════════ #
#  Pipeline execution
# ══════════════════════════════════════════════════════════════════════════ #

def _run_pipeline(query: str) -> None:
    """Execute the multi-agent pipeline sequentially."""
    import time

    if not AGENTS_AVAILABLE:
        st.error(
            "Agent modules not found. Make sure `agents/`, `models/`, and `security/` "
            "are on the Python path. The dashboard UI is fully functional — "
            "connect your agents to enable pipeline execution."
        )
        return

    st.session_state.pipeline_stage = "running_a1"
    st.session_state.current_query  = query
    append_log(f"Pipeline started for: {query[:80]}")
    st.rerun()


if run_clicked and query.strip():
    if st.session_state.pipeline_stage in ("idle", "done", "error"):
        _run_pipeline(query.strip())

# ── Agent execution blocks (run on rerun when stage is set) ───────────────
if AGENTS_AVAILABLE:
    import time

    if st.session_state.pipeline_stage == "running_a1":
        with st.spinner("🧠 Agent 1 — Parsing scientific intent…"):
            t0 = time.time()
            try:
                result = run_agent1(st.session_state.current_query)
                st.session_state.agent1_result = result
                st.session_state.timings["agent1"] = round(time.time() - t0, 2)
                append_log("Agent 1 complete.")
                enable_clarification = st.session_state.get("enable_clarification", True)
                st.session_state.pipeline_stage = "running_a2" if enable_clarification else "running_a3"
            except Exception as exc:
                st.session_state.pipeline_stage = "error"
                append_log(f"Agent 1 error: {exc}")
        st.rerun()

    elif st.session_state.pipeline_stage == "running_a2":
        with st.spinner("💬 Agent 2 — Generating clarification questions…"):
            t0 = time.time()
            try:
                a1 = st.session_state.agent1_result
                a2 = run_agent2(agent1_output=a1)
                st.session_state.agent2_result = a2
                st.session_state.timings["agent2"] = round(time.time() - t0, 2)
                append_log("Agent 2 complete — awaiting user answers.")
                needs_q = bool(getattr(a2, "prioritized_questions", []))
                st.session_state.pipeline_stage = "clarifying" if needs_q else "running_a3"
            except Exception as exc:
                st.session_state.pipeline_stage = "error"
                append_log(f"Agent 2 error: {exc}")
        st.rerun()

    elif st.session_state.pipeline_stage == "running_a3":
        with st.spinner("🔍 Agent 3 — Discovering datasets…"):
            t0 = time.time()
            try:
                a1 = st.session_state.agent1_result
                a2 = st.session_state.agent2_result
                history = st.session_state.get("clarification_history", [])
                a3 = run_agent3(
                    st.session_state.current_query,
                    a1,
                    clarification_history=history,
                    max_datasets=st.session_state.get("max_datasets", 10),
                )
                st.session_state.agent3_result = a3
                st.session_state.timings["agent3"] = round(time.time() - t0, 2)
                append_log(f"Agent 3 complete — {a3.total_candidates_found} datasets found.")
                st.session_state.pipeline_stage = "done"
                save_completed_analysis()
            except Exception as exc:
                st.session_state.pipeline_stage = "error"
                append_log(f"Agent 3 error: {exc}")
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════ #
#  Page routing
# ══════════════════════════════════════════════════════════════════════════ #

def _render_dashboard() -> None:
    """Main dashboard view."""
    st.markdown("## 🌊 Earth Intelligence Platform")
    st.caption("Multi-agent scientific dataset discovery system")

    render_metrics()
    st.divider()
    render_pipeline()

    stage = st.session_state.pipeline_stage

    # Clarification panel (mid-pipeline)
    if stage == "clarifying" and st.session_state.get("agent2_result"):
        submitted = render_clarification_panel(st.session_state.agent2_result)
        if submitted:
            round_num = st.session_state.get("clarification_round", 1)
            history   = st.session_state.get("clarification_history", [])
            history.append({
                "round_number":       round_num,
                "questions_asked":    [
                    {"question": q.question}
                    for q in st.session_state.agent2_result.prioritized_questions
                ],
                "responses_received": st.session_state.pending_user_responses,
            })
            st.session_state.clarification_history = history
            st.session_state.clarification_round   = round_num + 1
            st.session_state.pipeline_stage        = "running_a3"
            append_log(f"Clarification round {round_num} submitted.")
            st.rerun()

    # Agent output panels
    if st.session_state.get("agent1_result") and st.session_state.get("show_reasoning", True):
        with st.expander("🧠 Agent 1 — Scientific Intent", expanded=(stage == "running_a2")):
            render_agent1_panel(st.session_state.agent1_result)

    if st.session_state.get("agent3_result"):
        st.divider()
        render_agent3_panel(st.session_state.agent3_result)
        render_dataset_detail(st.session_state.agent3_result)

        with st.expander("🗺 Spatial Coverage Map", expanded=False):
            render_map(st.session_state.agent3_result)

    if stage == "idle":
        st.info(
            "👋 Enter a scientific query in the sidebar and click **Run Analysis** to start, "
            "or switch to the **🎙 Voice Interface** page to speak your query."
        )

    if stage == "error":
        st.error("Pipeline encountered an error. Check the execution log below.")

    st.divider()
    render_logs()


# ── Page dispatch ──────────────────────────────────────────────────────────
if active_page == "Dashboard":
    _render_dashboard()

elif active_page == "New Analysis":
    from dashboard.components.pages import render_page_intro
    render_page_intro("New Analysis", "Run a fresh scientific dataset discovery pipeline.")
    _render_dashboard()

elif active_page == "Voice Interface":
    render_voice_page()

elif active_page == "Previous Analyses":
    render_previous_analyses()

elif active_page == "Pipeline":
    render_pipeline_page(render_pipeline, render_logs)

elif active_page == "Agent Outputs":
    render_agent_outputs_page(render_agent1_panel, render_agent3_panel, render_dataset_detail)

elif active_page == "Security Reports":
    render_security_reports()

elif active_page == "Downloads":
    render_download_center()

elif active_page == "Settings":
    render_settings_page()

elif active_page == "Help":
    render_help_page()
