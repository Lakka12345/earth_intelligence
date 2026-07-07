"""
app.py — Scientific Dataset Discovery Dashboard

Wires the existing agent pipeline to a Streamlit frontend.
Agents are never modified — all logic lives in:
  agents/agent1.py         → run_agent1(query)
  agents/agent2.py         → run_agent2(agent1_output, user_responses, clarification_history, round_number)
  agents/agent3_discovery  → run_agent3(retrieval_request)

Agent 4 is not yet implemented; a placeholder section is shown.
"""

import time
import sys
import os
import streamlit as st

# ── Make sure the project root (parent of this file) is on sys.path ──────── #
# app.py lives inside  <project_root>/dashboard/
# The agents live in   <project_root>/agents/
# Adjust one level up so "from agents.agent1 import run_agent1" resolves.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Dashboard utilities ───────────────────────────────────────────────────── #
from utils.session_manager import init_session_state, append_log
from utils.styles import inject_styles

# ── Dashboard components ──────────────────────────────────────────────────── #
from components.sidebar      import render_sidebar
from components.pipeline     import render_pipeline
from components.logs         import render_logs
from components.metrics      import render_metrics
from components.clarification import render_clarification_panel
from components.dataset_cards import (
    render_agent1_panel,
    render_agent3_panel,
    render_dataset_detail,
)
from components.map import render_map

# ── Project agents ────────────────────────────────────────────────────────── #
from agents.agent1           import run_agent1
from agents.agent2           import run_agent2
from agents.agent3_discovery import run_agent3
from models.retrieval_request import build_retrieval_request
from security.input_validator import validate_input
from security.injection_guard import detect_prompt_injection


# ─────────────────────────────────────────────────────────────────────────── #
# Page config (must be the first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────── #
st.set_page_config(
    page_title="Scientific Dataset Discovery Assistant",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()
init_session_state()


# ─────────────────────────────────────────────────────────────────────────── #
# Sidebar
# ─────────────────────────────────────────────────────────────────────────── #
sidebar_query, run_clicked = render_sidebar()


# ─────────────────────────────────────────────────────────────────────────── #
# Page header
# ─────────────────────────────────────────────────────────────────────────── #
st.markdown(
    """
    <div style='padding: 4px 0 20px 0;'>
        <h1 style='margin:0;font-size:28px;font-weight:800;color:#0f172a;'>
            🌍 Scientific Dataset Discovery Assistant
        </h1>
        <p style='margin:4px 0 0 0;color:#64748b;font-size:14px;'>
            Multi-agent AI pipeline for intelligent scientific dataset discovery and ranking.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────── #
# Research Goal input (main area)
# ─────────────────────────────────────────────────────────────────────────── #
main_query = st.text_area(
    "Research Goal",
    placeholder=(
        "Describe your scientific data need in detail.\n"
        "Example: Find chlorophyll concentration datasets around Lakshadweep between 2018 and 2024."
    ),
    height=90,
    label_visibility="collapsed",
    key="main_query_input",
    value=sidebar_query,
)

analyze_col, _ = st.columns([2, 8])
with analyze_col:
    analyze_clicked = st.button(
        "🚀  Analyze",
        type="primary",
        use_container_width=True,
        disabled=(st.session_state.pipeline_stage not in ("idle", "done", "error")),
    )

# Merge query sources (sidebar Run or main Analyze button)
query_to_use = main_query or sidebar_query
trigger = run_clicked or analyze_clicked

st.divider()

# ─────────────────────────────────────────────────────────────────────────── #
# Pipeline status bar
# ─────────────────────────────────────────────────────────────────────────── #
st.markdown("<div class='section-header'>Pipeline Status</div>", unsafe_allow_html=True)
render_pipeline()
st.markdown("")


# ─────────────────────────────────────────────────────────────────────────── #
# Metrics row
# ─────────────────────────────────────────────────────────────────────────── #
render_metrics()
st.divider()


# ─────────────────────────────────────────────────────────────────────────── #
# Main content + log panel  (two-column layout)
# ─────────────────────────────────────────────────────────────────────────── #
content_col, log_col = st.columns([3, 1])


# ─────────────────────────────────────────────────────────────────────────── #
# PIPELINE EXECUTION
# ─────────────────────────────────────────────────────────────────────────── #

# ── Launch: user pressed Analyze / Run ────────────────────────────────────── #
if trigger and query_to_use.strip():

    if st.session_state.pipeline_stage not in ("idle", "done", "error"):
        st.warning("Pipeline is already running.")
        st.stop()

    st.session_state.pipeline_stage = "running_a1"
    st.session_state.current_query  = query_to_use.strip()
    st.session_state.logs           = []
    st.session_state.timings        = {}
    st.session_state.agent1_result  = None
    st.session_state.agent2_result  = None
    st.session_state.agent3_result  = None
    st.session_state.clarification_round   = 0
    st.session_state.clarification_history = []
    append_log("Pipeline started.")

    with content_col:
        # ── Agent 1 ──────────────────────────────────────────────────────── #
        with st.spinner("🧠 Agent 1 — Analysing scientific intent…"):
            append_log("Agent 1 started.")
            t0 = time.time()
            try:
                validated_query = validate_input(query_to_use.strip())
                detect_prompt_injection(validated_query)
                agent1_result = run_agent1(validated_query)
                elapsed = round(time.time() - t0, 2)
                st.session_state.agent1_result = agent1_result
                st.session_state.timings["agent1"] = elapsed
                append_log(f"Agent 1 complete ({elapsed}s).")
            except Exception as exc:
                st.session_state.pipeline_stage = "error"
                append_log(f"Agent 1 FAILED: {exc}")
                st.error(f"Agent 1 failed: {exc}")
                st.stop()

        # ── Agent 2 ──────────────────────────────────────────────────────── #
        st.session_state.pipeline_stage = "running_a2"
        with st.spinner("💬 Agent 2 — Checking for clarification needs…"):
            append_log("Agent 2 started.")
            t0 = time.time()
            try:
                agent2_result = run_agent2(
                    agent1_output=st.session_state.agent1_result,
                    round_number=1,
                )
                elapsed = round(time.time() - t0, 2)
                st.session_state.agent2_result       = agent2_result
                st.session_state.clarification_round = 1
                st.session_state.timings["agent2"]   = elapsed
                append_log(f"Agent 2 complete ({elapsed}s). Readiness: {agent2_result.retrieval_readiness}")
            except Exception as exc:
                st.session_state.pipeline_stage = "error"
                append_log(f"Agent 2 FAILED: {exc}")
                st.error(f"Agent 2 failed: {exc}")
                st.stop()

        needs_clarification = (
            st.session_state.enable_clarification
            and st.session_state.agent2_result.retrieval_readiness == "clarification_required"
        )

        if needs_clarification:
            st.session_state.pipeline_stage = "clarifying"
            append_log("Clarification required — waiting for user input.")
        else:
            # Skip straight to Agent 3
            st.session_state.pipeline_stage = "running_a3"

    st.rerun()

elif trigger and not query_to_use.strip():
    st.warning("Please enter a research query before running the analysis.")


# ─────────────────────────────────────────────────────────────────────────── #
# CLARIFICATION ROUND(S)
# ─────────────────────────────────────────────────────────────────────────── #
elif st.session_state.pipeline_stage == "clarifying":

    with content_col:
        if st.session_state.agent1_result:
            render_agent1_panel(st.session_state.agent1_result)
            st.divider()

        submitted = render_clarification_panel(st.session_state.agent2_result)

        if submitted:
            responses = st.session_state.pending_user_responses
            round_num = st.session_state.clarification_round

            # Record this round into history
            questions_asked = [
                {"question": q.question, "priority": str(q.priority)}
                for q in st.session_state.agent2_result.prioritized_questions
            ]
            responses_received = [
                {"field_name": r["field_name"], "user_answer": r["user_answer"]}
                for r in responses
            ]
            st.session_state.clarification_history.append({
                "round_number":       round_num,
                "questions_asked":    questions_asked,
                "responses_received": responses_received,
            })

            # Re-run Agent 2 with answers
            st.session_state.pipeline_stage = "running_a2"
            append_log(f"User submitted answers for round {round_num}.")

            with st.spinner(f"💬 Agent 2 — Processing answers (round {round_num + 1})…"):
                t0 = time.time()
                try:
                    from models.schemas import UserResponse, ClarificationRound
                    user_resp_objs = [UserResponse(**r) for r in responses]
                    agent2_result = run_agent2(
                        agent1_output=st.session_state.agent1_result,
                        user_responses=user_resp_objs,
                        clarification_history=st.session_state.clarification_history,
                        round_number=round_num + 1,
                    )
                    elapsed = round(time.time() - t0, 2)
                    st.session_state.agent2_result        = agent2_result
                    st.session_state.clarification_round  = round_num + 1
                    st.session_state.timings["agent2"]   += elapsed
                    append_log(f"Agent 2 round {round_num+1} complete ({elapsed}s). Readiness: {agent2_result.retrieval_readiness}")
                except Exception as exc:
                    st.session_state.pipeline_stage = "error"
                    append_log(f"Agent 2 (round {round_num+1}) FAILED: {exc}")
                    st.error(f"Agent 2 clarification failed: {exc}")
                    st.stop()

            if agent2_result.retrieval_readiness == "clarification_required":
                st.session_state.pipeline_stage = "clarifying"
            else:
                st.session_state.pipeline_stage = "running_a3"

            st.session_state.pending_user_responses = []
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────── #
# AGENT 3
# ─────────────────────────────────────────────────────────────────────────── #
elif st.session_state.pipeline_stage == "running_a3":

    with content_col:
        st.progress(0.0, text="🔍 Building retrieval request…")
        try:
            retrieval_request = build_retrieval_request(
                approved_plan=(
                    st.session_state.agent2_result
                    .refined_scientific_plan
                    .agent1_plan_preserved
                ),
                user_approved_retrieval=True,
            )
        except Exception as exc:
            st.session_state.pipeline_stage = "error"
            append_log(f"Retrieval request build FAILED: {exc}")
            st.error(f"Failed to build retrieval request: {exc}")
            st.stop()

        append_log("Agent 3 started — discovering datasets…")

        progress_bar = st.progress(0.1, text="🔍 Agent 3 — Discovering datasets…")
        t0 = time.time()
        try:
            agent3_result = run_agent3(retrieval_request)
            elapsed = round(time.time() - t0, 2)
            st.session_state.agent3_result = agent3_result
            st.session_state.timings["agent3"] = elapsed
            st.session_state.pipeline_stage = "done"
            progress_bar.progress(1.0, text="✅ Discovery complete.")
            append_log(
                f"Agent 3 complete ({elapsed}s). "
                f"Found {agent3_result.total_candidates_found} candidates, "
                f"{len(agent3_result.ranked_sources)} accepted."
            )
        except Exception as exc:
            st.session_state.pipeline_stage = "error"
            append_log(f"Agent 3 FAILED: {exc}")
            st.error(f"Agent 3 failed: {exc}")
            st.stop()

    st.rerun()


# ─────────────────────────────────────────────────────────────────────────── #
# RESULTS DISPLAY  (pipeline_stage == "done")
# ─────────────────────────────────────────────────────────────────────────── #

stage = st.session_state.pipeline_stage

with content_col:

    # ── Agent 1 panel ──────────────────────────────────────────────────────── #
    if st.session_state.agent1_result and stage in ("running_a2", "clarifying", "running_a3", "done"):
        render_agent1_panel(st.session_state.agent1_result)

        # Reasoning panel (Agent 1 readiness summary)
        if st.session_state.show_reasoning:
            a1 = st.session_state.agent1_result
            reasoning = getattr(a1, "reasoning_summary", None)
            if reasoning:
                with st.expander("🔎 Reasoning Summary", expanded=False):
                    for attr in ["goal_summary", "objective_summary", "variable_summary",
                                 "measurement_summary", "dataset_requirement_summary", "readiness_summary"]:
                        val = getattr(reasoning, attr, None)
                        if val:
                            st.markdown(
                                f"<div class='reasoning-item'>{val}</div>",
                                unsafe_allow_html=True,
                            )

        st.divider()

    # ── Agent 3 results ────────────────────────────────────────────────────── #
    if st.session_state.agent3_result and stage == "done":
        render_agent3_panel(st.session_state.agent3_result)

        # Map
        st.markdown("### 🗺 Coverage Map")
        render_map(st.session_state.agent3_result)

        # Dataset detail panel
        render_dataset_detail(st.session_state.agent3_result)

        st.divider()

        # Agent 4 placeholder
        st.markdown("### ✅ Agent 4 — Dataset Verification & Download")
        st.info(
            "**Agent 4 is not yet implemented.** "
            "Once available, it will verify dataset integrity, handle authentication, "
            "and prepare download pipelines for all accepted sources above."
        )

        # Export results
        with st.expander("💾 Export Results (JSON)", expanded=False):
            import json
            result_json = st.session_state.agent3_result.model_dump_json(indent=2)
            st.download_button(
                label="⬇️ Download Discovery Results",
                data=result_json,
                file_name=f"discovery_results_{int(time.time())}.json",
                mime="application/json",
            )

    # ── Error state ────────────────────────────────────────────────────────── #
    elif stage == "error":
        st.error("The pipeline encountered an error. Check the logs and reset to try again.")


# ─────────────────────────────────────────────────────────────────────────── #
# Log panel (right column — always visible)
# ─────────────────────────────────────────────────────────────────────────── #
with log_col:
    render_logs()
