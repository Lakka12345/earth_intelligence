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
# render_clarification_panel replaced by inline clarification form in pipeline block
from dashboard.components.dataset_cards import (
    render_agent1_panel,
    render_agent3_panel,
    render_dataset_detail,
)
from dashboard.components.map import render_map
from dashboard.components.agent4_results import render_agent4_panel
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
    from agents.agent4_orchestrator import run_agent4
    from models.plan_reconciliation import reconcile_agent1_plan
    from models.retrieval_request import build_retrieval_request
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

def _build_agent4_payload(a3_result, retrieval_request):
    """
    Build an Agent3ToAgent4Payload from a DiscoveryOutput + RetrievalRequest.
    DiscoveryOutput has no .agent4_payload attribute — the payload must be
    assembled here in the dashboard from the four ranked buckets.
    """
    from models.website_analysis_schemas import (
        Agent3ToAgent4Payload,
        Agent3ToAgent4Mode,
        SourceSnapshot,
        WebsiteAnalysisResult,
        AccessibilityProfile,
        AvailabilityProfile,
        AccuracyProfile,
        AccessClassification,
        CredentialEase,
        AvailabilityTimeline,
        DataAvailabilityStatus,
        TriState,
    )
    from models.discovery_schemas import AccessType

    # All three active buckets (accepted + auth-required + needs-eval).
    # Rejected sources are intentionally excluded — Agent 4 should not
    # attempt retrieval from sources Agent 3 already deemed unsuitable.
    all_scored = (
        list(a3_result.ranked_sources)
        + list(getattr(a3_result, "auth_required_sources", []))
        + list(getattr(a3_result, "needs_evaluation_sources", []))
    )

    source_snapshots: dict = {}
    website_analyses: dict = {}
    self_registerable: list = []
    real_creds_required: list = []
    paid: list = []
    unconfirmed: list = []

    requested_variables = (
        [v.variable for v in retrieval_request.variables]
        if retrieval_request else []
    )

    for scored in all_scored:
        c = scored.candidate
        sid = c.source_id

        source_snapshots[sid] = SourceSnapshot(
            source_id=sid,
            name=c.name,
            url=c.url,
            api_type=c.api_type if hasattr(c, "api_type") and c.api_type else "unknown",
            dataset_type=c.dataset_type,
            variables_available=list(c.variables_available),
            login_url=c.login_url,
            price_estimate=c.price_estimate,
        )

        # Derive AccessClassification from AccessType enum
        if c.access_type == AccessType.paid:
            access_cls = AccessClassification.paid_access
            paid.append(sid)
        elif c.access_type in (AccessType.registration, AccessType.api_key):
            access_cls = AccessClassification.simple_login_required
            if c.requires_login:
                self_registerable.append(sid)
        elif c.access_type == AccessType.unknown:
            access_cls = AccessClassification.free
            unconfirmed.append(sid)
        else:
            access_cls = AccessClassification.free

        website_analyses[sid] = WebsiteAnalysisResult(
            source_id=sid,
            website_name=c.name,
            accessibility=AccessibilityProfile(
                authentication_required=c.requires_login,
                anonymous_access_available=not c.requires_login,
                api_available=True,
                api_type=c.api_type if hasattr(c, "api_type") and c.api_type else "unknown",
                registration_required=c.requires_login,
                credential_ease=(
                    CredentialEase.no_credentials_needed if not c.requires_login
                    else CredentialEase.unknown
                ),
                payment_required=c.requires_payment,
                payment_notes=c.price_estimate or "",
                access_classification=access_cls,
                availability_timeline=AvailabilityTimeline.unknown,
                accessibility_composite_score=scored.final_score,
            ),
            availability=AvailabilityProfile(
                relevance_score=scored.final_score,
                completeness_score=scored.score_card.completeness.score,
                requested_variables=requested_variables,
                covered_variables=list(c.variables_available),
                missing_variables=[
                    v for v in requested_variables
                    if v.lower() not in [x.lower() for x in c.variables_available]
                ],
                variable_availability_score=scored.score_card.completeness.score,
                spatial_coverage_score=scored.score_card.geographic_match.score,
                temporal_coverage_score=scored.score_card.temporal_match.score,
                resolution_score=scored.score_card.resolution.score,
                availability_status=DataAvailabilityStatus.full if c.variables_available else DataAvailabilityStatus.unknown,
                availability_composite_score=scored.final_score,
            ),
            accuracy=AccuracyProfile(
                authority_score=scored.score_card.authority.score,
                credibility_score=(scored.score_card.authority.score + scored.score_card.historical_reliability.score) / 2,
                scientific_acceptance_score=scored.score_card.scientific_acceptance.score,
                consistency_score=scored.score_card.consistency.score,
                historical_reliability_score=scored.score_card.historical_reliability.score,
                metadata_quality_score=scored.score_card.metadata_quality.score,
                accuracy_composite_score=scored.final_score,
            ),
            data_policy_summary=getattr(c, "description", "")[:500],
        )

    ranked_ids = [s.candidate.source_id for s in a3_result.ranked_sources]

    return Agent3ToAgent4Payload(
        mode=Agent3ToAgent4Mode.ranked_selection,
        final_ranked_source_ids=ranked_ids,
        self_registerable_source_ids=self_registerable,
        real_credentials_required_source_ids=real_creds_required,
        paid_source_ids=paid,
        unconfirmed_credential_source_ids=unconfirmed,
        website_analyses=website_analyses,
        source_snapshots=source_snapshots,
        requested_variables=requested_variables,
        pre_collected_credentials=dict(getattr(a3_result, "retrieval_credentials", {})),
    )


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
                needs_q = bool(getattr(a2, "prioritized_questions", []))
                if needs_q:
                    append_log("Agent 2 complete — awaiting user answers.")
                    st.session_state.pipeline_stage = "clarifying"
                else:
                    append_log("Agent 2 complete — no questions needed, proceeding.")
                    st.session_state.pipeline_stage = "running_a3"
            except Exception as exc:
                st.session_state.pipeline_stage = "error"
                append_log(f"Agent 2 error: {exc}")
        st.rerun()

    elif st.session_state.pipeline_stage == "clarifying":
        # ── Render clarification questions prominently, blocking the pipeline ──
        a2 = st.session_state.get("agent2_result")
        if a2 and getattr(a2, "prioritized_questions", []):
            st.markdown("## 💬 Agent 2 — Clarification Questions")
            st.info(
                "Before searching for datasets, the system needs a few more details. "
                "Please answer the questions below and click **Submit Answers** to continue."
            )

            questions = a2.prioritized_questions
            responses = {}

            with st.form("clarification_form"):
                for i, q in enumerate(questions):
                    q_text = getattr(q, "question", str(q))
                    q_priority = getattr(q, "priority", "medium")
                    if q_priority in ("critical", "high"):
                        label = f"🔴 **Q{i+1} (Required):** {q_text}"
                    elif q_priority == "medium":
                        label = f"🟡 **Q{i+1}:** {q_text}"
                    else:
                        label = f"🟢 **Q{i+1} (Optional):** {q_text}"
                    responses[i] = st.text_area(label, key=f"clarif_q_{i}", height=80)

                submitted = st.form_submit_button("✅ Submit Answers", type="primary", use_container_width=True)

            if submitted:
                unanswered_critical = [
                    i + 1 for i, q in enumerate(questions)
                    if getattr(q, "priority", "") in ("critical", "high") and not responses[i].strip()
                ]
                if unanswered_critical:
                    st.error(f"Please answer the required question(s): {', '.join(f'Q{n}' for n in unanswered_critical)}")
                else:
                    user_responses = [
                        {
                            "question_id": getattr(q, "question_id", f"q_{i}"),
                            "question":    getattr(q, "question", str(q)),
                            "answer":      responses[i].strip(),
                        }
                        for i, q in enumerate(questions)
                    ]
                    st.session_state.pending_user_responses = user_responses
                    round_num = st.session_state.get("clarification_round", 1)
                    history   = st.session_state.get("clarification_history", [])
                    history.append({
                        "round_number":       round_num,
                        "questions_asked":    [{"question": getattr(q, "question", str(q))} for q in questions],
                        "responses_received": user_responses,
                    })
                    st.session_state.clarification_history = history
                    st.session_state.clarification_round   = round_num + 1
                    st.session_state.pipeline_stage        = "running_a3"
                    append_log(f"Clarification round {round_num} submitted — {len(user_responses)} answer(s) received.")
                    st.rerun()

            if st.session_state.get("agent1_result") and st.session_state.get("show_reasoning", True):
                with st.expander("🧠 Agent 1 — Scientific Intent (context)", expanded=False):
                    render_agent1_panel(st.session_state.agent1_result)

            st.divider()
            render_logs()
            st.stop()
        else:
            append_log("Agent 2 returned no questions — proceeding to Agent 3.")
            st.session_state.pipeline_stage = "running_a3"
            st.rerun()

    elif st.session_state.pipeline_stage == "running_a3":
        with st.spinner("🔍 Agent 3 — Discovering datasets…"):
            t0 = time.time()
            try:
                a1 = st.session_state.agent1_result
                a2 = st.session_state.agent2_result

                reconciled_plan = reconcile_agent1_plan(
                    a2.refined_scientific_plan.agent1_plan_preserved,
                    a2.resolved_information,
                )
                retrieval_request = build_retrieval_request(
                    approved_plan=reconciled_plan,
                    user_approved_retrieval=True,
                )
                st.session_state.last_retrieval_request = retrieval_request
                a3 = run_agent3(retrieval_request)
                st.session_state.agent3_result = a3
                st.session_state.timings["agent3"] = round(time.time() - t0, 2)
                append_log(f"Agent 3 complete — {a3.total_candidates_found} datasets found.")
                st.session_state.pipeline_stage = "running_a4"
            except Exception as exc:
                st.session_state.pipeline_stage = "error"
                append_log(f"Agent 3 error: {exc}")
        st.rerun()

    elif st.session_state.pipeline_stage == "running_a4":

        # ── Step A: Show confirmation UI and collect preferences BEFORE running ──
        if not st.session_state.get("a4_user_confirmed", False):
            st.subheader("📥 Agent 4 — Ready to Download")
            st.info(
                "Agent 3 has finished discovering datasets. "
                "Please confirm your download preferences before retrieval begins."
            )

            a3 = st.session_state.get("agent3_result")
            if a3:
                st.write(f"**{a3.total_candidates_found}** dataset candidate(s) found by Agent 3.")

            import os as _os
            default_path = _os.path.join(
                _os.path.dirname(_os.path.abspath(__file__)), "data"
            )
            save_path = st.text_input(
                "📁 Download folder (absolute path)",
                value=st.session_state.get("a4_save_path") or default_path,
                help="The folder where all downloaded dataset files will be saved.",
            )
            size_limit = st.number_input(
                "📦 Max file size per dataset (MB, 0 = no limit)",
                min_value=0,
                value=int(st.session_state.get("a4_size_limit_mb") or 0),
                help="Datasets larger than this will be auto-declined. Set 0 for no limit.",
            )
            wants_preprocessing = st.checkbox(
                "🔬 Send downloaded data to Agent 5 for pre-processing?",
                value=bool(st.session_state.get("a4_wants_preprocessing", False)),
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Confirm & Start Download", type="primary", use_container_width=True):
                    save_path = save_path.strip()
                    if not save_path:
                        st.error("Please enter a download folder path.")
                    else:
                        st.session_state.a4_save_path          = save_path
                        st.session_state.a4_size_limit_mb      = size_limit if size_limit > 0 else None
                        st.session_state.a4_wants_preprocessing = wants_preprocessing
                        st.session_state.a4_user_confirmed     = True
                        append_log(f"User confirmed download → {save_path}")
                        st.rerun()
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.pipeline_stage = "done"
                    append_log("User cancelled Agent 4 download.")
                    st.rerun()

            st.stop()   # hold here — do not proceed to Agent 4 until confirmed

        # ── Step B: User confirmed — reset flag then actually run Agent 4 ────────
        st.session_state.a4_user_confirmed = False   # reset for any future re-runs

        a4_choices = {
            "wants_preprocessing": st.session_state.get("a4_wants_preprocessing", False),
            "confirm_partial":     True,
            "size_limit_mb":       st.session_state.get("a4_size_limit_mb", None),
            "save_path":           st.session_state.get("a4_save_path"),
        }
        with st.spinner("📦 Agent 4 — Resolving access, estimating size, downloading datasets…"):
            t0 = time.time()
            try:
                a3 = st.session_state.agent3_result
                retrieval_request = st.session_state.get("last_retrieval_request")
                payload = _build_agent4_payload(a3, retrieval_request)
                a4 = run_agent4(payload, request=retrieval_request, ui_choices=a4_choices)
                st.session_state.agent4_result = a4
                st.session_state.timings["agent4"] = round(time.time() - t0, 2)
                append_log(
                    f"Agent 4 complete — {a4.coverage_percent:.1f}% variable coverage, "
                    f"{len([m for m in a4.manifest if m.success])} download(s) succeeded."
                )
                st.session_state.pipeline_stage = "done"
                save_completed_analysis()
            except Exception as exc:
                st.session_state.pipeline_stage = "error"
                append_log(f"Agent 4 error: {exc}")


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

    if st.session_state.get("agent4_result"):
        st.divider()
        render_agent4_panel(st.session_state.agent4_result)

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
