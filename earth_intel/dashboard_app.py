"""
dashboard_app.py
================
Streamlit entry point for the Design and Development of Multi Agent AI
Framework for Data Discovery and Retrieval dashboard.

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
    page_title="Design and Development of Multi Agent AI Framework for Data Discovery and Retrieval",
    page_icon="🤖",
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
    from agents.agent5 import run_agent5
    from formatting.understanding_summary import render_understanding_summary_banner
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
                    append_log("Agent 2 complete — no questions needed, proceeding to understanding confirmation.")
                    st.session_state.pipeline_stage = "confirm_understanding"
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
                            "field_name":  getattr(q, "question_id", f"q_{i}"),
                            "user_answer": responses[i].strip(),
                        }
                        for i, q in enumerate(questions)
                    ]
                    round_num = st.session_state.get("clarification_round", 1)
                    history   = st.session_state.get("clarification_history", [])
                    history.append({
                        "round_number":       round_num,
                        "questions_asked":    [{"question": getattr(q, "question", str(q))} for q in questions],
                        "responses_received": user_responses,
                    })
                    st.session_state.clarification_history = history
                    st.session_state.clarification_round   = round_num + 1
                    append_log(f"Clarification round {round_num} submitted — {len(user_responses)} answer(s) received.")

                    # ── Feed answers back into Agent 2 (previously skipped) ──
                    with st.spinner("💬 Agent 2 — Incorporating your answers…"):
                        try:
                            a1 = st.session_state.agent1_result
                            a2_updated = run_agent2(
                                agent1_output=a1,
                                user_responses=user_responses,
                                clarification_history=history,
                                round_number=round_num + 1,
                            )
                            st.session_state.agent2_result = a2_updated
                            if getattr(a2_updated, "retrieval_readiness", None) == "clarification_required" \
                                    and getattr(a2_updated, "prioritized_questions", []):
                                append_log("Agent 2 needs further clarification — another round required.")
                                st.session_state.pipeline_stage = "clarifying"
                            else:
                                append_log("Agent 2 updated — ready for understanding confirmation.")
                                st.session_state.pipeline_stage = "confirm_understanding"
                        except Exception as exc:
                            st.session_state.pipeline_stage = "error"
                            append_log(f"Agent 2 revision error: {exc}")
                    st.rerun()

            if st.session_state.get("agent1_result") and st.session_state.get("show_reasoning", True):
                with st.expander("🧠 Agent 1 — Scientific Intent (context)", expanded=False):
                    render_agent1_panel(st.session_state.agent1_result)

            st.divider()
            render_logs()
            st.stop()
        else:
            append_log("Agent 2 returned no questions — proceeding to understanding confirmation.")
            st.session_state.pipeline_stage = "confirm_understanding"
            st.rerun()

    elif st.session_state.pipeline_stage == "confirm_understanding":
        # ── Human-in-the-loop: confirm or revise the scientific understanding
        #    before Agent 3 discovery runs. Mirrors main.py's
        #    _run_confirmation_and_revision_loop(). ──
        a2 = st.session_state.get("agent2_result")
        revision_round = st.session_state.get("revision_round", 0)

        st.markdown("## 🧭 Scientific Understanding Summary")
        if revision_round:
            st.caption(f"Revision round {revision_round}")

        summary_text = render_understanding_summary_banner(
            a2, revision_round=revision_round if revision_round else None
        )
        st.text(summary_text)

        st.markdown("#### Is this understanding correct and sufficient to begin dataset discovery?")
        col1, col2 = st.columns(2)
        continue_clicked = col1.button("✅ Continue to Discovery", type="primary", use_container_width=True)
        modify_clicked = col2.button("✏️ Modify the Understanding", use_container_width=True)

        if continue_clicked:
            append_log("User confirmed understanding — proceeding to Agent 3.")
            st.session_state.pipeline_stage = "running_a3"
            st.rerun()

        if modify_clicked:
            st.session_state.show_modify_form = True

        MAX_REVISION_ROUNDS = 3
        if st.session_state.get("show_modify_form"):
            if revision_round >= MAX_REVISION_ROUNDS:
                st.warning(f"Maximum of {MAX_REVISION_ROUNDS} revision rounds reached. Proceeding with the current understanding.")
                st.session_state.pipeline_stage = "running_a3"
                st.rerun()
            else:
                with st.form("modify_understanding_form"):
                    change_request = st.text_area(
                        "What would you like to change? (Goal, Objectives, Variables, Measurements, "
                        "Location, Time Period, Dataset Requirements, Assumptions, or any other scientific detail)",
                        height=100,
                    )
                    submit_change = st.form_submit_button("Submit Change", type="primary", use_container_width=True)

                if submit_change and change_request.strip():
                    new_round = revision_round + 1
                    st.session_state.revision_round = new_round
                    st.session_state.show_modify_form = False

                    user_responses = [{
                        "field_name": f"revision_{new_round}_modification",
                        "user_answer": change_request.strip(),
                    }]
                    history = st.session_state.get("clarification_history", [])
                    history.append({
                        "round_number": len(history) + 1,
                        "questions_asked": [],
                        "responses_received": user_responses,
                    })
                    st.session_state.clarification_history = history

                    with st.spinner(f"💬 Agent 2 — Applying revision {new_round}…"):
                        try:
                            a1 = st.session_state.agent1_result
                            a2_updated = run_agent2(
                                agent1_output=a1,
                                user_responses=user_responses,
                                clarification_history=history,
                                round_number=len(history) + 1,
                            )
                            st.session_state.agent2_result = a2_updated
                            append_log(f"Understanding revised (round {new_round}).")
                            if getattr(a2_updated, "retrieval_readiness", None) == "clarification_required" \
                                    and getattr(a2_updated, "prioritized_questions", []):
                                append_log("Revision reopened a clarification requirement — returning to questions.")
                                st.session_state.pipeline_stage = "clarifying"
                            # else stay on confirm_understanding to show the updated summary
                        except Exception as exc:
                            st.session_state.pipeline_stage = "error"
                            append_log(f"Agent 2 revision error: {exc}")
                    st.rerun()

        st.divider()
        render_logs()
        st.stop()

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
                st.session_state.pipeline_stage = "agent3_ranking"
            except Exception as exc:
                st.session_state.pipeline_stage = "error"
                append_log(f"Agent 3 error: {exc}")
        st.rerun()

    elif st.session_state.pipeline_stage == "agent3_ranking":
        # ── Real interactive layer between Agent 3 and Agent 4:
        #    website analysis -> ranking preference -> adaptive ranking ->
        #    display -> override question -> build the real payload.
        #    Mirrors agents/agent3_interactive_runner.run_agent3_interactive(),
        #    but as Streamlit form steps instead of input().
        from discovery.website_analyzer import analyze_websites, get_requested_variables
        from agents.agent3_ranking_preference import adaptive_rank, AdaptiveRankingEntry
        from agents.agent3_override import _credential_buckets, _build_context
        from models.website_analysis_schemas import (
            RankingCriterion, RankingPreference,
            Agent3ToAgent4Mode, Agent3ToAgent4Payload,
        )

        a3 = st.session_state.agent3_result
        retrieval_request = st.session_state.get("last_retrieval_request")

        surviving_sources = (
            list(a3.ranked_sources)
            + list(getattr(a3, "auth_required_sources", []) or [])
            + list(getattr(a3, "needs_evaluation_sources", []) or [])
        )

        if not surviving_sources:
            st.warning("No sources survived discovery — nothing to rank.")
            st.session_state.agent3_payload = Agent3ToAgent4Payload(
                mode=Agent3ToAgent4Mode.ranked_selection,
                final_ranked_source_ids=[],
                notes=["No surviving sources to rank or override."],
            )
            st.session_state.pipeline_stage = "running_a4"
            st.rerun()

        # Cache the (potentially slow) website analysis so it only runs once
        if "a3_analyses" not in st.session_state:
            with st.spinner("🔎 Analyzing candidate websites (accessibility, accuracy, availability)…"):
                st.session_state.a3_analyses = analyze_websites(surviving_sources, retrieval_request)
            append_log(f"Website analysis complete for {len(surviving_sources)} source(s).")
        analyses = st.session_state.a3_analyses

        st.markdown("## 🔍 Agent 3 — Discovery Results")
        st.write(f"**{len(surviving_sources)}** source(s) survived discovery. On what basis should they be ranked?")

        criteria_options = {
            "Accuracy (Authority, Credibility, Scientific Acceptance, Consistency, Historical Reliability, Metadata Quality)": RankingCriterion.accuracy,
            "Accessibility (Authentication, Registration ease, API, Rate limits, Payment, Retrieval speed, Formats)": RankingCriterion.accessibility,
            "Availability (Relevance, Completeness, Variable/Spatial/Temporal coverage, Resolution, Historical coverage, Continuity)": RankingCriterion.availability,
        }

        if not st.session_state.get("a3_ranking_confirmed"):
            with st.form("ranking_preference_form"):
                selected_labels = st.multiselect(
                    "Ranking criteria (choose one or more; none selected = all three equally)",
                    options=list(criteria_options.keys()),
                )
                rank_submitted = st.form_submit_button("Rank Sources", type="primary", use_container_width=True)

            if rank_submitted:
                chosen = [criteria_options[l] for l in selected_labels] or list(criteria_options.values())
                preference = RankingPreference(selected_criteria=chosen)
                entries = adaptive_rank(surviving_sources, analyses, preference)
                st.session_state.a3_ranked_entries = entries
                st.session_state.a3_preference = preference
                st.session_state.a3_ranking_confirmed = True
                append_log(f"Sources ranked by: {', '.join(c.value for c in chosen)}.")
                st.rerun()

            st.divider()
            render_logs()
            st.stop()

        # ── Show the ranked table ──
        entries: list = st.session_state.a3_ranked_entries
        preference = st.session_state.a3_preference

        st.markdown("#### 📊 Ranked Sources")
        rows = []
        for e in entries:
            c = e.scored_source.candidate
            acc = e.analysis.accessibility
            tag = "💳 Paid" if acc.payment_required else ("🔐 Login required" if acc.authentication_required else "✅ Open")
            rows.append({
                "Rank": e.adaptive_rank,
                "Source": c.name,
                "Access": tag,
                "Adaptive Score": round(e.adaptive_score, 3),
            })
        st.dataframe(rows, use_container_width=True)

        if not st.session_state.get("a3_override_confirmed"):
            st.markdown("#### Would you like to obtain data from a specific website instead of the ranked recommendations?")
            with st.form("override_form"):
                use_override = st.checkbox("Yes, I want to specify a website instead")
                override_site = st.text_input("Preferred website (name or URL)", disabled=False)
                override_submitted = st.form_submit_button("Continue to Agent 4", type="primary", use_container_width=True)

            if override_submitted:
                if use_override and not override_site.strip():
                    st.error("Please enter a website name or URL, or uncheck the override option.")
                else:
                    requested_vars = get_requested_variables(retrieval_request)
                    real_cred, paid, unconfirmed = _credential_buckets(entries)
                    website_analyses, source_snapshots = _build_context(entries, requested_vars)
                    pre_collected = dict(getattr(a3, "retrieval_credentials", {}) or {})

                    if use_override:
                        payload = Agent3ToAgent4Payload(
                            mode=Agent3ToAgent4Mode.user_override,
                            final_ranked_source_ids=[e.scored_source.candidate.source_id for e in entries],
                            self_registerable_source_ids=[],
                            real_credentials_required_source_ids=real_cred,
                            paid_source_ids=paid,
                            unconfirmed_credential_source_ids=unconfirmed,
                            website_analyses=website_analyses,
                            source_snapshots=source_snapshots,
                            requested_variables=requested_vars,
                            pre_collected_credentials=pre_collected,
                            override_website=override_site.strip(),
                            ranking_preference=preference,
                            notes=[f"User overrode the ranked recommendations in favor of: {override_site.strip()}."],
                        )
                    else:
                        payload = Agent3ToAgent4Payload(
                            mode=Agent3ToAgent4Mode.ranked_selection,
                            final_ranked_source_ids=[e.scored_source.candidate.source_id for e in entries],
                            self_registerable_source_ids=[],
                            real_credentials_required_source_ids=real_cred,
                            paid_source_ids=paid,
                            unconfirmed_credential_source_ids=unconfirmed,
                            website_analyses=website_analyses,
                            source_snapshots=source_snapshots,
                            requested_variables=requested_vars,
                            pre_collected_credentials=pre_collected,
                            override_website=None,
                            ranking_preference=preference,
                            notes=[f"User accepted ranked recommendations ({', '.join(c.value for c in preference.selected_criteria)} basis)."],
                        )

                    st.session_state.agent3_payload = payload
                    st.session_state.a3_needs_credentials = list(dict.fromkeys(real_cred + unconfirmed))
                    st.session_state.a3_override_confirmed = True
                    append_log(f"Override choice submitted (override={use_override}).")
                    st.rerun()

            st.divider()
            render_logs()
            st.stop()

        # ── Credential collection for sources that need real user credentials ──
        needs_cred = st.session_state.get("a3_needs_credentials", [])
        if needs_cred and not st.session_state.get("a3_credentials_confirmed"):
            st.markdown("#### 🔐 Credentials Needed")
            st.info(
                f"{len(needs_cred)} source(s) require your own login credentials to access data. "
                "Enter them below, or leave blank to skip a source."
            )
            payload = st.session_state.agent3_payload
            with st.form("credentials_form"):
                collected = {}
                for sid in needs_cred:
                    name = payload.source_snapshots.get(sid).name if sid in payload.source_snapshots else sid
                    st.markdown(f"**{name}**")
                    c1, c2 = st.columns(2)
                    user = c1.text_input(f"Username / API key — {sid}", key=f"cred_user_{sid}")
                    pw = c2.text_input(f"Password / Token — {sid}", key=f"cred_pw_{sid}", type="password")
                    collected[sid] = {"username": user, "password": pw}
                creds_submitted = st.form_submit_button("Continue to Agent 4", type="primary", use_container_width=True)

            if creds_submitted:
                for sid, vals in collected.items():
                    if vals["username"] or vals["password"]:
                        payload.pre_collected_credentials[sid] = vals
                st.session_state.agent3_payload = payload
                st.session_state.a3_credentials_confirmed = True
                append_log("Credentials submitted for Agent 4.")
                st.session_state.pipeline_stage = "running_a4"
                st.rerun()

            st.divider()
            render_logs()
            st.stop()

        st.session_state.pipeline_stage = "running_a4"
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
                payload = st.session_state.get("agent3_payload") or _build_agent4_payload(a3, retrieval_request)
                a4 = run_agent4(payload, request=retrieval_request, ui_choices=a4_choices)
                st.session_state.agent4_result = a4
                st.session_state.timings["agent4"] = round(time.time() - t0, 2)
                append_log(
                    f"Agent 4 complete — {a4.coverage_percent:.1f}% variable coverage, "
                    f"{len([m for m in a4.manifest if m.success])} download(s) succeeded."
                )
                if getattr(a4, "send_to_agent5", False):
                    st.session_state.pipeline_stage = "running_a5"
                    st.rerun()
                else:
                    st.session_state.pipeline_stage = "done"
                    save_completed_analysis()
            except Exception as exc:
                st.session_state.pipeline_stage = "error"
                append_log(f"Agent 4 error: {exc}")

    elif st.session_state.pipeline_stage == "running_a5":
        with st.spinner("⚙️ Agent 5 — Pre-processing downloaded datasets…"):
            t0 = time.time()
            try:
                retrieval_request = st.session_state.get("last_retrieval_request")
                a4 = st.session_state.agent4_result
                a5 = run_agent5(retrieval_request, a4)
                st.session_state.agent5_result = a5
                st.session_state.timings["agent5"] = round(time.time() - t0, 2)
                append_log(f"Agent 5 complete — status: {a5.status.value}")
                st.session_state.pipeline_stage = "done"
                save_completed_analysis()
            except Exception as exc:
                st.session_state.pipeline_stage = "error"
                append_log(f"Agent 5 error: {exc}")


# ══════════════════════════════════════════════════════════════════════════ #
#  Page routing
# ══════════════════════════════════════════════════════════════════════════ #

def _render_dashboard() -> None:
    """Main dashboard view."""
    from dashboard.components.auth import PLATFORM_NAME
    st.markdown(f"## 🤖 {PLATFORM_NAME}")
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

    if st.session_state.get("agent5_result"):
        st.divider()
        st.subheader("⚙️ Agent 5 — Pre-processed Data")
        a5 = st.session_state.agent5_result
        st.write(f"Status: **{a5.status.value}**")
        clean_paths = getattr(a5, "clean_dataset_paths", None) or []
        if clean_paths:
            st.write(f"{len(clean_paths)} clean dataset(s) written:")
            for p in clean_paths:
                st.code(p)
        stop_reason = getattr(a5, "stop_reason", None)
        if stop_reason:
            st.warning(f"Agent 5 stopped early: {stop_reason}")

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
