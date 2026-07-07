"""
LangGraph graph definition for the Earth Intelligence system.

CHANGED in agent3_node:
  After Agent 3 completes, the node now also classifies each selected
  source by access type and writes three new state fields:
    - free_sources_to_retrieve
    - login_required_sources
    - payment_required_sources

  These tell Agent 4 exactly how to handle each source without
  re-running any discovery logic.
"""

from langgraph.graph import END, START, StateGraph

from agents.agent1 import run_agent1
from agents.agent2 import run_agent2
from agents.agent3_discovery import run_agent3
from agents.approval_gate import display_plan_for_approval
from agents.revision_loop import revise_scientific_understanding
from models.retrieval_request import build_retrieval_request
from models.state import EarthGraphState


# ------------------------------------------------------------------ #
# Node — Agent 1                                                       #
# ------------------------------------------------------------------ #

def agent1_node(state: EarthGraphState) -> EarthGraphState:
    print("\nRunning Agent 1 — Scientific Planning...")
    query = state["user_query"]
    agent1_output = run_agent1(query)
    return {
        **state,
        "agent1_output": agent1_output,
        "approved_scientific_plan": agent1_output,
        "current_stage": "agent1_complete",
    }


# ------------------------------------------------------------------ #
# Node — Agent 2                                                       #
# ------------------------------------------------------------------ #

def agent2_node(state: EarthGraphState) -> EarthGraphState:
    print("\nRunning Agent 2 — Clarification...")
    agent2_output = run_agent2(state["agent1_output"])
    clarified_plan = agent2_output.refined_scientific_plan.agent1_plan_preserved
    return {
        **state,
        "agent2_output": agent2_output,
        "approved_scientific_plan": clarified_plan,
        "current_stage": "agent2_complete",
    }


# ------------------------------------------------------------------ #
# Node — Approval gate                                                 #
# ------------------------------------------------------------------ #

def approval_gate_node(state: EarthGraphState) -> EarthGraphState:
    plan = state["approved_scientific_plan"]
    display_plan_for_approval(plan)

    print("\n" + "=" * 70)
    print("HUMAN APPROVAL GATE")
    print("=" * 70)
    print("Is this understanding correct?")
    print("Type YES to proceed to data discovery.")
    print("Type NO to revise the understanding.")
    print("=" * 70)

    answer = input("Your answer: ").strip().lower()
    approved = answer in ["yes", "y"]

    if approved:
        print("User approved. Agent 3 may begin data discovery.")
    else:
        print("User rejected. Moving to revision loop.")

    return {
        **state,
        "user_approved_retrieval": approved,
        "agent3_input_ready": approved,
        "current_stage": "approved_for_agent3" if approved else "revision_needed",
    }


# ------------------------------------------------------------------ #
# Node — Revision loop                                                 #
# ------------------------------------------------------------------ #

def revision_loop_node(state: EarthGraphState) -> EarthGraphState:
    revision_cycle = state.get("revision_cycle", 0)
    max_cycles = state.get("max_revision_cycles", 5)

    if revision_cycle >= max_cycles:
        print("Maximum revision cycles reached. Please restart with a clearer query.")
        return {
            **state,
            "current_stage": "revision_limit_reached",
            "agent3_input_ready": False,
        }

    print("\nPlease provide corrections.")
    user_feedback = input("\nYour corrections: ").strip()

    previous_plan = state["approved_scientific_plan"]
    updated_plan = revise_scientific_understanding(
        current_plan=previous_plan,
        user_feedback=user_feedback,
    )

    revision_history = list(state.get("revision_history", []))
    revision_history.append({
        "cycle": revision_cycle + 1,
        "user_feedback": user_feedback,
        "updated_goal": updated_plan.inferred_user_research_goal,
    })

    return {
        **state,
        "approved_scientific_plan": updated_plan,
        "user_feedback": user_feedback,
        "revision_history": revision_history,
        "revision_cycle": revision_cycle + 1,
        "current_stage": "understanding_revised",
    }


# ------------------------------------------------------------------ #
# Node — Agent 3 discovery                                            #
# CHANGED: now also classifies selected sources by access type        #
# ------------------------------------------------------------------ #

def agent3_node(state: EarthGraphState) -> EarthGraphState:
    print("\nRunning Agent 3 — Data Discovery...")

    retrieval_request = build_retrieval_request(
        approved_plan=state["approved_scientific_plan"],
        user_approved_retrieval=True,
    )

    discovery_output = run_agent3(retrieval_request)

    # Build ranked_sources summary (for display / Agent 4 reference)
    ranked_sources = [
        {
            "source_id": s.candidate.source_id,
            "name": s.candidate.name,
            "url": s.candidate.url,
            "final_score": s.final_score,
            "recommendation": s.recommendation.value,
            "rank": s.rank,
            # CHANGED: include access fields so Agent 4 can read them from state
            "access_type": s.candidate.access_type.value,
            "requires_login": s.candidate.requires_login,
            "requires_payment": s.candidate.requires_payment,
            "price_estimate": s.candidate.price_estimate,
            "login_url": s.candidate.login_url,
            "api_type": s.candidate.api_type.value,
            "api_docs": s.candidate.api_docs,
            "discovery_origin": s.candidate.discovery_origin,
        }
        for s in discovery_output.ranked_sources
    ]

    # CHANGED: classify selected sources by access type for Agent 4
    selected_ids = set(discovery_output.sources_selected_for_download)
    selected_sources = [
        s for s in discovery_output.ranked_sources
        if s.candidate.source_id in selected_ids
    ]

    free_sources = [
        s.candidate.source_id for s in selected_sources
        if not s.candidate.requires_login and not s.candidate.requires_payment
    ]
    login_required = [
        s.candidate.source_id for s in selected_sources
        if s.candidate.requires_login and not s.candidate.requires_payment
    ]
    payment_required = [
        s.candidate.source_id for s in selected_sources
        if s.candidate.requires_payment
    ]

    if login_required:
        print(f"\n[Agent 3 → Agent 4] {len(login_required)} source(s) need login.")
    if payment_required:
        print(f"[Agent 3 → Agent 4] {len(payment_required)} source(s) need payment approval.")

    return {
        **state,
        "retrieval_request": retrieval_request,
        "agent3_output": discovery_output,
        "ranked_sources": ranked_sources,
        "sources_to_download": list(selected_ids),
        "download_requested": discovery_output.download_requested,
        # CHANGED: three new access-classification lists for Agent 4
        "free_sources_to_retrieve": free_sources,
        "login_required_sources": login_required,
        "payment_required_sources": payment_required,
        "current_stage": "discovery_complete",
    }


# ------------------------------------------------------------------ #
# Routing functions                                                    #
# ------------------------------------------------------------------ #

def route_after_approval(state: EarthGraphState) -> str:
    if state.get("user_approved_retrieval"):
        return "agent3"
    return "revision"


def route_after_revision(state: EarthGraphState) -> str:
    if state.get("current_stage") == "revision_limit_reached":
        return "end"
    return "approval"


# ------------------------------------------------------------------ #
# Build graph                                                          #
# ------------------------------------------------------------------ #

def build_earth_graph():
    graph_builder = StateGraph(EarthGraphState)

    graph_builder.add_node("agent1", agent1_node)
    graph_builder.add_node("agent2", agent2_node)
    graph_builder.add_node("approval", approval_gate_node)
    graph_builder.add_node("revision", revision_loop_node)
    graph_builder.add_node("agent3", agent3_node)

    graph_builder.add_edge(START, "agent1")
    graph_builder.add_edge("agent1", "agent2")
    graph_builder.add_edge("agent2", "approval")

    graph_builder.add_conditional_edges(
        "approval",
        route_after_approval,
        {"agent3": "agent3", "revision": "revision"},
    )

    graph_builder.add_conditional_edges(
        "revision",
        route_after_revision,
        {"approval": "approval", "end": END},
    )

    graph_builder.add_edge("agent3", END)

    return graph_builder.compile()


earth_graph = build_earth_graph()
print("LangGraph compiled successfully.")
