"""
Agent 3 Interactive Runner — Discovery Agent extension entry point.
"""

from agents.agent3_output_formatter import format_ranked_output
from agents.agent3_override import ask_override_and_build_payload
from agents.agent3_ranking_preference import adaptive_rank, ask_ranking_preference
from discovery.website_analyzer import analyze_websites, get_requested_variables
from models.retrieval_request import RetrievalRequest
from models.website_analysis_schemas import Agent3ToAgent4Mode, Agent3ToAgent4Payload


def run_agent3_interactive(
    request: RetrievalRequest,
    discovery_output,  # DiscoveryOutput, as returned by the existing, unmodified run_agent3()
) -> Agent3ToAgent4Payload:
    # Scope: every source that passed the existing rejection process.
    surviving_sources = (
        list(discovery_output.ranked_sources)
        + list(getattr(discovery_output, "auth_required_sources", []) or [])
        + list(getattr(discovery_output, "needs_evaluation_sources", []) or [])
    )

    if not surviving_sources:
        print("\nNo sources survived discovery/rejection — nothing to analyze or rank.")
        return Agent3ToAgent4Payload(
            mode=Agent3ToAgent4Mode.ranked_selection,
            final_ranked_source_ids=[],
            notes=["No surviving sources to rank or override."],
        )

    analyses = analyze_websites(surviving_sources, request)
    preference = ask_ranking_preference()
    entries = adaptive_rank(surviving_sources, analyses, preference)

    preference_labels = [c.value.capitalize() for c in preference.selected_criteria]
    format_ranked_output(
        entries,
        preference_labels,
        requested_variables=get_requested_variables(request),
        analyses=analyses,
        selected_criteria=preference.selected_criteria,
    )

    requested_vars = get_requested_variables(request)
    pre_collected_credentials = dict(getattr(discovery_output, "retrieval_credentials", {}) or {})
    return ask_override_and_build_payload(entries, preference, requested_vars, pre_collected_credentials)
