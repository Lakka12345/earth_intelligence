"""
Agent 3 Interactive Runner — Discovery Agent extension entry point.
"""

from agents.agent3_output_formatter import format_ranked_output
from agents.agent3_override import ask_override_and_build_payload
from agents.agent3_ranking_preference import adaptive_rank, ask_ranking_preference
from discovery.website_analyzer import analyze_websites, get_requested_variables
from models.retrieval_request import RetrievalRequest
from models.website_analysis_schemas import Agent3ToAgent4Mode, Agent3ToAgent4Payload


def _enrich_requested_variables(
    raw_vars: list,
    request: RetrievalRequest,
) -> list:
    """
    ADDED: Guarantees that 'location' and 'time' are always present in the
    requested_variables list passed to Agent 4, even when get_requested_variables()
    (from website_analyzer.py) doesn't extract them.

    These two structural variables are mandatory for every retrieval task:
      - 'location' anchors the spatial query (bounding box / place name)
      - 'time'     anchors the temporal query (date range / timestamp)

    Without them, Agent 4's variable-coverage table always shows them as
    "Missing / no ranked Agent 3 source contributed this variable", which
    drags coverage below 50 % even for a perfectly executed retrieval.

    Also injects the concrete location string and date_range string (if
    present in the request) so Agent 4 can use them in its download queries
    rather than relying on generic tokens alone.
    """
    enriched = list(raw_vars)
    seen = {v.lower() for v in enriched}

    # Generic mandatory anchors
    for mandatory in ("location", "time"):
        if mandatory not in seen:
            enriched.append(mandatory)
            seen.add(mandatory)

    # Concrete location string (e.g. "bay of bengal")
    location_val = (
        request.spatial_requirements.get("location", "") or ""
    ).strip().lower()
    if location_val and location_val not in ("unknown", "unspecified", "") and location_val not in seen:
        enriched.append(location_val)
        seen.add(location_val)

    # Concrete date range string
    time_val = (
        request.temporal_requirements.get("date_range", "") or ""
    ).strip().lower()
    if time_val and time_val not in ("unknown", "unspecified", "") and time_val not in seen:
        enriched.append(time_val)
        seen.add(time_val)

    return enriched


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

    # CHANGED: enrich the variable list so 'location' and 'time' are always
    # present — get_requested_variables() only reads request.variables and
    # request.measurements, and misses the structural spatial/temporal fields.
    raw_requested_vars = get_requested_variables(request)
    requested_vars = _enrich_requested_variables(raw_requested_vars, request)

    pre_collected_credentials = dict(getattr(discovery_output, "retrieval_credentials", {}) or {})
    return ask_override_and_build_payload(entries, preference, requested_vars, pre_collected_credentials)
