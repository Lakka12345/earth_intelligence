"""
Output Formatter — Discovery Agent extension.
"""

from typing import Dict, List

from agents.agent3_ranking_preference import AdaptiveRankingEntry
from models.website_analysis_schemas import (
    CredentialEase,
    DataAvailabilityStatus,
    RankingCriterion,
    WebsiteAnalysisResult,
)
from discovery.website_analyzer import suggest_complementary_combination

_ACCESS_LABELS = {
    "free": "Free",
    "simple_login_required": "Login Required (self-service)",
    "credential_verification_required": "Credential Verification Required",
    "paid_access": "Paid Access",
}
_TIMELINE_LABELS = {
    "immediate": "Immediate",
    "manual_approval": "Manual Approval",
    "few_days_approval": "Approval Required (a few days)",
    "waiting_period": "Waiting Period Applies",
    "unknown": "Unknown",
}
_CREDENTIAL_LABELS = {
    CredentialEase.no_credentials_needed: "No credentials needed",
    CredentialEase.agent_can_self_register: "Agent 4 CAN self-register (any working email works)",
    CredentialEase.user_must_provide_real_credentials: "User MUST provide their own real credentials",
    CredentialEase.unknown: "Unclear from available metadata",
}
_TRISTATE_LABELS = {"yes": "Yes", "no": "No", "unknown": "Unknown"}


def format_ranked_output(
    entries: List[AdaptiveRankingEntry],
    preference_labels: List[str],
    requested_variables: List[str],
    analyses: Dict[str, WebsiteAnalysisResult],
    selected_criteria: List[RankingCriterion],
) -> None:
    print("\n" + "=" * 78)
    print("RANKED DATA SOURCES")
    print(f"(Ranking basis: {', '.join(preference_labels)})")
    print("=" * 78)

    # ---- Transparency note on ranking trade-offs ----------------------
    from models.website_analysis_schemas import RankingCriterion as RC
    _active = {c.value for c in selected_criteria}
    _note_lines = [
        "NOTE ON RANKING: Results are ordered by the criteria you selected.",
    ]
    if "accessibility" in _active and "accuracy" not in _active:
        _note_lines.append(
            "  ⚠  You ranked by Accessibility only. Sources that are easier to access "
            "may appear above sources with stronger scientific rigour or higher metadata "
            "quality. Consider adding Accuracy to the ranking if scientific provenance matters."
        )
    if "availability" in _active and "accuracy" not in _active:
        _note_lines.append(
            "  ⚠  You ranked by Availability only. Sources covering more of your requested "
            "variables may outrank those with better-validated or peer-reviewed data. "
            "Consider adding Accuracy if data quality is a priority."
        )
    if "accuracy" in _active and "accessibility" not in _active:
        _note_lines.append(
            "  ℹ  You ranked by Accuracy. Top results may require registration, "
            "credentials, or manual approval to access. Check the Accessibility section "
            "of each result for access requirements before attempting retrieval."
        )
    if len(_active) == 3:
        _note_lines.append(
            "  ℹ  All three criteria are weighted equally. Adaptive scores reflect a "
            "balanced trade-off between scientific quality, access ease, and variable coverage."
        )
    for line in _note_lines:
        print(line)
    print("-" * 78)

    if not entries:
        print("\nNo analyzed sources to display.")
        return

    for entry in entries:
        scored = entry.scored_source
        candidate = scored.candidate
        a = entry.analysis

        print(f"\n[{entry.adaptive_rank}] {candidate.name}")
        print(f"    Adaptive Score        : {entry.adaptive_score:.2f} / 1.00")
        print("      (dimension scores: " + ", ".join(f"{k}={v:.2f}" for k, v in entry.dimension_scores.items()) + ")")
        print(f"    Original Final Score  : {scored.final_score:.2f} / 1.00  (Phase 5, unchanged)")
        print(f"    URL                   : {candidate.url}")
        print(f"    Status (Phase 5)      : {scored.status.value}")

        # ---- ACCESSIBILITY ------------------------------------------------
        acc = a.accessibility
        print(f"\n    ACCESSIBILITY  (composite {acc.accessibility_composite_score:.2f}/1.00)")
        print(f"      Access type              : {_ACCESS_LABELS[acc.access_classification.value]}")
        print(f"      Credential ease          : {_CREDENTIAL_LABELS[acc.credential_ease]}")
        print(f"        -> {acc.credential_ease_notes}")
        print(f"      Payment required         : {'Yes — ' + acc.payment_notes if acc.payment_required else 'No'}")
        print(f"      Authentication required  : {'Yes' if acc.authentication_required else 'No'}")
        print(f"      Anonymous access         : {'Yes' if acc.anonymous_access_available else 'No'}")
        print(f"      API available            : {'Yes (' + acc.api_type + ')' if acc.api_available else 'No'}")
        print(f"      Download restrictions    : {_TRISTATE_LABELS[acc.download_restrictions.value]}")
        print(f"      Rate limits              : {_TRISTATE_LABELS[acc.rate_limits.value]}")
        print(f"      Real-time availability   : {acc.real_time_availability_score:.2f}/1.00")
        print(f"      Retrieval speed          : {acc.retrieval_speed_label}" + (f" (~{acc.retrieval_speed_ms:.0f}ms)" if acc.retrieval_speed_ms else ""))
        print(f"      Server uptime            : {_TRISTATE_LABELS[acc.server_uptime.value]}")
        print(f"      Supported formats        : {', '.join(acc.supported_download_formats) or 'Not specified'}")
        print(f"      Availability timeline    : {_TIMELINE_LABELS[acc.availability_timeline.value]} — {acc.timeline_notes}")

        # ---- AVAILABILITY ---------------------------------------------------
        av = a.availability
        status_label = {"full": "Full", "partial": "Partial", "not_available": "Not Available", "unknown": "Unknown (variable metadata not extracted for this source)"}[av.availability_status.value]
        print(f"\n    AVAILABILITY  (composite {av.availability_composite_score:.2f}/1.00, status: {status_label})")
        print(f"      Relevance                : {av.relevance_score:.2f}")
        print(f"      Completeness             : {av.completeness_score:.2f}")
        print(f"      Variable availability    : {av.variable_availability_score:.2f}  (covers {len(av.covered_variables)}/{len(av.requested_variables) or 'n/a'} requested variables)")
        if av.availability_status.value == "unknown":
            print("        Note: This source's variable list was not extracted by the pipeline -- coverage is UNCONFIRMED, not absent.")
        if av.covered_variables:
            print(f"        Covered : {', '.join(av.covered_variables)}")
        if av.missing_variables:
            print(f"        Missing : {', '.join(av.missing_variables)}")
        print(f"      Spatial coverage         : {av.spatial_coverage_score:.2f} — {av.spatial_coverage_notes}")
        print(f"      Temporal coverage        : {av.temporal_coverage_score:.2f} — {av.temporal_coverage_notes}")
        print(f"      Resolution               : {av.resolution_score:.2f} — {av.resolution_notes}")
        print(f"      Historical coverage      : {av.historical_coverage_score:.2f} (estimated, not verified)")
        print(f"      Data continuity          : {_TRISTATE_LABELS[av.data_continuity.value]} (not independently verifiable from metadata)")
        print(f"      Missing values           : {_TRISTATE_LABELS[av.missing_values.value]} (not independently verifiable from metadata)")

        # ---- ACCURACY ---------------------------------------------------
        ac = a.accuracy
        print(f"\n    ACCURACY  (composite {ac.accuracy_composite_score:.2f}/1.00)")
        print(f"      Authority                : {ac.authority_score:.2f}")
        print(f"      Credibility              : {ac.credibility_score:.2f} ({ac.credibility_notes})")
        print(f"      Scientific acceptance    : {ac.scientific_acceptance_score:.2f}")
        print(f"      Consistency              : {ac.consistency_score:.2f}")
        print(f"      Historical reliability   : {ac.historical_reliability_score:.2f}")
        print(f"      Metadata quality         : {ac.metadata_quality_score:.2f}")

    # ---- Complementary coverage note (only meaningful for availability) ----
    if RankingCriterion.availability in selected_criteria and requested_variables:
        combo = suggest_complementary_combination(analyses, requested_variables)
        if len(combo) > 1:
            names = [analyses[sid].website_name for sid in combo if sid in analyses]
            print("\n" + "-" * 78)
            print("COMPLEMENTARY COVERAGE SUGGESTION")
            print("-" * 78)
            print(
                f"No single source fully covers all requested variables "
                f"({', '.join(requested_variables)}). Using these {len(combo)} sources "
                f"TOGETHER covers the most of what was requested:"
            )
            for sid, name in zip(combo, names):
                covered = analyses[sid].availability.covered_variables
                print(f"  - {name}: provides {', '.join(covered) if covered else '(none matched)'}")

    print("\n" + "=" * 78)
