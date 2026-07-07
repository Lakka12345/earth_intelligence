"""
Agent 4 — Orchestrator.

Entry point: run_agent4(payload). Ties together, in order:

  1. Coverage optimization (rank-order respecting -- agent4_coverage_optimizer)
  2. Per-source access resolution (agent4_access_resolver) -- with
     RE-PLANNING: if the user declines a source's access path, that
     source is excluded and the coverage plan is rebuilt from the
     remaining ranked sources, so a lower-ranked alternative is tried
     for whatever it was providing, rather than just leaving a gap.
  3. Size probing + approval (agent4_size_estimator / agent4_size_approval)
     -- same re-planning behavior on decline.
  4. Completeness confirmation.
  5. Download location + format choice, then execution per approved source.
  6. Ask raw vs. Agent 5 preprocessing handoff.

HONEST SCOPE NOTE: the "user_override" mode (user asked for a specific
website instead of the ranked list) has no WebsiteAnalysisResult to
work from -- Agent 3 never analyzed that source. It's handled as a
best-effort single-source flow: access/size are treated as unconfirmed
and resolved interactively, exactly like Agent 3's own "unconfirmed"
category.
"""

from typing import Dict, List, Optional

from agents.agent4_access_resolver import resolve_access
from agents.agent4_connectors.base import Credentials
from agents.agent4_coverage_optimizer import build_coverage_plan, describe_plan
from agents.agent4_download_manager import ask_download_format, ask_download_location, download_source
from agents.agent4_query_dateparser import parse_date_range
from agents.agent4_query_geoparser import geocode_location
from agents.agent4_size_approval import ask_size_approval
from agents.agent4_size_estimator import estimate_size
from models.agent4_schemas import AccessDecisionType, Agent4Output, SourceDecision
from models.website_analysis_schemas import (
    Agent3ToAgent4Mode,
    Agent3ToAgent4Payload,
    AccessClassification,
    AccessibilityProfile,
    AccuracyProfile,
    AvailabilityProfile,
    AvailabilityTimeline,
    CredentialEase,
    DataAvailabilityStatus,
    SourceSnapshot,
    TriState,
    WebsiteAnalysisResult,
)


def _override_snapshot_and_analysis(website: str, requested_variables: List[str]):
    """Best-effort synthetic entry for a user-specified override website
    that Agent 3 never analyzed. Everything unknown is marked unknown,
    never guessed -- resolve_access already has an honest path for this
    (the 'unconfirmed' branch)."""
    source_id = "user_override_source"
    url = website if website.startswith(("http://", "https://")) else f"https://{website}"
    snapshot = SourceSnapshot(source_id=source_id, name=website, url=url)

    analysis = WebsiteAnalysisResult(
        source_id=source_id,
        website_name=website,
        accessibility=AccessibilityProfile(
            authentication_required=True,   # unknown -- treat conservatively, resolver will ask
            anonymous_access_available=False,
            api_available=False,
            credential_ease=CredentialEase.unknown,
            credential_ease_notes="User-specified override source -- Agent 3 never analyzed this site, so access requirements are entirely unconfirmed.",
            access_classification=AccessClassification.simple_login_required,
            availability_timeline=AvailabilityTimeline.unknown,
        ),
        availability=AvailabilityProfile(
            relevance_score=0.5, completeness_score=0.5,
            requested_variables=requested_variables,
            covered_variables=requested_variables,  # assume it provides what the user asked for -- they chose it deliberately
            spatial_coverage_score=0.5, temporal_coverage_score=0.5, resolution_score=0.5,
            availability_status=DataAvailabilityStatus.unknown,
        ),
        accuracy=AccuracyProfile(
            authority_score=0.5, credibility_score=0.5, scientific_acceptance_score=0.5,
            consistency_score=0.5, historical_reliability_score=0.5, metadata_quality_score=0.5,
        ),
        data_policy_summary="User override -- not analyzed by Agent 3.",
    )
    return snapshot, analysis


def _resolve_and_approve_plan(
    ranked_source_ids: List[str],
    website_analyses: Dict[str, WebsiteAnalysisResult],
    source_snapshots: Dict[str, SourceSnapshot],
    requested_variables: List[str],
    bounding_box=None,
    time_range=None,
    pre_collected_credentials=None,
) -> tuple:
    """
    Runs the coverage -> access -> size loop, rebuilding the plan each
    time a source is declined, until either every remaining variable is
    covered by APPROVED sources or no more ranked alternatives are left.

    Returns (approved_decisions: Dict[source_id, SourceDecision],
             approved_credentials: Dict[source_id, Optional[Credentials]] -- IN-MEMORY ONLY, never serialized,
             running_total_bytes: float,
             final_uncovered_variables: set)
    """
    excluded: set = set()
    approved: Dict[str, SourceDecision] = {}
    approved_credentials: Dict[str, Optional[Credentials]] = {}
    running_total = 0.0

    while True:
        plan = build_coverage_plan(ranked_source_ids, website_analyses, requested_variables, excluded_source_ids=excluded)

        # Only resolve sources not already approved in a previous iteration.
        newly_needed = [sid for sid in plan.selected_source_ids if sid not in approved]
        if not newly_needed:
            return approved, approved_credentials, running_total, plan.uncovered_variables

        any_declined_this_pass = False

        for sid in newly_needed:
            snapshot = source_snapshots.get(sid)
            analysis = website_analyses.get(sid)
            if snapshot is None or analysis is None:
                excluded.add(sid)
                any_declined_this_pass = True
                continue

            decision, credentials = resolve_access(snapshot, analysis, pre_collected_credentials)

            if decision.decision in (AccessDecisionType.skipped_declined, AccessDecisionType.skipped_unresolved):
                excluded.add(sid)
                any_declined_this_pass = True
                continue

            size_estimate = estimate_size(snapshot, requested_variables, bounding_box, time_range)
            decision.approved_size = size_estimate

            if not ask_size_approval(snapshot.name, size_estimate, running_total):
                excluded.add(sid)
                any_declined_this_pass = True
                continue

            approved[sid] = decision
            approved_credentials[sid] = credentials
            running_total += size_estimate.estimated_bytes or 0.0

        if not any_declined_this_pass:
            # Everyone in this pass was approved -- re-check coverage
            # once more in case the plan is now already complete.
            plan = build_coverage_plan(ranked_source_ids, website_analyses, requested_variables, excluded_source_ids=excluded)
            if not plan.uncovered_variables or all(sid in approved for sid in plan.selected_source_ids):
                return approved, approved_credentials, running_total, plan.uncovered_variables
        # else: loop again -- build_coverage_plan will try the next-ranked alternative(s)


def run_agent4(payload: Agent3ToAgent4Payload, request=None) -> Agent4Output:
    """
    `request` is the original RetrievalRequest (same object passed to
    run_agent3) -- optional only for backward compatibility / ad-hoc
    testing; without it, no bounding box or time range can be derived
    and every source falls back to full-download (still correct, just
    not storage-optimal).
    """
    print("\n" + "=" * 70)
    print("AGENT 4 — INTELLIGENT RETRIEVAL")
    print("=" * 70)

    website_analyses = dict(payload.website_analyses)
    source_snapshots = dict(payload.source_snapshots)
    ranked_ids = list(payload.final_ranked_source_ids)
    requested_variables = list(payload.requested_variables)

    bounding_box, time_range = None, None
    if request is not None:
        spatial = getattr(request, "spatial_requirements", {}) or {}
        temporal = getattr(request, "temporal_requirements", {}) or {}
        bounding_box = geocode_location(spatial.get("location", ""), spatial.get("geographic_extent", ""))
        time_range = parse_date_range(temporal.get("date_range", ""), temporal.get("historical_baseline", ""))
        if bounding_box:
            print(f"\nResolved region to a bounding box for subsetting: {bounding_box}")
        else:
            print("\nCould not resolve a specific region for subsetting -- sources will use their full spatial extent.")
        if time_range:
            print(f"Resolved time range for subsetting: {time_range[0]} to {time_range[1]}")
        else:
            print("Could not resolve a specific date range for subsetting -- sources will use their full time extent.")

    if payload.mode == Agent3ToAgent4Mode.user_override:
        snapshot, analysis = _override_snapshot_and_analysis(payload.override_website, requested_variables)
        website_analyses[snapshot.source_id] = analysis
        source_snapshots[snapshot.source_id] = snapshot
        ranked_ids = [snapshot.source_id]
        print(f"\nUser requested a specific website instead of the ranked list: {payload.override_website}")
        print("Access and size for this source are unconfirmed since Agent 3 never analyzed it.")

    print("\nStep 1 — Selecting the most efficient combination of sources (rank-order respected)...")
    preview_plan = build_coverage_plan(ranked_ids, website_analyses, requested_variables)
    print(describe_plan(preview_plan, source_snapshots))

    print("\nStep 2 — Resolving access and confirming size per source (you'll be asked as we go)...")
    approved, approved_credentials, total_bytes, still_uncovered = _resolve_and_approve_plan(
        ranked_ids, website_analyses, source_snapshots, requested_variables, bounding_box, time_range,
        pre_collected_credentials=payload.pre_collected_credentials,
    )

    if not approved:
        print("\nNo sources were approved -- nothing to download.")
        return Agent4Output(
            plan_source_ids=[], source_decisions=[], covers_full_query=False,
            uncovered_variables=sorted(still_uncovered) if still_uncovered else requested_variables,
            notes=["User declined every candidate source; no data retrieved."],
        )

    from models.agent4_schemas import format_bytes
    print(f"\nApproved {len(approved)} source(s), total estimated size: {format_bytes(total_bytes)}")
    if still_uncovered:
        print(f"NOTE: the following requested variables are not covered by any approved source: {', '.join(sorted(still_uncovered))}")
    else:
        print("Every requested variable is covered by the approved sources.")

    print("\nStep 3 — Download setup")
    location = ask_download_location()
    ask_download_format()  # currently informational only -- see agent4_download_manager's honest scope note

    print("\nStep 4 — Downloading...")
    manifest = []
    for sid, decision in approved.items():
        snapshot = source_snapshots[sid]
        creds = approved_credentials.get(sid)
        entry = download_source(snapshot, requested_variables, location, creds, bounding_box, time_range)
        manifest.append(entry)
        status = "OK" if entry.success else f"FAILED ({entry.error})"
        print(f"  {snapshot.name}: {status}"
              + (f" -> {entry.local_path} ({entry.fetch_method.value})" if entry.success else ""))

        # NEW: CandidateSource.health_score's docstring documents this
        # exact contract -- "Agent 4 calls: source.health_score *= 0.9
        # on failure, then persists" -- via Qdrant so it survives across
        # runs. Never implemented until now. Non-fatal if Qdrant is
        # unavailable (update_source_reliability already swallows its
        # own errors and returns False).
        try:
            from memory.qdrant_store import update_source_reliability
            update_source_reliability(sid, succeeded=entry.success)
        except Exception as exc:
            print(f"  (reliability history not updated for {sid}: {exc})")

    print("\nStep 5 — Next step")
    wants_preprocessing = input(
        "Do you want the raw downloaded files, or should Agent 5 preprocess/combine them first? "
        "(raw / preprocess): "
    ).strip().lower().startswith("p")

    return Agent4Output(
        plan_source_ids=list(approved.keys()),
        source_decisions=list(approved.values()),
        manifest=manifest,
        total_size_bytes=total_bytes,
        covers_full_query=not bool(still_uncovered),
        uncovered_variables=sorted(still_uncovered) if still_uncovered else [],
        download_location=location,
        send_to_agent5=wants_preprocessing,
        notes=[
            f"{len([m for m in manifest if m.success])}/{len(manifest)} download(s) succeeded.",
        ],
    )
