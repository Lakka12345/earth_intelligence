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

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from agents.agent4_access_resolver import resolve_access
from connectors.base_connector import Credentials
from agents.agent4_coverage_optimizer import build_coverage_plan, describe_plan
from agents.agent4_download_manager import (
    DEFAULT_MANAGED_FOLDER,
    ask_download_format,
    ask_download_location,
    download_source,
    plan_download_batches,
)
from agents.agent4_query_dateparser import parse_date_range
from agents.agent4_query_geoparser import geocode_location
from agents.agent4_size_approval import ask_size_approval
from agents.agent4_size_estimator import discover_source_datasets, estimate_size, probe_dataset_metadata
from models.agent4_schemas import AccessDecisionType, Agent4Output, SourceDecision, format_bytes
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

DOWNLOADABLE_DECISIONS = {
    AccessDecisionType.free_access,
    AccessDecisionType.agent_self_registered,
    AccessDecisionType.user_provided_credentials,
}


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
    initial_excluded_source_ids=None,
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
    excluded: set = set(initial_excluded_source_ids or set())
    approved: Dict[str, SourceDecision] = {}
    approved_credentials: Dict[str, Optional[Credentials]] = {}
    running_total = 0.0

    while True:
        plan = build_coverage_plan(
            ranked_source_ids,
            website_analyses,
            requested_variables,
            excluded_source_ids=excluded,
            source_snapshots=source_snapshots,
        )

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

            decision.variables_expected = list(plan.per_source_new_coverage.get(sid, []))

            if decision.decision in (AccessDecisionType.skipped_declined, AccessDecisionType.skipped_unresolved):
                approved[sid] = decision
                excluded.add(sid)
                any_declined_this_pass = True
                continue

            if decision.decision == AccessDecisionType.payment_redirect:
                approved[sid] = decision
                excluded.add(sid)
                any_declined_this_pass = True
                print("  Payment is pending. Agent 4 will continue with remaining ranked alternatives.")
                continue

            dataset_candidates = discover_source_datasets(snapshot, requested_variables, bounding_box, time_range)
            if dataset_candidates:
                best_dataset = dataset_candidates[0]
                print(
                    "  Dataset selected: "
                    f"{best_dataset.dataset_name} ({best_dataset.dataset_id})"
                )
            else:
                print("  Dataset discovery: no structured dataset descriptor available; using connector metadata probe.")

            dataset_metadata = probe_dataset_metadata(snapshot, requested_variables, bounding_box, time_range)
            decision.dataset_metadata = dataset_metadata
            if dataset_metadata.unavailable_reason:
                print(f"  Metadata: unavailable ({dataset_metadata.unavailable_reason})")
            else:
                print(
                    "  Metadata: "
                    f"dataset={dataset_metadata.dataset_id or 'unknown'}, "
                    f"format={dataset_metadata.file_format or dataset_metadata.content_type or 'unknown'}, "
                    f"size={format_bytes(dataset_metadata.file_size_bytes)}"
                )

            size_estimate = estimate_size(snapshot, requested_variables, bounding_box, time_range)
            if size_estimate.estimated_bytes is None and dataset_metadata.file_size_bytes is not None:
                size_estimate.estimated_bytes = dataset_metadata.file_size_bytes
                size_estimate.is_exact = True
                size_estimate.method = dataset_metadata.retrieval_method or "dataset metadata"
                size_estimate.human_readable = format_bytes(dataset_metadata.file_size_bytes)
            decision.approved_size = size_estimate

            if not ask_size_approval(snapshot.name, size_estimate, running_total):
                decision.decision = AccessDecisionType.skipped_declined
                decision.notes = "User declined this source because of estimated download size."
                approved[sid] = decision
                excluded.add(sid)
                any_declined_this_pass = True
                continue

            approved[sid] = decision
            approved_credentials[sid] = credentials
            running_total += size_estimate.estimated_bytes or 0.0

        if not any_declined_this_pass:
            # Everyone in this pass was approved -- re-check coverage
            # once more in case the plan is now already complete.
            plan = build_coverage_plan(
                ranked_source_ids,
                website_analyses,
                requested_variables,
                excluded_source_ids=excluded,
                source_snapshots=source_snapshots,
            )
            if not plan.uncovered_variables or all(sid in approved for sid in plan.selected_source_ids):
                return approved, approved_credentials, running_total, plan.uncovered_variables
        # else: loop again -- build_coverage_plan will try the next-ranked alternative(s)


def _source_access_label(decision: SourceDecision) -> str:
    mapping = {
        AccessDecisionType.free_access: "Public",
        AccessDecisionType.agent_self_registered: "Self-registration",
        AccessDecisionType.user_provided_credentials: "Credentials required",
        AccessDecisionType.payment_redirect: "Payment required",
        AccessDecisionType.skipped_declined: "Skipped",
        AccessDecisionType.skipped_unresolved: "Unresolved",
    }
    return mapping.get(decision.decision, decision.decision.value)


def _build_coverage_table(
    requested_variables: List[str],
    source_decisions: List[SourceDecision],
    source_snapshots: Dict[str, SourceSnapshot],
    manifest,
) -> tuple:
    downloaded = {m.source_id: m for m in manifest if m.success}
    rows = []
    retrieved = set()

    for variable in requested_variables:
        key = variable.lower().strip()
        matched_decision = None
        for decision in source_decisions:
            if key in {v.lower().strip() for v in decision.variables_expected}:
                matched_decision = decision
                break

        if matched_decision is None:
            rows.append({
                "Variable": variable,
                "Retrieved From": "",
                "Coverage Status": "Missing",
                "Access Type": "Unavailable",
                "Download Status": "Not attempted",
                "Reason if unavailable": "No ranked Agent 3 source contributed this variable.",
            })
            continue

        snapshot = source_snapshots.get(matched_decision.source_id)
        manifest_entry = downloaded.get(matched_decision.source_id)
        if manifest_entry:
            retrieved.add(variable)
            coverage_status = "Retrieved"
            download_status = "Downloaded"
            reason = ""
        elif matched_decision.decision == AccessDecisionType.payment_redirect:
            coverage_status = "Pending"
            download_status = "Waiting for user payment"
            reason = matched_decision.notes
        elif matched_decision.decision in (AccessDecisionType.skipped_declined, AccessDecisionType.skipped_unresolved):
            coverage_status = "Skipped"
            download_status = "Not downloaded"
            reason = matched_decision.notes
        else:
            coverage_status = "Failed"
            download_status = "Download failed"
            reason = matched_decision.notes or "All available connector download methods failed."

        rows.append({
            "Variable": variable,
            "Retrieved From": snapshot.name if snapshot else matched_decision.source_id,
            "Coverage Status": coverage_status,
            "Access Type": _source_access_label(matched_decision),
            "Download Status": download_status,
            "Reason if unavailable": reason,
        })

    coverage_percent = (len(retrieved) / len(requested_variables) * 100.0) if requested_variables else 100.0
    return rows, sorted(retrieved), round(coverage_percent, 2)


def _print_retrieval_report(output: Agent4Output, source_snapshots: Dict[str, SourceSnapshot]) -> None:
    print("\n" + "=" * 70)
    print("AGENT 4 RETRIEVAL REPORT")
    print("=" * 70)
    print(f"Sources successfully retrieved: {len([m for m in output.manifest if m.success])}")
    print(f"Sources skipped/pending/failed: {len([d for d in output.source_decisions if d.decision not in DOWNLOADABLE_DECISIONS])}")
    print(f"Variables retrieved: {', '.join(output.retrieved_variables) if output.retrieved_variables else '(none)'}")
    print(f"Variables still missing: {', '.join(output.uncovered_variables) if output.uncovered_variables else '(none)'}")
    print(f"Overall scientific coverage: {output.coverage_percent:.2f}%")
    print(f"Estimated download size: {format_bytes(output.total_size_bytes)}")
    print(f"Actual downloaded size: {format_bytes(output.actual_downloaded_bytes)}")
    print(f"Retries attempted: {sum(m.retries_attempted for m in output.manifest)}")
    print(f"Validation failures: {len([m for m in output.manifest if m.validation_notes and not m.success])}")
    print("\nVariable coverage table:")
    for row in output.coverage_table:
        print(
            f"  {row['Variable']}: {row['Coverage Status']} | "
            f"{row['Retrieved From'] or 'No source'} | {row['Access Type']} | "
            f"{row['Download Status']}"
        )


def _confirm_incomplete_coverage(uncovered_variables) -> bool:
    missing = sorted(uncovered_variables or [])
    if not missing:
        return True

    print("\n" + "=" * 70)
    print("INCOMPLETE COVERAGE")
    print("=" * 70)
    print(f"Coverage is incomplete. Missing variables: {', '.join(missing)}")
    print("Agent 4 has already walked the ranked Agent 3 source list available in this handoff.")
    print("Choose:")
    print("  1. Stop and return to source discovery/ranking")
    print("  2. Continue anyway with partial coverage")

    while True:
        choice = input("Your choice [1/2]: ").strip().lower()
        if choice in ("1", "stop", "no", "n"):
            return False
        if choice in ("2", "continue", "yes", "y"):
            return True
        print("Please choose 1 to stop or 2 to continue anyway.")


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
    preview_plan = build_coverage_plan(ranked_ids, website_analyses, requested_variables, source_snapshots=source_snapshots)
    print(describe_plan(preview_plan, source_snapshots))

    print("\nStep 2 — Resolving access and confirming size per source (you'll be asked as we go)...")
    approved, approved_credentials, total_bytes, still_uncovered = _resolve_and_approve_plan(
        ranked_ids, website_analyses, source_snapshots, requested_variables, bounding_box, time_range,
        pre_collected_credentials=payload.pre_collected_credentials,
    )

    source_decisions = list(approved.values())
    downloadable_ids = [
        sid for sid, decision in approved.items()
        if decision.decision in DOWNLOADABLE_DECISIONS
    ]

    if not source_decisions:
        print("\nNo sources were approved or left pending -- nothing to retrieve.")
        output = Agent4Output(
            plan_source_ids=[], source_decisions=[], covers_full_query=False,
            uncovered_variables=sorted(still_uncovered) if still_uncovered else requested_variables,
            notes=["Every candidate source was unavailable or declined; no data retrieved."],
        )
        _print_retrieval_report(output, source_snapshots)
        return output

    print(f"\nResolved {len(source_decisions)} source decision(s), total estimated downloadable size: {format_bytes(total_bytes)}")
    if still_uncovered:
        print(f"NOTE: the following requested variables are not covered by any approved source: {', '.join(sorted(still_uncovered))}")
        if not _confirm_incomplete_coverage(still_uncovered):
            coverage_table, retrieved_variables, coverage_percent = _build_coverage_table(
                requested_variables, source_decisions, source_snapshots, []
            )
            output = Agent4Output(
                plan_source_ids=list(approved.keys()),
                source_decisions=source_decisions,
                manifest=[],
                total_size_bytes=total_bytes,
                actual_downloaded_bytes=0.0,
                covers_full_query=False,
                uncovered_variables=[row["Variable"] for row in coverage_table if row["Coverage Status"] != "Retrieved"],
                retrieved_variables=retrieved_variables,
                coverage_percent=coverage_percent,
                coverage_table=coverage_table,
                download_location=None,
                send_to_agent5=False,
                notes=[
                    "Retrieval stopped because approved Agent 3 ranked sources did not cover every requested variable.",
                ],
            )
            _print_retrieval_report(output, source_snapshots)
            return output
    else:
        print("Every requested variable is covered by the approved sources.")

    print("\nStep 3 -- Next step")
    wants_preprocessing = input(
        "What would you like to do? Type 'raw' to download retrieved raw datasets, "
        "or 'preprocess' to continue directly to Agent 5 preprocessing: "
    ).strip().lower().startswith("p")

    location = None
    if downloadable_ids:
        if wants_preprocessing:
            location = DEFAULT_MANAGED_FOLDER
            print("Agent 5 preprocessing selected. Agent 4 will use the managed project data folder without asking for a custom location.")
        else:
            print(f"\nTotal download size: {format_bytes(total_bytes)}")
            print(f"Estimated local storage required: {format_bytes(total_bytes)}")
            print("Estimated download time depends on provider throughput and network speed.")
            location = ask_download_location()
        ask_download_format()  # currently informational only -- see agent4_download_manager's honest scope note

    print("\nStep 4 -- Downloading...")
    manifest = []
    failed_source_ids = set()
    downloaded_source_ids = set()
    successful_variables = set()
    while True:
        pending_ids = [
            sid for sid in downloadable_ids
            if sid not in downloaded_source_ids and sid in approved and sid in source_snapshots
        ]
        if not pending_ids:
            break
        batches = plan_download_batches(pending_ids, source_snapshots)
        if not batches:
            break
        batch = batches[0]
        print(
            f"  Download batch {batch[0].batch_id + 1}: "
            + ", ".join(source_snapshots[item.source_id].name for item in batch)
            + (" (parallel)" if len(batch) > 1 else "")
        )

        def _execute(sid: str):
            snapshot = source_snapshots[sid]
            decision = approved[sid]
            creds = approved_credentials.get(sid)
            return sid, download_source(
                snapshot,
                requested_variables,
                location,
                creds,
                bounding_box,
                time_range,
                dataset_metadata=decision.dataset_metadata,
            )

        batch_results = []
        if len(batch) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(batch))) as pool:
                future_map = {pool.submit(_execute, item.source_id): item.source_id for item in batch}
                for future in as_completed(future_map):
                    batch_results.append(future.result())
        else:
            batch_results.append(_execute(batch[0].source_id))

        for sid, entry in batch_results:
            snapshot = source_snapshots[sid]
            decision = approved[sid]
            manifest.append(entry)
            downloaded_source_ids.add(sid)
            status = "OK" if entry.success else f"FAILED ({entry.error})"
            print(f"  {snapshot.name}: {status}"
                  + (f" -> {entry.local_path} ({entry.fetch_method.value})" if entry.success else ""))

            if entry.success:
                successful_variables.update(decision.variables_expected)
            else:
                failed_source_ids.add(sid)
                replacement_variables = [
                    variable for variable in decision.variables_expected
                    if variable not in successful_variables
                ]
                if replacement_variables:
                    print(
                        "  Searching remaining ranked sources for alternatives covering: "
                        + ", ".join(replacement_variables)
                    )
                    new_decisions, new_credentials, added_bytes, _ = _resolve_and_approve_plan(
                        ranked_ids,
                        website_analyses,
                        source_snapshots,
                        replacement_variables,
                        bounding_box,
                        time_range,
                        pre_collected_credentials=payload.pre_collected_credentials,
                        initial_excluded_source_ids=failed_source_ids,
                    )
                    total_bytes += added_bytes
                    for new_sid, new_decision in new_decisions.items():
                        if new_sid in approved:
                            continue
                        approved[new_sid] = new_decision
                        source_decisions.append(new_decision)
                        if new_sid in new_credentials:
                            approved_credentials[new_sid] = new_credentials[new_sid]
                        if new_decision.decision in DOWNLOADABLE_DECISIONS:
                            downloadable_ids.append(new_sid)

            try:
                from memory.qdrant_store import update_source_reliability
                update_source_reliability(sid, succeeded=entry.success)
            except Exception as exc:
                print(f"  (reliability history not updated for {sid}: {exc})")

    for decision in source_decisions:
        if decision.decision in DOWNLOADABLE_DECISIONS:
            continue
        snapshot = source_snapshots.get(decision.source_id)
        print(f"  {snapshot.name if snapshot else decision.source_id}: not downloaded ({decision.decision.value})")

    coverage_table, retrieved_variables, coverage_percent = _build_coverage_table(
        requested_variables, source_decisions, source_snapshots, manifest
    )
    missing_variables = [
        row["Variable"] for row in coverage_table
        if row["Coverage Status"] != "Retrieved"
    ]
    actual_downloaded = sum(m.size_bytes or 0 for m in manifest if m.success)

    output = Agent4Output(
        plan_source_ids=list(approved.keys()),
        source_decisions=source_decisions,
        manifest=manifest,
        total_size_bytes=total_bytes,
        actual_downloaded_bytes=actual_downloaded,
        covers_full_query=not bool(missing_variables),
        uncovered_variables=missing_variables,
        retrieved_variables=retrieved_variables,
        coverage_percent=coverage_percent,
        coverage_table=coverage_table,
        download_location=location,
        send_to_agent5=wants_preprocessing,
        notes=[
            f"{len([m for m in manifest if m.success])}/{len(downloadable_ids)} download(s) succeeded.",
            "Agent 3 remained the planner; Agent 4 executed access, size, retrieval, retry, validation, and reporting.",
        ],
    )
    _print_retrieval_report(output, source_snapshots)
    return output
