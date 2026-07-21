"""
Scientific Understanding Summary -- pure formatter, no LLM call.

Renders a human-readable summary of what the pipeline currently
understands about the user's scientific request, built entirely from a
validated ClarificationAgentOutput instance (i.e. only data that has
already passed Agent 1's and Agent 2's pydantic validation).

This module performs NO inference, NO scientific reasoning, and makes
NO model calls. It only reads fields that already exist on the
validated object and arranges them for display. If a field is empty or
"Unknown", the summary says so plainly rather than guessing or filling
in a default -- inventing information here would defeat the entire
point of a confirmation step.

RECONCILIATION: Agent 1's plan (agent1_plan_preserved) is captured
before the user answers anything, and is never mutated in place (see
agent2.py). Without reconciliation, this summary would keep showing
Agent 1's PRE-ANSWER values (e.g. "Global" spatial coverage) even after
the user narrowed the area and gave a date range -- which would mean
the confirmation screen and the actual retrieval request could disagree
about what's being retrieved. To prevent that, this module calls
models.plan_reconciliation.reconcile_agent1_plan() to compute the same
effective values that models/retrieval_request.py will use, and
displays those explicitly wherever they differ from Agent 1's raw
values, so nothing is silently substituted without being shown.

Canonical source choice: ClarificationAgentOutput carries
active_assumptions / remaining_gaps / resolved_information at BOTH the
top level and inside refined_scientific_plan (the schema does not
assert these two copies are equal -- only retrieval_readiness and
completeness_score are schema-enforced equal between the two). This
formatter treats the TOP-LEVEL fields as canonical for display, since
those are the fields validate_agent2_contract's own consistency checks
reason about (e.g. the critical-conflict check compares
self.resolved_information against self.remaining_gaps, not the nested
copies), and since main.py already treats the top-level
ClarificationAgentOutput as Agent 2's primary output.
"""

from typing import List, Optional

from models.schemas import ClarificationAgentOutput
from models.plan_reconciliation import (
    bucket_resolved_information,
    reconcile_agent1_plan,
)


def _risk_bucket(confidence: float) -> str:
    """
    Derives a coarse, clearly-labeled qualitative risk bucket from
    Assumption.confidence. This is a DERIVED display label, not a field
    the LLM produced -- the summary always shows it alongside the raw
    confidence value so nothing is hidden or asserted as LLM output.
    """
    if confidence >= 0.8:
        return "Low risk"
    if confidence >= 0.5:
        return "Medium risk"
    return "High risk"


def _line(label: str, value: str, indent: str = "") -> str:
    return f"{indent}{label}: {value}"


def render_understanding_summary(output: ClarificationAgentOutput) -> str:
    """
    Renders the full Scientific Understanding summary as plain text.

    Args:
        output: a validated ClarificationAgentOutput (must already have
            passed model_validate() -- this function does not validate
            anything itself).

    Returns:
        A formatted multi-section string suitable for printing to the
        console (or adapting to another display surface later).
    """
    plan = output.refined_scientific_plan
    agent1 = plan.agent1_plan_preserved

    # Deterministic, no-LLM reconciliation of Agent 1's frozen plan with
    # the user's resolved location/time-period answers. This is the
    # SAME function main.py calls before building the retrieval
    # request, so what's displayed here is guaranteed to match what
    # will actually be retrieved.
    reconciled_agent1 = reconcile_agent1_plan(
        agent1, output.resolved_information
    )

    sections: List[str] = []

    # ---------------------------------------------------------------- #
    # Research Goal                                                     #
    # ---------------------------------------------------------------- #
    sections.append("RESEARCH GOAL")
    sections.append(f"  {agent1.inferred_user_research_goal}")

    # ---------------------------------------------------------------- #
    # Scientific Objectives                                             #
    # ---------------------------------------------------------------- #
    sections.append("\nSCIENTIFIC OBJECTIVES")
    if agent1.scientific_objectives:
        for i, obj in enumerate(agent1.scientific_objectives, start=1):
            sections.append(f"  {i}. {obj.objective}  [priority: {obj.priority.value}]")
            sections.append(f"     Rationale: {obj.rationale}")
    else:
        sections.append("  None recorded.")

    # ---------------------------------------------------------------- #
    # Research Questions                                                #
    # ---------------------------------------------------------------- #
    sections.append("\nRESEARCH QUESTIONS")
    if agent1.research_questions:
        for i, rq in enumerate(agent1.research_questions, start=1):
            sections.append(f"  {i}. {rq.question}  [importance: {rq.importance.value}]")
    else:
        sections.append("  None recorded.")

    # ---------------------------------------------------------------- #
    # Scientific Variables                                              #
    # ---------------------------------------------------------------- #
    sections.append("\nSCIENTIFIC VARIABLES")
    if agent1.scientific_variables:
        for v in agent1.scientific_variables:
            sections.append(f"  - {v.variable}  [priority: {v.priority.value}]")
            sections.append(f"    Meaning: {v.scientific_meaning}")
            sections.append(f"    Relevance: {v.relevance}")
    else:
        sections.append("  None recorded.")

    # ---------------------------------------------------------------- #
    # Measurements                                                      #
    # ---------------------------------------------------------------- #
    sections.append("\nMEASUREMENTS")
    if agent1.measurements:
        for m in agent1.measurements:
            sections.append(f"  - {m.measurement_name} (measures: {m.variable_measured})")
            sections.append(f"    Why needed: {m.why_measurement_is_needed}")
            sections.append(
                f"    Preferred unit/representation: {m.preferred_unit_or_representation}"
                f"  |  Required resolution: {m.required_resolution}"
            )
    else:
        sections.append("  None recorded.")

    # ---------------------------------------------------------------- #
    # Location                                                          #
    # ---------------------------------------------------------------- #
    resolved_location, resolved_temporal, resolved_other = bucket_resolved_information(
        output.resolved_information
    )

    sections.append("\nLOCATION")
    sc = agent1.spatial_context
    sections.append(f"  Location (Agent 1, before your answers): {sc.location}")
    sections.append(f"  Geographic extent: {sc.geographic_extent}")
    sections.append(f"  Study boundary type: {sc.study_boundary_type}")
    sections.append(f"  Coordinate requirements: {sc.coordinate_requirements}")
    sections.append(f"  Spatial resolution requirements: {sc.spatial_resolution_requirements}")
    sections.append(f"  Spatial context category: {sc.spatial_context}")
    if resolved_location:
        sections.append("  Resolved by Agent 2:")
        for item in resolved_location:
            sections.append(
                f"    - {item.field_name}: {item.resolved_value}  "
                f"[source: {item.resolution_source.value}]"
            )
    # Always show the effective value that will actually be used for
    # retrieval -- this is the whole point of reconciliation: nothing
    # about what gets retrieved should be inferable only by re-reading
    # the Q&A above.
    sections.append(
        f"  >> Effective area used for retrieval: "
        f"{reconciled_agent1.spatial_context.location}"
    )

    # ---------------------------------------------------------------- #
    # Time Period                                                       #
    # ---------------------------------------------------------------- #
    sections.append("\nTIME PERIOD")
    tc = agent1.temporal_context
    sections.append(f"  Date range (Agent 1, before your answers): {tc.date_range}")
    sections.append(f"  Event window: {tc.event_window}")
    sections.append(f"  Historical baseline: {tc.historical_baseline}")
    sections.append(f"  Temporal resolution: {tc.temporal_resolution}")
    sections.append(f"  Temporal analysis type: {tc.temporal_analysis_type}")
    if resolved_temporal:
        sections.append("  Resolved by Agent 2:")
        for item in resolved_temporal:
            sections.append(
                f"    - {item.field_name}: {item.resolved_value}  "
                f"[source: {item.resolution_source.value}]"
            )
    sections.append(
        f"  >> Effective time period used for retrieval: "
        f"{reconciled_agent1.temporal_context.date_range}"
    )

    # ---------------------------------------------------------------- #
    # Dataset Requirements                                              #
    # ---------------------------------------------------------------- #
    # Shows the RECONCILED spatial/temporal coverage (what will
    # actually be sent to Agent 3), and explicitly flags when that
    # differs from Agent 1's original value, rather than silently
    # displaying stale "Global" / "Unknown" coverage as if it were
    # still accurate.
    sections.append("\nDATASET REQUIREMENTS")
    if agent1.dataset_requirements:
        for dr, rdr in zip(
            agent1.dataset_requirements,
            reconciled_agent1.dataset_requirements,
        ):
            sections.append(f"  - {dr.requirement_name}  [type: {dr.dataset_type}]")
            sections.append(f"    Variables/measurements needed: {', '.join(dr.variables_or_measurements_needed)}")

            if rdr.spatial_coverage_needed != dr.spatial_coverage_needed:
                sections.append(
                    f"    Spatial coverage needed: {rdr.spatial_coverage_needed}  "
                    f'(updated from "{dr.spatial_coverage_needed}" based on your answer)'
                )
            else:
                sections.append(f"    Spatial coverage needed: {dr.spatial_coverage_needed}")

            if rdr.temporal_coverage_needed != dr.temporal_coverage_needed:
                sections.append(
                    f"    Temporal coverage needed: {rdr.temporal_coverage_needed}  "
                    f'(updated from "{dr.temporal_coverage_needed}" based on your answer)'
                )
            else:
                sections.append(f"    Temporal coverage needed: {dr.temporal_coverage_needed}")

            sections.append(f"    Why required: {dr.why_required}")
    else:
        sections.append("  None recorded.")

    # ---------------------------------------------------------------- #
    # Decision Context                                                  #
    # ---------------------------------------------------------------- #
    sections.append("\nDECISION CONTEXT")
    dc = agent1.decision_context
    sections.append(f"  Inferred decision context: {dc.inferred_decision_context}")
    sections.append(f"  Intended use case: {dc.intended_use_case}")

    # ---------------------------------------------------------------- #
    # Other resolved information (anything not bucketed as location/time) #
    # ---------------------------------------------------------------- #
    if resolved_other:
        sections.append("\nOTHER RESOLVED INFORMATION")
        for item in resolved_other:
            sections.append(
                f"  - {item.field_name}: {item.resolved_value}  "
                f"[source: {item.resolution_source.value}]"
            )

    # ---------------------------------------------------------------- #
    # Assumptions + Risk Level                                          #
    # ---------------------------------------------------------------- #
    sections.append("\nASSUMPTIONS")
    if output.active_assumptions:
        for a in output.active_assumptions:
            risk_label = _risk_bucket(a.confidence)
            sections.append(f"  - {a.assumption}")
            sections.append(f"    Reason: {a.reason}")
            sections.append(
                f"    Confidence: {a.confidence:.2f}  "
                f"[{risk_label} -- derived from confidence, not LLM-stated]"
            )
            sections.append(f"    Risk if wrong: {a.risk_if_wrong}")
    else:
        sections.append("  None recorded.")

    # ---------------------------------------------------------------- #
    # Remaining Gaps                                                    #
    # ---------------------------------------------------------------- #
    sections.append("\nREMAINING GAPS")
    if output.remaining_gaps:
        for g in output.remaining_gaps:
            blocks = "blocks retrieval" if g.blocks_retrieval else "does not block retrieval"
            sections.append(f"  - {g.gap_name}  [{g.severity.value}, {blocks}]")
            sections.append(f"    {g.description}")
    else:
        sections.append("  None recorded.")

    # ---------------------------------------------------------------- #
    # Retrieval Readiness                                               #
    # ---------------------------------------------------------------- #
    sections.append("\nRETRIEVAL READINESS")
    sections.append(f"  Status: {output.retrieval_readiness.value}")
    sections.append(f"  Completeness score: {output.completeness_score:.2f}")
    sections.append(f"  Confidence score: {output.confidence_score:.2f}")

    return "\n".join(sections)


def render_understanding_summary_banner(
    output: ClarificationAgentOutput,
    revision_round: Optional[int] = None,
) -> str:
    """
    Wraps render_understanding_summary with a banner header, optionally
    noting which revision round this is (round 0 / None = initial
    summary, before any "Modify" requests).
    """
    header = "=" * 70
    title = "SCIENTIFIC UNDERSTANDING SUMMARY"
    if revision_round:
        title += f"  (revision {revision_round})"

    readiness = output.retrieval_readiness.value
    if readiness == "proceed_with_assumptions":
        notice = (
            "⚠️  Proceeding with assumptions -- not all gaps were resolved "
            "(e.g. the clarification round limit was reached). Review the "
            "ASSUMPTIONS and REMAINING GAPS sections below carefully before "
            "continuing."
        )
    elif readiness == "clarification_required":
        notice = (
            "⚠️  Clarification is still required -- this summary reflects "
            "the current best understanding, but Agent 2 has open questions "
            "that were not resolved."
        )
    else:
        notice = "✅  Understanding is fully resolved -- no outstanding gaps."

    return (
        f"{header}\n{title}\n{header}\n\n"
        f"{notice}\n\n"
        f"{render_understanding_summary(output)}\n\n"
        f"{header}"
    )
