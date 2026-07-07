"""
Reconciles Agent 1's original ScientificIntentOutput with the location,
time-period, and scientific modifications the user gave to Agent 2.

WHY THIS EXISTS
----------------
Agent 1's plan (agent1_plan_preserved) is captured once, before the user
has answered anything, and agent2.py deliberately never mutates it in
place (see _inject_python_derived_fields in agent2.py) so Agent 1's
original reasoning is never silently rewritten by the LLM -- that field
stays a true, untouched record of what Agent 1 originally produced.

But that means, without an explicit reconciliation step, downstream
consumers would keep showing/using Agent 1's PRE-ANSWER values (e.g.
"Global" spatial coverage, "Unknown" temporal coverage, or "Unknown"
variables) even after the user has clarified them. This module is a 
pure, deterministic Python transformation -- no LLM call -- so both 
the summary and the retrieval request builder can call it and always 
see the same, correct effective values.
"""
"""
Reconciles Agent 1's original ScientificIntentOutput with the location,
time-period, and scientific modifications the user gave to Agent 2.
"""

from typing import List, Tuple

from models.schemas import (
    ScientificIntentOutput, 
    ResolvedInformation,
    ScientificVariable,
    Measurement,
    DatasetRequirement,
    PriorityLevel
)

_LOCATION_KEYWORDS = [
    "location", "spatial", "region", "area", "coordinate", "geograph",
    "where",
]
_TEMPORAL_KEYWORDS = [
    "date", "time", "temporal", "period", "season", "baseline", "when",
]

_GOAL_KEYWORDS = ["goal", "research goal", "intent", "objective"]
_VARIABLE_KEYWORDS = ["variable", "scientific variable", "parameter"]
_MEASUREMENT_KEYWORDS = ["measurement", "metric", "observation"]
_REQUIREMENT_KEYWORDS = ["dataset requirement", "data requirement", "dataset"]

_VAGUE_MARKERS = {
    "", "unknown", "not specified", "n/a", "none", "not provided",
    "global",
}

def bucket_resolved_information(
    resolved: List[ResolvedInformation],
) -> Tuple[
    List[ResolvedInformation],
    List[ResolvedInformation],
    List[ResolvedInformation],
]:
    location, temporal, other = [], [], []

    for item in resolved:
        name = item.field_name.lower()
        if any(k in name for k in _LOCATION_KEYWORDS):
            location.append(item)
        elif any(k in name for k in _TEMPORAL_KEYWORDS):
            temporal.append(item)
        else:
            other.append(item)
    return location, temporal, other

def reconcile_agent1_plan(
    agent1_plan: ScientificIntentOutput,
    resolved_information: List[ResolvedInformation],
) -> ScientificIntentOutput:
    location_items, temporal_items, other_items = bucket_resolved_information(
        resolved_information
    )

    reconciled = agent1_plan.model_copy(deep=True)

    if location_items:
        resolved_location_value = location_items[-1].resolved_value
        reconciled.spatial_context.location = resolved_location_value
        reconciled.spatial_context.geographic_extent = resolved_location_value
        if reconciled.spatial_context.coordinate_requirements.strip().lower() in _VAGUE_MARKERS:
            reconciled.spatial_context.coordinate_requirements = f"Derived from user-specified area: {resolved_location_value}"

    if temporal_items:
        resolved_temporal_value = temporal_items[-1].resolved_value
        reconciled.temporal_context.date_range = resolved_temporal_value

    for item in other_items:
        name = item.field_name.lower()
        val = item.resolved_value

        if any(k in name for k in _GOAL_KEYWORDS):
            reconciled.inferred_user_research_goal = val

        elif any(k in name for k in _VARIABLE_KEYWORDS):
            new_var = ScientificVariable(
                variable=val[:50] if len(val) > 50 else val,
                scientific_meaning=f"User modified: {val}",
                relevance="Provided by user via clarification",
                priority=PriorityLevel.high
            )
            replaced = False
            for i, v in enumerate(reconciled.scientific_variables):
                if v.variable.strip().lower() in _VAGUE_MARKERS:
                    reconciled.scientific_variables[i] = new_var
                    replaced = True
                    break
            if not replaced:
                reconciled.scientific_variables.append(new_var)

        elif any(k in name for k in _MEASUREMENT_KEYWORDS):
            new_meas = Measurement(
                measurement_name=val[:50] if len(val) > 50 else val,
                variable_measured="User specified",
                why_measurement_is_needed=f"User modified: {val}"
            )
            replaced = False
            for i, m in enumerate(reconciled.measurements):
                if m.measurement_name.strip().lower() in _VAGUE_MARKERS:
                    reconciled.measurements[i] = new_meas
                    replaced = True
                    break
            if not replaced:
                reconciled.measurements.append(new_meas)

        elif any(k in name for k in _REQUIREMENT_KEYWORDS):
            new_req = DatasetRequirement(
                requirement_name=val[:50] if len(val) > 50 else val,
                dataset_type="User specified",
                variables_or_measurements_needed=["User specified"],
                spatial_coverage_needed=reconciled.spatial_context.location,
                temporal_coverage_needed=reconciled.temporal_context.date_range,
                metadata_needed=["Standard"],
                quality_constraints=["Standard"],
                why_required=f"User modified: {val}"
            )
            replaced = False
            for i, r in enumerate(reconciled.dataset_requirements):
                if r.requirement_name.strip().lower() in _VAGUE_MARKERS:
                    reconciled.dataset_requirements[i] = new_req
                    replaced = True
                    break
            if not replaced:
                reconciled.dataset_requirements.append(new_req)

    if location_items or temporal_items:
        updated_requirements = []
        for req in reconciled.dataset_requirements:
            updated = req.model_copy(deep=True)
            if location_items and updated.spatial_coverage_needed.strip().lower() in _VAGUE_MARKERS:
                updated.spatial_coverage_needed = location_items[-1].resolved_value
            if temporal_items and updated.temporal_coverage_needed.strip().lower() in _VAGUE_MARKERS:
                updated.temporal_coverage_needed = temporal_items[-1].resolved_value
            updated_requirements.append(updated)
        reconciled.dataset_requirements = updated_requirements

    return reconciled
