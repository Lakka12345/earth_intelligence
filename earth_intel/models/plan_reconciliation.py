"""
Reconciles Agent 1's original ScientificIntentOutput with accepted Agent
2 clarifications and modifications.

The reconciliation stays deterministic: it never calls an LLM and it
never guesses when the available information is not specific enough.
"""

import re
from typing import Any, Iterable, List, Optional, Tuple

from models.schemas import (
    DatasetRequirement,
    Measurement,
    PriorityLevel,
    ResolvedInformation,
    ScientificIntentOutput,
    ScientificObjective,
    ScientificVariable,
    VariablePriority,
)

_LOCATION_KEYWORDS = [
    "location", "spatial", "region", "area", "coordinate", "geograph",
    "where",
]
_TEMPORAL_KEYWORDS = [
    "date", "time", "temporal", "period", "season", "baseline", "when",
]

_GOAL_KEYWORDS = ["goal", "research goal", "intent"]
_VARIABLE_KEYWORDS = ["variable", "scientific variable", "parameter"]
_MEASUREMENT_KEYWORDS = ["measurement", "metric", "observation"]
_REQUIREMENT_KEYWORDS = ["dataset requirement", "data requirement", "dataset"]
_OBJECTIVE_KEYWORDS = ["objective", "scientific objective"]
_HAZARD_KEYWORDS = ["hazard", "event", "cyclone", "flood", "drought"]

_REMOVE_WORDS = {"remove", "delete", "drop", "exclude", "omit"}

_VAGUE_MARKERS = {
    "", "unknown", "not specified", "n/a", "none", "not provided",
    "global",
}

_RICH_PLAN_ATTRS = (
    "updated_scientific_plan",
    "final_scientific_plan",
    "reconciled_scientific_plan",
    "approved_scientific_plan",
    "scientific_plan",
)


def bucket_resolved_information(
    resolved: List[ResolvedInformation],
) -> Tuple[
    List[ResolvedInformation],
    List[ResolvedInformation],
    List[ResolvedInformation],
]:
    location, temporal, other = [], [], []

    for item in resolved:
        text = _entry_text(item)
        if any(k in text for k in _LOCATION_KEYWORDS):
            location.append(item)
        elif any(k in text for k in _TEMPORAL_KEYWORDS):
            temporal.append(item)
        else:
            other.append(item)
    return location, temporal, other


def _entry_text(item: Any) -> str:
    return (
        f"{getattr(item, 'field_name', '')} "
        f"{getattr(item, 'resolved_value', '')}"
    ).lower()


def _entry_value(item: Any) -> str:
    return str(getattr(item, "resolved_value", "") or "").strip()


def _has_any(text: str, keywords: Iterable[str]) -> bool:
    text = text.lower()
    return any(k in text for k in keywords)


def _is_vague(value: Any) -> bool:
    return str(value or "").strip().lower() in _VAGUE_MARKERS


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def _short(value: str, max_len: int = 50) -> str:
    value = (value or "").strip()
    return value[:max_len] if len(value) > max_len else value


def _model_key(value: Any) -> str:
    if hasattr(value, "model_dump_json"):
        return value.model_dump_json()
    return repr(value)


def _iter_resolved_information(source: Any) -> List[ResolvedInformation]:
    if source is None:
        return []

    if isinstance(source, list):
        return source

    resolved = getattr(source, "resolved_information", None)
    if isinstance(resolved, list):
        return resolved

    refined = getattr(source, "refined_scientific_plan", None)
    resolved = getattr(refined, "resolved_information", None)
    if isinstance(resolved, list):
        return resolved

    return []


def _extract_rich_plan(source: Any) -> Optional[ScientificIntentOutput]:
    """
    Future-compatible extractor.

    Current callers usually pass a list of ResolvedInformation, but if a
    newer Agent 2 object carries a fully refined ScientificIntentOutput,
    prefer that structured plan over reconstructing from keyword buckets.
    The historical agent1_plan_preserved field is intentionally ignored:
    it is the older Agent 1 snapshot, not accepted user modifications.
    """
    if isinstance(source, ScientificIntentOutput):
        return source

    refined = getattr(source, "refined_scientific_plan", None)
    for container in (source, refined):
        if container is None:
            continue
        for attr in _RICH_PLAN_ATTRS:
            candidate = getattr(container, attr, None)
            if isinstance(candidate, ScientificIntentOutput):
                return candidate

    return None


def _field_has_signal(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return not _is_vague(value)
    if isinstance(value, list):
        return len(value) > 0
    return True


def _deep_copy_value(value: Any) -> Any:
    if hasattr(value, "model_copy"):
        return value.model_copy(deep=True)
    if isinstance(value, list):
        return [
            item.model_copy(deep=True) if hasattr(item, "model_copy") else item
            for item in value
        ]
    return value


def _overlay_rich_plan(
    reconciled: ScientificIntentOutput,
    rich_plan: Optional[ScientificIntentOutput],
) -> ScientificIntentOutput:
    if rich_plan is None:
        return reconciled

    updated = reconciled.model_copy(deep=True)

    for field_name in ScientificIntentOutput.model_fields:
        if field_name == "query":
            continue

        incoming = getattr(rich_plan, field_name, None)
        if _field_has_signal(incoming):
            setattr(updated, field_name, _deep_copy_value(incoming))

    return updated


def _extract_target_after_action(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""

    if "-" in value:
        value = value.split("-", 1)[1].strip()
    if ":" in value:
        value = value.split(":", 1)[1].strip()

    pattern = (
        r"\b(add|include|use|consider|remove|delete|drop|exclude|omit|"
        r"change|modify|update|replace|set)\b"
        r"\s+(?:the\s+|an?\s+|existing\s+|scientific\s+|new\s+)?"
    )
    value = re.sub(pattern, "", value, count=1, flags=re.IGNORECASE).strip()

    generic = {
        "variable", "variables", "measurement", "measurements",
        "objective", "objectives", "dataset", "requirement",
        "requirements", "goal", "location", "time period",
    }
    return "" if _normalise_name(value) in generic else value


def _append_unique_constraint(
    reconciled: ScientificIntentOutput,
    value: str,
) -> None:
    value = (value or "").strip()
    if not value:
        return

    note = f"User clarification/modification preserved: {value}"
    existing = {_normalise_name(item) for item in reconciled.critical_constraints}
    if _normalise_name(note) not in existing:
        reconciled.critical_constraints.append(note)


def _remove_matching_variables(
    reconciled: ScientificIntentOutput,
    target: str,
) -> bool:
    target_key = _normalise_name(target)
    if not target_key:
        return False

    kept = []
    removed = False
    for variable in reconciled.scientific_variables:
        variable_key = _normalise_name(variable.variable)
        if variable_key and (
            variable_key == target_key
            or target_key in variable_key
            or variable_key in target_key
        ):
            removed = True
            continue
        kept.append(variable)

    # Keep schema validity: ScientificIntentOutput requires at least one
    # scientific variable.
    if not removed or not kept:
        return False

    reconciled.scientific_variables = kept
    reconciled.variable_priorities = [
        priority
        for priority in reconciled.variable_priorities
        if not (
            (priority_key := _normalise_name(priority.variable))
            and (
                priority_key == target_key
                or target_key in priority_key
                or priority_key in target_key
            )
        )
    ]
    reconciled.measurements = [
        measurement
        for measurement in reconciled.measurements
        if not (
            target_key in _normalise_name(measurement.variable_measured)
            or target_key in _normalise_name(measurement.measurement_name)
        )
    ] or reconciled.measurements
    return True


def _upsert_variable(
    reconciled: ScientificIntentOutput,
    value: str,
) -> bool:
    value = _extract_target_after_action(value)
    if not value:
        return False

    value_key = _normalise_name(value)
    for existing in reconciled.scientific_variables:
        if _normalise_name(existing.variable) == value_key:
            return True

    new_var = ScientificVariable(
        variable=_short(value),
        scientific_meaning=f"User modified: {value}",
        relevance="Provided by user via clarification",
        priority=PriorityLevel.high,
    )

    for i, variable in enumerate(reconciled.scientific_variables):
        if _is_vague(variable.variable):
            reconciled.scientific_variables[i] = new_var
            break
    else:
        reconciled.scientific_variables.append(new_var)

    priority_keys = {
        _normalise_name(priority.variable)
        for priority in reconciled.variable_priorities
    }
    if value_key not in priority_keys:
        reconciled.variable_priorities.append(
            VariablePriority(
                variable=new_var.variable,
                priority=PriorityLevel.high,
                justification="Provided by user via clarification",
            )
        )
    return True


def _upsert_measurement(
    reconciled: ScientificIntentOutput,
    value: str,
) -> bool:
    value = _extract_target_after_action(value)
    if not value:
        return False

    value_key = _normalise_name(value)
    for existing in reconciled.measurements:
        if _normalise_name(existing.measurement_name) == value_key:
            return True

    new_meas = Measurement(
        measurement_name=_short(value),
        variable_measured="User specified",
        why_measurement_is_needed=f"User modified: {value}",
    )

    for i, measurement in enumerate(reconciled.measurements):
        if _is_vague(measurement.measurement_name):
            reconciled.measurements[i] = new_meas
            break
    else:
        reconciled.measurements.append(new_meas)
    return True


def _upsert_dataset_requirement(
    reconciled: ScientificIntentOutput,
    value: str,
) -> bool:
    value = _extract_target_after_action(value)
    if not value:
        return False

    value_key = _normalise_name(value)
    for existing in reconciled.dataset_requirements:
        if _normalise_name(existing.requirement_name) == value_key:
            return True

    new_req = DatasetRequirement(
        requirement_name=_short(value),
        dataset_type="User specified",
        variables_or_measurements_needed=["User specified"],
        spatial_coverage_needed=reconciled.spatial_context.location,
        temporal_coverage_needed=reconciled.temporal_context.date_range,
        metadata_needed=["Standard"],
        quality_constraints=["Standard"],
        why_required=f"User modified: {value}",
    )

    for i, requirement in enumerate(reconciled.dataset_requirements):
        if _is_vague(requirement.requirement_name):
            reconciled.dataset_requirements[i] = new_req
            break
    else:
        reconciled.dataset_requirements.append(new_req)
    return True


def _append_objective(
    reconciled: ScientificIntentOutput,
    value: str,
) -> bool:
    value = _extract_target_after_action(value)
    if not value:
        return False

    value_key = _normalise_name(value)
    for existing in reconciled.scientific_objectives:
        if _normalise_name(existing.objective) == value_key:
            return True

    reconciled.scientific_objectives.append(
        ScientificObjective(
            objective=value,
            rationale="Provided by user via clarification",
            priority=PriorityLevel.high,
        )
    )
    return True


def _apply_resolved_item(
    reconciled: ScientificIntentOutput,
    item: ResolvedInformation,
) -> None:
    name = str(getattr(item, "field_name", "") or "").lower()
    val = _entry_value(item)
    if not val:
        return

    item_text = f"{name} {val}".lower()
    target = _extract_target_after_action(val)

    if _has_any(name, _GOAL_KEYWORDS):
        reconciled.inferred_user_research_goal = val
    elif _has_any(name, _VARIABLE_KEYWORDS):
        if any(word in item_text for word in _REMOVE_WORDS):
            if not _remove_matching_variables(reconciled, target or val):
                _append_unique_constraint(reconciled, val)
        elif not _upsert_variable(reconciled, val):
            _append_unique_constraint(reconciled, val)
    elif _has_any(name, _MEASUREMENT_KEYWORDS):
        if not _upsert_measurement(reconciled, val):
            _append_unique_constraint(reconciled, val)
    elif _has_any(name, _REQUIREMENT_KEYWORDS):
        if not _upsert_dataset_requirement(reconciled, val):
            _append_unique_constraint(reconciled, val)
    elif _has_any(name, _OBJECTIVE_KEYWORDS):
        if not _append_objective(reconciled, val):
            _append_unique_constraint(reconciled, val)
    elif _has_any(name, _HAZARD_KEYWORDS):
        reconciled.hazard_mechanism_reasoning = val
    else:
        _append_unique_constraint(reconciled, val)


def _merge_unique_models(existing: List[Any], incoming: List[Any]) -> List[Any]:
    seen = {_model_key(item) for item in existing}
    merged = list(existing)
    for item in incoming:
        key = _model_key(item)
        if key not in seen:
            merged.append(item)
            seen.add(key)
    return merged


def _update_requirement_scope(
    reconciled: ScientificIntentOutput,
    location_items: List[ResolvedInformation],
    temporal_items: List[ResolvedInformation],
) -> None:
    if not location_items and not temporal_items:
        return

    updated_requirements = []
    for req in reconciled.dataset_requirements:
        updated = req.model_copy(deep=True)
        if (
            location_items
            and _is_vague(updated.spatial_coverage_needed)
        ):
            updated.spatial_coverage_needed = _entry_value(location_items[-1])
        if (
            temporal_items
            and _is_vague(updated.temporal_coverage_needed)
        ):
            updated.temporal_coverage_needed = _entry_value(temporal_items[-1])
        updated_requirements.append(updated)
    reconciled.dataset_requirements = updated_requirements


def reconcile_agent1_plan(
    agent1_plan: ScientificIntentOutput,
    resolved_information: List[ResolvedInformation],
) -> ScientificIntentOutput:
    """
    Return the effective plan Agent 3 should receive.

    The second argument remains backward compatible with the original
    List[ResolvedInformation] contract, but may also be an Agent 2
    output/refined object. If a richer structured plan is available, it
    is copied first; resolved location/time and other accepted entries
    are then applied deterministically on top.
    """
    resolved_items = _iter_resolved_information(resolved_information)
    location_items, temporal_items, other_items = bucket_resolved_information(
        resolved_items
    )

    reconciled = agent1_plan.model_copy(deep=True)
    reconciled = _overlay_rich_plan(
        reconciled,
        _extract_rich_plan(resolved_information),
    )

    if location_items:
        resolved_location_value = _entry_value(location_items[-1])
        if resolved_location_value:
            reconciled.spatial_context.location = resolved_location_value
            reconciled.spatial_context.geographic_extent = resolved_location_value
            if _is_vague(reconciled.spatial_context.coordinate_requirements):
                reconciled.spatial_context.coordinate_requirements = (
                    "Derived from user-specified area: "
                    f"{resolved_location_value}"
                )

    if temporal_items:
        resolved_temporal_value = _entry_value(temporal_items[-1])
        if resolved_temporal_value:
            reconciled.temporal_context.date_range = resolved_temporal_value

    for item in other_items:
        _apply_resolved_item(reconciled, item)

    # If a rich plan and resolved entries both supplied list updates,
    # keep both sets rather than letting either path silently erase the
    # other. Duplicate Pydantic objects are ignored deterministically.
    rich_plan = _extract_rich_plan(resolved_information)
    if rich_plan is not None:
        reconciled.scientific_objectives = _merge_unique_models(
            reconciled.scientific_objectives,
            rich_plan.scientific_objectives,
        )
        reconciled.scientific_variables = _merge_unique_models(
            reconciled.scientific_variables,
            rich_plan.scientific_variables,
        )
        reconciled.variable_priorities = _merge_unique_models(
            reconciled.variable_priorities,
            rich_plan.variable_priorities,
        )
        reconciled.measurements = _merge_unique_models(
            reconciled.measurements,
            rich_plan.measurements,
        )
        reconciled.dataset_requirements = _merge_unique_models(
            reconciled.dataset_requirements,
            rich_plan.dataset_requirements,
        )
        reconciled.critical_constraints = list(dict.fromkeys(
            reconciled.critical_constraints + rich_plan.critical_constraints
        ))

    _update_requirement_scope(reconciled, location_items, temporal_items)

    return reconciled
