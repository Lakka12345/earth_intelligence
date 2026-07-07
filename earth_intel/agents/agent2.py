import json
import re as _re
import time
import os

from groq import Groq
from pydantic import BaseModel, ValidationError

from config.settings import MAX_RETRIES
from prompts.agent2_prompt import build_agent2_system_prompt

from models.schemas import (
    ScientificIntentOutput,
    ClarificationAgentOutput,
    UserResponse,
    ClarificationRound,
)
from security.hallucination_guard import (
    validate_agent2_consistency,
    HallucinationException,
)


def pydantic_schema_for_anthropic(model_class: type[BaseModel]) -> dict:
    schema = model_class.model_json_schema()

    def clean_schema(obj):
        if isinstance(obj, dict):
            obj.pop("title", None)

            for value in obj.values():
                clean_schema(value)

        elif isinstance(obj, list):

            for item in obj:
                clean_schema(item)

    clean_schema(schema)

    return schema


def _strip_agent1_plan_preserved_schema(schema: dict) -> dict:
    """
    ClarificationAgentOutput.refined_scientific_plan.agent1_plan_preserved
    is typed as the FULL nested ScientificIntentOutput model (see
    schemas.py). Because run_agent2()'s _inject_python_derived_fields()
    ALWAYS overwrites this field with the real Python-held value BEFORE
    final pydantic validation -- unconditionally, regardless of what the
    LLM returned for it -- the LLM never needs to see, understand, or
    satisfy this field's schema at all.

    Without this stripping, json.dumps(AGENT2_SCHEMA) pulls in ~15 extra
    nested model definitions purely as unused schema text: ResearchQuestion,
    ScientificObjective, ScientificVariable, DatasetRequirement,
    PlanningStep (which alone requires >=23 items), SpatialContext,
    TemporalContext, etc. This gets sent on EVERY Agent 2 call, every
    round -- it was very likely the single largest contributor to the
    413 "request too large" errors, bigger than the round-1 full-plan
    context block.

    This only changes what we SHOW the model in the prompt. The real
    ClarificationAgentOutput (with the true, complete agent1_plan_preserved)
    is still what gets validated after Python injects it -- nothing about
    correctness changes, only prompt size.
    """
    defs = schema.get("$defs", {})

    if "RefinedScientificPlan" in defs:
        rsp_props = defs["RefinedScientificPlan"].get("properties", {})
        if "agent1_plan_preserved" in rsp_props:
            rsp_props["agent1_plan_preserved"] = {
                "type": "object",
                "description": (
                    "DO NOT GENERATE THIS FIELD YOURSELF. It is filled "
                    "in automatically by the system after your response "
                    "is received. Omit it entirely, or leave it as an "
                    "empty object -- either is fine, it will be replaced."
                ),
            }

    # Drop $defs that are only reachable through ScientificIntentOutput
    # and are now orphaned after the rewrite above, to shrink the schema
    # further. Safe: nothing else in ClarificationAgentOutput's own tree
    # references any of these (verified against schemas.py).
    orphaned_defs = {
        "ScientificIntentOutput", "ResearchQuestion", "DecisionContext",
        "ScientificObjective", "ScientificVariable", "VariablePriority",
        "DomainConfidence", "CrossDomainDependency", "VariableDependency",
        "SpatialContext", "TemporalContext", "Measurement",
        "DatasetRequirement", "PlanningStep", "UncertaintyItem",
        "ClarificationQuestion", "ReasoningSummary", "UserIntentType",
        "AmbiguityLevel", "RetrievalReadiness", "PriorityLevel",
        "DomainName", "EventType",
    }

    for key in orphaned_defs:
        defs.pop(key, None)

    schema["$defs"] = defs

    return schema


AGENT2_SCHEMA = _strip_agent1_plan_preserved_schema(
    pydantic_schema_for_anthropic(
        ClarificationAgentOutput
    )
)


def normalize_agent1_plan(agent1_output) -> ScientificIntentOutput:
    """
    Accepts either an already-validated ScientificIntentOutput instance
    or a raw dict (e.g. loaded from JSON) and returns a validated
    ScientificIntentOutput.
    """

    if isinstance(agent1_output, ScientificIntentOutput):
        return agent1_output

    if isinstance(agent1_output, dict):
        return ScientificIntentOutput.model_validate(agent1_output)

    raise TypeError(
        "agent1_output must be a ScientificIntentOutput "
        "instance or a dict, got "
        f"{type(agent1_output).__name__}."
    )


def normalize_user_responses(user_responses) -> list[dict]:
    """
    Accepts None, a list of dicts, or a list of UserResponse instances.
    Returns a list of plain dicts suitable for JSON serialization.
    """

    if not user_responses:
        return []

    normalized = []

    for item in user_responses:

        if isinstance(item, UserResponse):
            normalized.append(item.model_dump())

        elif isinstance(item, dict):
            normalized.append(
                UserResponse.model_validate(item).model_dump()
            )

        else:
            raise TypeError(
                "Each user response must be a UserResponse "
                f"instance or a dict, got {type(item).__name__}."
            )

    return normalized


def normalize_clarification_history(clarification_history) -> list[dict]:
    """
    Accepts None, a list of dicts, or a list of ClarificationRound
    instances. Returns a list of plain dicts suitable for JSON
    serialization.
    """

    if not clarification_history:
        return []

    normalized = []

    for item in clarification_history:

        if isinstance(item, ClarificationRound):
            normalized.append(item.model_dump())

        elif isinstance(item, dict):
            normalized.append(
                ClarificationRound.model_validate(item).model_dump()
            )

        else:
            raise TypeError(
                "Each clarification round must be a "
                "ClarificationRound instance or a dict, got "
                f"{type(item).__name__}."
            )

    return normalized


def extract_json_from_text(text: str) -> dict:
    text = text.strip()

    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    import re

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        raise ValueError("No JSON object found in model response.")

    return json.loads(match.group(0))


def _fix_clarification_question_dict(question_dict: dict) -> dict:
    """
    Fixes a single Agent2ClarificationQuestion-shaped dict in place
    (returns the same dict, mutated).

    PROBLEM: the LLM sometimes emits the key "importance" instead of
    "priority" for objects that must validate against
    Agent2ClarificationQuestion (used in prioritized_questions and
    clarification_history[].questions_asked). Since "priority" is a
    required field with no default, pydantic raises a "field required"
    ValidationError before Agent2ClarificationQuestion.normalize_priority
    ever runs -- that validator only fires once a value for "priority"
    already exists in the input, so it cannot rescue a missing key.

    FIX SCOPE: this function must ONLY be applied to dicts we already
    know are headed for Agent2ClarificationQuestion validation
    (prioritized_questions / clarification_history[].questions_asked).
    It must NEVER be applied broadly across the whole parsed payload,
    because "importance" is also a real, correct field name on Agent 1's
    own ResearchQuestion and ClarificationQuestion models (preserved
    verbatim inside refined_scientific_plan.agent1_plan_preserved). A
    blind global rename would corrupt Agent 1's untouched data.

    Does not touch the dict at all if "priority" is already present
    (the common case) -- this is a targeted repair, not a rewrite.
    """
    if not isinstance(question_dict, dict):
        return question_dict

    if "priority" not in question_dict and "importance" in question_dict:
        question_dict["priority"] = question_dict.pop("importance")
    else:
        # "priority" already present (or neither key present, which will
        # surface as a normal, legitimate "field required" validation
        # error -- we must not paper over that case).
        question_dict.pop("importance", None)

    return question_dict


_LOCATION_CATCHALL_TEXT = "None of the above - type your own answer"


def _fix_question_options(question_dict: dict) -> dict:
    """
    Cleans and hardens the "options" list on a single
    Agent2ClarificationQuestion-shaped dict:

      - Drops any leading numbering the LLM might still emit inside an
        option string (e.g. "1. Daily" -> "Daily"), since numbering is
        the UI's job (main.py), not the model's.
      - Guarantees that whenever options are present, the LAST one is
        always a free-text catch-all ("None of the above - type your
        own answer"). If the LLM forgot it, or phrased it differently,
        we append our own rather than trusting the LLM to remember —
        this is what makes the "always has a type-your-own-answer
        option" behavior reliable instead of best-effort.

    Leaves an empty/missing options list alone (schema default is []).
    """
    if not isinstance(question_dict, dict):
        return question_dict

    options = question_dict.get("options")
    if not isinstance(options, list):
        options = []

    cleaned = []
    for opt in options:
        if not isinstance(opt, str):
            continue
        stripped = _re.sub(r"^\s*\d+[\.\)]\s*", "", opt).strip()
        if stripped:
            cleaned.append(stripped)

    if cleaned and not any(
        "none of the above" in c.lower() for c in cleaned
    ):
        cleaned.append(_LOCATION_CATCHALL_TEXT)

    question_dict["options"] = cleaned

    return question_dict


def _build_location_question_dict() -> dict:
    """
    The deterministic, Python-owned version of the area/location
    clarification question. We do not rely on the LLM to remember to
    ask this in the right shape every time -- we build it ourselves and
    inject it whenever Agent 1's spatial_context shows no usable area,
    so the behavior is guaranteed rather than best-effort.
    """
    return {
        "question": (
            "Which area would you like to study? You can either give us "
            "exact coordinates, or just tell us the place name and we'll "
            "work out the location for you."
        ),
        "reason": (
            "No clear study area (place name or coordinates) was found "
            "in the query, and data cannot be retrieved without one."
        ),
        "priority": "critical",
        "resolves_gaps": ["study_area"],
        "options": [
            "I know the exact latitude and longitude - I'll type it in",
            "I'll type the name of the place, region, or coastline instead",
            _LOCATION_CATCHALL_TEXT,
        ],
        "is_location_question": True,
    }


def _location_is_missing(agent1_plan: ScientificIntentOutput) -> bool:
    """
    True when Agent 1's spatial_context gives us neither a usable place
    name nor usable coordinates -- i.e. we genuinely cannot locate the
    study area on a map yet.
    """
    vague_markers = {
        "", "unknown", "not specified", "n/a", "none", "not provided",
    }

    spatial = agent1_plan.spatial_context

    location = (spatial.location or "").strip().lower()
    coords = (spatial.coordinate_requirements or "").strip().lower()

    return location in vague_markers and coords in vague_markers


def _ensure_location_question_present(
    parsed: dict,
    agent1_plan: ScientificIntentOutput,
    round_number: int,
) -> dict:
    """
    Guarantees the area/location clarification question is present as
    the FIRST prioritized question whenever the study area is missing --
    regardless of whether the LLM remembered to ask it this time.

    Only applies on round 1 (before the user has answered anything).
    Later rounds rely on the merged "ALREADY RESOLVED" block in the
    prompt to know the area question was already answered.
    """
    if round_number != 1:
        return parsed

    if not _location_is_missing(agent1_plan):
        return parsed

    questions = parsed.get("prioritized_questions")
    if not isinstance(questions, list):
        questions = []

    already_present = any(
        isinstance(q, dict) and q.get("is_location_question")
        for q in questions
    )

    if already_present:
        return parsed

    questions = [_build_location_question_dict()] + questions

    # Hard cap: never exceed 6 questions total, even after injection.
    parsed["prioritized_questions"] = questions[:6]

    parsed["clarification_needed"] = True
    parsed["retrieval_readiness"] = "clarification_required"

    if isinstance(parsed.get("refined_scientific_plan"), dict):
        parsed["refined_scientific_plan"]["retrieval_readiness"] = (
            "clarification_required"
        )

    critical_gaps = parsed.get("critical_gaps")
    if not isinstance(critical_gaps, list):
        critical_gaps = []

    has_area_gap = any(
        isinstance(g, dict)
        and g.get("gap_name", "").lower() == "study_area"
        for g in critical_gaps
    )

    if not has_area_gap:
        critical_gaps.append(
            {
                "gap_name": "study_area",
                "description": (
                    "No clear study area (place name or coordinates) "
                    "was provided."
                ),
                "severity": "critical",
                "blocks_retrieval": True,
            }
        )

    parsed["critical_gaps"] = critical_gaps

    return parsed


def _time_period_is_missing(agent1_plan: ScientificIntentOutput) -> bool:
    """
    True when Agent 1's temporal_context gives us no concrete date
    range, event window, or resolvable time reference -- i.e. we
    genuinely don't know WHEN to pull data for yet.
    """
    vague_markers = {
        "", "unknown", "not specified", "n/a", "none", "not provided",
    }

    temporal = agent1_plan.temporal_context

    date_range = (temporal.date_range or "").strip().lower()
    event_window = (temporal.event_window or "").strip().lower()

    return date_range in vague_markers and event_window in vague_markers


def _build_time_period_question_dict() -> dict:
    """
    The deterministic, Python-owned version of the time-period
    clarification question. Mirrors _build_location_question_dict --
    we do not trust the LLM to remember this every time, and critically
    we never let a vague category like "a range of years" stand in as
    the final answer; main.py always follows up for a concrete value
    (see _ask_time_period_question).
    """
    return {
        "question": (
            "What time period would you like to study? You can give an "
            "exact date range, a single year or season, or name a "
            "specific event."
        ),
        "reason": (
            "No concrete time period was found in the query, and data "
            "cannot be retrieved without one."
        ),
        "priority": "critical",
        "resolves_gaps": ["time_period"],
        "options": [
            "A specific date range - I'll give the start and end dates",
            "A single year or season",
            "A specific event or occurrence (e.g. a named storm or flood)",
            _LOCATION_CATCHALL_TEXT,
        ],
        "is_time_period_question": True,
    }


def _ensure_time_period_question_present(
    parsed: dict,
    agent1_plan: ScientificIntentOutput,
    round_number: int,
) -> dict:
    """
    Guarantees the time-period clarification question is present
    (immediately after the location question, if any) whenever no
    concrete time period exists yet -- regardless of whether the LLM
    remembered to ask it. Mirrors _ensure_location_question_present.

    Only applies on round 1, for the same reason as the location
    version: later rounds rely on the "ALREADY RESOLVED" prompt block.
    """
    if round_number != 1:
        return parsed

    if not _time_period_is_missing(agent1_plan):
        return parsed

    questions = parsed.get("prioritized_questions")
    if not isinstance(questions, list):
        questions = []

    already_present = any(
        isinstance(q, dict) and q.get("is_time_period_question")
        for q in questions
    )

    if already_present:
        return parsed

    time_question = _build_time_period_question_dict()

    # Insert right after the location-related question if one is
    # present -- either the "which area" question (is_location_question,
    # fires when area is missing) or the "entire vs specific part"
    # scope question (is_scope_question, fires when area was given but
    # is broad). These two never co-occur (see
    # _ensure_scope_question_present), but time-period injection must
    # still land after WHICHEVER one is present, not just the first
    # one -- otherwise WHEN can end up asked before WHERE is settled.
    location_idx = next(
        (
            i for i, q in enumerate(questions)
            if isinstance(q, dict)
            and (q.get("is_location_question") or q.get("is_scope_question"))
        ),
        None,
    )

    if location_idx is not None:
        questions = (
            questions[: location_idx + 1]
            + [time_question]
            + questions[location_idx + 1 :]
        )
    else:
        questions = [time_question] + questions

    # Hard cap: never exceed 6 questions total, even after injection.
    # The area and time period questions always take priority over any
    # other question the LLM proposed.
    parsed["prioritized_questions"] = questions[:6]

    parsed["clarification_needed"] = True
    parsed["retrieval_readiness"] = "clarification_required"

    if isinstance(parsed.get("refined_scientific_plan"), dict):
        parsed["refined_scientific_plan"]["retrieval_readiness"] = (
            "clarification_required"
        )

    critical_gaps = parsed.get("critical_gaps")
    if not isinstance(critical_gaps, list):
        critical_gaps = []

    has_time_gap = any(
        isinstance(g, dict)
        and g.get("gap_name", "").lower() == "time_period"
        for g in critical_gaps
    )

    if not has_time_gap:
        critical_gaps.append(
            {
                "gap_name": "time_period",
                "description": (
                    "No concrete time period (date range, year, season, "
                    "or named event) was provided."
                ),
                "severity": "critical",
                "blocks_retrieval": True,
            }
        )

    parsed["critical_gaps"] = critical_gaps

    return parsed


_BROAD_LOCATION_MARKERS = [
    # Seas / oceans / gulfs -- naming one of these alone is never a
    # retrievable area on its own.
    "bay of bengal",
    "arabian sea",
    "indian ocean",
    "pacific ocean",
    "atlantic ocean",
    "south china sea",
    "mediterranean sea",
    "red sea",
    "persian gulf",
    "gulf of mexico",
    "north sea",
    "caribbean sea",
    # Whole-country / whole-coastline / global phrasing.
    "world",
    "global",
    "globally",
    "entire coastline",
    "whole coastline",
    "entire coast",
    "whole coast",
    "all of india",
    "pan india",
    "pan-india",
    "throughout india",
    "across india",
]


def is_broad_location_text(text: str) -> bool:
    """
    Public helper: True if the given free-text location string matches
    a known broad-region marker (ocean, sea, gulf, whole-country
    phrasing, etc.).

    This is deliberately exposed at the plain-text level (not just via
    _location_is_broad(agent1_plan) below) because broad regions can
    show up in TWO different places:

      1. In Agent 1's extracted spatial_context.location -- caught by
         _ensure_scope_question_present() during round 1.
      2. In the user's own ANSWER to the "which area" clarification
         question (e.g. they type "Arabian Sea" when asked to name a
         place). This answer never passes back through Agent 1, and
         because the clarification loop is hard-capped at one round,
         there is no further LLM round in which Agent 2 could catch it
         and ask again. So main.py must catch this case itself, in the
         same turn, using this same marker list -- see
         _maybe_narrow_broad_place() in main.py.
    """
    normalized = (text or "").strip().lower()

    if not normalized:
        return False

    return any(marker in normalized for marker in _BROAD_LOCATION_MARKERS)


def _location_is_broad(agent1_plan: ScientificIntentOutput) -> bool:
    """
    True when a study area WAS stated in Agent 1's output, but matches
    one of the known broad-region markers (a sea, ocean, whole
    country/coastline, etc.) -- meaning it's stated but not yet scoped.
    This is a deliberately narrow, high-confidence heuristic list;
    broader cases the LLM might catch (e.g. "all of Tamil Nadu") are
    handled via the Step 3c prompt instruction instead of hardcoded
    here, since guessing at every possible broad phrasing in Python
    would be too brittle.
    """
    spatial = agent1_plan.spatial_context

    return is_broad_location_text(spatial.location)


def _build_scope_question_dict(location_text: str) -> dict:
    """
    The deterministic, Python-owned version of the scope-narrowing
    question, used when the heuristic list above catches a known broad
    region. Mirrors the location/time-period question builders.
    """
    return {
        "question": (
            f'You mentioned "{location_text}" - do you want data for '
            "the entire area, or a specific part of it?"
        ),
        "reason": (
            f'"{location_text}" is a very large area, and pulling data '
            "for all of it could be far more than you actually need."
        ),
        "priority": "critical",
        "resolves_gaps": ["study_area_scope"],
        "options": [
            f"The entire {location_text}",
            "Just a specific country's coastline/waters within it (I'll name it)",
            "A specific city, district, or shorter stretch (I'll name it)",
            _LOCATION_CATCHALL_TEXT,
        ],
        "is_scope_question": True,
    }


def _ensure_scope_question_present(
    parsed: dict,
    agent1_plan: ScientificIntentOutput,
    round_number: int,
) -> dict:
    """
    Guarantees the scope-narrowing question is present whenever the
    stated study area matches a known broad-region marker (see
    _BROAD_LOCATION_MARKERS) -- regardless of whether the LLM remembered
    to ask it. Only relevant when the location is NOT missing (that
    case is handled by _ensure_location_question_present instead); a
    broad-but-present location and a missing location never fire
    together.

    Inserted right after the location question (if present) and before
    the time period question, matching the WHERE -> HOW MUCH OF WHERE ->
    WHEN priority order described in the prompt.

    Only applies on round 1, for the same reason as the other
    deterministic questions.
    """
    if round_number != 1:
        return parsed

    if _location_is_missing(agent1_plan):
        # No area was given at all -- that's the missing-location case,
        # not the broad-but-stated case. Nothing to narrow yet.
        return parsed

    if not _location_is_broad(agent1_plan):
        return parsed

    questions = parsed.get("prioritized_questions")
    if not isinstance(questions, list):
        questions = []

    already_present = any(
        isinstance(q, dict) and q.get("is_scope_question")
        for q in questions
    )

    if already_present:
        return parsed

    location_text = (agent1_plan.spatial_context.location or "").strip()
    scope_question = _build_scope_question_dict(location_text)

    location_idx = next(
        (
            i for i, q in enumerate(questions)
            if isinstance(q, dict) and q.get("is_location_question")
        ),
        None,
    )

    if location_idx is not None:
        questions = (
            questions[: location_idx + 1]
            + [scope_question]
            + questions[location_idx + 1 :]
        )
    else:
        questions = [scope_question] + questions

    # Hard cap: never exceed 6 questions total, even after injection.
    parsed["prioritized_questions"] = questions[:6]

    parsed["clarification_needed"] = True
    parsed["retrieval_readiness"] = "clarification_required"

    if isinstance(parsed.get("refined_scientific_plan"), dict):
        parsed["refined_scientific_plan"]["retrieval_readiness"] = (
            "clarification_required"
        )

    critical_gaps = parsed.get("critical_gaps")
    if not isinstance(critical_gaps, list):
        critical_gaps = []

    has_scope_gap = any(
        isinstance(g, dict)
        and g.get("gap_name", "").lower() == "study_area_scope"
        for g in critical_gaps
    )

    if not has_scope_gap:
        critical_gaps.append(
            {
                "gap_name": "study_area_scope",
                "description": (
                    f'"{location_text}" was named but is too broad to '
                    "retrieve sensibly without narrowing."
                ),
                "severity": "critical",
                "blocks_retrieval": True,
            }
        )

    parsed["critical_gaps"] = critical_gaps

    return parsed


def preprocess_agent2_payload(parsed: dict) -> dict:
    """
    Applies scoped, pre-validation fixes to the raw parsed JSON dict
    before ClarificationAgentOutput.model_validate(parsed).

    Currently fixes exactly one known LLM quirk (see
    _fix_clarification_question_dict), applied only at the two known
    locations where Agent2ClarificationQuestion objects appear:
      - parsed["prioritized_questions"]            (List[...])
      - parsed["clarification_history"][i]["questions_asked"]  (List[...])

    Mutates and returns the same dict. Safe to call even if these keys
    are missing or malformed in some other way -- in that case we leave
    the data untouched and let model_validate() raise its own, real
    validation error rather than masking it here.
    """
    prioritized = parsed.get("prioritized_questions")
    if isinstance(prioritized, list):
        for q in prioritized:
            _fix_clarification_question_dict(q)
            _fix_question_options(q)

    history = parsed.get("clarification_history")
    if isinstance(history, list):
        for round_entry in history:
            if not isinstance(round_entry, dict):
                continue
            questions_asked = round_entry.get("questions_asked")
            if isinstance(questions_asked, list):
                for q in questions_asked:
                    _fix_clarification_question_dict(q)
                    _fix_question_options(q)

    return parsed


def _build_merged_resolved_information(
    history: list[dict],
    latest_responses: list[dict],
) -> list[dict]:
    """
    Builds a deduplicated, field-name-keyed list of resolved information
    entries by pairing every question that was asked in history with the
    answer the user gave in the same round.

    Why this is needed
    ------------------
    clarification_history contains ClarificationRound objects, which only
    carry questions_asked + responses_received.  The resolved_information
    list lives on ClarificationAgentOutput, which is NOT threaded back
    into subsequent run_agent2 calls — so the LLM has no direct way to
    see what was already resolved unless we reconstruct it here in Python
    and inject it explicitly into the prompt.

    Strategy
    --------
    Within each past round we pair question[i] with response[i] by
    position (the loop in main.py appends one UserResponse per question
    in the same order).  We use the question text as the key label and
    the user_answer as the resolved value.

    For the latest (current) round responses we use field_name directly
    since they haven't been paired with questions yet — they are passed
    separately and the LLM will handle them as the new input.

    Deduplication: later rounds win.  A dict keyed on a normalised label
    ensures that if the same conceptual field appears in two rounds
    (e.g. the user changed their answer in a revision round), only the
    most recent value is kept.
    """
    merged: dict[str, dict] = {}   # label → resolved entry dict

    for round_dict in history:
        questions = round_dict.get("questions_asked") or []
        responses = round_dict.get("responses_received") or []

        if questions:
            # Standard clarification round: pair question[i] with
            # response[i] by position.
            for i, q in enumerate(questions):
                if i >= len(responses):
                    break
                answer = responses[i].get("user_answer", "").strip()
                if not answer:
                    continue

                # Use the question text as the field_name (required by
                # ResolvedInformation schema). Truncate to keep the
                # injected block compact.
                label = q.get("question", f"question_{i}")[:80]
                normalised_key = label.lower().strip()

                merged[normalised_key] = {
                    "field_name": label,          # required by ResolvedInformation schema
                    "resolved_value": answer,
                    "resolution_source": "user",
                    "round": round_dict.get("round_number", "?"),
                }
        else:
            # "Modify the understanding" round: main.py's
            # _run_confirmation_and_revision_loop always appends
            # ClarificationRound(questions_asked=[], responses_received=[...])
            # for these, since a modification is free text, not an
            # answer to a specific prioritized question. Without this
            # branch, the loop above never executes (nothing to
            # enumerate), and the modification is silently dropped from
            # every LATER round's context -- which is exactly what
            # caused a second "Modify" request to behave as if the
            # first one never happened: Agent 2 genuinely had no memory
            # of it once it aged out of latest_round_block (which only
            # ever shows the single most recent round).
            for i, r in enumerate(responses):
                answer = r.get("user_answer", "").strip()
                if not answer:
                    continue

                raw_field_name = r.get("field_name") or f"modification_{i}"
                round_num = round_dict.get("round_number", "?")
                # Distinct label per round so successive modifications
                # don't overwrite each other here -- reconciliation and
                # the LLM both need to see the FULL sequence of changes,
                # not just the latest one, in case an earlier
                # modification touched a different topic than the
                # latest one.
                normalised_key = f"{raw_field_name.lower().strip()}_{round_num}"

                merged[normalised_key] = {
                    "field_name": (
                        f"User-requested change (round {round_num})"
                    ),
                    "resolved_value": answer,
                    "resolution_source": "user",
                    "round": round_num,
                }

    # Also surface the latest round's raw responses so nothing is lost
    # if the caller passes them both in history AND in latest_responses.
    # We do NOT add them again here — they are already in the prompt
    # under "Optional user clarification responses (latest round)" and
    # the LLM will process them as new input for this round.

    return list(merged.values())


def _extract_agent1_compact_context(plan: ScientificIntentOutput) -> dict:
    """
    Returns only the fields from Agent 1's ScientificIntentOutput that
    Agent 2 strictly needs, stripping out empty and null values to save tokens.
    """
    d = plan.model_dump(exclude_none=True)
    rsp = d.get("refined_scientific_plan") or {}

    compact = {
        "user_query": d.get("user_query"),
        "research_goal": d.get("research_goal"),
        "scientific_objectives": d.get("scientific_objectives"),
        "scientific_variables": d.get("scientific_variables"),
        "spatial_context": rsp.get("spatial_context") or d.get("spatial_context"),
        "temporal_context": rsp.get("temporal_context") or d.get("temporal_context"),
    }
    
    return {k: v for k, v in compact.items() if v}


def _inject_python_derived_fields(
    parsed: dict,
    agent1_plan: ScientificIntentOutput,
    merged_resolved: list[dict],
    round_number: int,
) -> dict:
    """
    Populates fields in the LLM's parsed output that must come from
    authoritative Python state rather than from LLM generation.

    This is called AFTER preprocess_agent2_payload() and BEFORE
    ClarificationAgentOutput.model_validate(), so Pydantic sees a
    complete, schema-valid dict without the LLM ever having to
    regenerate large objects.

    Fields injected
    ---------------
    refined_scientific_plan.agent1_plan_preserved
        Required by the schema (ScientificIntentOutput).  We always own
        the original agent1_plan in Python, so there is zero reason to
        ask the LLM to reconstruct it — doing so wastes tokens AND risks
        hallucinated drift from the original.  We write it directly.

    refined_scientific_plan.updated_missing_information
        If the LLM omitted this field (common when the prompt is compact),
        we fall back to the merged_resolved list that was computed in
        Python from the full clarification history.  This ensures the
        field is never empty due to context truncation.

    Nothing else is touched — all other LLM-generated fields are left
    exactly as returned.
    """
    # Ensure the top-level key exists (LLM may have omitted it entirely
    # on early rounds when there is nothing to refine yet).
    if "refined_scientific_plan" not in parsed or not isinstance(
        parsed.get("refined_scientific_plan"), dict
    ):
        parsed["refined_scientific_plan"] = {}

    rsp = parsed["refined_scientific_plan"]

    # ── 1. agent1_plan_preserved ────────────────────────────────────────
    # Always overwrite: the LLM must never be trusted to reproduce this
    # verbatim, and it should never need to — we have the ground truth.
    rsp["agent1_plan_preserved"] = agent1_plan.model_dump()

    # ── 2. updated_missing_information ──────────────────────────────────
    # Use the LLM's value when present and non-empty; otherwise fall back
    # to the Python-computed merged_resolved list so the field is never
    # silently dropped due to prompt truncation.
    if not rsp.get("updated_missing_information") and merged_resolved:
        rsp["updated_missing_information"] = [
            entry.get("field_name", "")
            for entry in merged_resolved
            if entry.get("field_name")
        ]

    return parsed


def _ensure_resolved_information_complete(
    parsed: dict,
    merged_resolved: list[dict],
) -> dict:
    """
    Guarantees every Python-computed merged_resolved entry (location,
    time period, and anything else answered across all rounds) is
    present in the TOP-LEVEL resolved_information list, even if the LLM
    dropped some despite being instructed to carry them over verbatim
    (see the "ALREADY RESOLVED" prompt block).

    This matters beyond display: models/plan_reconciliation.py and
    main.py's retrieval-request build both read
    agent2_result.resolved_information (top-level, the canonical copy
    per understanding_summary.py's own docstring) to figure out what
    the user actually said about location and time period. If an entry
    silently went missing here, reconciliation would silently fall back
    to Agent 1's original ("Global" / "Unknown") values -- exactly the
    bug this function exists to prevent.

    Mutates and returns the same dict. Existing entries are never
    overwritten -- only genuinely missing ones are appended.
    """
    existing = parsed.get("resolved_information")
    if not isinstance(existing, list):
        existing = []

    existing_keys = {
        (item.get("field_name") or "").strip().lower()
        for item in existing
        if isinstance(item, dict)
    }

    for entry in merged_resolved:
        key = (entry.get("field_name") or "").strip().lower()

        if key and key not in existing_keys:
            existing.append(
                {
                    "field_name": entry.get("field_name", ""),
                    "resolved_value": entry.get("resolved_value", ""),
                    "resolution_source": entry.get(
                        "resolution_source", "user"
                    ),
                }
            )
            existing_keys.add(key)

    parsed["resolved_information"] = existing

    return parsed


def run_agent2(
    agent1_output,
    user_responses=None,
    clarification_history=None,
    round_number: int = 1,
) -> ClarificationAgentOutput:

    agent1_plan = normalize_agent1_plan(
        agent1_output
    )

    responses = normalize_user_responses(
        user_responses
    )

    history = normalize_clarification_history(
        clarification_history
    )

    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

    # ------------------------------------------------------------------ #
    # Build a merged view of ALL previously resolved information.         #
    # This is injected into the prompt as a separate, explicit section    #
    # so the LLM cannot "forget" earlier answers even when the history    #
    # JSON is long.  The merge is done in Python — not delegated to the   #
    # LLM — so it is deterministic and cannot be silently dropped.        #
    # ------------------------------------------------------------------ #
    merged_resolved = _build_merged_resolved_information(history, responses)

    merged_resolved_block = (
        json.dumps(merged_resolved, separators=(',', ':'))
        if merged_resolved
        else "[]"
    )

    compact = _extract_agent1_compact_context(agent1_plan)
    agent1_context_block = f"""Research context:
{json.dumps(compact, separators=(',', ':'))}"""

    if history:
        latest_round = history[-1]
        latest_round_block = f"""Latest round ({latest_round.get('round_number', '?')}):
{json.dumps(latest_round, separators=(',', ':'))}"""
    else:
        latest_round_block = "No previous rounds."

    prompt = f"""
{build_agent2_system_prompt()}

════════════════════════════════════════
CURRENT STATE
════════════════════════════════════════
Current clarification round : {round_number}
Rounds remaining            : {max(0, 3 - round_number)} (hard cap = 3)

If current round >= 3 and only non-critical gaps remain:
  → Set retrieval_readiness = "proceed_with_assumptions"
  → Set prioritized_questions = []
  → Do NOT output clarification_required

════════════════════════════════════════
ALREADY RESOLVED
════════════════════════════════════════
{merged_resolved_block}

════════════════════════════════════════

Return only valid JSON matching this schema:
{json.dumps(AGENT2_SCHEMA, separators=(',', ':'))}

{agent1_context_block}

Optional user clarification responses (latest round):
{json.dumps(responses, separators=(',', ':'))}

{latest_round_block}
"""

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
                max_tokens=1500,
                response_format={
                    "type": "json_object"
                },
            )
            raw = (
                response
                .choices[0]
                .message.content
            )

            parsed = extract_json_from_text(
                raw
            )

            parsed = preprocess_agent2_payload(
                parsed
            )

            # ---------------------------------------------------------- #
            # Guarantee the area/location question whenever the study    #
            # area is genuinely missing, regardless of whether the LLM   #
            # remembered to ask it this round.                            #
            # ---------------------------------------------------------- #
            parsed = _ensure_location_question_present(
                parsed,
                agent1_plan,
                round_number,
            )

            parsed = _ensure_scope_question_present(
                parsed,
                agent1_plan,
                round_number,
            )

            parsed = _ensure_time_period_question_present(
                parsed,
                agent1_plan,
                round_number,
            )

            # ---------------------------------------------------------- #
            # Guarantee top-level resolved_information carries every     #
            # answer the user has given so far, even if the LLM dropped  #
            # one -- reconciliation (models/plan_reconciliation.py) and  #
            # the retrieval request both depend on this being complete.  #
            # ---------------------------------------------------------- #
            parsed = _ensure_resolved_information_complete(
                parsed,
                merged_resolved,
            )

            # ---------------------------------------------------------- #
            # Inject authoritative Python-derived fields before           #
            # validation.  The LLM is never asked to regenerate these;    #
            # we own the ground-truth values in Python.                   #
            # ---------------------------------------------------------- #
            parsed = _inject_python_derived_fields(
                parsed,
                agent1_plan,
                merged_resolved,
                round_number,
            )

            print(
                "\n===== AGENT 2 RAW PARSED JSON ====="
            )

            print(
                json.dumps(
                    parsed,
                    indent=2
                )
            )

            print(
                "====================================\n"
            )

            validated_output = (
                ClarificationAgentOutput
                .model_validate(parsed)
            )

            validate_agent2_consistency(
                validated_output
            )

            return validated_output

        except (ValidationError, HallucinationException):
            raise

        except Exception as exc:

            last_error = exc

            wait_time = (
                2 ** (attempt + 1)
            )

            print(
                f"Groq failed "
                f"(attempt {attempt + 1}/{MAX_RETRIES}). "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(
                wait_time
            )

            continue

    raise RuntimeError(
        f"Agent 2 failed after retries: "
        f"{last_error}"
    )