"""
Agent 5 LLM prompts.

Only two things in Agent 5's pipeline genuinely need LLM judgment
rather than deterministic code:
  1. Deciding WHICH preprocessing steps this specific request needs,
     and in what order (Step 3 of the system prompt: "never a fixed
     pipeline").
  2. Deciding whether a detected data-quality problem is survivable
     (handle automatically) or objective-breaking (stop and report) --
     Step 2/4's "Autonomous Decision Making" section.

Everything else in the system prompt (unit conversion, CRS transforms,
resampling, interpolation, merging, statistics) is mechanical once the
plan says to do it, and is implemented directly in agent5.py with
xarray/pandas/rioxarray -- no LLM call needed, and no LLM call wanted,
since burning a model call on arithmetic that's already fully
specified would be slower, costlier, and less reproducible than doing
it in code.
"""

SYSTEM_PROMPT = """You are the planning/judgment component of Agent 5, \
an Intelligent Scientific Data Preparation & Analysis Agent.

You are NEVER asking the user anything. The user has already chosen to \
have their data preprocessed -- your only job is to decide, given the \
data profile and validation results you're shown, which preprocessing \
steps are scientifically appropriate for THIS request, and whether any \
detected issue is severe enough that the requested scientific objective \
cannot be reliably achieved.

You must respond with ONLY a JSON object, no preamble, no markdown \
fences, matching exactly this shape:

{
  "objective_achievable": true | false,
  "stop_reason": "<string, only meaningful if objective_achievable is false>",
  "preprocessing_steps": [
    {
      "step": "<one of: missing_value_handling, duplicate_removal, \
invalid_value_filtering, unit_conversion, coordinate_normalization, \
crs_transformation, variable_standardization, time_alignment, \
spatial_alignment, resampling, interpolation, dataset_merging, \
feature_extraction, derived_variable_computation>",
      "applies_to_source_ids": ["<source_id>", ...],
      "rationale": "<why this step is needed for this request>",
      "method": "<the specific scientifically appropriate method to use>"
    }
  ],
  "plan_rationale": "<2-3 sentence summary of the overall plan>"
}

RULES:
- Only include a step if the data profile/validation actually shows a \
need for it. Do not include steps "just in case".
- Order steps in the list in the order they should execute (e.g. \
invalid_value_filtering and missing_value_handling before resampling; \
unit_conversion/coordinate_normalization before dataset_merging; \
dataset_merging only when more than one source is involved and the \
user's objective requires combining them).
- objective_achievable is false ONLY for: required variables entirely \
absent with no scientifically defensible way to derive them, \
completely corrupted data, unsupported formats, irrecoverable \
metadata, or insufficient data volume for the requested analysis. \
Minor/moderate imperfections (some missing values, some outliers, \
unit mismatches, resolution differences) are NEVER a reason to set \
this false -- those are handled automatically via the steps above.
- Never invent a source_id that wasn't shown to you.
"""


def build_planning_prompt(
    scientific_goal: str,
    expected_scientific_outputs: list,
    required_data_fusion: list,
    dataset_profiles: list,   # List[DatasetProfile], already .model_dump()-able
    validation_results: list,  # List[DatasetValidationResult]
) -> str:
    import json

    payload = {
        "scientific_goal": scientific_goal,
        "expected_scientific_outputs": expected_scientific_outputs,
        "required_data_fusion": required_data_fusion,
        "dataset_profiles": [p.model_dump() for p in dataset_profiles],
        "validation_results": [v.model_dump() for v in validation_results],
    }
    return (
        "Here is the data profile and validation output for this "
        "request. Decide the preprocessing plan per your instructions.\n\n"
        + json.dumps(payload, indent=2, default=str)
    )


def parse_planning_response(raw_text: str) -> dict:
    """
    Parses the LLM's JSON response. Raises ValueError with a clear
    message on malformed output rather than silently returning a
    partial/empty plan -- a silently-empty plan here would look
    identical to "no preprocessing needed", which is a dangerous
    failure mode to hide.
    """
    import json

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Agent 5 planning LLM did not return valid JSON: {exc}\n"
            f"Raw response: {raw_text[:500]}"
        )

    required_keys = {"objective_achievable", "preprocessing_steps", "plan_rationale"}
    missing = required_keys - data.keys()
    if missing:
        raise ValueError(
            f"Agent 5 planning LLM response missing required key(s): {missing}"
        )

    return data
