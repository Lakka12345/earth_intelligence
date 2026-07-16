"""
Agent 3 prompt — the single LLM call for scoring candidate sources.

This prompt scores ONLY three parameters per candidate:
  - relevance      (how well does this source match the query?)
  - completeness   (does this source cover what's needed?)
  - consistency    (does this source agree with others?)

Everything else is computed in pure Python.

CHANGED: Strengthened as a SECONDARY signal, per the original spec's
"Alternative Considered" section -- the deterministic Python validators
(discovery/geo_validator.py, temporal_validator.py, platform_validator.py,
wired into Agent 3 Phase 5) are now the PRIMARY mechanism for catching
geographic/temporal/platform mismatches, including a hard-reject override
for confident geographic mismatches that no LLM recommendation can
overturn. This prompt is not relied on alone to reject anything -- it's
strengthened so its own relevance/completeness reasoning stays consistent
with what the deterministic layer will ultimately decide, rather than
contradicting it (e.g. an LLM giving a wrong-region source a high
completeness score, with a justification that reads strangely once
Phase 5's hard-reject overrides the recommendation anyway).

Two concrete additions, both requested in the approved refinements:
  1. An explicit worked example showing a region mismatch must be scored
     low on completeness/relevance REGARDLESS OF variable match quality
     (e.g. "Ireland ERDDAP station for a Bay of Bengal query").
  2. An explicit instruction to weight dataset-type/platform preference
     (satellite vs. in-situ) when the user's request states one --
     mirroring discovery/platform_validator.py's symmetric penalty.
"""

import json
from typing import List, Optional

from models.discovery_schemas import CandidateSource, LLMScoringOutput


def build_scoring_prompt(
    goal: str,
    user_intent_type: str,
    variables_needed: List[str],
    dataset_types_needed: List[str],
    spatial_requirements: dict,
    temporal_requirements: dict,
    candidates: List[CandidateSource],
    relevant_sensors: Optional[List[str]] = None,
) -> str:
    """
    Builds the LLM scoring prompt.

    CHANGED: Compact format to stay within the 12 000 TPM limit.
      - Context block appears ONCE at the top (not repeated per candidate).
      - Each candidate is a tight 6-field summary, not a full JSON dump.
      - Inline JSON schema removed — Groq already enforces JSON output via
        response_format; the schema added ~400 tokens per call with no benefit
        since parse_scoring_response validates the output, not the LLM.
      - Sensor instruction kept but condensed.
    """
    # Location / time — pull the most useful fields only
    location   = spatial_requirements.get("location", "") or spatial_requirements.get("geographic_extent", "") or "unspecified"
    date_range = temporal_requirements.get("date_range", "") or temporal_requirements.get("historical_baseline", "") or "unspecified"

    # Compact candidate list — one entry per candidate, 6 fields only.
    # Description is capped at 120 chars to prevent one verbose provider
    # from inflating the prompt disproportionately.
    candidate_lines = []
    for i, c in enumerate(candidates, 1):
        vars_str  = ", ".join(c.variables_available[:6]) or "unknown"
        desc      = (c.description or "")[:120].replace("\n", " ")
        candidate_lines.append(
            f"{i}. id={c.source_id}\n"
            f"   name={c.name}\n"
            f"   vars={vars_str}\n"
            f"   spatial={c.spatial_coverage}\n"
            f"   temporal={c.temporal_coverage}\n"
            f"   desc={desc}"
        )
    candidates_block = "\n\n".join(candidate_lines)

    sensor_line = ""
    if relevant_sensors:
        sensor_line = (
            f"\nPreferred sensors for these variables: {', '.join(relevant_sensors[:6])}. "
            "Candidates derived from these sensors score higher on relevance — this is a "
            "positive signal, not a hard requirement.\n"
        )

    # Required output format — described inline, no full JSON schema
    output_format = (
        '{"scored_candidates": [{'
        '"source_id": "...", '
        '"relevance_score": 0.0-1.0, "relevance_explanation": "...", '
        '"completeness_score": 0.0-1.0, "completeness_explanation": "...", '
        '"consistency_score": 0.0-1.0, "consistency_explanation": "...", '
        '"recommendation": "use"|"consider", '
        '"selection_justification": "...", '
        '"rejection_reason": "..."|null, '
        '"failed_criteria": ["..."]|[] , '
        '"rejection_confidence": 0.0-1.0|null'
        '}, ...]}'
    )

    return f"""You are the Source Scoring Agent for an Earth Intelligence Platform.
Score each candidate on relevance, completeness, and consistency. Return only JSON.

=== QUERY CONTEXT ===
Goal: {goal}
Intent: {user_intent_type}
Variables needed: {', '.join(variables_needed)}
Dataset types: {', '.join(dataset_types_needed)}
Location: {location}
Time period: {date_range}
{sensor_line}
=== SCORING RULES ===
- relevance: does this source match the scientific goal and variables? (0.0-1.0)
- completeness: does it cover the required variables, region, and time period? (0.0-1.0)
- consistency: does it agree with / complement the other candidates? (0.0-1.0)
- recommendation: "use" (clearly fits) or "consider" (partial fit). Never emit "reject"; deterministic Python decides rejection status.
- Be strict: missing a required variable → completeness < 0.5
- Geographic mismatch should significantly reduce relevance and completeness but should not automatically reject a source. Global or overlapping datasets may still be useful. Only clearly irrelevant geographic coverage should receive very low scores.
  WORKED EXAMPLE: A candidate covering only Irish coastal waters (ERDDAP station data)
  is scored LOW on both relevance (~0.1-0.2) and completeness (~0.1-0.2) for a query about
  Bay of Bengal / Indian Ocean conditions -- REGARDLESS of how well its variables match
  (e.g. even if it has the exact sea_surface_temperature/salinity variables requested).
  Variable match quality never overrides a clear regional mismatch. Apply this same strictness
  whenever a candidate's spatial_coverage names a region that does not overlap the requested
  location -- do not give credit for "the right kind of data in the wrong place."
- Platform preference: if the request implies satellite data, in-situ sources score lower, and vice versa. Only apply when the request states a preference.
- AUTHENTICATION IS NOT A REJECTION CRITERION. A source that requires login, API keys, registration, or restricted download is NOT less relevant. Score it purely on scientific fit. Do NOT recommend "reject" simply because access requires authentication. Score it as "use" or "consider" if scientifically appropriate.
- Missing metadata fields (Unknown spatial coverage, Unknown temporal coverage, etc.) alone do not justify rejection. Score based on available evidence. Lower completeness where metadata is genuinely missing, but do not reject solely for missing fields.
- Score ALL {len(candidates)} candidates. Do not skip any.

=== URL TYPE REQUIREMENT — HARD RULE ===
Every candidate URL passed to Agent 4 MUST point to a machine-readable data endpoint.
NEVER score as "use" or "consider" a candidate whose URL is:
  * A website homepage (e.g. https://gdacs.org/, https://mausam.imd.gov.in/)
  * An interactive web map, dashboard, or data explorer (a page that opens a browser map)
  * An HTML portal or catalog browsing page requiring human interaction to find data
  * A documentation, "about", or dataset landing/description page
  * A download page requiring a human to click a button
ONLY score as usable if the URL is one of:
  * A REST API endpoint returning JSON/CSV/XML/binary directly (e.g. https://api.open-meteo.com/v1/forecast?...)
  * An OPeNDAP / THREDDS / ERDDAP data URL (e.g. ending in .nc, griddap, tabledap)
  * A STAC API endpoint (/search, /collections, /items)
  * A direct file download URL ending in .nc, .csv, .json, .tif, .grib, .h5, .zarr
  * An S3 or cloud-storage direct object URL
If a candidate is a homepage or HTML portal with no direct API endpoint in its metadata,
set completeness_score=0.0, add "Broken Resource" to failed_criteria, and set
rejection_confidence=0.95. Do NOT pass HTML portals to Agent 4 as data sources.


=== DO NOT RECOMMEND "reject" ===
1. Completely unrelated scientific domain (e.g. financial market data, agricultural yield statistics, or any dataset with no scientific connection to {', '.join(variables_needed) or 'the requested variables'}).
2. Duplicate of another candidate in this list (same underlying dataset, same provider, no added value).
3. Broken or inaccessible resource (the description/metadata indicates the link or service is dead, deprecated, or returns no real data).
4. Invalid, empty, or corrupted dataset (metadata indicates there is no actual data, or it is a placeholder/test entry).
5. Obvious spam or a non-scientific website (marketing page, unrelated commercial product, link farm).
A source that is merely a weaker fit (lower resolution, partial variable coverage, less authoritative provider, or in the wrong region/time period — already covered above) should be scored low and marked "consider" or left for the deterministic ranking to handle, NOT "reject". Reject is for sources that are clearly USELESS for this query, not sources that are merely WORSE than another candidate. Never reject a relevant source just because a better one exists in this list.

=== LEGACY NOTE: recommendation "reject" is disabled ===
You must also populate:
- "failed_criteria": a list of the specific criteria this source failed, using ONLY these labels: "Relevance", "Scientific Domain", "Required Variables", "Duplicate", "Broken Resource", "Invalid Dataset", "Spam / Non-Scientific". Include every label that applies (usually 1-2). Leave this as [] for sources that are fine.
- "rejection_confidence": how confident you are this source is genuinely unusable for this query, 0.0-1.0. Use a high value (0.9+) only when the evidence is unambiguous (e.g. a finance dataset for an oceanography query). Use a lower value (0.5-0.7) if there is some doubt. Leave this as null for sources that are fine.
Always set "rejection_reason" to null -- that field is filled in by deterministic Python from your failed_criteria and selection_justification, not by you. Do NOT leave failed_criteria/rejection_confidence empty just because you're not emitting "reject" as the recommendation -- a source can be "consider" with populated failed_criteria if it has real, specific problems worth flagging, even though final rejection is decided by the deterministic layer, not by you.

=== CANDIDATES ({len(candidates)} total) ===
{candidates_block}

=== OUTPUT FORMAT ===
{output_format}

Return only the JSON. No markdown, no prose outside JSON.
"""


def parse_scoring_response(response_text: str) -> LLMScoringOutput:
    """
    Parses and validates the LLM scoring response.
    Raises ValidationError if output doesn't match schema.
    """
    import re
    from pydantic import ValidationError

    text = response_text.strip()

    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON found in LLM scoring response.")
        text = match.group(0)

    parsed = json.loads(text)
    return LLMScoringOutput.model_validate(parsed)
