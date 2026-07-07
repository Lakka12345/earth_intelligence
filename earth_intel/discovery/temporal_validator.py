"""
Temporal Validator — deterministic temporal-overlap check for Agent 3
candidate sources.

Problem this solves:
  Phase 5's _compute_freshness_score only looks at how RECENT a candidate's
  temporal_coverage looks (does it mention 2024/2025/"present"?). Nothing
  ever compares that coverage against what the request actually asked for
  (TemporalContext.date_range / historical_baseline). A dataset that ended
  in 2010 could score well on "freshness-of-the-text" while completely
  failing to cover a 2023 storm-surge analysis window.

Design (small, pure-Python, no date-parsing library dependency):
  - Extracts plausible 4-digit years (1900-2099) from both the request's
    requested date range and the candidate's temporal_coverage string.
  - Builds an approximate [min_year, max_year] span for each side.
    "Present" / "real-time" / "ongoing" extends the candidate's span to
    the current effective year so open-ended live feeds aren't penalized.
  - Compares spans for overlap.

This is intentionally a SOFTER validator than geography (per the original
spec): unknown/unparseable coverage on either side lowers confidence
rather than rejecting, since temporal text in the wild ("varies by
dataset", "see THREDDS catalog") is far less standardized than region
names, and a hard-reject here would have a much higher false-positive
rate than the geographic case.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

ONGOING_TOKENS = ["present", "real-time", "real time", "ongoing", "current", "now"]
UNKNOWN_TOKENS = [
    "unknown", "see thredds catalog", "see stac item geometry",
    "see portal metadata", "from web search", "varies", "varies by dataset",
    "n/a", "",
]

# Effective "current year" ceiling for open-ended ("present") coverage.
# Kept as a constant rather than calling datetime.now() so validator
# behavior is deterministic/testable; bump periodically.
CURRENT_YEAR_CEILING = 2026


@dataclass
class TemporalValidationResult:
    score: float                 # 0.0 - 1.0
    explanation: str


def _normalize(text: Optional[str]) -> str:
    if not text:
        return ""
    return text.lower().strip()


def _extract_years(text: str) -> List[int]:
    return [int(m) for m in re.findall(r"\b(?:19|20)\d{2}\b", text)]


def _is_unknown(text: str) -> bool:
    if text.strip() in UNKNOWN_TOKENS:
        return True
    return any(tok in text for tok in UNKNOWN_TOKENS if tok)


def _is_ongoing(text: str) -> bool:
    return any(tok in text for tok in ONGOING_TOKENS)


def _extract_span(text: str) -> Optional[Tuple[int, int]]:
    """
    Returns an approximate (start_year, end_year) span parsed from free
    text, or None if no years could be found.

    "Present"/"real-time"/"ongoing" anywhere in the text extends the
    upper bound to CURRENT_YEAR_CEILING.
    """
    years = _extract_years(text)
    ongoing = _is_ongoing(text)

    if not years:
        if ongoing:
            # e.g. "real-time" with no year mentioned -- treat as
            # current-only, not a full historical span.
            return (CURRENT_YEAR_CEILING, CURRENT_YEAR_CEILING)
        return None

    start, end = min(years), max(years)
    if ongoing:
        end = max(end, CURRENT_YEAR_CEILING)
    return (start, end)


def _spans_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def validate_temporal(
    requested_date_range,
    requested_historical_baseline,
    candidate_temporal_coverage,
) -> TemporalValidationResult:
    """
    Compares the request's temporal intent against a candidate's
    temporal_coverage string.

    Args:
        requested_date_range: TemporalContext.date_range.
        requested_historical_baseline: TemporalContext.historical_baseline,
            included because a baseline period can extend the relevant
            requested span beyond date_range alone.
        candidate_temporal_coverage: CandidateSource.temporal_coverage.

    Returns:
        TemporalValidationResult with a 0-1 score. No hard_reject flag --
        this validator only ever lowers confidence, per the softer-signal
        design described in the module docstring.
    """
    requested_text = " ".join(
        filter(None, [
            _normalize(requested_date_range),
            _normalize(requested_historical_baseline),
        ])
    )
    candidate_text = _normalize(candidate_temporal_coverage)

    if not requested_text or requested_text in ("unknown", ""):
        return TemporalValidationResult(
            score=0.75,
            explanation="Request has no specific temporal constraint; temporal coverage not evaluated.",
        )

    if _is_unknown(candidate_text) and not _extract_years(candidate_text):
        return TemporalValidationResult(
            score=0.55,
            explanation=(
                "Candidate's temporal coverage is not yet known structurally; "
                "temporal match unverified, treated as neutral."
            ),
        )

    requested_span = _extract_span(requested_text)
    candidate_span = _extract_span(candidate_text)

    if requested_span is None:
        return TemporalValidationResult(
            score=0.55,
            explanation=(
                f"Could not parse a year range from requested period '{requested_text}'; "
                "temporal match unverified."
            ),
        )

    if candidate_span is None:
        return TemporalValidationResult(
            score=0.55,
            explanation=(
                f"Could not parse a year range from candidate coverage '{candidate_text}'; "
                "temporal match unverified."
            ),
        )

    if _spans_overlap(requested_span, candidate_span):
        return TemporalValidationResult(
            score=0.90,
            explanation=(
                f"Candidate coverage {candidate_span} overlaps requested period {requested_span}."
            ),
        )

    return TemporalValidationResult(
        score=0.20,
        explanation=(
            f"Candidate coverage {candidate_span} does not overlap requested period "
            f"{requested_span}. Likely temporal mismatch (soft penalty, not a hard reject)."
        ),
    )
