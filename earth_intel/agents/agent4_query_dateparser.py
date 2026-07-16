"""
Agent 4 — Natural Language -> Temporal Resolution (Problem 9).

WHAT THIS IS
  A small, self-contained preprocessing helper. Its only job is:
  given the free-text time fields already present on the
  RetrievalRequest (`temporal_requirements["date_range"]` /
  `temporal_requirements["historical_baseline"]`), resolve them into a
  normalized (ISO start date, ISO end date) tuple that connectors
  supporting server-side subsetting can use.

  Nothing here talks to connectors or touches any other agent's logic.
  It is called exactly once per run, from `run_agent4()`, before the
  coverage/access loop starts (see agent4_orchestrator.py).

WHY A DEDICATED PARSER INSTEAD OF A NEW DEPENDENCY
  requirements.txt does not include a natural-language date library
  (e.g. dateparser). Rather than widen the dependency surface for a
  bounded set of expression shapes, this module implements the
  required patterns directly with the stdlib `re`/`datetime`/
  `calendar`. It's easy to extend with more patterns later without
  adding a new package.

SUPPORTED EXPRESSIONS (per spec)
  - "2015 to 2020" / "2015-2020" / "between 2015 and 2020"
  - "January 2024" / "Jan 2024"
  - "last month"
  - "last five years" / "last 5 years"
  - "summer 2022" (meteorological summer, hemisphere-agnostic default:
    Jun-Aug -- see _SEASON_MONTHS note below)
  - "monsoon 2021" (South Asian monsoon convention: Jun-Sep)
  - Plain single years ("2020"), ISO dates, and ISO date ranges are
    also supported since they're common and cheap to handle.

FALLBACK BEHAVIOR (per spec: "if geocoding fails, preserve the
existing fallback behavior" -- the same principle applies here)
  `parse_date_range()` returns None on any failure (empty input,
  unrecognized expression, malformed date). The orchestrator already
  treats a None time_range as "no specific date range -- sources use
  their full time extent" (see agent4_orchestrator.run_agent4). This
  module never raises, so that fallback path is always preserved.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta
from typing import Optional, Tuple

DateRange = Tuple[str, str]  # (ISO start date, ISO end date)

_MONTH_NAMES = {
    name.lower(): index
    for index, name in enumerate(calendar.month_name) if name
}
_MONTH_ABBR = {
    name.lower(): index
    for index, name in enumerate(calendar.month_abbr) if name
}

# Meteorological seasons, Northern Hemisphere convention. Most
# scientific data-retrieval queries phrased in English default to this
# convention (e.g. "summer 2022" almost always means Jun-Aug in
# climate/earth-science contexts); flip this table if the caller's
# domain is known to be Southern Hemisphere.
_SEASON_MONTHS = {
    "spring": (3, 5),
    "summer": (6, 8),
    "autumn": (9, 11),
    "fall": (9, 11),
    "winter": (12, 2),  # wraps into the next year
}

# South Asian monsoon convention (June-September), the standard usage
# of the bare word "monsoon" in scientific/retrieval contexts.
_MONSOON_MONTHS = (6, 9)


def _iso(d: date) -> str:
    return d.isoformat()


def _last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _month_range(year: int, month: int) -> DateRange:
    start = date(year, month, 1)
    end = date(year, month, _last_day_of_month(year, month))
    return _iso(start), _iso(end)


def _year_range(year: int) -> DateRange:
    return _iso(date(year, 1, 1)), _iso(date(year, 12, 31))


def _season_range(year: int, season: str) -> Optional[DateRange]:
    months = _SEASON_MONTHS.get(season)
    if not months:
        return None
    start_month, end_month = months
    if season == "winter":
        # Dec (year) -> Feb (year+1)
        start = date(year, 12, 1)
        end_year = year + 1
        end = date(end_year, 2, _last_day_of_month(end_year, 2))
    else:
        start = date(year, start_month, 1)
        end = date(year, end_month, _last_day_of_month(year, end_month))
    return _iso(start), _iso(end)


def _monsoon_range(year: int) -> DateRange:
    start_month, end_month = _MONSOON_MONTHS
    start = date(year, start_month, 1)
    end = date(year, end_month, _last_day_of_month(year, end_month))
    return _iso(start), _iso(end)


def _today() -> date:
    return date.today()


def _add_months(d: date, months: int) -> date:
    month_index = d.month - 1 + months
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    day = min(d.day, _last_day_of_month(year, month))
    return date(year, month, day)


_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_YEAR_RE = re.compile(r"^\d{4}$")
_YEAR_RANGE_RE = re.compile(r"(\d{4})\s*(?:to|-|through|until|–|—)\s*(\d{4})")
_ISO_RANGE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\s*(?:to|-|through|until|–|—)\s*(\d{4}-\d{2}-\d{2})")
_BETWEEN_YEARS_RE = re.compile(r"between\s+(\d{4})\s+and\s+(\d{4})")
_MONTH_YEAR_RE = re.compile(
    r"\b(" + "|".join(sorted(list(_MONTH_NAMES) + list(_MONTH_ABBR), key=len, reverse=True)) + r")\.?\s+(\d{4})\b"
)
_SEASON_YEAR_RE = re.compile(
    r"\b(spring|summer|autumn|fall|winter)\s+(\d{4})\b"
)
_MONSOON_YEAR_RE = re.compile(r"\bmonsoon\s+(\d{4})\b")
_LAST_N_YEARS_RE = re.compile(
    r"\blast\s+(\d+|" + "|".join(_WORD_NUMBERS) + r")\s+years?\b"
)
_LAST_N_MONTHS_RE = re.compile(
    r"\blast\s+(\d+|" + "|".join(_WORD_NUMBERS) + r")\s+months?\b"
)


def _resolve_month_token(token: str) -> Optional[int]:
    token = token.lower()
    return _MONTH_NAMES.get(token) or _MONTH_ABBR.get(token)


def _parse_relative(text: str) -> Optional[DateRange]:
    text = text.strip().lower()
    today = _today()

    if text in ("last month", "previous month", "past month"):
        first_of_this_month = today.replace(day=1)
        last_month_end = first_of_this_month - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        return _iso(last_month_start), _iso(last_month_end)

    if text in ("last year", "previous year", "past year"):
        return _year_range(today.year - 1)

    if text in ("last week", "previous week", "past week"):
        end = today - timedelta(days=1)
        start = end - timedelta(days=6)
        return _iso(start), _iso(end)

    if text in ("today",):
        return _iso(today), _iso(today)

    if text in ("yesterday",):
        y = today - timedelta(days=1)
        return _iso(y), _iso(y)

    match = _LAST_N_YEARS_RE.search(text)
    if match:
        raw = match.group(1)
        n = int(raw) if raw.isdigit() else _WORD_NUMBERS.get(raw, 0)
        if n > 0:
            start = _add_months(today, -12 * n) + timedelta(days=1)
            return _iso(start), _iso(today)

    match = _LAST_N_MONTHS_RE.search(text)
    if match:
        raw = match.group(1)
        n = int(raw) if raw.isdigit() else _WORD_NUMBERS.get(raw, 0)
        if n > 0:
            start = _add_months(today, -n)
            return _iso(start), _iso(today)

    return None


def _parse_single_expression(text: str) -> Optional[DateRange]:
    """Attempts every supported pattern against one text field."""
    if not text:
        return None

    raw = text.strip()
    lowered = raw.lower()

    # ISO range: "2015-01-01 to 2020-12-31"
    match = _ISO_RANGE_RE.search(raw)
    if match:
        return match.group(1), match.group(2)

    # Plain ISO date: use as both start and end
    if _ISO_DATE_RE.match(raw):
        return raw, raw

    # "between 2015 and 2020"
    match = _BETWEEN_YEARS_RE.search(lowered)
    if match:
        start_year, end_year = int(match.group(1)), int(match.group(2))
        return _year_range(start_year)[0], _year_range(end_year)[1]

    # "2015 to 2020" / "2015-2020" / "2015 through 2020"
    match = _YEAR_RANGE_RE.search(lowered)
    if match:
        start_year, end_year = int(match.group(1)), int(match.group(2))
        return _year_range(start_year)[0], _year_range(end_year)[1]

    # "monsoon 2021"
    match = _MONSOON_YEAR_RE.search(lowered)
    if match:
        return _monsoon_range(int(match.group(1)))

    # "summer 2022", "winter 2021", etc.
    match = _SEASON_YEAR_RE.search(lowered)
    if match:
        season, year = match.group(1), int(match.group(2))
        result = _season_range(year, season)
        if result:
            return result

    # "January 2024" / "Jan 2024"
    match = _MONTH_YEAR_RE.search(lowered)
    if match:
        month = _resolve_month_token(match.group(1))
        year = int(match.group(2))
        if month:
            return _month_range(year, month)

    # Relative expressions: "last month", "last five years", etc.
    relative = _parse_relative(lowered)
    if relative:
        return relative

    # Plain single year: "2020"
    if _YEAR_RE.match(raw):
        return _year_range(int(raw))

    return None


def parse_date_range(date_range: str, historical_baseline: str = "") -> Optional[DateRange]:
    """
    Resolves a free-text time expression into a normalized
    (ISO start date, ISO end date) tuple.

    Tries `date_range` first, falling back to `historical_baseline` if
    the former is empty or unparseable, since Agent 1/2's temporal
    context sometimes populates only one of the two fields.

    Returns None if both are empty or neither could be parsed --
    callers (agent4_orchestrator.run_agent4) already treat None as
    "use each source's full time extent," which is the existing
    fallback behavior this function must preserve.
    """
    for candidate in (date_range, historical_baseline):
        if not candidate or not candidate.strip():
            continue
        try:
            result = _parse_single_expression(candidate)
        except Exception:
            # Any parsing failure must degrade to "no time range,"
            # never propagate -- same non-fatal contract as the
            # geoparser.
            result = None
        if result:
            return result

    return None
