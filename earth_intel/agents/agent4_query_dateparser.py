"""
Agent 4 — Query Dateparser.

RetrievalRequest carries free-text temporal strings ("last 5 years",
"2015-2020", "since 2010", "present"). Connectors need actual ISO
start/end dates to build a subsetting query. This module handles the
common patterns and returns None (never a guessed date) for anything
it can't confidently parse -- caller falls back to full download.

Deliberately dependency-free (no dateutil requirement) -- year/month
arithmetic here is approximate (30-day months, 365-day years), which
is precise enough for choosing a subsetting window and is documented
as such rather than presented as exact calendar arithmetic.
"""

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple

DateRange = Tuple[str, str]  # (start_iso, end_iso)


def _today() -> datetime:
    return datetime.utcnow()


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def parse_date_range(date_range_text: str, historical_baseline_text: str = "") -> Optional[DateRange]:
    """
    Tries, in order:
      1. Explicit ISO "YYYY-MM-DD to YYYY-MM-DD"
      2. Explicit "YYYY-YYYY" / "YYYY to YYYY"
      3. Relative "last/past N years|months|days|decades"
      4. "since YYYY"
    Falls back to historical_baseline_text with the same patterns if
    date_range_text doesn't match anything. Returns None -- not a
    guess -- if nothing matches.
    """
    for text in (date_range_text, historical_baseline_text):
        if not text:
            continue
        text = text.strip().lower()

        result = (
            _try_explicit_iso_range(text)
            or _try_explicit_year_range(text)
            or _try_relative_range(text)
            or _try_since_year(text)
        )
        if result:
            return result

    return None


def _try_explicit_iso_range(text: str) -> Optional[DateRange]:
    match = re.search(r"(\d{4}-\d{2}-\d{2})\s*(?:to|-|–|through)\s*(\d{4}-\d{2}-\d{2})", text)
    if match:
        return match.group(1), match.group(2)
    return None


def _try_explicit_year_range(text: str) -> Optional[DateRange]:
    match = re.search(r"\b(\d{4})\s*(?:to|-|–|through)\s*(\d{4})\b", text)
    if match:
        start_year, end_year = match.group(1), match.group(2)
        return f"{start_year}-01-01", f"{end_year}-12-31"
    return None


def _try_relative_range(text: str) -> Optional[DateRange]:
    match = re.search(r"\b(?:last|past)\s+(\d+)?\s*(year|month|day|decade)s?\b", text)
    if not match:
        return None
    n = int(match.group(1)) if match.group(1) else 1
    unit = match.group(2)
    now = _today()

    if unit == "year":
        start = now - timedelta(days=365 * n)
    elif unit == "decade":
        start = now - timedelta(days=3650 * n)
    elif unit == "month":
        start = now - timedelta(days=30 * n)
    else:  # day
        start = now - timedelta(days=n)

    return _iso(start), _iso(now)


def _try_since_year(text: str) -> Optional[DateRange]:
    match = re.search(r"\bsince\s+(\d{4})\b", text)
    if match:
        return f"{match.group(1)}-01-01", _iso(_today())
    return None
