"""
Agent 4 — Query Geoparser.

RetrievalRequest only carries free-text location strings ("chennai",
"bay of bengal", etc.) -- but connectors that support server-side
subsetting (the whole point of the "least storage" design) need an
actual numeric bounding box: (min_lon, min_lat, max_lon, max_lat).

This module bridges that gap using Nominatim (OpenStreetMap's free
geocoding API, no key required). It NEVER fabricates coordinates: if
geocoding fails or the text is too ambiguous, it returns None and the
caller (agent4_orchestrator) falls back to full-download, which is
called out explicitly rather than silently subsetting against a wrong
or made-up region.

Nominatim usage policy: max ~1 request/second, and a descriptive
User-Agent is required (not optional) -- both are respected here.
"""

import time
from typing import Optional, Tuple

import requests

BoundingBox = Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_USER_AGENT = "earth-intel-agent4/1.0 (dataset retrieval geocoding)"
_MIN_SECONDS_BETWEEN_REQUESTS = 1.1  # Nominatim policy: max 1 req/sec, small margin added

_cache: dict = {}
_last_request_time = 0.0


def _rate_limit():
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_SECONDS_BETWEEN_REQUESTS:
        time.sleep(_MIN_SECONDS_BETWEEN_REQUESTS - elapsed)
    _last_request_time = time.time()


def geocode_location(location_text: str, geographic_extent_text: str = "") -> Optional[BoundingBox]:
    """
    Returns a bounding box for the given free-text location, or None if
    it can't be resolved. Prefers `geographic_extent_text` if given
    (usually more specific), falls back to `location_text`.

    Results are cached in-memory for the process lifetime -- the same
    query text won't be re-geocoded twice in one run.
    """
    query = (geographic_extent_text or location_text or "").strip()
    if not query or query.lower() in ("global", "worldwide", "unspecified", ""):
        return None  # global coverage needs no bounding box at all

    cache_key = query.lower()
    if cache_key in _cache:
        return _cache[cache_key]

    _rate_limit()
    try:
        resp = requests.get(
            _NOMINATIM_URL,
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": _USER_AGENT},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json()
        if not results:
            print(f"[Geoparser] No geocoding match for '{query}' -- subsetting will fall back to full download for this region.")
            _cache[cache_key] = None
            return None

        bbox_raw = results[0].get("boundingbox")  # Nominatim returns [south, north, west, east] as strings
        if not bbox_raw or len(bbox_raw) != 4:
            _cache[cache_key] = None
            return None

        south, north, west, east = (float(x) for x in bbox_raw)
        bbox = (west, south, east, north)
        _cache[cache_key] = bbox
        return bbox

    except Exception as exc:
        print(f"[Geoparser] Geocoding failed for '{query}' (non-fatal): {exc} -- falling back to full download for this region.")
        _cache[cache_key] = None
        return None
