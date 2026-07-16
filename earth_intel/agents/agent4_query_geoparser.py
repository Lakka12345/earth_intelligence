"""
Agent 4 — Natural Language -> Geographic Resolution (Problem 9).

WHAT THIS IS
  A small, self-contained preprocessing helper. Its only job is:
  given the free-text location fields already present on the
  RetrievalRequest (`spatial_requirements["location"]` /
  `spatial_requirements["geographic_extent"]`), resolve them into a
  bounding box that connectors supporting server-side subsetting can
  use.

  Nothing here talks to connectors, changes any request/response
  models, or touches Agent 3's discovery/ranking logic. It is called
  exactly once per run, from `run_agent4()`, before the coverage/access
  loop starts (see agent4_orchestrator.py).

WHY NOMINATIM (OpenStreetMap)
  - No API key / registration required (respects a single required
    User-Agent header per OSM's usage policy).
  - Covers cities, states, countries, oceans, seas, rivers, and named
    landmarks -- all listed in the Problem 9 requirements -- because it
    indexes OSM's full place/waterway/natural feature dataset, not just
    administrative boundaries.
  - Returns a ready-made bounding box (`boundingbox`) for every result,
    so no separate polygon-to-bbox conversion step is needed for the
    common case. This keeps the implementation small and avoids adding
    a new heavy geospatial dependency to requirements.txt.

FALLBACK BEHAVIOR (per spec: "if geocoding fails, preserve the
existing fallback behavior")
  `geocode_location()` returns None on any failure (network error,
  no match, empty input, timeout). The orchestrator already treats a
  None bounding_box as "no specific region -- sources use their full
  spatial extent" (see agent4_orchestrator.run_agent4). This module
  never raises, so that fallback path is always preserved.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

import requests

BoundingBox = Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_USER_AGENT = "Agent4-RetrievalPreprocessor/1.0 (scientific data retrieval assistant)"
_TIMEOUT_SECONDS = 8

# In-process cache so repeated calls for the same place within one run
# (e.g. re-planning after a declined source) don't re-hit the network.
_geocode_cache: dict[str, Optional[BoundingBox]] = {}

# A handful of well-known large-scale features that are frequently
# requested in scientific queries but are sometimes ambiguous or
# poorly ranked by a plain Nominatim free-text search (e.g. "Pacific"
# alone can return small unrelated places). Hard-coded as a
# first-pass lookup; falls through to Nominatim for anything else.
_KNOWN_REGIONS: dict[str, BoundingBox] = {
    "pacific ocean": (-180.0, -60.0, 180.0, 66.0),
    "atlantic ocean": (-80.0, -60.0, 20.0, 70.0),
    "indian ocean": (20.0, -60.0, 147.0, 30.0),
    "arctic ocean": (-180.0, 66.0, 180.0, 90.0),
    "southern ocean": (-180.0, -90.0, 180.0, -60.0),
    "global": (-180.0, -90.0, 180.0, 90.0),
    "worldwide": (-180.0, -90.0, 180.0, 90.0),
    "world": (-180.0, -90.0, 180.0, 90.0),
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _query_nominatim(place: str) -> Optional[BoundingBox]:
    try:
        response = requests.get(
            _NOMINATIM_URL,
            params={
                "q": place,
                "format": "json",
                "limit": 1,
                "polygon_geojson": 0,
            },
            headers={"User-Agent": _USER_AGENT},
            timeout=_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        results = response.json()
        if not results:
            return None

        top = results[0]
        bbox = top.get("boundingbox")  # [south_lat, north_lat, west_lon, east_lon] as strings
        if not bbox or len(bbox) != 4:
            return None

        south_lat, north_lat, west_lon, east_lon = (float(v) for v in bbox)
        return (west_lon, south_lat, east_lon, north_lat)
    except Exception:
        # Network failure, timeout, malformed response, bad JSON, etc.
        # -- always fall back to "no bounding box" rather than raising.
        return None


def geocode_location(location: str, geographic_extent: str = "") -> Optional[BoundingBox]:
    """
    Resolves a free-text location (city, state, country, ocean, river,
    landmark, etc.) into a bounding box: (min_lon, min_lat, max_lon, max_lat).

    Tries `location` first, falling back to `geographic_extent` if the
    former is empty or fails to resolve, since Agent 1/2's spatial
    context sometimes populates only one of the two fields.

    Returns None if both are empty or neither could be resolved --
    callers (agent4_orchestrator.run_agent4) already treat None as
    "use each source's full spatial extent," which is the existing
    fallback behavior this function must preserve.
    """
    for candidate in (location, geographic_extent):
        normalized = _normalize(candidate)
        if not normalized:
            continue

        if normalized in _geocode_cache:
            cached = _geocode_cache[normalized]
            if cached is not None:
                return cached
            continue

        if normalized in _KNOWN_REGIONS:
            bbox = _KNOWN_REGIONS[normalized]
            _geocode_cache[normalized] = bbox
            return bbox

        bbox = _query_nominatim(candidate.strip())
        _geocode_cache[normalized] = bbox
        if bbox is not None:
            return bbox

    return None
