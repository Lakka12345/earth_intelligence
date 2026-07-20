"""
components/map.py
Optional PyDeck map showing spatial coverage of discovered datasets.
Only renders when at least one source has parseable bbox coverage.
"""

from __future__ import annotations
import os
import re
import time
import streamlit as st
from functools import lru_cache

# ---------------------------------------------------------------------------
# Known ocean/hemisphere regions that Nominatim won't return useful bboxes
# for — keep these hardcoded since they aren't addressable places.
# Cities, countries, rivers, bays: resolved via Nominatim at runtime.
# ---------------------------------------------------------------------------
_FIXED_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "global":         (-180.0, -90.0,  180.0,  90.0),
    "worldwide":      (-180.0, -90.0,  180.0,  90.0),
    "indian ocean":   ( 20.0,  -60.0,  120.0,  30.0),
    "pacific ocean":  (120.0,  -60.0,  280.0,  60.0),
    "atlantic ocean": (-80.0,  -60.0,   20.0,  60.0),
    "southern ocean": (-180.0, -90.0,  180.0, -60.0),
    "arctic ocean":   (-180.0,  60.0,  180.0,  90.0),
}

# Nominatim rate-limit: max 1 request/second per OSM policy.
_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_LAST_NOMINATIM_CALL: list[float] = [0.0]   # mutable container for closure


@lru_cache(maxsize=256)
def _geocode(place: str) -> tuple[float, float, float, float] | None:
    """
    Resolve a place name to a bounding box via Nominatim (OSM).
    Returns (min_lon, min_lat, max_lon, max_lat) or None on failure.
    Results are cached in-process so repeated queries for the same
    place don't hit the network.
    """
    import urllib.request
    import urllib.parse
    import json

    # Respect Nominatim's 1 req/s policy
    elapsed = time.monotonic() - _LAST_NOMINATIM_CALL[0]
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)

    params = urllib.parse.urlencode({
        "q": place,
        "format": "json",
        "limit": 1,
        "addressdetails": 0,
    })
    url = f"{_NOMINATIM_URL}?{params}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "EarthIntelligencePlatform/1.0 (research tool)"},
    )
    try:
        _LAST_NOMINATIM_CALL[0] = time.monotonic()
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return None

    if not data:
        return None

    bb = data[0].get("boundingbox")  # [min_lat, max_lat, min_lon, max_lon]
    if bb and len(bb) == 4:
        try:
            min_lat, max_lat, min_lon, max_lon = (float(x) for x in bb)
            return (min_lon, min_lat, max_lon, max_lat)
        except (ValueError, TypeError):
            return None
    return None


def _parse_bbox(spatial_coverage: str) -> tuple[float, float, float, float] | None:
    """
    Extract a bounding box from spatial_coverage strings.

    Resolution order:
      1. Fixed ocean/global regions (hardcoded — Nominatim won't help here)
      2. Explicit bbox format:  "bbox [-180.0,-90.0,180.0,90.0]"
      3. Lat/lon range format:  "lat [5.0 to 23.0], lon [80.0 to 100.0]"
      4. Nominatim geocoding:   any named place — city, country, bay, etc.
    """
    text = (spatial_coverage or "").strip()
    if not text:
        return None
    lower = text.lower()

    # 1. Fixed ocean/global regions
    for name in sorted(_FIXED_REGIONS, key=len, reverse=True):
        if name in lower:
            return _FIXED_REGIONS[name]

    # 2. bbox [minlon, minlat, maxlon, maxlat]
    m = re.search(
        r"bbox\s*\[?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)",
        lower,
    )
    if m:
        return (float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))

    # 3. lat [min to max], lon [min to max]
    lat = re.search(r"lat\s*\[?\s*(-?\d+\.?\d*)\s*(?:to|–|-)\s*(-?\d+\.?\d*)", lower)
    lon = re.search(r"lon\s*\[?\s*(-?\d+\.?\d*)\s*(?:to|–|-)\s*(-?\d+\.?\d*)", lower)
    if lat and lon:
        return (float(lon.group(1)), float(lat.group(1)), float(lon.group(2)), float(lat.group(2)))

    # 4. Nominatim geocoding — works for any named place
    return _geocode(text)


def _bbox_area_deg2(min_lon, min_lat, max_lon, max_lat) -> float:
    return abs(max_lon - min_lon) * abs(max_lat - min_lat)


def _zoom_for_bbox(min_lon, min_lat, max_lon, max_lat) -> int:
    span = max(abs(max_lon - min_lon), abs(max_lat - min_lat))
    if span < 0.5:
        return 11
    if span < 1:
        return 10
    if span < 3:
        return 8
    if span < 10:
        return 6
    if span < 30:
        return 5
    if span < 90:
        return 4
    return 2


def _score_color(final_score: float, alpha: int = 120) -> list[int]:
    r = int(255 * (1.0 - final_score))
    g = int(255 * final_score)
    return [r, g, 50, alpha]


def render_map(result, agent4_result=None) -> None:
    """Render a PyDeck map of accepted dataset coverage areas if coords exist.

    When agent4_result carries a resolved_bounding_box (the geocoded query
    region — e.g. the bounding box for "Chennai"), that box is drawn as a
    distinct highlight layer and used to centre/zoom the map.  Dataset
    coverage polygons are rendered as faint context behind it.  Without
    agent4_result the old behaviour is preserved.
    """
    try:
        import pydeck as pdk
    except ImportError:
        st.caption("Install pydeck to enable the coverage map: `pip install pydeck`")
        return

    # ── Resolved query bounding box from Agent 4 (the actual place asked about) ──
    query_bbox: tuple[float, float, float, float] | None = (
        agent4_result.resolved_bounding_box
        if agent4_result and getattr(agent4_result, "resolved_bounding_box", None)
        else None
    )

    all_sources = result.ranked_sources + getattr(result, "auth_required_sources", [])

    skipped = []
    raw_polygons: list[tuple[float, dict]] = []

    for src in all_sources:
        c = src.candidate
        bbox = _parse_bbox(c.spatial_coverage)
        if bbox is None:
            skipped.append(c.name)
            continue

        min_lon, min_lat, max_lon, max_lat = bbox
        area = _bbox_area_deg2(min_lon, min_lat, max_lon, max_lat)
        raw_polygons.append((area, {
            "name": c.name,
            "score": round(src.final_score * 100, 1),
            "spatial_coverage": c.spatial_coverage or "—",
            "coordinates": [[
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat],
            ]],
            "_bbox": (min_lon, min_lat, max_lon, max_lat),
            "_score": src.final_score,
        }))

    if not raw_polygons and query_bbox is None:
        st.caption(
            "No parseable spatial bounds found — map not displayed. "
            f"Datasets without recognised bounds: {', '.join(skipped) or 'none'}."
        )
        return

    # ── Alpha-fade dataset polygons by relative size ─────────────────────
    # When a query_bbox is present the dataset polygons are background
    # context only, so cap their alpha lower (max 60) to keep them ghosted.
    # Without a query_bbox use the original range (max 160).
    bg_alpha_max = 60 if query_bbox else 160

    min_area = min((a for a, _ in raw_polygons), default=1.0)
    max_area = max((a for a, _ in raw_polygons), default=1.0)
    area_range = max(max_area - min_area, 1.0)

    polygons: list[tuple[float, dict]] = []
    for area, p in raw_polygons:
        relative_size = (area - min_area) / area_range
        alpha = int(bg_alpha_max - relative_size * (bg_alpha_max - 15))
        alpha = max(15, min(bg_alpha_max, alpha))
        p["fill_color"] = _score_color(p["_score"], alpha=alpha)
        p.pop("_bbox", None)
        p.pop("_score", None)
        polygons.append((area, p))

    final_polygons = [p for _, p in polygons]

    # ── Centre / zoom ────────────────────────────────────────────────────
    # Priority: query_bbox > smallest dataset polygon.
    if query_bbox:
        q_min_lon, q_min_lat, q_max_lon, q_max_lat = query_bbox
        centre_lon = (q_min_lon + q_max_lon) / 2
        centre_lat = (q_min_lat + q_max_lat) / 2
        zoom = _zoom_for_bbox(q_min_lon, q_min_lat, q_max_lon, q_max_lat)
    elif polygons:
        focus_area, focus_poly = min(polygons, key=lambda x: x[0])
        focus_coords = focus_poly["coordinates"][0]
        focus_lons = [pt[0] for pt in focus_coords]
        focus_lats = [pt[1] for pt in focus_coords]
        centre_lon = (min(focus_lons) + max(focus_lons)) / 2
        centre_lat = (min(focus_lats) + max(focus_lats)) / 2
        zoom = _zoom_for_bbox(min(focus_lons), min(focus_lats), max(focus_lons), max(focus_lats))
    else:
        centre_lon, centre_lat, zoom = 0.0, 0.0, 2

    # ── Layers ───────────────────────────────────────────────────────────
    layers: list = []

    # Background: faint dataset coverage polygons
    if final_polygons:
        layers.append(pdk.Layer(
            "PolygonLayer",
            data=final_polygons,
            get_polygon="coordinates",
            get_fill_color="fill_color",
            get_line_color=[59, 130, 246, 80],
            line_width_min_pixels=1,
            pickable=True,
            auto_highlight=True,
        ))

    # Foreground: the actual resolved query region (solid green highlight)
    if query_bbox:
        q_min_lon, q_min_lat, q_max_lon, q_max_lat = query_bbox
        query_polygon = [{
            "name": "Query region",
            "coordinates": [[
                [q_min_lon, q_min_lat],
                [q_max_lon, q_min_lat],
                [q_max_lon, q_max_lat],
                [q_min_lon, q_max_lat],
                [q_min_lon, q_min_lat],
            ]],
            "fill_color": [34, 139, 34, 160],   # forest green, clearly visible
        }]
        layers.append(pdk.Layer(
            "PolygonLayer",
            data=query_polygon,
            get_polygon="coordinates",
            get_fill_color="fill_color",
            get_line_color=[0, 80, 0, 255],
            line_width_min_pixels=2,
            pickable=True,
            auto_highlight=True,
        ))

    view = pdk.ViewState(
        longitude=centre_lon,
        latitude=centre_lat,
        zoom=zoom,
        pitch=0,
    )

    mapbox_token = os.environ.get("MAPBOX_API_KEY", "")
    if mapbox_token:
        pdk.settings.mapbox_key = mapbox_token
        map_style = "mapbox://styles/mapbox/light-v9"
    else:
        map_style = "road"

    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=view,
            tooltip={"text": "{name}\nScore: {score}%\nCoverage: {spatial_coverage}"},
            map_style=map_style,
        )
    )

    caption = f"Showing coverage for {len(final_polygons)} dataset(s). Colour: green = high score, red = low score."
    if query_bbox:
        caption += " Dark green box = resolved query region."
    if skipped:
        caption += f" ({len(skipped)} dataset(s) with unrecognised bounds not shown.)"
    st.caption(caption)
