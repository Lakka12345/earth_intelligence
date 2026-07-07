"""
components/map.py
Optional PyDeck map showing spatial coverage of discovered datasets.
Only renders when at least one source has parseable bbox coverage.
"""

from __future__ import annotations
import re
import streamlit as st


def _parse_bbox(spatial_coverage: str) -> tuple[float, float, float, float] | None:
    """
    Try to extract a bounding box from spatial_coverage strings like:
      "bbox [-180.0,-90.0,180.0,90.0]"
      "lat [5.0 to 23.0], lon [80.0 to 100.0]"
      "Global"  → returns world bbox
    """
    text = (spatial_coverage or "").lower().strip()

    if "global" in text or "worldwide" in text:
        return (-180.0, -90.0, 180.0, 90.0)

    # bbox [minlon, minlat, maxlon, maxlat]
    m = re.search(
        r"bbox\s*\[?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)",
        text,
    )
    if m:
        return (float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))

    # lat [min to max], lon [min to max]
    lat = re.search(r"lat\s*\[?\s*(-?\d+\.?\d*)\s*(?:to|–|-)\s*(-?\d+\.?\d*)", text)
    lon = re.search(r"lon\s*\[?\s*(-?\d+\.?\d*)\s*(?:to|–|-)\s*(-?\d+\.?\d*)", text)
    if lat and lon:
        return (float(lon.group(1)), float(lat.group(1)), float(lon.group(2)), float(lat.group(2)))

    return None


def render_map(result) -> None:
    """Render a PyDeck map of accepted dataset coverage areas if coords exist."""
    try:
        import pydeck as pdk
    except ImportError:
        st.caption("Install pydeck to enable the coverage map: `pip install pydeck`")
        return

    all_sources = (
        result.ranked_sources +
        getattr(result, "auth_required_sources", [])
    )

    polygons = []
    for src in all_sources:
        c    = src.candidate
        bbox = _parse_bbox(c.spatial_coverage)
        if bbox is None:
            continue

        min_lon, min_lat, max_lon, max_lat = bbox
        score = int(src.final_score * 255)
        polygons.append({
            "name":       c.name,
            "score":      round(src.final_score * 100, 1),
            "coordinates": [[
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat],
            ]],
            "fill_color": [59, 130, score, 80],
        })

    if not polygons:
        st.caption("No parseable spatial bounds found — map not displayed.")
        return

    # Centre view on first polygon
    first = polygons[0]["coordinates"][0]
    lons = [p[0] for p in first]
    lats = [p[1] for p in first]
    centre_lon = (min(lons) + max(lons)) / 2
    centre_lat = (min(lats) + max(lats)) / 2

    layer = pdk.Layer(
        "PolygonLayer",
        data=polygons,
        get_polygon="coordinates",
        get_fill_color="fill_color",
        get_line_color=[59, 130, 246, 200],
        line_width_min_pixels=1,
        pickable=True,
        auto_highlight=True,
    )

    view = pdk.ViewState(
        longitude=centre_lon,
        latitude=centre_lat,
        zoom=3,
        pitch=0,
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            tooltip={"text": "{name}\nScore: {score}%"},
            map_style="mapbox://styles/mapbox/light-v9",
        )
    )
    st.caption(f"Showing coverage for {len(polygons)} dataset(s). Colour intensity = discovery score.")
