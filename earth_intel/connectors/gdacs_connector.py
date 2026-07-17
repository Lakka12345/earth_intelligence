"""
GDACS Connector — Global Disaster Alert and Coordination System.

Real API endpoints (confirmed against GDACS's published API quickstart
guide and swagger docs, not guessed):

  Event search (custom date range / type / alert level), GeoJSON:
    https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH
      ?eventlist=EQ;TC;FL;VO;WF;DR
      &fromdate=YYYY-MM-DD&todate=YYYY-MM-DD
      &alertlevel=green;orange;red

  Most recent 100 events in the last 4 days:
    https://www.gdacs.org/gdacsapi/api/events/geteventlist/EVENTS4APP

  Single event detail:
    https://www.gdacs.org/gdacsapi/api/events/geteventdata
      ?eventtype=FL&eventid=1102983

  Swagger reference: https://www.gdacs.org/gdacsapi/swagger/index.html

Event types: TC (Tropical Cyclones), EQ (Earthquakes), FL (Floods),
VO (Volcanoes), WF (Wild Fires), DR (Droughts).

HONEST SCOPE NOTE: GDACS's SEARCH endpoint is documented with eventlist /
fromdate / todate / alertlevel parameters. No server-side bounding-box
parameter is documented anywhere I could confirm, so this connector does
NOT fabricate one -- it requests by date/type and applies the requested
bounding box as a client-side filter against each returned feature's
geometry instead. The API caps results at 100 records per call, ordered
by "todate" descending (most recent first); for a query that needs more
than that, only the most recent 100 matching events are returned.

Terms of use: GDACS asks that the source be acknowledged as "Global
Disaster Alert and Coordination System, GDACS" wherever this data is used.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from connectors.base_connector import (
    BaseConnector,
    ConnectorDescriptor,
    ConnectorMatch,
    Credentials,
    FetchRequest,
)
from connectors.connector_registry import register_connector
from connectors.connector_types import (
    AccessType,
    AuthenticationType,
    CapabilityFlags,
    ConnectorType,
    DatasetType,
)
from connectors.dataset_matching import any_token_matches
from models.agent4_schemas import DatasetDescriptor, DatasetMetadata, SizeEstimate, format_bytes
from models.website_analysis_schemas import SourceSnapshot

logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────

_GDACS_API_BASE   = "https://www.gdacs.org/gdacsapi/api"
_SEARCH_ENDPOINT  = f"{_GDACS_API_BASE}/events/geteventlist/SEARCH"
_RECENT_ENDPOINT  = f"{_GDACS_API_BASE}/events/geteventlist/EVENTS4APP"
_EVENT_ENDPOINT   = f"{_GDACS_API_BASE}/events/geteventdata"

REQUEST_TIMEOUT = 30
MAX_RECORDS_PER_CALL = 100  # documented API cap, not our choice

_EVENT_TYPE_KEYWORDS: List[Tuple[Tuple[str, ...], str]] = [
    (("flood", "flood extent", "inundation", "river discharge", "water level"), "FL"),
    (("earthquake", "seismic", "quake", "magnitude"), "EQ"),
    (("cyclone", "hurricane", "typhoon", "tropical storm", "wind speed"), "TC"),
    (("volcano", "eruption", "ash"), "VO"),
    (("wildfire", "wild fire", "fire", "burn"), "WF"),
    (("drought", "dry spell", "aridity"), "DR"),
]

_ALL_EVENT_TYPES = ("EQ", "TC", "FL", "VO", "WF", "DR")


def _event_types_for_variables(variables: Optional[List[str]]) -> List[str]:
    """Map requested variables to GDACS event-type codes. Falls back to
    all event types if nothing maps -- better to over-fetch and filter
    than silently return nothing for a variable this list doesn't know."""
    if not variables:
        return list(_ALL_EVENT_TYPES)
    matched: List[str] = []
    for var in variables:
        var_l = var.lower()
        for keywords, code in _EVENT_TYPE_KEYWORDS:
            if any(kw in var_l for kw in keywords) and code not in matched:
                matched.append(code)
    return matched or list(_ALL_EVENT_TYPES)


def _parse_date(value: Optional[str]) -> Optional[str]:
    """Normalize a time_range endpoint to YYYY-MM-DD, or None if unparseable."""
    if not value:
        return None
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(value))
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m2 = re.match(r"(\d{4})", str(value))
    if m2:
        return f"{m2.group(1)}-01-01"
    return None


def _feature_intersects_bbox(feature: dict, bbox: Optional[Tuple[float, float, float, float]]) -> bool:
    """
    Client-side bbox filter (no documented server-side bbox param -- see
    module docstring). bbox is (west, south, east, north). Returns True
    (keep the feature) if no bbox was requested, or the feature's geometry
    can't be parsed -- we'd rather over-include than silently drop events
    due to a geometry shape we don't handle (Point vs Polygon vs
    MultiPolygon all appear in GDACS output).
    """
    if not bbox or len(bbox) != 4:
        return True
    geom = feature.get("geometry") or {}
    coords = geom.get("coordinates")
    gtype = geom.get("type", "")
    if coords is None:
        return True

    west, south, east, north = bbox

    def _point_in_bbox(lon: float, lat: float) -> bool:
        return west <= lon <= east and south <= lat <= north

    try:
        if gtype == "Point":
            lon, lat = coords[0], coords[1]
            return _point_in_bbox(lon, lat)
        # For Polygon/MultiPolygon/LineString etc., check if ANY vertex
        # falls inside the requested bbox -- an approximation, but a safe
        # one for a "does this event touch my region of interest" filter.
        flat: List[Any] = []

        def _flatten(c):
            if isinstance(c, (int, float)):
                return
            if len(c) == 2 and all(isinstance(x, (int, float)) for x in c):
                flat.append(c)
                return
            for item in c:
                _flatten(item)

        _flatten(coords)
        return any(_point_in_bbox(lon, lat) for lon, lat in flat)
    except Exception:
        return True


def _get(url: str, params: Optional[dict] = None) -> Optional[requests.Response]:
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT, verify=False,
                             headers={"Accept": "application/json", "User-Agent": "EarthIntelligenceAgent/1.0"})
        return resp
    except Exception as exc:
        logger.debug("gdacs _get: request failed for %s — %s", url, exc)
        return None


def _search_events(
    event_types: List[str],
    from_date: Optional[str],
    to_date: Optional[str],
    alert_levels: Tuple[str, ...] = ("green", "orange", "red"),
) -> Optional[dict]:
    """Call the real SEARCH endpoint. Returns the raw GeoJSON FeatureCollection dict, or None on failure."""
    params = {
        "eventlist": ";".join(event_types),
        "alertlevel": ";".join(alert_levels),
    }
    if from_date:
        params["fromdate"] = from_date
    if to_date:
        params["todate"] = to_date

    resp = _get(_SEARCH_ENDPOINT, params)
    if resp is None or not resp.ok:
        logger.warning("gdacs _search_events: SEARCH failed (status=%s)",
                        resp.status_code if resp is not None else "no response")
        return None
    try:
        return resp.json()
    except Exception as exc:
        logger.warning("gdacs _search_events: response was not valid JSON — %s", exc)
        return None


def _recent_events() -> Optional[dict]:
    """Fallback: most recent 100 events in the last 4 days, no filters."""
    resp = _get(_RECENT_ENDPOINT)
    if resp is None or not resp.ok:
        return None
    try:
        return resp.json()
    except Exception:
        return None


# ── connector ────────────────────────────────────────────────────────────

class GDACSConnector(BaseConnector):
    descriptor = ConnectorDescriptor(
        connector_id="gdacs",
        provider_name="GDACS (Global Disaster Alerts)",
        connector_type=ConnectorType.provider_api,
        supported_access_types=(AccessType.public,),
        supported_dataset_types=(DatasetType.event, DatasetType.vector) if hasattr(DatasetType, "event") else (DatasetType.unknown,),
        supported_authentication=(AuthenticationType.none,),
        capabilities=(
            CapabilityFlags.supports_metadata
            | CapabilityFlags.supports_api
            | CapabilityFlags.supports_dataset_search
            | CapabilityFlags.supports_download
        ),
        priority=40,
    )
    provider_keywords = ("gdacs", "global disaster alert", "disaster alert and coordination")
    api_keywords = ("gdacs", "gdacsapi")

    def can_handle(self, snapshot: SourceSnapshot) -> bool:
        provider_text = f"{snapshot.name} {snapshot.url}"
        return any_token_matches(self.provider_keywords, provider_text)

    def match_score(self, snapshot: SourceSnapshot, context: Optional[Dict[str, Any]] = None) -> ConnectorMatch:
        if not self.can_handle(snapshot):
            return ConnectorMatch(score=0, reason="Provider keywords did not match.")
        return ConnectorMatch(score=max(10, 1000 - self.descriptor.priority),
                               reason="GDACS provider match.")

    # ── dataset discovery / metadata ────────────────────────────────────

    def discover_datasets(self, snapshot: SourceSnapshot, context: Optional[Dict[str, Any]] = None) -> List[DatasetDescriptor]:
        ctx = context if isinstance(context, dict) else {}
        variables = ctx.get("variables") or list(getattr(snapshot, "variables_available", None) or [])
        event_types = _event_types_for_variables(variables)
        return [
            DatasetDescriptor(
                provider="GDACS",
                dataset_name=f"GDACS {code} events",
                collection_name="GDACS Event Search",
                dataset_id=f"gdacs-{code.lower()}",
                api_endpoint=_SEARCH_ENDPOINT,
                metadata_endpoint=_EVENT_ENDPOINT,
                download_endpoint=_SEARCH_ENDPOINT,
                supported_variables=[kw for kws, c in _EVENT_TYPE_KEYWORDS if c == code for kw in kws],
                temporal_coverage="2004-present",
                spatial_coverage="Global",
                supported_formats=["GeoJSON"],
                authentication_required=False,
            )
            for code in event_types
        ]

    def probe_metadata(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> DatasetMetadata:
        event_types = _event_types_for_variables(list(fetch_request.variables or []))
        from_date = _parse_date(fetch_request.time_range[0] if fetch_request.time_range else None)
        to_date = _parse_date(fetch_request.time_range[1] if fetch_request.time_range and len(fetch_request.time_range) > 1 else None)

        data = _search_events(event_types, from_date, to_date)
        unavailable_reason = ""
        n_features = 0
        if data is None:
            data = _recent_events()
            if data is not None:
                unavailable_reason = (
                    "SEARCH with the requested date range returned nothing usable; "
                    "falling back to the most recent 100 events (last 4 days, unfiltered by date)."
                )
        if data is not None:
            n_features = len(data.get("features", []) or [])
        else:
            unavailable_reason = "GDACS API did not return usable event data (SEARCH and EVENTS4APP both failed)."

        return DatasetMetadata(
            source_id=snapshot.source_id,
            dataset_id=f"gdacs-{'-'.join(t.lower() for t in event_types)}",
            collection="GDACS Event Search",
            product=f"GDACS events ({', '.join(event_types)})",
            download_endpoint=_SEARCH_ENDPOINT,
            api_endpoint=_SEARCH_ENDPOINT,
            metadata_endpoint=_EVENT_ENDPOINT,
            variables=list(fetch_request.variables or []),
            spatial_coverage="Global (client-side bbox filter applied on download)",
            temporal_coverage=f"{from_date or 'unbounded'} to {to_date or 'unbounded'}",
            file_format="GeoJSON",
            content_type="application/geo+json",
            license="GDACS open data -- acknowledge source as 'Global Disaster Alert and Coordination System, GDACS'.",
            retrieval_method="GDACS SEARCH API" if not unavailable_reason else "GDACS EVENTS4APP (fallback)",
            unavailable_reason=unavailable_reason if n_features == 0 else "",
            authentication_required=False,
        )

    def probe_size(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> SizeEstimate:
        # GDACS returns small JSON event records, not gridded data -- a
        # rough per-event byte estimate is more honest here than pretending
        # we can HEAD a fixed file (there isn't one; SEARCH is a live query).
        event_types = _event_types_for_variables(list(fetch_request.variables or []))
        from_date = _parse_date(fetch_request.time_range[0] if fetch_request.time_range else None)
        to_date = _parse_date(fetch_request.time_range[1] if fetch_request.time_range and len(fetch_request.time_range) > 1 else None)
        data = _search_events(event_types, from_date, to_date)
        if data is None:
            return SizeEstimate(
                source_id=snapshot.source_id,
                method="GDACS size unavailable (SEARCH request failed)",
                human_readable="Unknown",
            )
        n_features = len(data.get("features", []) or [])
        # ~2 KB/event GeoJSON feature is a reasonable order-of-magnitude
        # estimate for GDACS's event geometry + properties payload.
        est_bytes = max(200, n_features * 2048)
        return SizeEstimate(
            source_id=snapshot.source_id,
            estimated_bytes=float(est_bytes),
            is_exact=False,
            method=f"GDACS event count estimate ({n_features} events × ~2KB)",
            human_readable=format_bytes(est_bytes),
        )

    # ── asset resolution / download ─────────────────────────────────────

    def resolve_download_asset(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> Optional[str]:
        # GDACS is a live search API, not a static pre-signed file URL --
        # return the SEARCH endpoint itself as the informational "asset",
        # matching the pattern CDS uses (job/query-based, not URL-based).
        return _SEARCH_ENDPOINT

    def fetch_subset(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> str:
        """
        "Subsetting" for GDACS means: filter by event type (from requested
        variables), by date range (server-side, via SEARCH's fromdate/
        todate), and by bounding box (client-side -- see module docstring
        for why there's no server-side bbox param to use here).
        """
        event_types = _event_types_for_variables(list(fetch_request.variables or []))
        from_date = _parse_date(fetch_request.time_range[0] if fetch_request.time_range else None)
        to_date = _parse_date(fetch_request.time_range[1] if fetch_request.time_range and len(fetch_request.time_range) > 1 else None)

        data = _search_events(event_types, from_date, to_date)
        used_fallback = False
        if data is None or not data.get("features"):
            fallback = _recent_events()
            if fallback is not None:
                data = fallback
                used_fallback = True

        if data is None:
            raise RuntimeError(
                "GDACS connector: both the SEARCH and EVENTS4APP endpoints failed to return data."
            )

        features = data.get("features", []) or []
        if fetch_request.bounding_box:
            features = [f for f in features if _feature_intersects_bbox(f, fetch_request.bounding_box)]

        out = {
            "type": "FeatureCollection",
            "gdacs_query": {
                "event_types": event_types,
                "from_date": from_date,
                "to_date": to_date,
                "bounding_box": list(fetch_request.bounding_box) if fetch_request.bounding_box else None,
                "used_recent_fallback": used_fallback,
                "source": "Global Disaster Alert and Coordination System, GDACS",
            },
            "features": features,
        }
        return self._write_geojson(out, fetch_request.dest_path)

    def fetch_full(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> str:
        # No meaningful distinction between "full" and "subset" for a
        # search API -- full just means no bbox filter applied.
        event_types = _event_types_for_variables(list(fetch_request.variables or []))
        from_date = _parse_date(fetch_request.time_range[0] if fetch_request.time_range else None)
        to_date = _parse_date(fetch_request.time_range[1] if fetch_request.time_range and len(fetch_request.time_range) > 1 else None)

        data = _search_events(event_types, from_date, to_date)
        if data is None or not data.get("features"):
            data = _recent_events()
        if data is None:
            raise RuntimeError(
                "GDACS connector: both the SEARCH and EVENTS4APP endpoints failed to return data."
            )
        data.setdefault("gdacs_query", {})
        data["gdacs_query"].update({
            "event_types": event_types,
            "from_date": from_date,
            "to_date": to_date,
            "source": "Global Disaster Alert and Coordination System, GDACS",
        })
        return self._write_geojson(data, fetch_request.dest_path)

    def _write_geojson(self, data: dict, dest_path: str) -> str:
        if dest_path:
            os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
            fpath = dest_path
        else:
            out_dir = tempfile.mkdtemp(prefix="gdacs_")
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            fpath = os.path.join(out_dir, f"gdacs_events_{ts}.geojson")

        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f)

        self.validate_download(fpath)
        return fpath

    # ── validation ───────────────────────────────────────────────────────

    def validate_download(self, path: str, metadata=None) -> None:
        if not path or not os.path.exists(path):
            raise RuntimeError(f"GDACS: downloaded file not found: {path}")
        size = os.path.getsize(path)
        if size == 0:
            raise RuntimeError(f"GDACS: downloaded file is empty: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                parsed = json.load(f)
        except Exception as exc:
            raise RuntimeError(f"GDACS: downloaded file is not valid JSON: {exc}")
        if parsed.get("type") != "FeatureCollection" or "features" not in parsed:
            raise RuntimeError(
                "GDACS: downloaded file is valid JSON but not a GeoJSON FeatureCollection "
                f"(got type={parsed.get('type')!r})."
            )


register_connector(GDACSConnector)
