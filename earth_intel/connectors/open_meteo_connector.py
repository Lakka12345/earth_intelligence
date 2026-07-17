"""
Open-Meteo Connector — Real Provider Implementation
====================================================
Replaces the static stub with a live connector that:
  • Dynamically routes to Forecast or Archive API based on dates
  • Builds request URLs from FetchRequest fields (lat, lon, dates, variables)
  • Maps scientific variable names → Open-Meteo parameter names
  • Implements probe_metadata(), probe_size(), fetch_full()
  • Validates downloaded payloads
  • Supports both JSON and CSV output formats
  • Returns meaningful errors for HTTP 4xx/5xx and bad data
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from connectors.base_connector import ConnectorDescriptor
from connectors.connector_registry import register_connector
from connectors.connector_types import (
    AccessType,
    AuthenticationType,
    CapabilityFlags,
    ConnectorType,
    DatasetType,
)
from connectors.dataset_matching import StaticDatasetConnector
from models.agent4_schemas import DatasetDescriptor, DatasetMetadata, SizeEstimate, format_bytes

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FORECAST_API = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_API  = "https://archive-api.open-meteo.com/v1/archive"

# Archive API requires data to be at least 5 days old
ARCHIVE_LAG_DAYS = 5

# Default request timeout (seconds)
REQUEST_TIMEOUT = 30

# ---------------------------------------------------------------------------
# Variable mapping: scientific / user-friendly names → Open-Meteo API names
# ---------------------------------------------------------------------------

_HOURLY_VAR_MAP: Dict[str, str] = {
    # temperature
    "temperature":             "temperature_2m",
    "temperature anomaly":     "temperature_2m",
    "air temperature":         "temperature_2m",
    "2m temperature":          "temperature_2m",
    "temperature_2m":          "temperature_2m",
    # humidity
    "humidity":                "relative_humidity_2m",
    "relative humidity":       "relative_humidity_2m",
    "relative_humidity_2m":    "relative_humidity_2m",
    # dew point
    "dew point":               "dew_point_2m",
    "dewpoint":                "dew_point_2m",
    "dew_point_2m":            "dew_point_2m",
    # apparent temperature
    "apparent temperature":    "apparent_temperature",
    "feels like":              "apparent_temperature",
    "apparent_temperature":    "apparent_temperature",
    # precipitation
    "precipitation":           "precipitation",
    "rain":                    "rain",
    "rainfall":                "rain",
    "snowfall":                "snowfall",
    "snow":                    "snowfall",
    "precipitation probability":  "precipitation_probability",
    "precipitation_probability":  "precipitation_probability",
    # pressure
    "pressure":                "surface_pressure",
    "surface pressure":        "surface_pressure",
    "surface_pressure":        "surface_pressure",
    "sea level pressure":      "pressure_msl",
    "msl pressure":            "pressure_msl",
    "pressure_msl":            "pressure_msl",
    # wind
    "wind speed":              "wind_speed_10m",
    "wind":                    "wind_speed_10m",
    "wind_speed_10m":          "wind_speed_10m",
    "wind speed 10m":          "wind_speed_10m",
    "wind direction":          "wind_direction_10m",
    "wind_direction_10m":      "wind_direction_10m",
    "wind gusts":              "wind_gusts_10m",
    "wind_gusts_10m":          "wind_gusts_10m",
    # cloud cover
    "cloud cover":             "cloud_cover",
    "cloudcover":              "cloud_cover",
    "cloud_cover":             "cloud_cover",
    # radiation
    "shortwave radiation":     "shortwave_radiation",
    "shortwave_radiation":     "shortwave_radiation",
    "solar radiation":         "shortwave_radiation",
    "radiation":               "shortwave_radiation",
    "direct radiation":        "direct_radiation",
    "direct_radiation":        "direct_radiation",
    "diffuse radiation":       "diffuse_radiation",
    "diffuse_radiation":       "diffuse_radiation",
    # soil moisture
    "soil moisture":           "soil_moisture_0_to_7cm",
    "soil_moisture":           "soil_moisture_0_to_7cm",
    "soil_moisture_0_to_7cm":  "soil_moisture_0_to_7cm",
    # evapotranspiration
    "evapotranspiration":      "et0_fao_evapotranspiration",
    "et0":                     "et0_fao_evapotranspiration",
    "et0_fao_evapotranspiration": "et0_fao_evapotranspiration",
    # misc
    "visibility":              "visibility",
    "cape":                    "cape",
}

_DAILY_VAR_MAP: Dict[str, str] = {
    "temperature max":              "temperature_2m_max",
    "temperature_2m_max":           "temperature_2m_max",
    "max temperature":              "temperature_2m_max",
    "temperature min":              "temperature_2m_min",
    "temperature_2m_min":           "temperature_2m_min",
    "min temperature":              "temperature_2m_min",
    "precipitation sum":            "precipitation_sum",
    "precipitation_sum":            "precipitation_sum",
    "rain sum":                     "rain_sum",
    "rain_sum":                     "rain_sum",
    "wind speed max":               "wind_speed_10m_max",
    "wind_speed_10m_max":           "wind_speed_10m_max",
    "wind gusts max":               "wind_gusts_10m_max",
    "wind_gusts_10m_max":           "wind_gusts_10m_max",
    "sunrise":                      "sunrise",
    "sunset":                       "sunset",
    "uv index":                     "uv_index_max",
    "uv_index_max":                 "uv_index_max",
    "shortwave radiation sum":      "shortwave_radiation_sum",
    "shortwave_radiation_sum":      "shortwave_radiation_sum",
    "et0 sum":                      "et0_fao_evapotranspiration_sum",
    "et0_fao_evapotranspiration_sum": "et0_fao_evapotranspiration_sum",
}

_DEFAULT_HOURLY = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "surface_pressure",
]


# ---------------------------------------------------------------------------
# Helper utilities (module-level, reusable)
# ---------------------------------------------------------------------------

def _map_variables(
    requested: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """Map user-supplied variable names to Open-Meteo API parameter names.

    Returns:
        hourly_vars  – validated Open-Meteo hourly parameter names
        daily_vars   – validated Open-Meteo daily parameter names
        unsupported  – original names that have no Open-Meteo equivalent
                       (never sent to the API)
    """
    hourly:      List[str] = []
    daily:       List[str] = []
    unsupported: List[str] = []

    for raw in requested:
        key = raw.strip().lower()
        if key in _HOURLY_VAR_MAP:
            v = _HOURLY_VAR_MAP[key]
            if v not in hourly:
                hourly.append(v)
        elif key in _DAILY_VAR_MAP:
            v = _DAILY_VAR_MAP[key]
            if v not in daily:
                daily.append(v)
        else:
            logger.warning(
                "open_meteo: variable %r has no Open-Meteo equivalent — "
                "it will NOT be sent to the API and is recorded as unsupported.",
                raw,
            )
            unsupported.append(raw)

    return hourly, daily, unsupported


def _choose_api(start_date: Optional[str], end_date: Optional[str]) -> str:
    """Return ARCHIVE_API when the period is fully in the past, else FORECAST_API.

    Open-Meteo exposes two endpoints:
      • Forecast API  — current conditions and up to 16-day forecasts
      • Archive API   — ERA5 reanalysis back to 1940, with a 5-day lag

    Routing rules (in order):
      1. If start_date is provided AND it predates the archive cutoff →
         Archive API, regardless of whether end_date is also provided.
         This is the key fix: the old code only routed to Archive when
         end_date was absent, so a historical request with BOTH dates
         (the normal case from Agent 4) always fell through to Forecast.
      2. If only end_date is provided and it predates the cutoff → Archive.
      3. Otherwise → Forecast.

    Also handles date objects in addition to ISO strings, and silently
    truncates datetime strings (e.g. "2013-06-01T00:00:00") to date-only
    before parsing so fromisoformat() never raises on those inputs.
    """
    cutoff = date.today() - timedelta(days=ARCHIVE_LAG_DAYS)

    def _to_date(val) -> Optional[date]:
        if val is None:
            return None
        if isinstance(val, date):
            return val
        try:
            # Truncate to 10 chars covers "YYYY-MM-DD" and "YYYY-MM-DDTHH:MM:SS…"
            return date.fromisoformat(str(val)[:10])
        except (ValueError, TypeError):
            return None

    sd = _to_date(start_date)
    ed = _to_date(end_date)

    # Rule 1: start_date is in the past → always Archive
    if sd is not None and sd < cutoff:
        logger.debug(
            "open_meteo: selected Archive API (start_date %s < cutoff %s)", sd, cutoff
        )
        return ARCHIVE_API

    # Rule 2: no start_date but end_date is in the past → Archive
    if sd is None and ed is not None and ed < cutoff:
        logger.debug(
            "open_meteo: selected Archive API (end_date %s < cutoff %s, no start_date)", ed, cutoff
        )
        return ARCHIVE_API

    logger.debug("open_meteo: selected Forecast API")
    return FORECAST_API


def _build_url(
    base_url: str,
    latitude: float,
    longitude: float,
    start_date: Optional[str],
    end_date: Optional[str],
    hourly_vars: List[str],
    daily_vars: List[str],
    timezone: str = "UTC",
    output_format: str = "json",
) -> str:
    """Assemble a complete Open-Meteo request URL."""
    params: Dict[str, Any] = {
        "latitude":  latitude,
        "longitude": longitude,
        "timezone":  timezone,
    }
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    if hourly_vars:
        params["hourly"] = ",".join(hourly_vars)
    if daily_vars:
        params["daily"] = ",".join(daily_vars)
    if output_format.lower() == "csv":
        params["format"] = "csv"
    return f"{base_url}?{urlencode(params)}"


def _http_get(url: str, timeout: int = REQUEST_TIMEOUT) -> requests.Response:
    """GET with descriptive errors for common HTTP status codes."""
    try:
        resp = requests.get(url, timeout=timeout, verify=False)
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"open_meteo: request timed out after {timeout}s — {url}"
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"open_meteo: connection error — {url}: {exc}") from exc

    if resp.status_code == 400:
        raise ValueError(
            f"open_meteo: bad request (400) — check coordinates, dates, or variable names. "
            f"URL: {url}  Response: {resp.text[:300]}"
        )
    if resp.status_code == 404:
        raise ValueError(f"open_meteo: not found (404) — {url}")
    if resp.status_code == 429:
        raise RuntimeError("open_meteo: rate limited (429). Please retry later.")
    if resp.status_code >= 500:
        raise RuntimeError(
            f"open_meteo: server error ({resp.status_code}) — {url}  "
            f"Response: {resp.text[:200]}"
        )
    if not resp.ok:
        raise RuntimeError(
            f"open_meteo: HTTP {resp.status_code} — {url}  Response: {resp.text[:200]}"
        )
    return resp


def _validate_json_payload(
    data: dict,
    requested_hourly: List[str],
    requested_daily: List[str],
) -> None:
    """Raise ValueError when the JSON payload looks wrong or empty."""
    if not isinstance(data, dict):
        raise ValueError("open_meteo: response is not a JSON object")
    if "error" in data:
        raise ValueError(f"open_meteo: API returned error — {data.get('reason', data)}")
    if "latitude" not in data or "longitude" not in data:
        raise ValueError("open_meteo: response missing latitude/longitude")
    hourly = data.get("hourly", {})
    daily  = data.get("daily", {})
    if not hourly and not daily:
        raise ValueError("open_meteo: response contains no hourly or daily data")
    if hourly and "time" not in hourly:
        raise ValueError("open_meteo: hourly block missing time series")
    if daily and "time" not in daily:
        raise ValueError("open_meteo: daily block missing time series")
    present      = set(hourly.keys()) | set(daily.keys())
    all_requested = set(requested_hourly) | set(requested_daily)
    if all_requested and not all_requested.intersection(present):
        raise ValueError(
            f"open_meteo: none of the requested variables {sorted(all_requested)} "
            f"present in response (got: {sorted(present)})"
        )


def _validate_csv_payload(text: str) -> None:
    """Raise ValueError when a CSV response looks wrong."""
    if not text or len(text.strip()) < 10:
        raise ValueError("open_meteo: CSV response is empty")
    low = text[:500].lower()
    if "<html" in low or "<!doctype" in low:
        raise ValueError("open_meteo: response is HTML, not CSV")
    if "error" in low and "latitude" not in low:
        raise ValueError(f"open_meteo: API error in CSV response: {text[:200]}")
    if "latitude" not in low and "time" not in low:
        raise ValueError("open_meteo: CSV does not appear to contain weather data")


def _http_head(url: str, timeout: int = REQUEST_TIMEOUT) -> Optional[requests.Response]:
    """
    Issue an HTTP HEAD request.  Returns the response on success, or None
    when HEAD is explicitly unsupported (405 Method Not Allowed, 501 Not
    Implemented) or any other transport/server error occurs.
    """
    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True, verify=False)
        if resp.status_code in (405, 501):
            logger.debug("open_meteo: HEAD not supported (%s) for %s", resp.status_code, url)
            return None
        if not resp.ok:
            logger.debug("open_meteo: HEAD returned %s for %s", resp.status_code, url)
            return None
        return resp
    except Exception as exc:
        logger.debug("open_meteo: HEAD failed for %s: %s", url, exc)
        return None


def _estimate_size(resp: requests.Response) -> Tuple[Optional[int], str, float]:
    """
    Try several strategies to estimate payload size.
    Returns (size_bytes, method_name, confidence_0_to_1).

    Confidence levels:
        1.00 – actual response body measured directly
        0.95 – Content-Length header from HEAD response
        0.90 – Content-Length header from GET response
        0.70 – response body measured from GET (no Content-Length)
        0.00 – unknown (last resort)
    """
    # 1. Attempt HEAD first to get Content-Length without downloading the body
    head_resp = _http_head(resp.url)
    if head_resp is not None:
        cl = head_resp.headers.get("Content-Length")
        if cl and cl.isdigit():
            logger.debug("open_meteo: size from HEAD Content-Length: %s bytes", cl)
            return int(cl), "HEAD Content-Length", 0.95

    # 2. Content-Length from the GET response we already have
    cl = resp.headers.get("Content-Length")
    if cl and cl.isdigit():
        logger.debug("open_meteo: size from GET Content-Length: %s bytes", cl)
        return int(cl), "GET Content-Length", 0.90

    # 3. Measure the actual GET response body
    body = resp.content
    if body:
        logger.debug("open_meteo: size from GET response body: %d bytes", len(body))
        return len(body), "GET response body", 0.70

    return None, "unknown", 0.0


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class OpenMeteoConnector(StaticDatasetConnector):
    """
    Real Open-Meteo provider connector.

    Inherits StaticDatasetConnector for dataset-discovery/keyword matching.
    All live API methods (probe_metadata, probe_size, fetch_full) are
    implemented here and actually call the Open-Meteo API.
    """

    descriptor = ConnectorDescriptor(
        connector_id="open_meteo",
        provider_name="Open-Meteo",
        connector_type=ConnectorType.provider_api,
        supported_access_types=(AccessType.public,),
        supported_dataset_types=(DatasetType.time_series, DatasetType.gridded),
        supported_authentication=(AuthenticationType.none,),
        capabilities=(
            CapabilityFlags.supports_metadata
            | CapabilityFlags.supports_api
            | CapabilityFlags.supports_dataset_search
        ),
        priority=30,
    )

    provider_keywords = ("open-meteo", "open meteo")
    api_keywords      = ("open-meteo", "rest")

    datasets = [
        DatasetDescriptor(
            provider="Open-Meteo",
            dataset_name="Open-Meteo Forecast API",
            collection_name="Forecast",
            dataset_id="open-meteo-forecast",
            api_endpoint=FORECAST_API,
            metadata_endpoint="https://open-meteo.com/en/docs",
            download_endpoint=FORECAST_API,
            supported_variables=[
                "temperature", "precipitation", "wind speed", "humidity",
                "pressure", "soil moisture", "cloud cover", "radiation",
                "dew point", "apparent temperature", "visibility", "cape",
            ],
            temporal_coverage="7-day forecast (up to 16 days, model dependent)",
            spatial_coverage="Global",
            supported_formats=["JSON", "CSV"],
            authentication_required=False,
            access_notes="Official public REST API. No API key required.",
        ),
        DatasetDescriptor(
            provider="Open-Meteo",
            dataset_name="Open-Meteo Historical Weather API",
            collection_name="Historical Weather",
            dataset_id="open-meteo-archive",
            api_endpoint=ARCHIVE_API,
            metadata_endpoint="https://open-meteo.com/en/docs/historical-weather-api",
            download_endpoint=ARCHIVE_API,
            supported_variables=[
                "temperature", "precipitation", "wind speed", "humidity",
                "pressure", "radiation", "soil moisture", "dew point",
                "apparent temperature",
            ],
            temporal_coverage="1940-present (ERA5 reanalysis, 5-day lag)",
            spatial_coverage="Global",
            supported_formats=["JSON", "CSV"],
            authentication_required=False,
            access_notes="Official public REST API. No API key required.",
        ),
    ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_request_params(self, fetch_request) -> Dict[str, Any]:
        """Safely extract normalised params from a FetchRequest or dict.

        Coordinate handling (strict priority, no silent defaults):
          1. latitude/longitude fields from Agent 3 / Agent 4 / orchestrator
          2. Centre of a bounding_box when only a bbox is supplied
          3. ValueError — never substitute synthetic coordinates

        Bounding-box index convention (Bug 1 fix):
          Agent 4's download_manager builds bounding_box as a 4-tuple in
          standard GeoJSON / WGS84 order:
              [min_lon, min_lat, max_lon, max_lat]  ← index 0 = LON, 1 = LAT
          The old code had the comment "[min_lat, min_lon, max_lat, max_lon]"
          but used indices [0]/[2] for lat and [1]/[3] for lon — correct only
          if the caller uses lat-first order.  Since Agent 4 uses lon-first,
          this was silently passing Chennai's longitude (80) as latitude and
          its latitude (13) as longitude.

          Fix: try to auto-detect the order.  A value whose absolute magnitude
          is > 90 cannot be a latitude, so if bbox[0] > 90 we know the tuple
          is [lon, lat, lon, lat].  This is robust against both conventions and
          prevents the swap for all future callers.

        Date extraction (Bug 2 extension):
          Agent 4 may pass dates either as top-level start_date/end_date fields
          OR inside a time_range tuple/list/dict.  The old code only checked
          the top-level fields, so when Agent 4 populated only time_range the
          dates were None and _choose_api() always picked the Forecast API.
        """
        if isinstance(fetch_request, dict):
            r = fetch_request
        else:
            r = vars(fetch_request) if hasattr(fetch_request, "__dict__") else {}

        def _get(*keys, default=None):
            for k in keys:
                v = r.get(k)
                if v is not None:
                    return v
            return default

        # ── Coordinates ───────────────────────────────────────────────
        _lat = _get("latitude", "lat")
        _lon = _get("longitude", "lon")

        if _lat is None or _lon is None:
            bbox = _get("bounding_box", "bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                v0, v1, v2, v3 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

                # Auto-detect lon-first (GeoJSON/WGS84) vs lat-first order.
                # Latitudes are always in [-90, 90]; longitudes can reach ±180.
                # If the first element exceeds ±90 it must be a longitude.
                if abs(v0) > 90 or abs(v2) > 90:
                    # lon-first: [min_lon, min_lat, max_lon, max_lat]
                    _lon = (v0 + v2) / 2.0
                    _lat = (v1 + v3) / 2.0
                    logger.info(
                        "open_meteo: bbox detected as lon-first "
                        "[%.4f, %.4f, %.4f, %.4f] → centre lat=%.4f lon=%.4f",
                        v0, v1, v2, v3, _lat, _lon,
                    )
                else:
                    # lat-first: [min_lat, min_lon, max_lat, max_lon]
                    _lat = (v0 + v2) / 2.0
                    _lon = (v1 + v3) / 2.0
                    logger.info(
                        "open_meteo: bbox detected as lat-first "
                        "[%.4f, %.4f, %.4f, %.4f] → centre lat=%.4f lon=%.4f",
                        v0, v1, v2, v3, _lat, _lon,
                    )

            elif isinstance(bbox, dict):
                # bbox as dict: {min_lat, min_lon, max_lat, max_lon} or similar keys
                _lat_min = bbox.get("min_lat", bbox.get("south", bbox.get("lat_min")))
                _lat_max = bbox.get("max_lat", bbox.get("north", bbox.get("lat_max")))
                _lon_min = bbox.get("min_lon", bbox.get("west",  bbox.get("lon_min")))
                _lon_max = bbox.get("max_lon", bbox.get("east",  bbox.get("lon_max")))
                if None not in (_lat_min, _lat_max, _lon_min, _lon_max):
                    _lat = (float(_lat_min) + float(_lat_max)) / 2.0
                    _lon = (float(_lon_min) + float(_lon_max)) / 2.0
                    logger.info(
                        "open_meteo: derived centre coordinates (%.4f, %.4f) from bbox dict",
                        _lat, _lon,
                    )

        if _lat is None or _lon is None:
            raise ValueError(
                "Open-Meteo request cannot be built because no valid coordinates or "
                "bounding box were provided. Supply latitude/longitude (or a bounding_box) "
                "via Agent 3, Agent 4, or the orchestrator."
            )

        lat = float(_lat)
        lon = float(_lon)

        # ── Dates (Bug 2 fix: also unpack time_range) ─────────────────
        start = _get("start_date", "date_start", "from_date")
        end   = _get("end_date",   "date_end",   "to_date")

        # Agent 4 may supply dates inside a time_range field as a
        # tuple (start, end), list [start, end], or dict {"start": ..., "end": ...}
        if start is None or end is None:
            time_range = _get("time_range")
            if isinstance(time_range, (list, tuple)) and len(time_range) >= 2:
                if start is None:
                    start = str(time_range[0])[:10] if time_range[0] else None
                if end is None:
                    end   = str(time_range[1])[:10] if time_range[1] else None
                logger.debug(
                    "open_meteo: extracted dates from time_range tuple/list: start=%s end=%s",
                    start, end,
                )
            elif isinstance(time_range, dict):
                if start is None:
                    raw = time_range.get("start") or time_range.get("start_date") or time_range.get("from")
                    start = str(raw)[:10] if raw else None
                if end is None:
                    raw = time_range.get("end") or time_range.get("end_date") or time_range.get("to")
                    end = str(raw)[:10] if raw else None
                logger.debug(
                    "open_meteo: extracted dates from time_range dict: start=%s end=%s",
                    start, end,
                )

        # Normalise: strip any time component so fromisoformat() is always happy
        if start:
            start = str(start)[:10]
        if end:
            end   = str(end)[:10]

        variables = _get("variables", "requested_variables", default=[]) or []
        if isinstance(variables, str):
            variables = [v.strip() for v in variables.split(",") if v.strip()]
        fmt = (_get("output_format", "format", "file_format") or "json").lower()
        tz  = _get("timezone", default="UTC") or "UTC"

        return {
            "latitude":   lat,
            "longitude":  lon,
            "start_date": start,
            "end_date":   end,
            "variables":  variables,
            "format":     fmt,
            "timezone":   tz,
        }

    def _build_request_url(self, params: Dict[str, Any]) -> str:
        """Choose API endpoint and build the complete request URL.

        Variable mapping converts user-friendly / scientific names into the
        exact parameter names the Open-Meteo API accepts (e.g. "temperature"
        → "temperature_2m").  Variables with no mapping are excluded from the
        request and logged as ignored.
        """
        base = _choose_api(params["start_date"], params["end_date"])
        if params["variables"]:
            hourly, daily, unsupported = _map_variables(params["variables"])
            if unsupported:
                logger.warning(
                    "open_meteo: variables ignored (no Open-Meteo equivalent) — %s",
                    unsupported,
                )
            if hourly or daily:
                logger.debug(
                    "open_meteo: variables mapped — hourly=%s daily=%s",
                    hourly, daily,
                )
            if not hourly and not daily:
                raise ValueError(
                    "Open-Meteo cannot provide any of the requested scientific variables: "
                    + str(params["variables"]) + ". None map to a supported Open-Meteo parameter. "
                    "Unsupported variables: " + str(unsupported)
                )
        else:
            hourly, daily = _DEFAULT_HOURLY, []
        return _build_url(
            base_url=base,
            latitude=params["latitude"],
            longitude=params["longitude"],
            start_date=params["start_date"],
            end_date=params["end_date"],
            hourly_vars=hourly,
            daily_vars=daily,
            timezone=params["timezone"],
            output_format=params["format"],
        )

    # ------------------------------------------------------------------
    # probe_metadata
    # ------------------------------------------------------------------

    def probe_metadata(self, snapshot=None, fetch_request=None, **kwargs) -> DatasetMetadata:
        """
        Call the Open-Meteo API and return live metadata as a DatasetMetadata object.
        """
        source_id = getattr(snapshot, "source_id", None) or "open_meteo"
        params = self._extract_request_params(fetch_request or kwargs)
        url    = self._build_request_url(params)
        logger.info("open_meteo probe_metadata GET %s", url)

        try:
            resp = _http_get(url)
        except Exception as exc:
            logger.warning("open_meteo probe_metadata: request failed — %s", exc)
            return DatasetMetadata(
                source_id=source_id,
                api_endpoint=url,
                file_size_bytes=50 * 1024 * 1024,
                variables=params.get("variables") or [],
                retrieval_method="unavailable",
                unavailable_reason=str(exc),
            )

        content_type = resp.headers.get("Content-Type", "")
        size_bytes, size_method, confidence = _estimate_size(resp)

        variables_returned: List[str] = []
        dataset_info: Dict[str, Any] = {}
        # Parse JSON metadata; catch only expected parsing errors so unexpected
        # exceptions propagate and are not silently swallowed.
        try:
            data = resp.json()
            variables_returned = (
                [k for k in data.get("hourly", {}).keys() if k != "time"]
                + [k for k in data.get("daily",  {}).keys() if k != "time"]
            )
            dataset_info = {
                "latitude":             data.get("latitude"),
                "longitude":            data.get("longitude"),
                "elevation":            data.get("elevation"),
                "utc_offset_seconds":   data.get("utc_offset_seconds"),
                "timezone":             data.get("timezone"),
                "timezone_abbreviation":data.get("timezone_abbreviation"),
                "generationtime_ms":    data.get("generationtime_ms"),
            }
        except (ValueError, json.JSONDecodeError) as exc:
            # Response was not valid JSON (e.g. CSV format requested, or API error
            # returned plain text). Return partial metadata with a warning.
            logger.warning("open_meteo probe_metadata: could not parse JSON response — %s", exc)

        api_base = _choose_api(params["start_date"], params["end_date"])

        # Build the full variable breakdown for transparent reporting:
        #   supported_variables_used  — what was actually sent to the API
        #   unsupported_variables     — requested but has no Open-Meteo equivalent
        #   ignored_variables         — sent to API but absent from the response
        _hourly_used: List[str] = []
        _daily_used:  List[str] = []
        _unsupported_meta: List[str] = []
        if params["variables"]:
            _hourly_used, _daily_used, _unsupported_meta = _map_variables(params["variables"])
        _supported_used = _hourly_used + _daily_used
        _ignored = [v for v in _supported_used if v not in variables_returned]

        return DatasetMetadata(
            source_id=source_id,
            dataset_id="Historical Weather API" if api_base == ARCHIVE_API else "Forecast API",
            product="Open-Meteo",
            download_endpoint=url,
            api_endpoint=api_base,
            metadata_endpoint=url,
            file_size_bytes=size_bytes,
            variables=variables_returned or _supported_used,
            file_format=params["format"].upper(),
            content_type=content_type,
            retrieval_method=size_method or "Open-Meteo API",
            unavailable_reason="",
        )

    # ------------------------------------------------------------------
    # probe_size
    # ------------------------------------------------------------------

    def probe_size(self, snapshot=None, fetch_request=None, **kwargs) -> SizeEstimate:
        """
        Estimate download size.  Algorithm:
          1. HEAD → Content-Length (exact, HIGH confidence).
          2. GET body measurement (exact, HIGH confidence).
          3. Dynamic estimate: hours × variables × bytes-per-value (LOW confidence).
        Returns SizeEstimate for Agent 4 compatibility.
        """
        source_id = getattr(snapshot, "source_id", None) or "open_meteo"

        try:
            params = self._extract_request_params(fetch_request or kwargs)
        except (ValueError, Exception) as exc:
            return SizeEstimate(
                source_id=source_id,
                method=f"Open-Meteo: cannot build request — {exc}",
                human_readable="Unknown",
            )

        url = self._build_request_url(params)
        logger.info("open_meteo probe_size — attempting HEAD first: %s", url)

        # Priority 1: HEAD Content-Length
        head_resp = _http_head(url)
        if head_resp is not None:
            cl = head_resp.headers.get("Content-Length")
            if cl and cl.isdigit():
                sz = int(cl)
                logger.info("open_meteo probe_size: HEAD Content-Length = %d bytes", sz)
                return SizeEstimate(
                    source_id=source_id,
                    estimated_bytes=float(sz),
                    is_exact=True,
                    method="HEAD Content-Length",
                    human_readable=format_bytes(sz),
                )

        # Priority 2: GET body measurement
        logger.info("open_meteo probe_size GET %s", url)
        try:
            resp = _http_get(url)
            sz, method, _ = _estimate_size(resp)
            if sz is not None:
                return SizeEstimate(
                    source_id=source_id,
                    estimated_bytes=float(sz),
                    is_exact=True,
                    method=f"Open-Meteo GET body measurement ({method})",
                    human_readable=format_bytes(sz),
                )
        except Exception as exc:
            logger.warning("open_meteo probe_size: GET failed — %s", exc)

        # Priority 3: Dynamic estimate from request parameters
        # Open-Meteo JSON: each hourly value ≈ 8 bytes (float64 in JSON text),
        # plus ~100 bytes per variable for key/time overhead.
        try:
            variables = params.get("variables") or []
            hourly_vars, daily_vars, _ = _map_variables(variables)
            n_hourly = len(hourly_vars)
            n_daily  = len(daily_vars)

            start_str = params.get("start_date") or "2020-01-01"
            end_str   = params.get("end_date")   or "2020-01-07"
            from datetime import datetime as _dt
            n_days = max(1, (_dt.fromisoformat(end_str[:10]) - _dt.fromisoformat(start_str[:10])).days + 1)

            # ~8 bytes per number in JSON, 24 hourly slots/day, 1 daily slot/day
            est_bytes = int(
                n_hourly * n_days * 24 * 8
                + n_daily  * n_days * 1  * 8
                + (n_hourly + n_daily) * 100  # key/time overhead
                + 500  # base JSON envelope
            )
            if est_bytes < 100:
                # No variables resolved — return unknown rather than misleading 0
                raise ValueError("no resolvable variables")

            logger.info(
                "open_meteo probe_size: dynamic estimate %d bytes "
                "(hourly_vars=%d, daily_vars=%d, days=%d)",
                est_bytes, n_hourly, n_daily, n_days,
            )
            return SizeEstimate(
                source_id=source_id,
                estimated_bytes=float(est_bytes),
                is_exact=False,
                method="Open-Meteo dynamic estimate (vars × hours × 8 bytes/value)",
                human_readable=format_bytes(est_bytes),
            )
        except Exception as exc:
            logger.warning("open_meteo probe_size: dynamic estimate failed — %s", exc)

        return SizeEstimate(
            source_id=source_id,
            method="Open-Meteo size unavailable",
            human_readable="Unknown",
        )

    # ------------------------------------------------------------------
    # fetch_full
    # ------------------------------------------------------------------
    def fetch_full(self, snapshot=None, fetch_request=None, credentials=None, output_dir: Optional[str] = None, **kwargs) -> str:
        """
        Download weather data and save to disk.
        Returns the local file path string (BaseConnector contract).
        Raises RuntimeError on failure so the download manager can retry.
        """
        params = self._extract_request_params(fetch_request or kwargs)
        url    = self._build_request_url(params)
        fmt    = params["format"]
        logger.info("open_meteo fetch_full GET %s", url)

        try:
            resp = _http_get(url)
        except Exception as exc:
            raise RuntimeError(f"Open-Meteo HTTP request failed: {exc}") from exc

        content = resp.text

        hourly_vars, daily_vars, unsupported_vars = (
            _map_variables(params["variables"]) if params["variables"]
            else (_DEFAULT_HOURLY, [], [])
        )

        try:
            if fmt == "csv":
                _validate_csv_payload(content)
            else:
                data = resp.json()
                _validate_json_payload(data, hourly_vars, daily_vars)
        except ValueError as exc:
            raise RuntimeError(f"Open-Meteo payload validation failed: {exc}") from exc

        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="open_meteo_")
        os.makedirs(output_dir, exist_ok=True)

        ext      = "csv" if fmt == "csv" else "json"
        ts       = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"open_meteo_{ts}.{ext}"
        file_path = os.path.join(output_dir, filename)

        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write(content)

        size_bytes = os.path.getsize(file_path)
        logger.info("open_meteo fetch_full: saved %d bytes → %s", size_bytes, file_path)

        # Post-save validation: ensure the file on disk is structurally sound.
        # If it fails, delete the file so no corrupted data is left behind.
        validation = self.validate_download(file_path, variables=params["variables"] or None)
        if not validation["valid"]:
            logger.error(
                "open_meteo fetch_full: post-save validation FAILED for %s — issues: %s. "
                "Deleting file.",
                file_path, validation["issues"],
            )
            try:
                os.remove(file_path)
            except OSError as rm_exc:
                logger.warning("open_meteo fetch_full: could not delete invalid file %s — %s", file_path, rm_exc)
            raise RuntimeError("Open-Meteo post-save validation failed: " + "; ".join(validation["issues"]))

        logger.info("open_meteo fetch_full: validation passed — %s", file_path)
        # BaseConnector contract: fetch_full returns the local file path as a str.
        # Returning a dict here previously caused "AttributeError: 'dict' object
        # has no attribute 'endswith'" in download_manager._trim_local_file().
        return file_path

    # ------------------------------------------------------------------
    # discover_datasets — dynamic routing
    # ------------------------------------------------------------------

    def discover_datasets(self, snapshot=None, context=None, **kwargs):
        """Return the descriptor appropriate for the requested time range."""
        src = context or kwargs or None
        try:
            params = self._extract_request_params(src) if src else {}
        except ValueError:
            params = {}
        api_base = _choose_api(params.get("start_date"), params.get("end_date"))
        return [self.datasets[1]] if api_base == ARCHIVE_API else [self.datasets[0]]

    # ------------------------------------------------------------------
    # resolve_download_asset
    # ------------------------------------------------------------------

    def resolve_download_asset(self, snapshot=None, fetch_request=None, credentials=None, **kwargs) -> Optional[str]:
        """
        Return the concrete Open-Meteo request URL that will be fetched.

        Per BaseConnector's contract, this must return a plain URL string
        (the caller treats it as such, e.g. checking file extension with
        .endswith()) — NOT a dict. Returning a dict here previously caused
        FAILED ('dict' object has no attribute 'endswith') downstream.
        """
        params = self._extract_request_params(fetch_request or kwargs)
        return self._build_request_url(params)

    # ------------------------------------------------------------------
    # validate_download — post-download check
    # ------------------------------------------------------------------

    def validate_download(
        self,
        file_path: str,
        variables: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Validate a downloaded Open-Meteo file.

        Validation workflow:
          1. File must exist and be non-empty.
          2. Content must not be an HTML error page.
          3. JSON files are parsed and checked for required fields (latitude,
             longitude, hourly/daily blocks with a time series).
          4. CSV files are checked for expected header keywords.

        Returns {"valid": True/False, "issues": [...], "size_bytes": N}.
        Called automatically by fetch_full() after every save.
        """
        issues: List[str] = []

        if not os.path.exists(file_path):
            return {"valid": False, "issues": [f"File not found: {file_path}"]}
        size = os.path.getsize(file_path)
        if size == 0:
            return {"valid": False, "issues": ["File is empty"]}

        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()

        low = content[:500].lower()
        if "<html" in low or "<!doctype" in low:
            return {"valid": False, "issues": ["File contains HTML, not weather data"]}

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".json" or content.lstrip().startswith("{"):
            try:
                data = json.loads(content)
            except json.JSONDecodeError as exc:
                return {"valid": False, "issues": [f"Invalid JSON: {exc}"]}
            hourly_vars, daily_vars, _unsupported = _map_variables(variables or [])
            try:
                _validate_json_payload(data, hourly_vars, daily_vars)
            except ValueError as exc:
                issues.append(str(exc))
        else:
            try:
                _validate_csv_payload(content)
            except ValueError as exc:
                issues.append(str(exc))

        if issues:
            logger.warning("open_meteo validate_download: validation FAILED — %s", issues)
            return {"valid": False, "issues": issues}
        logger.info("open_meteo validate_download: validation passed (%d bytes) — %s", size, file_path)
        return {"valid": True, "issues": [], "size_bytes": size}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_connector(OpenMeteoConnector)
