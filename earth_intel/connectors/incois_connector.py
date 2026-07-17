"""
INCOIS Connector — Real Provider Implementation
===============================================
Uses INCOIS public ERDDAP and OPeNDAP endpoints to:
  • Discover ocean datasets (SST, currents, salinity, waves, wind)
  • Probe metadata from ERDDAP info endpoints
  • Resolve downloadable NetCDF/CSV assets
  • Estimate size via HEAD + ERDDAP metadata
  • Download with optional spatial/temporal subsetting

INCOIS hosts several datasets via ERDDAP at:
  https://erddap.incois.gov.in/erddap/

OPeNDAP access (institutional):
  https://thredds.incois.gov.in/thredds/

Public data portal:
  https://incois.gov.in/portal/datainfo/datainfo.jsp

Note: INCOIS has inconsistent public API availability.
The connector uses the ERDDAP endpoint as the primary interface
and falls back to direct OPeNDAP URLs where possible.
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, quote

import requests
import urllib3

# Suppress the SSL InsecureRequestWarning to keep console clean
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
from models.agent4_schemas import DatasetDescriptor, SizeEstimate, format_bytes, DatasetMetadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — INCOIS ERDDAP
# ---------------------------------------------------------------------------

ERDDAP_BASE      = "https://erddap.incois.gov.in/erddap"
ERDDAP_INFO      = f"{ERDDAP_BASE}/info"
ERDDAP_SEARCH    = f"{ERDDAP_BASE}/search/index.json"
ERDDAP_TABLEDAP  = f"{ERDDAP_BASE}/tabledap"
ERDDAP_GRIDDAP   = f"{ERDDAP_BASE}/griddap"

REQUEST_TIMEOUT  = 30

# ---------------------------------------------------------------------------
# Known INCOIS datasets (fallback when ERDDAP is unavailable)
# ---------------------------------------------------------------------------

_KNOWN_DATASETS: List[Dict[str, Any]] = [
    {
        "dataset_id":  "INCOIS_SST_Daily",
        "title":       "INCOIS Sea Surface Temperature Daily Analysis",
        "variables":   ["sea surface temperature", "sst"],
        "format":      "NetCDF",
        "endpoint":    f"{ERDDAP_GRIDDAP}/INCOIS_SST_Daily",
        "temporal":    "2000-present",
        "spatial":     "Indian Ocean (30E-120E, 40S-30N)",
    },
    {
        "dataset_id":  "INCOIS_OC",
        "title":       "INCOIS Ocean Current Analysis",
        "variables":   ["ocean current", "current velocity", "u", "v"],
        "format":      "NetCDF",
        "endpoint":    f"{ERDDAP_GRIDDAP}/INCOIS_OC",
        "temporal":    "Dataset dependent",
        "spatial":     "Indian Ocean",
    },
    {
        "dataset_id":  "INCOIS_SALT",
        "title":       "INCOIS Sea Surface Salinity",
        "variables":   ["salinity", "sea surface salinity"],
        "format":      "NetCDF",
        "endpoint":    f"{ERDDAP_GRIDDAP}/INCOIS_SALT",
        "temporal":    "Dataset dependent",
        "spatial":     "Indian Ocean",
    },
    {
        "dataset_id":  "INCOIS_WAVES",
        "title":       "INCOIS Wave Analysis",
        "variables":   ["waves", "significant wave height", "wave period", "wave direction"],
        "format":      "NetCDF",
        "endpoint":    f"{ERDDAP_GRIDDAP}/INCOIS_WAVES",
        "temporal":    "Dataset dependent",
        "spatial":     "Indian Ocean",
    },
    {
        "dataset_id":  "INCOIS_WIND",
        "title":       "INCOIS Surface Wind Analysis",
        "variables":   ["wind", "wind speed", "wind direction"],
        "format":      "NetCDF",
        "endpoint":    f"{ERDDAP_GRIDDAP}/INCOIS_WIND",
        "temporal":    "Dataset dependent",
        "spatial":     "Indian Ocean",
    },
]


def _var_to_dataset_id(variables: List[str]) -> str:
    """Match requested variables to best known dataset_id.

    NOTE: this only searches the hardcoded _KNOWN_DATASETS fallback list.
    The IDs in that list (INCOIS_SST_Daily, INCOIS_OC, ...) are guesses
    that do not match the live ERDDAP server's actual naming convention
    (e.g. incois_tmi_3day, incois_quickscat_daily). Callers should prefer
    _resolve_live_dataset_id() and only fall back to this as a last resort.
    """
    vl = [v.lower() for v in variables]
    for ds in _KNOWN_DATASETS:
        for dv in ds["variables"]:
            if any(dv in v or v in dv for v in vl):
                return ds["dataset_id"]
    return _KNOWN_DATASETS[0]["dataset_id"]  # default: SST


def _dataset_id_exists(dataset_id: str) -> bool:
    """Check the ERDDAP info endpoint to confirm a dataset_id is actually live."""
    try:
        resp = _get(f"{ERDDAP_INFO}/{dataset_id}/index.json")
        return resp.ok
    except Exception:
        return False


# Module-level cache for the full ERDDAP dataset list.
# Populated once on first call to _live_dataset_list(); never re-fetched
# in the same process run (datasets change rarely and we don't want to
# hammer the server on every variable lookup).
_ERDDAP_DATASET_CACHE: Optional[List[Dict[str, Any]]] = None

# Memoizes _resolve_live_dataset_id() results per (variables, requested_id)
# -- see that function's docstring for why this matters.
_RESOLVED_DATASET_ID_CACHE: Dict[Tuple[Tuple[str, ...], str], str] = {}


def _live_dataset_list() -> List[Dict[str, Any]]:
    """
    Fetch the complete INCOIS ERDDAP dataset catalogue once and cache it.

    ERDDAP's /search/index.json returns HTTP 404 when a free-text query
    matches zero datasets — that's documented ERDDAP behavior, not an
    outage. Long natural-language variable names like "chlorophyll-a
    concentration" or "nutrient concentrations (e.g., nitrogen,
    phosphorus)" almost never match ERDDAP's indexed titles, so every
    per-variable search was 404-ing and falling through to a hardcoded
    guess that doesn't exist on the live server.

    The fix: fetch /griddap/index.json once (the full catalogue, no
    search term, so it never 404s on zero results) and match locally
    using token overlap. This is cheaper than N ERDDAP search round-trips
    and more reliable.
    """
    global _ERDDAP_DATASET_CACHE
    if _ERDDAP_DATASET_CACHE is not None:
        return _ERDDAP_DATASET_CACHE

    datasets: List[Dict[str, Any]] = []
    for endpoint in (f"{ERDDAP_GRIDDAP}/index.json", f"{ERDDAP_TABLEDAP}/index.json"):
        try:
            resp = requests.get(endpoint, timeout=20, verify=False,
                                headers={"Accept": "application/json"})
            if not resp.ok:
                continue
            data = resp.json()
            cols = data.get("table", {}).get("columnNames", [])
            rows = data.get("table", {}).get("rows", [])
            for row in rows:
                record = dict(zip(cols, row))
                if record.get("Dataset ID"):
                    datasets.append(record)
        except Exception as exc:
            logger.debug("incois _live_dataset_list: catalogue fetch failed for %s — %s", endpoint, exc)

    _ERDDAP_DATASET_CACHE = datasets
    if datasets:
        logger.info("incois _live_dataset_list: cached %d datasets from ERDDAP", len(datasets))
    else:
        logger.warning("incois _live_dataset_list: ERDDAP catalogue unavailable; will use static fallback")
    return datasets


def _score_dataset_match(record: Dict[str, Any], variables: List[str]) -> int:
    """Score an ERDDAP catalogue row against requested variables (higher = better)."""
    haystack = " ".join([
        str(record.get("Dataset ID", "")),
        str(record.get("Title", "")),
        str(record.get("Summary", "")),
        str(record.get("Institution", "")),
    ]).lower()
    score = 0
    for var in variables:
        # Normalise: lowercase, drop parenthetical detail, split on punctuation
        tokens = [t for t in var.lower().replace("(", " ").replace(")", " ")
                  .replace(",", " ").replace("-", " ").split() if len(t) > 2]
        for token in tokens:
            if token in haystack:
                score += 1
    return score


def _resolve_live_dataset_id(variables: List[str], requested_id: Optional[str] = None) -> str:
    """
    Resolve a dataset_id that is confirmed to exist on the live ERDDAP server.

    Uses the cached full catalogue (fetched once) and local token matching
    instead of per-query ERDDAP search calls that 404 on zero results.

    Order of preference:
      1. requested_id, if given and confirmed live.
      2. Best local match from the cached ERDDAP catalogue.
      3. Static _KNOWN_DATASETS guess confirmed live.
      4. Static _KNOWN_DATASETS guess anyway (will likely 404 but preserves
         old fallback behavior rather than hard-failing).

    CHANGED: results are memoized per (sorted variables tuple, requested_id).
    Previously this re-resolved from scratch on every call, and different
    call sites in this file (probe_metadata, resolve_download_asset,
    fetch_subset, fetch_full) could each be invoked with a *different*
    variables list for the same logical request as it flowed through the
    orchestrator -- e.g. metadata probing seeing one variable subset and
    the actual download seeing another. That produced a different
    dataset_id at download time than the one reported during the access/
    size steps, and the download would 404 against a dataset nobody was
    told about. This cache doesn't fix a caller passing genuinely
    different variables on purpose, but it does guarantee the *same*
    variable list always resolves to the *same* dataset_id within a
    single process run, and makes any upstream inconsistency visible in
    the logs (variables differ -> different cache key -> different id)
    rather than silently masked by fresh network calls each time.
    """
    cache_key = (tuple(sorted(v.lower() for v in variables)), requested_id or "")
    cached = _RESOLVED_DATASET_ID_CACHE.get(cache_key)
    if cached is not None:
        return cached

    resolved = _resolve_live_dataset_id_uncached(variables, requested_id)
    _RESOLVED_DATASET_ID_CACHE[cache_key] = resolved
    return resolved


def _resolve_live_dataset_id_uncached(variables: List[str], requested_id: Optional[str] = None) -> str:
    if requested_id and _dataset_id_exists(requested_id):
        return requested_id

    live = _live_dataset_list()
    if live:
        scored = [(record, _score_dataset_match(record, variables)) for record in live]
        best_record, best_score = max(scored, key=lambda x: x[1])
        if best_score > 0:
            ds_id = str(best_record.get("Dataset ID", ""))
            logger.info(
                "incois _resolve_live_dataset_id: local catalogue match score=%d, variables=%s -> %s",
                best_score, variables, ds_id,
            )
            return ds_id

    static_guess = _var_to_dataset_id(variables)
    if _dataset_id_exists(static_guess):
        return static_guess

    logger.warning(
        "incois _resolve_live_dataset_id: no live match found for variables=%s; "
        "falling back to static guess %r which is NOT confirmed to exist on the "
        "live ERDDAP server and will likely 404.", variables, static_guess,
    )
    return static_guess


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict = None, timeout: int = REQUEST_TIMEOUT) -> requests.Response:
    try:
        # Added verify=False to bypass SSL errors
        resp = requests.get(url, params=params, timeout=timeout,
                            headers={"Accept": "application/json"}, verify=False)
    except requests.exceptions.Timeout:
        raise RuntimeError(f"incois: timeout — {url}")
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"incois: connection error — {url}: {exc}")
    if not resp.ok:
        raise RuntimeError(f"incois: HTTP {resp.status_code} — {url}: {resp.text[:200]}")
    return resp


def _head(url: str, timeout: int = REQUEST_TIMEOUT) -> Optional[requests.Response]:
    try:
        # Added verify=False to bypass SSL errors
        resp = requests.head(url, timeout=timeout, allow_redirects=True, verify=False)
        if resp.status_code in (405, 501) or not resp.ok:
            return None
        return resp
    except Exception as exc:
        logger.debug("incois: HEAD failed for %s — %s", url, exc)
        return None


def _build_erddap_url(dataset_id: str, fmt: str = "nc",
                      lat_min=None, lat_max=None,
                      lon_min=None, lon_max=None,
                      start_date=None, end_date=None) -> str:
    """
    Build an ERDDAP griddap download URL with optional spatial/temporal constraints.

    ERDDAP griddap URL format:
      {base}/griddap/{dataset_id}.{format}?[variable][time_range][lat_range][lon_range]
    """
    base_url = f"{ERDDAP_GRIDDAP}/{dataset_id}.{fmt}"
    # When no constraints given, return the base URL (ERDDAP will use defaults)
    constraints: List[str] = []
    if start_date:
        constraints.append(f"time>={start_date}T00:00:00Z")
    if end_date:
        constraints.append(f"time<={end_date}T23:59:59Z")
    if lat_min is not None:
        constraints.append(f"latitude>={lat_min}")
    if lat_max is not None:
        constraints.append(f"latitude<={lat_max}")
    if lon_min is not None:
        constraints.append(f"longitude>={lon_min}")
    if lon_max is not None:
        constraints.append(f"longitude<={lon_max}")
    if constraints:
        return base_url + "?" + "&".join(constraints)
    return base_url


def _validate_content(content: bytes) -> List[str]:
    issues: List[str] = []
    if not content:
        issues.append("Downloaded content is empty")
        return issues
    head = content[:512].lower()
    if b"<html" in head or b"<!doctype" in head:
        issues.append("Response is an HTML page, not a data file")
    if b"error" in head and b"cdf" not in content[:4] and b"\x89hdf" not in content[:4]:
        logger.debug("incois validate: possible error in response")
    return issues


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class INCOISConnector(StaticDatasetConnector):
    """
    Real INCOIS provider connector.

    Uses the INCOIS ERDDAP instance for dataset discovery, metadata,
    size estimation, and data download.  Supports spatial and temporal
    subsetting via ERDDAP griddap constraint expressions.
    """

    descriptor = ConnectorDescriptor(
        connector_id="incois",
        provider_name="INCOIS",
        connector_type=ConnectorType.provider_api,
        supported_access_types=(AccessType.public, AccessType.user_credentials_required),
        supported_dataset_types=(DatasetType.gridded, DatasetType.time_series),
        supported_authentication=(AuthenticationType.none, AuthenticationType.user_credentials),
        capabilities=(
            CapabilityFlags.supports_metadata
            | CapabilityFlags.supports_api
            | CapabilityFlags.supports_dataset_search
        ),
        priority=40,
    )

    provider_keywords = ("incois", "indian national centre for ocean information services")
    api_keywords      = ("incois", "erddap", "opendap")

    datasets = [
        DatasetDescriptor(
            provider="INCOIS",
            dataset_name=ds["title"],
            collection_name="INCOIS ERDDAP",
            dataset_id=ds["dataset_id"],
            api_endpoint=ds["endpoint"],
            metadata_endpoint=f"{ERDDAP_INFO}/{ds['dataset_id']}/index.json",
            download_endpoint=ds["endpoint"],
            supported_variables=ds["variables"],
            temporal_coverage=ds["temporal"],
            spatial_coverage=ds["spatial"],
            supported_formats=[ds["format"], "CSV"],
            authentication_required=False,
            access_notes="INCOIS ERDDAP public endpoint.",
        )
        for ds in _KNOWN_DATASETS
    ]

    # ------------------------------------------------------------------
    # discover_datasets
    # ------------------------------------------------------------------

    def discover_datasets(self, snapshot=None, context=None, **kwargs) -> List[DatasetDescriptor]:
        """
        Search the INCOIS ERDDAP catalogue for matching datasets using the
        cached full dataset list instead of per-query ERDDAP search calls.

        ERDDAP's /search/index.json returns HTTP 404 when zero results match
        a query — that's normal ERDDAP behavior, but it caused the connector
        to log spurious errors and fall through to hardcoded IDs that don't
        exist on the live server. The fix: fetch /griddap/index.json once
        (full catalogue, never 404s), cache it, and match locally.
        """
        r = self._as_dict(context or kwargs)
        keywords = r.get("keywords") or r.get("variables") or []
        if isinstance(keywords, str):
            keywords = [keywords]

        # Try catalogue-based local matching
        live = _live_dataset_list()
        if live:
            scored = [
                (record, _score_dataset_match(record, keywords))
                for record in live
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            descriptors: List[DatasetDescriptor] = []
            for record, score in scored[:5]:  # top 5 matches
                if score == 0 and descriptors:
                    break  # stop once we've exhausted meaningful matches
                ds_id = str(record.get("Dataset ID", ""))
                title = str(record.get("Title", ds_id))
                endpoint = f"{ERDDAP_GRIDDAP}/{ds_id}"
                logger.info("incois: discovered ERDDAP dataset (score=%d) %s — %s", score, ds_id, title)
                descriptors.append(DatasetDescriptor(
                    provider="INCOIS",
                    dataset_name=title[:120],
                    collection_name="INCOIS ERDDAP",
                    dataset_id=ds_id,
                    api_endpoint=endpoint,
                    metadata_endpoint=f"{ERDDAP_INFO}/{ds_id}/index.json",
                    download_endpoint=endpoint,
                    supported_variables=keywords,
                    temporal_coverage=str(record.get("Min Time", "Dataset dependent")),
                    spatial_coverage="Indian Ocean region",
                    supported_formats=["NetCDF", "CSV"],
                    authentication_required=False,
                    access_notes=f"INCOIS ERDDAP dataset ID: {ds_id}",
                ))
            if descriptors:
                return descriptors

        logger.warning("incois discover_datasets: ERDDAP catalogue unavailable. Returning known datasets.")
        # Match known datasets by variable
        if keywords:
            matched_id = _var_to_dataset_id(keywords)
            return [d for d in self.datasets if d.dataset_id == matched_id] or self.datasets
        return self.datasets

    # ------------------------------------------------------------------
    # probe_metadata
    # ------------------------------------------------------------------

    def probe_metadata(self, snapshot=None, fetch_request=None, **kwargs) -> DatasetMetadata:
        """
        Retrieve live metadata from the ERDDAP info endpoint for the dataset.
        Returns a DatasetMetadata object instead of a raw dictionary.
        """
        r = self._as_dict(fetch_request or kwargs)
        variables = r.get("variables") or r.get("keywords") or []
        if isinstance(variables, str):
            variables = [variables]
        dataset_id = _resolve_live_dataset_id(variables, r.get("dataset_id"))
        source_id = getattr(snapshot, "source_id", "incois")

        info_url = f"{ERDDAP_INFO}/{dataset_id}/index.json"
        logger.info("incois probe_metadata: GET %s", info_url)

        unavailable_reason = ""
        try:
            resp = _get(info_url)
            erddap_meta = resp.json()
        except Exception as exc:
            logger.warning("incois probe_metadata: ERDDAP info failed — %s", exc)
            unavailable_reason = f"ERDDAP info failed: {exc}"

        # Build a download URL for size estimation
        download_url = _build_erddap_url(
            dataset_id,
            lat_min=r.get("lat_min") or r.get("south"),
            lat_max=r.get("lat_max") or r.get("north"),
            lon_min=r.get("lon_min") or r.get("west"),
            lon_max=r.get("lon_max") or r.get("east"),
            start_date=r.get("start_date"),
            end_date=r.get("end_date"),
        )
        logger.info("incois probe_metadata: resolved asset URL = %s", download_url)

        size_bytes, size_method, size_conf = self._estimate_size(download_url)
        logger.info("incois probe_metadata: size=%s method=%s", size_bytes, size_method)

        # Return a structured DatasetMetadata object to avoid AttributeError
        return DatasetMetadata(
            source_id=source_id,
            dataset_id=dataset_id,
            collection="INCOIS ERDDAP",
            product=dataset_id,
            download_endpoint=download_url,
            api_endpoint=f"{ERDDAP_GRIDDAP}/{dataset_id}",
            metadata_endpoint=info_url,
            file_size_bytes=float(size_bytes) if size_bytes else 50.0 * 1024 * 1024,
            variables=variables,
            file_format="NetCDF",
            checksum=None,
            content_type="application/x-netcdf",
            retrieval_method=size_method or "INCOIS API",
            unavailable_reason=unavailable_reason,
        )

    # ------------------------------------------------------------------
    # probe_size
    # ------------------------------------------------------------------

    def probe_size(self, snapshot=None, fetch_request=None, **kwargs) -> SizeEstimate:
        """
        Estimate size via HEAD on ERDDAP download URL.
        Fallback: ERDDAP rows × variables × datatype size estimation.
        Returns SizeEstimate for Agent 4 compatibility.
        """
        source_id = getattr(snapshot, "source_id", None) or "incois"
        r = self._as_dict(fetch_request or kwargs)
        variables = r.get("variables") or r.get("keywords") or []
        if isinstance(variables, str):
            variables = [variables]
        dataset_id = _resolve_live_dataset_id(variables, r.get("dataset_id"))

        download_url = _build_erddap_url(
            dataset_id,
            lat_min=r.get("lat_min") or r.get("south"),
            lat_max=r.get("lat_max") or r.get("north"),
            lon_min=r.get("lon_min") or r.get("west"),
            lon_max=r.get("lon_max") or r.get("east"),
            start_date=r.get("start_date"),
            end_date=r.get("end_date"),
        )

        # Priority 1: HEAD Content-Length
        logger.info("incois probe_size: HEAD %s", download_url)
        size_bytes, method, _ = self._estimate_size(download_url)
        if size_bytes is not None:
            return SizeEstimate(
                source_id=source_id,
                estimated_bytes=float(size_bytes),
                is_exact=True,
                method=f"INCOIS ERDDAP {method}",
                human_readable=format_bytes(size_bytes),
            )

        # Priority 2: ERDDAP griddap dimension-based estimation
        # Indian Ocean grid: ~90° lat × 150° lon at 0.25° = 360×600 = 216,000 grid points
        # Default to Indian Ocean spatial extent if no bbox given
        lat_min = float(r.get("lat_min") or r.get("south") or -40)
        lat_max = float(r.get("lat_max") or r.get("north") or 30)
        lon_min = float(r.get("lon_min") or r.get("west") or 30)
        lon_max = float(r.get("lon_max") or r.get("east") or 120)
        lat_pts = max(1, int((lat_max - lat_min) / 0.25))
        lon_pts = max(1, int((lon_max - lon_min) / 0.25))
        grid_pts = lat_pts * lon_pts

        # Temporal: days in range
        try:
            from datetime import datetime as _dt
            start_str = r.get("start_date") or "2020-01-01"
            end_str   = r.get("end_date")   or "2020-01-31"
            t_days = max(1, (_dt.fromisoformat(end_str[:10]) - _dt.fromisoformat(start_str[:10])).days + 1)
        except Exception:
            t_days = 30  # safe default

        n_vars = max(1, len(variables) if variables else 1)
        # 4 bytes per float32 value, light NetCDF overhead factor ~1.05
        est_bytes = int(grid_pts * t_days * n_vars * 4 * 1.05)
        logger.info(
            "incois probe_size: ERDDAP grid estimate %d bytes "
            "(lat=%d×lon=%d×days=%d×vars=%d×4B)",
            est_bytes, lat_pts, lon_pts, t_days, n_vars,
        )
        return SizeEstimate(
            source_id=source_id,
            estimated_bytes=float(est_bytes),
            is_exact=False,
            method="INCOIS ERDDAP grid estimation (lat×lon×days×vars×4 bytes)",
            human_readable=format_bytes(est_bytes),
        )

    # ------------------------------------------------------------------
    # resolve_download_asset
    # ------------------------------------------------------------------

    def resolve_download_asset(self, snapshot=None, fetch_request=None, credentials=None, **kwargs) -> Dict[str, Any]:
        """Resolve the ERDDAP griddap download URL for the requested dataset."""
        r = self._as_dict(fetch_request or kwargs)
        variables = r.get("variables") or r.get("keywords") or []
        if isinstance(variables, str):
            variables = [variables]
        dataset_id = _resolve_live_dataset_id(variables, r.get("dataset_id"))

        url = _build_erddap_url(
            dataset_id,
            lat_min=r.get("lat_min") or r.get("south"),
            lat_max=r.get("lat_max") or r.get("north"),
            lon_min=r.get("lon_min") or r.get("west"),
            lon_max=r.get("lon_max") or r.get("east"),
            start_date=r.get("start_date"),
            end_date=r.get("end_date"),
        )
        logger.info("incois resolve_download_asset: %s → %s", dataset_id, url)
        return {"download_url": url, "dataset_id": dataset_id, "format": "NetCDF"}

    # ------------------------------------------------------------------
    # fetch_subset  (ERDDAP griddap supports server-side subsetting natively)
    # ------------------------------------------------------------------

    def fetch_subset(self, snapshot=None, fetch_request=None, credentials=None, output_dir=None, **kwargs) -> str:
        """
        Download a spatial/temporal subset via ERDDAP griddap constraint expressions.
        ERDDAP handles subsetting server-side; no local clipping required.

        Returns the local file path as a string, matching the BaseConnector
        contract (fetch_subset/fetch_full -> str). Previously this returned
        the raw result dict from _download_url(), which download_manager's
        _move_to_dest() then passed straight into os.path.exists() expecting
        a string -- raising "_path_exists: path should be string, bytes,
        os.PathLike or integer, not dict".
        """
        r = self._as_dict(fetch_request or kwargs)
        variables = r.get("variables") or r.get("keywords") or []
        if isinstance(variables, str):
            variables = [variables]
        dataset_id = _resolve_live_dataset_id(variables, r.get("dataset_id"))

        url = _build_erddap_url(
            dataset_id,
            lat_min=r.get("lat_min") or r.get("south"),
            lat_max=r.get("lat_max") or r.get("north"),
            lon_min=r.get("lon_min") or r.get("west"),
            lon_max=r.get("lon_max") or r.get("east"),
            start_date=r.get("start_date"),
            end_date=r.get("end_date"),
        )
        logger.info("incois fetch_subset: server-side subset via ERDDAP — %s", url)
        result = self._download_url(url, output_dir)
        if not result.get("success"):
            raise RuntimeError(result.get("error") or f"incois fetch_subset failed for {url}")
        return result["file_path"]

    # ------------------------------------------------------------------
    # fetch_full
    # ------------------------------------------------------------------

    def fetch_full(self, snapshot=None, fetch_request=None, credentials=None, output_dir=None, **kwargs) -> str:
        """Download the full INCOIS dataset via ERDDAP griddap.

        Returns the local file path as a string (see fetch_subset docstring
        for why this must not be the raw result dict).
        """
        r = self._as_dict(fetch_request or kwargs)
        variables = r.get("variables") or r.get("keywords") or []
        if isinstance(variables, str):
            variables = [variables]
        dataset_id = _resolve_live_dataset_id(variables, r.get("dataset_id"))

        url = _build_erddap_url(dataset_id)
        logger.info("incois fetch_full: GET %s", url)
        result = self._download_url(url, output_dir)
        if not result.get("success"):
            raise RuntimeError(result.get("error") or f"incois fetch_full failed for {url}")
        return result["file_path"]

    # ------------------------------------------------------------------
    # validate_download
    # ------------------------------------------------------------------

    def validate_download(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Validate a downloaded INCOIS NetCDF or CSV file."""
        if not os.path.exists(file_path):
            return {"valid": False, "issues": [f"File not found: {file_path}"]}
        size = os.path.getsize(file_path)
        if size == 0:
            return {"valid": False, "issues": ["File is empty"]}

        with open(file_path, "rb") as fh:
            head = fh.read(512)

        issues = _validate_content(head)
        if issues:
            logger.warning("incois validate_download: FAILED — %s", issues)
            return {"valid": False, "issues": issues}

        logger.info("incois validate_download: passed (%d bytes) — %s", size, file_path)
        return {"valid": True, "issues": [], "size_bytes": size}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_dict(obj) -> dict:
        if isinstance(obj, dict):
            d = dict(obj)
        elif hasattr(obj, "__dict__"):
            d = dict(vars(obj))
        else:
            d = {}
        meta = d.get("metadata")
        if isinstance(meta, dict):
            merged = dict(meta)
            merged.update({k: v for k, v in d.items() if k != "metadata" and v is not None})
            return merged
        return d

    @staticmethod
    def _estimate_size(url: str) -> Tuple[Optional[int], str, float]:
        """HEAD → GET body measurement → unknown."""
        head_resp = _head(url)
        if head_resp:
            cl = head_resp.headers.get("Content-Length")
            if cl and cl.isdigit():
                logger.debug("incois: HEAD Content-Length = %s bytes", cl)
                return int(cl), "HEAD Content-Length", 0.95
        return None, "unknown", 0.0

    @staticmethod
    def _size_dict(size_bytes: int, method: str, confidence: float,
                   url: str) -> Dict[str, Any]:
        if size_bytes < 1024:
            human = f"{size_bytes} B"
        elif size_bytes < 1024 ** 2:
            human = f"{size_bytes / 1024:.1f} KB"
        else:
            human = f"{size_bytes / (1024 ** 2):.2f} MB"
        return {"size_bytes": size_bytes, "size_human": human,
                "confidence": confidence, "method": method, "request_url": url}

    def _download_url(self, url: str, output_dir) -> Dict[str, Any]:
        """Download a URL, validate, and return result dict."""
        logger.info("incois _download_url: GET %s", url)
        try:
            # Added verify=False to bypass SSL errors during download
            resp = requests.get(url, timeout=300, stream=True, verify=False)
        except Exception as exc:
            return {"success": False, "error": str(exc), "request_url": url}

        if not resp.ok:
            return {"success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                    "request_url": url}

        content = resp.content
        issues  = _validate_content(content)
        if issues:
            return {"success": False, "error": "; ".join(issues), "request_url": url}

        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="incois_")
        os.makedirs(output_dir, exist_ok=True)

        ct    = resp.headers.get("Content-Type", "")
        ext   = "csv" if "csv" in ct else "nc"
        ts    = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        fpath = os.path.join(output_dir, f"incois_{ts}.{ext}")

        with open(fpath, "wb") as fh:
            fh.write(content)

        size_bytes = os.path.getsize(fpath)
        logger.info("incois: saved %d bytes → %s", size_bytes, fpath)

        validation = self.validate_download(fpath)
        if not validation["valid"]:
            logger.error("incois: post-save validation FAILED — %s. Deleting.", validation["issues"])
            try:
                os.remove(fpath)
            except OSError:
                pass
            return {"success": False,
                    "error": "Post-save validation failed: " + "; ".join(validation["issues"]),
                    "request_url": url}

        logger.info("incois: validation passed")
        return {
            "success":     True,
            "file_path":   fpath,
            "size_bytes":  size_bytes,
            "request_url": url,
            "format":      ext.upper(),
            "content_type": ct,
        }


register_connector(INCOISConnector)