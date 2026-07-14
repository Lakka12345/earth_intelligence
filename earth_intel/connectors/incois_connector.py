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
from models.agent4_schemas import DatasetDescriptor

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
    """Match requested variables to best known dataset_id."""
    vl = [v.lower() for v in variables]
    for ds in _KNOWN_DATASETS:
        for dv in ds["variables"]:
            if any(dv in v or v in dv for v in vl):
                return ds["dataset_id"]
    return _KNOWN_DATASETS[0]["dataset_id"]  # default: SST


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict = None, timeout: int = REQUEST_TIMEOUT) -> requests.Response:
    try:
        resp = requests.get(url, params=params, timeout=timeout,
                            headers={"Accept": "application/json"})
    except requests.exceptions.Timeout:
        raise RuntimeError(f"incois: timeout — {url}")
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"incois: connection error — {url}: {exc}")
    if not resp.ok:
        raise RuntimeError(f"incois: HTTP {resp.status_code} — {url}: {resp.text[:200]}")
    return resp


def _head(url: str, timeout: int = REQUEST_TIMEOUT) -> Optional[requests.Response]:
    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
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

    def discover_datasets(self, fetch_request=None, **kwargs) -> List[DatasetDescriptor]:
        """
        Search ERDDAP for matching datasets.  Falls back to _KNOWN_DATASETS
        when the ERDDAP search endpoint is unavailable.
        """
        r = self._as_dict(fetch_request or kwargs)
        keywords = r.get("keywords") or r.get("variables") or []
        if isinstance(keywords, str):
            keywords = [keywords]

        # Try ERDDAP search
        search_term = " ".join(keywords) if keywords else "ocean"
        logger.info("incois discover_datasets: ERDDAP search = %r", search_term)
        try:
            resp = _get(ERDDAP_SEARCH, params={"searchFor": search_term, "page": 1,
                                                "itemsPerPage": 10})
            data = resp.json()
            rows = data.get("table", {}).get("rows", [])
            cols = data.get("table", {}).get("columnNames", [])
            idx  = {c: i for i, c in enumerate(cols)}

            descriptors: List[DatasetDescriptor] = []
            for row in rows:
                ds_id    = row[idx.get("Dataset ID", 0)] if idx else row[0]
                title    = row[idx.get("Title", 1)]      if idx else row[1]
                endpoint = f"{ERDDAP_GRIDDAP}/{ds_id}"
                logger.info("incois: discovered ERDDAP dataset %s — %s", ds_id, title)
                descriptors.append(DatasetDescriptor(
                    provider="INCOIS",
                    dataset_name=str(title)[:120],
                    collection_name="INCOIS ERDDAP",
                    dataset_id=str(ds_id),
                    api_endpoint=endpoint,
                    metadata_endpoint=f"{ERDDAP_INFO}/{ds_id}/index.json",
                    download_endpoint=endpoint,
                    supported_variables=keywords,
                    temporal_coverage="Dataset dependent",
                    spatial_coverage="Indian Ocean region",
                    supported_formats=["NetCDF", "CSV"],
                    authentication_required=False,
                    access_notes=f"INCOIS ERDDAP dataset ID: {ds_id}",
                ))
            if descriptors:
                return descriptors
        except Exception as exc:
            logger.warning("incois discover_datasets: ERDDAP search failed — %s. "
                           "Returning known datasets.", exc)

        # Match known datasets by variable
        if keywords:
            matched_id = _var_to_dataset_id(keywords)
            return [d for d in self.datasets if d.dataset_id == matched_id] or self.datasets
        return self.datasets

    # ------------------------------------------------------------------
    # probe_metadata
    # ------------------------------------------------------------------

    def probe_metadata(self, fetch_request=None, **kwargs) -> Dict[str, Any]:
        """
        Retrieve live metadata from the ERDDAP info endpoint for the dataset.
        """
        r = self._as_dict(fetch_request or kwargs)
        variables = r.get("variables") or r.get("keywords") or []
        if isinstance(variables, str):
            variables = [variables]
        dataset_id = r.get("dataset_id") or _var_to_dataset_id(variables)

        info_url = f"{ERDDAP_INFO}/{dataset_id}/index.json"
        logger.info("incois probe_metadata: GET %s", info_url)

        erddap_meta: dict = {}
        try:
            resp  = _get(info_url)
            erddap_meta = resp.json()
        except Exception as exc:
            logger.warning("incois probe_metadata: ERDDAP info failed — %s", exc)

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

        return {
            "provider":              "INCOIS",
            "dataset_id":            dataset_id,
            "erddap_info":           erddap_meta,
            "download_url":          download_url,
            "size_bytes":            size_bytes,
            "size_estimation_method": size_method,
            "size_confidence":       size_conf,
            "api_endpoint":          f"{ERDDAP_GRIDDAP}/{dataset_id}",
            "info_endpoint":         info_url,
            "authentication_required": False,
        }

    # ------------------------------------------------------------------
    # probe_size
    # ------------------------------------------------------------------

    def probe_size(self, fetch_request=None, **kwargs) -> Dict[str, Any]:
        """
        Estimate size via HEAD on ERDDAP download URL.
        """
        r = self._as_dict(fetch_request or kwargs)
        variables = r.get("variables") or r.get("keywords") or []
        if isinstance(variables, str):
            variables = [variables]
        dataset_id = r.get("dataset_id") or _var_to_dataset_id(variables)

        download_url = _build_erddap_url(
            dataset_id,
            lat_min=r.get("lat_min") or r.get("south"),
            lat_max=r.get("lat_max") or r.get("north"),
            lon_min=r.get("lon_min") or r.get("west"),
            lon_max=r.get("lon_max") or r.get("east"),
            start_date=r.get("start_date"),
            end_date=r.get("end_date"),
        )

        logger.info("incois probe_size: HEAD %s", download_url)
        size_bytes, method, confidence = self._estimate_size(download_url)

        if size_bytes is None:
            return {"size_bytes": None, "size_human": "Unknown", "confidence": 0.0,
                    "method": "unknown", "request_url": download_url}

        return self._size_dict(size_bytes, method, confidence, download_url)

    # ------------------------------------------------------------------
    # resolve_download_asset
    # ------------------------------------------------------------------

    def resolve_download_asset(self, fetch_request=None, **kwargs) -> Dict[str, Any]:
        """Resolve the ERDDAP griddap download URL for the requested dataset."""
        r = self._as_dict(fetch_request or kwargs)
        variables = r.get("variables") or r.get("keywords") or []
        if isinstance(variables, str):
            variables = [variables]
        dataset_id = r.get("dataset_id") or _var_to_dataset_id(variables)

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

    def fetch_subset(self, fetch_request=None, output_dir=None, **kwargs) -> Dict[str, Any]:
        """
        Download a spatial/temporal subset via ERDDAP griddap constraint expressions.
        ERDDAP handles subsetting server-side; no local clipping required.
        """
        r = self._as_dict(fetch_request or kwargs)
        variables = r.get("variables") or r.get("keywords") or []
        if isinstance(variables, str):
            variables = [variables]
        dataset_id = r.get("dataset_id") or _var_to_dataset_id(variables)

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
        result["subset_note"] = "Spatial/temporal subsetting applied server-side via ERDDAP."
        return result

    # ------------------------------------------------------------------
    # fetch_full
    # ------------------------------------------------------------------

    def fetch_full(self, fetch_request=None, output_dir=None, **kwargs) -> Dict[str, Any]:
        """Download the full INCOIS dataset via ERDDAP griddap."""
        r = self._as_dict(fetch_request or kwargs)
        variables = r.get("variables") or r.get("keywords") or []
        if isinstance(variables, str):
            variables = [variables]
        dataset_id = r.get("dataset_id") or _var_to_dataset_id(variables)

        url = _build_erddap_url(dataset_id)
        logger.info("incois fetch_full: GET %s", url)
        return self._download_url(url, output_dir)

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
            return obj
        return vars(obj) if hasattr(obj, "__dict__") else {}

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
            resp = requests.get(url, timeout=300, stream=True)
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
