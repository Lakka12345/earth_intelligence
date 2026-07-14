"""
DataONE Connector — Real Provider Implementation
=================================================
Uses the DataONE Coordinating Node Solr REST API to:
  • Search and discover datasets by keyword / time / bbox
  • Resolve object identifiers to downloadable resources
  • Probe metadata and size via HEAD + Content-Length
  • Fetch full objects and validate them
  • Report checksums from DataONE system metadata

Official API: https://cn.dataone.org/cn/v2/query/solr/
Object resolve: https://cn.dataone.org/cn/v2/resolve/{pid}
System metadata: https://cn.dataone.org/cn/v2/meta/{pid}
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlencode

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
# Constants
# ---------------------------------------------------------------------------

CN_SOLR      = "https://cn.dataone.org/cn/v2/query/solr/"
CN_RESOLVE   = "https://cn.dataone.org/cn/v2/resolve/"
CN_META      = "https://cn.dataone.org/cn/v2/meta/"
REQUEST_TIMEOUT = 30

# Fields to request from Solr
_SOLR_FIELDS = (
    "id,identifier,title,abstract,author,authoritativeMN,"
    "formatId,formatType,size,checksum,checksumAlgorithm,"
    "dateUploaded,dateModified,beginDate,endDate,"
    "northBoundCoord,southBoundCoord,eastBoundCoord,westBoundCoord,"
    "origin,dataUrl,resourceMap,isDocumentedBy"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict = None, timeout: int = REQUEST_TIMEOUT) -> requests.Response:
    """GET with error handling."""
    try:
        resp = requests.get(url, params=params, timeout=timeout,
                            headers={"Accept": "application/json"})
    except requests.exceptions.Timeout:
        raise RuntimeError(f"dataone: timeout — {url}")
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"dataone: connection error — {url}: {exc}")
    if not resp.ok:
        raise RuntimeError(f"dataone: HTTP {resp.status_code} — {url}: {resp.text[:200]}")
    return resp


def _head(url: str, timeout: int = REQUEST_TIMEOUT) -> Optional[requests.Response]:
    """HEAD with graceful failure."""
    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
        if resp.status_code in (405, 501) or not resp.ok:
            return None
        return resp
    except Exception as exc:
        logger.debug("dataone: HEAD failed for %s — %s", url, exc)
        return None


def _solr_search(query: str, extra: dict = None, rows: int = 5) -> dict:
    """Run a Solr query and return parsed JSON."""
    params = {
        "q":    query,
        "fl":   _SOLR_FIELDS,
        "rows": rows,
        "wt":   "json",
    }
    if extra:
        params.update(extra)
    resp = _get(CN_SOLR, params=params)
    return resp.json()


def _estimate_size(pid: str) -> Tuple[Optional[int], str, float]:
    """
    Estimate download size for a DataONE object.
    Priority: Solr `size` field → HEAD Content-Length → unknown.
    """
    # Try HEAD on the resolve URL
    resolve_url = CN_RESOLVE + quote(pid, safe="")
    head_resp = _head(resolve_url)
    if head_resp:
        cl = head_resp.headers.get("Content-Length")
        if cl and cl.isdigit():
            logger.debug("dataone: size from HEAD Content-Length = %s bytes (pid=%s)", cl, pid)
            return int(cl), "HEAD Content-Length", 0.95
    return None, "unknown", 0.0


def _resolve_download_url(pid: str) -> str:
    """Return the CN resolve URL for a data object PID."""
    return CN_RESOLVE + quote(pid, safe="")


def _validate_content(content: bytes, file_path: str) -> List[str]:
    """Return a list of issue strings; empty means valid."""
    issues: List[str] = []
    if not content:
        issues.append("Downloaded content is empty")
        return issues
    head = content[:512].lower()
    if b"<html" in head or b"<!doctype" in head:
        issues.append("Response is an HTML page, not a data file")
    if b"404 not found" in head:
        issues.append("Response contains 404 error page")
    return issues


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class DataONEConnector(StaticDatasetConnector):
    """
    Real DataONE provider connector.

    Uses the DataONE Coordinating Node Solr API for search and object
    resolution.  Downloads are retrieved via the CN resolve endpoint which
    redirects to the authoritative member node.
    """

    descriptor = ConnectorDescriptor(
        connector_id="dataone",
        provider_name="DataONE",
        connector_type=ConnectorType.provider_api,
        supported_access_types=(AccessType.public, AccessType.user_credentials_required),
        supported_dataset_types=(DatasetType.tabular, DatasetType.vector,
                                  DatasetType.raster, DatasetType.time_series),
        supported_authentication=(AuthenticationType.none, AuthenticationType.user_credentials),
        capabilities=(
            CapabilityFlags.supports_metadata
            | CapabilityFlags.supports_api
            | CapabilityFlags.supports_dataset_search
        ),
        priority=45,
    )

    provider_keywords = ("dataone", "data one", "knb", "arctic data center")
    api_keywords      = ("dataone", "solr")

    datasets = [
        DatasetDescriptor(
            provider="DataONE",
            dataset_name="DataONE Coordinating Node Search Result",
            collection_name="DataONE Member Node Catalog",
            dataset_id="dataone-cn-search",
            api_endpoint=CN_SOLR,
            metadata_endpoint="https://search.dataone.org/",
            download_endpoint=CN_RESOLVE,
            supported_variables=["ecology", "biodiversity", "land use", "climate",
                                  "environmental observations"],
            temporal_coverage="Dataset dependent",
            spatial_coverage="Dataset dependent",
            supported_formats=["Science metadata", "CSV", "NetCDF", "GeoTIFF",
                                "dataset dependent"],
            authentication_required=False,
            access_notes="Public metadata via DataONE API; object access depends on member node policy.",
        )
    ]

    # ------------------------------------------------------------------
    # discover_datasets
    # ------------------------------------------------------------------

    def discover_datasets(self, fetch_request=None, **kwargs) -> List[DatasetDescriptor]:
        """
        Search the DataONE Solr index and return matching dataset descriptors.
        Supports keyword, temporal (start_date/end_date), and bbox filtering.
        """
        r = self._as_dict(fetch_request or kwargs)
        query_parts = ["formatType:DATA"]

        keywords = r.get("keywords") or r.get("variables") or []
        if isinstance(keywords, str):
            keywords = [keywords]
        if keywords:
            kw_clause = " OR ".join(f'"{k}"' for k in keywords)
            query_parts.append(f"({kw_clause})")

        if r.get("start_date"):
            query_parts.append(f"beginDate:[{r['start_date']}T00:00:00Z TO *]")
        if r.get("end_date"):
            query_parts.append(f"endDate:[* TO {r['end_date']}T23:59:59Z]")

        # Bounding box
        lat_min = r.get("lat_min") or r.get("south")
        lat_max = r.get("lat_max") or r.get("north")
        lon_min = r.get("lon_min") or r.get("west")
        lon_max = r.get("lon_max") or r.get("east")
        if all(v is not None for v in (lat_min, lat_max, lon_min, lon_max)):
            query_parts.append(
                f"northBoundCoord:[{lat_min} TO 90] AND southBoundCoord:[-90 TO {lat_max}] "
                f"AND eastBoundCoord:[{lon_min} TO 180] AND westBoundCoord:[-180 TO {lon_max}]"
            )

        query = " AND ".join(query_parts)
        logger.info("dataone discover_datasets: Solr query = %r", query)

        try:
            result = _solr_search(query, rows=10)
        except Exception as exc:
            logger.warning("dataone discover_datasets: search failed — %s", exc)
            return self.datasets

        docs = result.get("response", {}).get("docs", [])
        if not docs:
            logger.info("dataone discover_datasets: no results; returning catalog descriptor")
            return self.datasets

        descriptors: List[DatasetDescriptor] = []
        for doc in docs:
            pid = doc.get("identifier") or doc.get("id", "")
            logger.info("dataone: discovered dataset pid=%s title=%s", pid, doc.get("title", ""))
            descriptors.append(DatasetDescriptor(
                provider="DataONE",
                dataset_name=doc.get("title", pid)[:120],
                collection_name=doc.get("authoritativeMN", "DataONE Member Node"),
                dataset_id=pid,
                api_endpoint=CN_SOLR,
                metadata_endpoint=CN_META + quote(pid, safe=""),
                download_endpoint=_resolve_download_url(pid),
                supported_variables=[],
                temporal_coverage=(
                    f"{doc.get('beginDate', '?')} — {doc.get('endDate', '?')}"
                ),
                spatial_coverage=(
                    f"N:{doc.get('northBoundCoord')} S:{doc.get('southBoundCoord')} "
                    f"E:{doc.get('eastBoundCoord')} W:{doc.get('westBoundCoord')}"
                    if doc.get("northBoundCoord") else "Dataset dependent"
                ),
                supported_formats=[doc.get("formatId", "unknown")],
                authentication_required=False,
                access_notes=f"DataONE CN resolve — PID: {pid}",
            ))
        return descriptors

    # ------------------------------------------------------------------
    # probe_metadata
    # ------------------------------------------------------------------

    def probe_metadata(self, fetch_request=None, **kwargs) -> Dict[str, Any]:
        """
        Retrieve live metadata from the DataONE CN Solr index for the given PID.
        Falls back to a catalog-level descriptor when no PID is provided.
        """
        r = self._as_dict(fetch_request or kwargs)
        pid = r.get("dataset_id") or r.get("identifier") or r.get("pid")

        if not pid:
            logger.info("dataone probe_metadata: no PID — returning catalog descriptor")
            return {
                "provider":    "DataONE",
                "api_endpoint": CN_SOLR,
                "note":        "Supply dataset_id/pid for object-level metadata",
            }

        logger.info("dataone probe_metadata: querying CN Solr for pid=%s", pid)
        try:
            result = _solr_search(f'id:"{pid}"', rows=1)
        except Exception as exc:
            return {"error": str(exc), "pid": pid}

        docs = result.get("response", {}).get("docs", [])
        doc  = docs[0] if docs else {}

        size_bytes, size_method, size_conf = _estimate_size(pid)
        resolve_url = _resolve_download_url(pid)

        logger.info("dataone probe_metadata: pid=%s size=%s method=%s", pid, size_bytes, size_method)
        return {
            "provider":             "DataONE",
            "pid":                  pid,
            "title":                doc.get("title"),
            "author":               doc.get("author"),
            "format_id":            doc.get("formatId"),
            "size_bytes":           size_bytes or doc.get("size"),
            "size_estimation_method": size_method,
            "size_confidence":      size_conf,
            "checksum":             doc.get("checksum"),
            "checksum_algorithm":   doc.get("checksumAlgorithm"),
            "date_uploaded":        doc.get("dateUploaded"),
            "date_modified":        doc.get("dateModified"),
            "begin_date":           doc.get("beginDate"),
            "end_date":             doc.get("endDate"),
            "north":                doc.get("northBoundCoord"),
            "south":                doc.get("southBoundCoord"),
            "east":                 doc.get("eastBoundCoord"),
            "west":                 doc.get("westBoundCoord"),
            "authoritative_mn":     doc.get("authoritativeMN"),
            "resolve_url":          resolve_url,
            "metadata_url":         CN_META + quote(pid, safe=""),
        }

    # ------------------------------------------------------------------
    # probe_size
    # ------------------------------------------------------------------

    def probe_size(self, fetch_request=None, **kwargs) -> Dict[str, Any]:
        """
        Estimate download size.
        Priority: HEAD Content-Length on resolve URL → Solr `size` field → unknown.
        """
        r = self._as_dict(fetch_request or kwargs)
        pid = r.get("dataset_id") or r.get("pid")
        if not pid:
            return {"size_bytes": None, "size_human": "Unknown", "confidence": 0.0,
                    "method": "no PID supplied"}

        resolve_url = _resolve_download_url(pid)
        logger.info("dataone probe_size: HEAD %s", resolve_url)

        # Step 1: HEAD
        head_resp = _head(resolve_url)
        if head_resp:
            cl = head_resp.headers.get("Content-Length")
            if cl and cl.isdigit():
                sz = int(cl)
                logger.info("dataone probe_size: HEAD Content-Length = %d bytes", sz)
                return self._size_dict(sz, "HEAD Content-Length", 0.95, resolve_url)

        # Step 2: Solr size field
        try:
            result = _solr_search(f'id:"{pid}"', rows=1)
            doc = (result.get("response", {}).get("docs") or [{}])[0]
            sz = doc.get("size")
            if sz:
                logger.info("dataone probe_size: Solr size field = %s bytes", sz)
                return self._size_dict(int(sz), "provider metadata (Solr size field)", 0.80,
                                       resolve_url)
        except Exception as exc:
            logger.debug("dataone probe_size: Solr lookup failed — %s", exc)

        logger.warning("dataone probe_size: could not determine size for pid=%s", pid)
        return {"size_bytes": None, "size_human": "Unknown", "confidence": 0.0,
                "method": "unknown", "request_url": resolve_url}

    # ------------------------------------------------------------------
    # resolve_download_asset
    # ------------------------------------------------------------------

    def resolve_download_asset(self, fetch_request=None, **kwargs) -> Dict[str, Any]:
        """Return the CN resolve URL for the requested data object."""
        r = self._as_dict(fetch_request or kwargs)
        pid = r.get("dataset_id") or r.get("pid")
        if not pid:
            return {"error": "No dataset_id / pid provided"}
        url = _resolve_download_url(pid)
        logger.info("dataone resolve_download_asset: pid=%s → %s", pid, url)
        return {"download_url": url, "pid": pid, "resolve_endpoint": CN_RESOLVE}

    # ------------------------------------------------------------------
    # fetch_subset  (DataONE has no server-side subsetting)
    # ------------------------------------------------------------------

    def fetch_subset(self, fetch_request=None, output_dir=None, **kwargs) -> Dict[str, Any]:
        """
        DataONE does not support server-side spatial/temporal subsetting.
        Downloads the full object and reports that local subsetting is required.
        """
        logger.info("dataone fetch_subset: no server-side subsetting — performing full download")
        result = self.fetch_full(fetch_request, output_dir=output_dir, **kwargs)
        result["subset_note"] = (
            "DataONE provides no server-side subsetting. Full object downloaded; "
            "apply spatial/temporal filters locally."
        )
        return result

    # ------------------------------------------------------------------
    # fetch_full
    # ------------------------------------------------------------------

    def fetch_full(self, fetch_request=None, output_dir=None, **kwargs) -> Dict[str, Any]:
        """Download a DataONE data object via the CN resolve endpoint."""
        r = self._as_dict(fetch_request or kwargs)
        pid = r.get("dataset_id") or r.get("pid")
        if not pid:
            return {"success": False, "error": "No dataset_id / pid provided"}

        url = _resolve_download_url(pid)
        logger.info("dataone fetch_full: GET %s", url)

        try:
            resp = requests.get(url, timeout=120, stream=True)
        except Exception as exc:
            return {"success": False, "error": str(exc), "request_url": url}

        if not resp.ok:
            return {"success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                    "request_url": url}

        content = resp.content
        issues  = _validate_content(content, "")
        if issues:
            return {"success": False, "error": "; ".join(issues), "request_url": url}

        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="dataone_")
        os.makedirs(output_dir, exist_ok=True)

        # Determine extension from Content-Type or format ID
        ct  = resp.headers.get("Content-Type", "")
        ext = "nc" if "netcdf" in ct else ("csv" if "csv" in ct else "bin")
        ts  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        fname = f"dataone_{ts}.{ext}"
        fpath = os.path.join(output_dir, fname)

        with open(fpath, "wb") as fh:
            fh.write(content)

        size_bytes = os.path.getsize(fpath)
        logger.info("dataone fetch_full: saved %d bytes → %s", size_bytes, fpath)

        validation = self.validate_download(fpath)
        if not validation["valid"]:
            logger.error("dataone fetch_full: post-save validation FAILED — %s. Deleting.",
                         validation["issues"])
            try:
                os.remove(fpath)
            except OSError:
                pass
            return {"success": False,
                    "error": "Post-save validation failed: " + "; ".join(validation["issues"]),
                    "request_url": url}

        logger.info("dataone fetch_full: validation passed")
        return {
            "success":     True,
            "file_path":   fpath,
            "size_bytes":  size_bytes,
            "request_url": url,
            "pid":         pid,
            "content_type": ct,
        }

    # ------------------------------------------------------------------
    # validate_download
    # ------------------------------------------------------------------

    def validate_download(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Validate a downloaded DataONE file."""
        if not os.path.exists(file_path):
            return {"valid": False, "issues": [f"File not found: {file_path}"]}
        size = os.path.getsize(file_path)
        if size == 0:
            return {"valid": False, "issues": ["File is empty"]}

        with open(file_path, "rb") as fh:
            head = fh.read(512)

        issues = _validate_content(head, file_path)
        if issues:
            logger.warning("dataone validate_download: FAILED — %s", issues)
            return {"valid": False, "issues": issues}

        logger.info("dataone validate_download: passed (%d bytes) — %s", size, file_path)
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
    def _size_dict(size_bytes: int, method: str, confidence: float,
                   url: str) -> Dict[str, Any]:
        if size_bytes < 1024:
            human = f"{size_bytes} B"
        elif size_bytes < 1024 ** 2:
            human = f"{size_bytes / 1024:.1f} KB"
        else:
            human = f"{size_bytes / (1024 ** 2):.2f} MB"
        return {
            "size_bytes":  size_bytes,
            "size_human":  human,
            "confidence":  confidence,
            "method":      method,
            "request_url": url,
        }


register_connector(DataONEConnector)
