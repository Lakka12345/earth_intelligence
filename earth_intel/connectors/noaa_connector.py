"""
NOAA connector — live multi-API implementation.

Provider APIs used:
  NCEI OISST directory   https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/
  NCEI CDO API           https://www.ncei.noaa.gov/cdo-web/api/v2/
  ERDDAP (NOAA)          https://coastwatch.pfeg.noaa.gov/erddap/
  NOAA ERDDAP (IOOS)     https://erddap.ioos.us/erddap/
  NOMADS GFS             https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/

Implements:
  discover_datasets     → ERDDAP allDatasets search
  probe_metadata        → live HEAD / directory / ERDDAP metadata
  probe_size            → HEAD Content-Length / ERDDAP metadata
  resolve_download_asset→ real file URL
  fetch_subset          → ERDDAP griddap/tabledap constraint expression
  fetch_full            → streaming NetCDF download
  validate_download     → MIME / magic / HTML rejection
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

import requests

from connectors.base_connector import ConnectorDescriptor, Credentials, FetchRequest
from connectors.connector_registry import register_connector
from connectors.connector_types import (
    AccessType, AuthenticationType, CapabilityFlags, ConnectorType, DatasetType,
)
from connectors.dataset_matching import StaticDatasetConnector
from models.agent4_schemas import DatasetDescriptor, DatasetMetadata, SizeEstimate, format_bytes
from models.website_analysis_schemas import SourceSnapshot

# ── constants ──────────────────────────────────────────────────────────────────

_OISST_BASE   = "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/"
_CDO_BASE     = "https://www.ncei.noaa.gov/cdo-web/api/v2"
_ERDDAP_CW    = "https://coastwatch.pfeg.noaa.gov/erddap"
_ERDDAP_IOOS  = "https://erddap.ioos.us/erddap"
_NOMADS_GFS   = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"

_HTML_SIGNATURES = (b"<!DOCTYPE", b"<html", b"<HTML", b"<head", b"<body")


# ── helpers ────────────────────────────────────────────────────────────────────

def _estimate_size_head(url: str, headers: Optional[dict] = None) -> Optional[int]:
    try:
        r = requests.head(url, headers=headers or {}, timeout=15, allow_redirects=True)
        cl = r.headers.get("Content-Length") or r.headers.get("content-length")
        if cl:
            return int(cl)
    except Exception:
        pass
    return None


def _is_html_content(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(512)
        return any(sig in chunk for sig in _HTML_SIGNATURES)
    except Exception:
        return False


# ── ERDDAP helpers ─────────────────────────────────────────────────────────────

def _erddap_search(base: str, query: str, limit: int = 5) -> List[dict]:
    """Full-text search on an ERDDAP server; returns list of dataset dicts."""
    try:
        r = requests.get(
            f"{base}/search/index.json",
            params={"searchFor": query, "page": 1, "itemsPerPage": limit},
            timeout=20,
        )
        if not r.ok:
            return []
        data = r.json()
        # ERDDAP returns table structure: columnNames + rows
        cols = data.get("table", {}).get("columnNames", [])
        rows = data.get("table", {}).get("rows", [])
        return [dict(zip(cols, row)) for row in rows]
    except Exception:
        return []


def _erddap_griddap_url(
    base: str,
    dataset_id: str,
    variables: Optional[List[str]],
    bbox=None,
    time_range=None,
    fmt: str = ".nc",
) -> str:
    """Build an ERDDAP griddap download URL with constraint expressions."""
    var_part = ",".join(variables) if variables else ""
    constraints = []
    if time_range:
        if time_range[0]:
            constraints.append(f"time>={time_range[0]}")
        if len(time_range) > 1 and time_range[1]:
            constraints.append(f"time<={time_range[1]}")
    if bbox and len(bbox) == 4:
        west, south, east, north = bbox
        constraints.append(f"longitude>={west}")
        constraints.append(f"longitude<={east}")
        constraints.append(f"latitude>={south}")
        constraints.append(f"latitude<={north}")
    ce = f"[({')[('.join(constraints)})]" if constraints else ""
    return f"{base}/griddap/{dataset_id}{fmt}?{var_part}{ce}"


def _erddap_tabledap_url(
    base: str,
    dataset_id: str,
    variables: Optional[List[str]],
    bbox=None,
    time_range=None,
    fmt: str = ".csv",
) -> str:
    """Build an ERDDAP tabledap URL with constraint expressions."""
    var_part = ",".join(variables) if variables else ""
    constraints = []
    if time_range and time_range[0]:
        constraints.append(f"&time>={time_range[0]}")
    if time_range and len(time_range) > 1 and time_range[1]:
        constraints.append(f"&time<={time_range[1]}")
    if bbox and len(bbox) == 4:
        west, south, east, north = bbox
        constraints.append(f"&longitude>={west}&longitude<={east}")
        constraints.append(f"&latitude>={south}&latitude<={north}")
    ce = "".join(constraints)
    return f"{base}/tabledap/{dataset_id}{fmt}?{var_part}{ce}"


# ── OISST directory resolver ───────────────────────────────────────────────────

def _resolve_oisst_url(time_range=None) -> Optional[str]:
    """
    Navigate NCEI directory listing to find a real OISST NetCDF file.
    If time_range is given, try to pick the relevant year/month.
    """
    try:
        r = requests.get(_OISST_BASE, timeout=20)
        if not r.ok:
            return None
        years = re.findall(r'href="(\d{4})/"', r.text)
        if not years:
            return None

        # Pick year from time_range or latest
        target_year = None
        if time_range and time_range[0]:
            m = re.match(r"(\d{4})", time_range[0])
            if m and m.group(1) in years:
                target_year = m.group(1)
        year = target_year or max(years)

        r2 = requests.get(f"{_OISST_BASE}{year}/", timeout=15)
        months = re.findall(r'href="(\d{2})/"', r2.text)
        if not months:
            return None
        month = months[-1]

        r3 = requests.get(f"{_OISST_BASE}{year}/{month}/", timeout=15)
        # e.g. oisst-avhrr-v02r01.19810901.nc
        files = re.findall(r'href="(oisst-avhrr[^"]+\.nc)"', r3.text)
        if not files:
            return None
        # Use last file in the listing (most recent day in that month)
        return f"{_OISST_BASE}{year}/{month}/{files[-1]}"
    except Exception:
        return None


# ── NOMADS GFS resolver ────────────────────────────────────────────────────────

def _resolve_nomads_gfs_url() -> Optional[str]:
    """Pick latest GFS 0.25deg forecast run from NOMADS."""
    try:
        r = requests.get(_NOMADS_GFS, timeout=15)
        if not r.ok:
            return None
        runs = re.findall(r'href="(gfs\.\d{8}/)"', r.text)
        if not runs:
            return None
        latest_run_dir = max(runs)
        r2 = requests.get(f"{_NOMADS_GFS}{latest_run_dir}", timeout=15)
        hours = re.findall(r'href="(\d{2}/)"', r2.text)
        if not hours:
            return None
        latest_hour = max(hours)
        r3 = requests.get(f"{_NOMADS_GFS}{latest_run_dir}{latest_hour}atmos/", timeout=15)
        files = re.findall(
            r'href="(gfs\.t\d+z\.pgrb2\.0p25\.f\d+)"',
            r3.text,
        )
        if not files:
            return None
        # f000 = analysis (most compact)
        f000 = [f for f in files if f.endswith("f000")]
        chosen = f000[0] if f000 else files[0]
        return f"{_NOMADS_GFS}{latest_run_dir}{latest_hour}atmos/{chosen}"
    except Exception:
        return None


# ── connector ──────────────────────────────────────────────────────────────────

class NOAAConnector(StaticDatasetConnector):
    descriptor = ConnectorDescriptor(
        connector_id="noaa",
        provider_name="NOAA",
        connector_type=ConnectorType.provider_api,
        supported_access_types=(AccessType.public, AccessType.user_credentials_required),
        supported_dataset_types=(DatasetType.gridded, DatasetType.time_series, DatasetType.raster),
        supported_authentication=(AuthenticationType.none, AuthenticationType.api_key),
        capabilities=(
            CapabilityFlags.supports_metadata
            | CapabilityFlags.supports_api
            | CapabilityFlags.supports_dataset_search
            | CapabilityFlags.supports_download
            | CapabilityFlags.supports_subsetting
        ),
        priority=35,
    )
    provider_keywords = ("noaa", "ncei", "ncdc", "coastwatch", "psl", "nomads", "gfs", "oisst")
    api_keywords = ("noaa", "ncei", "erddap", "nomads")

    datasets = [
        DatasetDescriptor(
            provider="NOAA",
            dataset_name="NOAA Optimum Interpolation SST v2.1 (OISST)",
            collection_name="OISST",
            dataset_id="oisst-avhrr-v02r01",
            doi="10.25921/RE9P-PT57",
            api_endpoint="https://www.ncei.noaa.gov/access/services",
            metadata_endpoint="https://www.ncei.noaa.gov/products/optimum-interpolation-sst",
            download_endpoint=_OISST_BASE,
            supported_variables=["sea surface temperature", "sst", "temperature anomaly", "ice concentration"],
            temporal_coverage="1981-present",
            spatial_coverage="Global ocean",
            supported_formats=["NetCDF"],
            authentication_required=False,
        ),
        DatasetDescriptor(
            provider="NOAA",
            dataset_name="NOAA ERDDAP CoastWatch SST (erdATssta3day)",
            collection_name="CoastWatch SST",
            dataset_id="erdATssta3day",
            api_endpoint=_ERDDAP_CW,
            metadata_endpoint=f"{_ERDDAP_CW}/info/erdATssta3day/index.html",
            download_endpoint=f"{_ERDDAP_CW}/griddap/erdATssta3day.nc",
            supported_variables=["sea surface temperature", "sst"],
            temporal_coverage="2002-present",
            spatial_coverage="Global ocean",
            supported_formats=["NetCDF", "CSV", "JSON"],
            authentication_required=False,
        ),
        DatasetDescriptor(
            provider="NOAA",
            dataset_name="NOAA Climate Data Online Daily Summaries (GHCN-D)",
            collection_name="GHCN-Daily",
            dataset_id="GHCND",
            api_endpoint=_CDO_BASE,
            metadata_endpoint="https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily",
            download_endpoint=f"{_CDO_BASE}/data",
            supported_variables=["temperature", "precipitation", "snow", "wind"],
            temporal_coverage="1763-present",
            spatial_coverage="Global stations",
            supported_formats=["JSON", "CSV"],
            authentication_required=True,
        ),
        DatasetDescriptor(
            provider="NOAA",
            dataset_name="NOAA GFS 0.25 Degree Global Forecast",
            collection_name="GFS",
            dataset_id="gfs-0p25",
            api_endpoint=_NOMADS_GFS,
            metadata_endpoint="https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php",
            download_endpoint=_NOMADS_GFS,
            supported_variables=["temperature", "wind", "pressure", "precipitation", "humidity"],
            temporal_coverage="Latest 10-day forecast",
            spatial_coverage="Global",
            supported_formats=["GRIB2"],
            authentication_required=False,
        ),
    ]

    # ── internal helpers ───────────────────────────────────────────────────────

    def _pick_dataset(self, fetch_request: FetchRequest) -> Optional[DatasetDescriptor]:
        ds = self._best_dataset(None, fetch_request)  # type: ignore[arg-type]
        return ds

    def _resolve_url_for(
        self,
        dataset: DatasetDescriptor,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> Optional[str]:
        did = dataset.dataset_id

        if did == "oisst-avhrr-v02r01":
            return _resolve_oisst_url(fetch_request.time_range)

        if did == "erdATssta3day":
            return _erddap_griddap_url(
                _ERDDAP_CW, "erdATssta3day",
                variables=list(fetch_request.variables or ["sst"]),
                bbox=fetch_request.bounding_box,
                time_range=fetch_request.time_range,
                fmt=".nc",
            )

        if did == "GHCND":
            if not (credentials and credentials.api_key):
                return None
            params = {
                "datasetid": "GHCND",
                "datatypeid": "TMAX,TMIN,PRCP",
                "limit": 1000,
                "units": "metric",
            }
            if fetch_request.time_range and fetch_request.time_range[0]:
                params["startdate"] = fetch_request.time_range[0][:10]
                params["enddate"] = (
                    fetch_request.time_range[1] if len(fetch_request.time_range) > 1
                    else fetch_request.time_range[0]
                )[:10]
            return f"{_CDO_BASE}/data?" + "&".join(f"{k}={v}" for k, v in params.items())

        if did == "gfs-0p25":
            return _resolve_nomads_gfs_url()

        return dataset.download_endpoint or None

    # ── public interface ───────────────────────────────────────────────────────

    def discover_datasets(
        self,
        query: str,
        bbox=None,
        time_range=None,
        credentials: Optional[Credentials] = None,
    ) -> List[DatasetDescriptor]:
        """Search ERDDAP CoastWatch for datasets matching query."""
        results_raw = _erddap_search(_ERDDAP_CW, query, limit=8)
        out = []
        for row in results_raw:
            did = row.get("Dataset ID", "")
            title = row.get("Title", did)
            out.append(DatasetDescriptor(
                provider="NOAA",
                dataset_name=title,
                collection_name=row.get("Institution", "NOAA"),
                dataset_id=did,
                api_endpoint=_ERDDAP_CW,
                metadata_endpoint=f"{_ERDDAP_CW}/info/{did}/index.html",
                download_endpoint=f"{_ERDDAP_CW}/griddap/{did}.nc",
                supported_variables=[],
                temporal_coverage=f"{row.get('Min Time','')} – {row.get('Max Time','')}",
                spatial_coverage="See metadata",
                supported_formats=["NetCDF", "CSV"],
                authentication_required=False,
            ))
        return out or self.datasets

    def probe_metadata(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> DatasetMetadata:
        dataset = self._best_dataset(snapshot, fetch_request)
        if dataset is None:
            return DatasetMetadata(
                source_id=snapshot.source_id,
                dataset_id=snapshot.source_id,
                variables=list(snapshot.variables_available or fetch_request.variables or []),
                retrieval_method="NOAA static catalog",
                unavailable_reason="No matching NOAA dataset found.",
            )

        url = self._resolve_url_for(dataset, fetch_request, credentials)

        # Size via HEAD
        auth_headers = {}
        if credentials and credentials.api_key and dataset.dataset_id == "GHCND":
            auth_headers["token"] = credentials.api_key
        size = _estimate_size_head(url, auth_headers) if url else None

        return DatasetMetadata(
            source_id=snapshot.source_id,
            dataset_id=dataset.dataset_id,
            collection=dataset.collection_name,
            product=dataset.dataset_name,
            download_endpoint=url or dataset.download_endpoint,
            api_endpoint=dataset.api_endpoint,
            metadata_endpoint=dataset.metadata_endpoint,
            file_size_bytes=size,
            variables=list(dataset.supported_variables),
            spatial_coverage=dataset.spatial_coverage,
            temporal_coverage=dataset.temporal_coverage,
            file_format=", ".join(dataset.supported_formats),
            content_type="application/x-netcdf" if "NetCDF" in dataset.supported_formats else "application/octet-stream",
            license="NOAA open data",
            retrieval_method="NOAA directory listing / ERDDAP griddap / CDO API / NOMADS",
            unavailable_reason="" if url else "Could not resolve download URL. CDO token may be required.",
        )

    def probe_size(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> SizeEstimate:
        meta = self.probe_metadata(snapshot, fetch_request, credentials)
        if meta.file_size_bytes:
            return SizeEstimate(
                source_id=snapshot.source_id,
                estimated_bytes=meta.file_size_bytes,
                is_exact=True,
                method="HEAD Content-Length",
                human_readable=format_bytes(meta.file_size_bytes),
            )
        return SizeEstimate(
            source_id=snapshot.source_id,
            method="HEAD returned no Content-Length (streaming or auth-gated endpoint)",
        )

    def resolve_download_asset(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> Optional[str]:
        dataset = self._best_dataset(snapshot, fetch_request)
        if dataset is None:
            return None
        return self._resolve_url_for(dataset, fetch_request, credentials)

    def fetch_subset(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> str:
        """
        Use ERDDAP server-side subsetting where available.
        OISST and GFS receive a full download (server-side subset not supported).
        """
        dataset = self._best_dataset(snapshot, fetch_request)
        if dataset is None:
            raise RuntimeError("NOAA connector: no dataset matched for subsetting.")

        did = dataset.dataset_id

        # ERDDAP griddap — server-side spatial+temporal subset
        if did in ("erdATssta3day",):
            url = _erddap_griddap_url(
                _ERDDAP_CW, did,
                variables=list(fetch_request.variables or ["sst"]),
                bbox=fetch_request.bounding_box,
                time_range=fetch_request.time_range,
                fmt=".nc",
            )
            return self._stream_download(url, snapshot, fetch_request, credentials)

        # CDO API already returns a subset
        if did == "GHCND":
            url = self._resolve_url_for(dataset, fetch_request, credentials)
            if not url:
                raise RuntimeError("NOAA CDO: API token required for GHCND subsetting.")
            return self._stream_download(url, snapshot, fetch_request, credentials,
                                         extra_headers={"token": credentials.api_key} if credentials and credentials.api_key else {})

        # Fallback: full download
        return self.fetch_full(snapshot, fetch_request, credentials)

    def _stream_download(
        self,
        url: str,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials],
        extra_headers: Optional[dict] = None,
    ) -> str:
        from agents.agent4_download_engine import DownloadEngine, DownloadTask
        task = DownloadTask(
            url=url,
            dest_path=fetch_request.dest_path,
            source_id=snapshot.source_id,
            provider=snapshot.name,
            connector_id=self.name,
            protocol=self.descriptor.connector_type.value,
            extra_headers=extra_headers or {},
        )
        result = DownloadEngine().download_one(task, credentials)
        fetch_request.metadata["download_result"] = result
        if not result.success:
            raise RuntimeError(result.error or "NOAA download failed.")
        self.validate_download(result.dest_path)
        return result.dest_path

    def fetch_full(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> str:
        meta = self.probe_metadata(snapshot, fetch_request, credentials)
        url = meta.download_endpoint
        if not url or not url.startswith("http"):
            raise RuntimeError(
                f"NOAA connector: no downloadable URL for {snapshot.name}. "
                "CDO API token may be required."
            )
        extra_headers = {}
        if credentials and credentials.api_key:
            extra_headers["token"] = credentials.api_key
        return self._stream_download(url, snapshot, fetch_request, credentials, extra_headers)

    def validate_download(self, path: str) -> None:
        if not path or not os.path.exists(path):
            raise RuntimeError(f"Downloaded file not found: {path}")
        size = os.path.getsize(path)
        if size == 0:
            raise RuntimeError(f"Downloaded file is empty: {path}")
        if _is_html_content(path):
            raise RuntimeError(
                f"NOAA download returned HTML (login page or error): {path}. "
                "Check CDO API token or ERDDAP availability."
            )
        with open(path, "rb") as f:
            header = f.read(4)
        # GRIB2 magic: GRIB
        # NetCDF3: CDF\x01
        # HDF5: \x89HDF
        # JSON (CDO): {
        # CSV: starts with printable ASCII
        bad = (
            size < 64
            and not any([
                header[:3] == b"CDF",
                header[:4] == b"\x89HDF",
                header[:4] == b"GRIB",
                header[:1] == b"{",
                header[:1] == b"s",  # CSV "station" header
            ])
        )
        if bad:
            raise RuntimeError(
                f"NOAA download appears invalid ({size} bytes, header={header!r}): {path}"
            )


register_connector(NOAAConnector)
