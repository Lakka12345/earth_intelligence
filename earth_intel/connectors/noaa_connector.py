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
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
        r = requests.head(url, headers=headers or {}, timeout=15, allow_redirects=True, verify=False)
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
            verify=False,
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
    Build a direct OISST NetCDF URL without directory scraping.

    The old approach scraped the NCEI directory listing for year/month/file
    links, but NCEI's directory server now returns 500 Server Error on that
    path, which the DownloadEngine correctly rejects as "HTML response
    rejected". Constructing the URL directly from time_range avoids the
    scrape entirely and matches the real NCEI file-naming convention.

    File naming: oisst-avhrr-v02r01.YYYYMMDD.nc
    Path:        {_OISST_BASE}{YYYY}/{MM}/oisst-avhrr-v02r01.{YYYYMMDD}.nc
    """
    import datetime

    # Determine target date from time_range, defaulting to yesterday
    target_date = None
    if time_range and time_range[0]:
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(time_range[0]))
        if m:
            try:
                target_date = datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass
    if target_date is None:
        target_date = datetime.date.today() - datetime.timedelta(days=14)

    year  = target_date.strftime("%Y")
    month = target_date.strftime("%m")
    day   = target_date.strftime("%Y%m%d")
    return f"{_OISST_BASE}{year}/{month}/oisst-avhrr-v02r01.{day}.nc"


# ── NOMADS GFS resolver ────────────────────────────────────────────────────────

def _resolve_nomads_gfs_url() -> Optional[str]:
    """Pick latest GFS 0.25deg forecast run from NOMADS."""
    try:
        r = requests.get(_NOMADS_GFS, timeout=15, verify=False)
        if not r.ok:
            return None
        runs = re.findall(r'href="(gfs\.\d{8}/)"', r.text)
        if not runs:
            return None
        latest_run_dir = max(runs)
        r2 = requests.get(f"{_NOMADS_GFS}{latest_run_dir}", timeout=15, verify=False)
        hours = re.findall(r'href="(\d{2}/)"', r2.text)
        if not hours:
            return None
        latest_hour = max(hours)
        r3 = requests.get(f"{_NOMADS_GFS}{latest_run_dir}{latest_hour}atmos/", timeout=15, verify=False)
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


# ── ERDDAP info/index.json metadata (units, bbox, coverage, institution) ───────

def _erddap_info(base: str, dataset_id: str) -> dict:
    """Fetch and reduce ERDDAP info/{dataset_id}/index.json. Never raises."""
    try:
        r = requests.get(f"{base}/info/{dataset_id}/index.json", timeout=20, verify=False)
        if not r.ok:
            return {}
        data = r.json()
    except Exception:
        return {}
    rows = data.get("table", {}).get("rows", [])
    cols = data.get("table", {}).get("columnNames", [])
    try:
        ri, vi, ai, vali = (cols.index("Row Type"), cols.index("Variable Name"),
                            cols.index("Attribute Name"), cols.index("Value"))
    except ValueError:
        return {}
    global_attrs: dict = {}
    var_units: dict = {}
    variables: List[str] = []
    for row in rows:
        row_type, var_name, attr_name, value = row[ri], row[vi], row[ai], row[vali]
        if row_type == "attribute" and var_name == "NC_GLOBAL":
            global_attrs[attr_name] = value
        elif row_type == "variable" and var_name and var_name not in variables:
            variables.append(var_name)
        elif row_type == "attribute" and attr_name == "units" and var_name:
            var_units[var_name] = value
    return {"global_attrs": global_attrs, "var_units": var_units, "variables": variables}


def _cdo_dataset_metadata(dataset_id: str, credentials: Optional["Credentials"] = None) -> dict:
    """Fetch NCEI CDO API's /datasets/{id} record (name, coverage dates, data coverage)."""
    if not (credentials and credentials.api_key):
        return {}
    try:
        r = requests.get(
            f"{_CDO_BASE}/datasets/{dataset_id}",
            headers={"token": credentials.api_key},
            timeout=20,
            verify=False,
        )
        if r.ok:
            return r.json()
    except Exception:
        pass
    return {}


def _oisst_coverage() -> dict:
    """Derive OISST temporal coverage from the live NCEI directory listing."""
    try:
        r = requests.get(_OISST_BASE, timeout=20, verify=False)
        if not r.ok:
            return {}
        years = re.findall(r'href="(\d{4})/"', r.text)
        if not years:
            return {}
        return {"start_date": f"{min(years)}-09-01", "end_date": f"{max(years)}-12-31"}
    except Exception:
        return {}


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

    # CHANGED: this connector bundles four unrelated NOAA products (OISST,
    # ERDDAP CoastWatch SST, GHCN-Daily, GFS) behind one connector class.
    # _best_dataset() (inherited from StaticDatasetConnector) scores
    # fetch_request.variables against each dataset's supported_variables
    # and picks the best match -- it has no way to know which of the four
    # products a *specific source* (e.g. a source explicitly ranked as
    # "NOAA GFS via NOMADS") was actually supposed to mean. Since SST-type
    # variables dominate most queries, OISST's supported_variables list
    # wins the scoring almost every time, so every NOAA source in a plan
    # -- including ones explicitly meant to be GFS or GHCN-D -- silently
    # resolved to OISST. That's why "NOAA GFS via NOMADS" kept printing
    # "Dataset selected: NOAA Optimum Interpolation SST v2.1 (OISST)".
    #
    # Fix: if the snapshot's name/url clearly identifies one of the four
    # products, pin to that dataset directly and skip the generic
    # variable-based scoring entirely. Only fall back to _best_dataset()
    # when the snapshot gives no such signal (e.g. a generic "NOAA" entry
    # with no sub-product indicated).
    _SNAPSHOT_DATASET_HINTS: List[tuple] = [
        (("gfs", "nomads"), "gfs-0p25"),
        (("ghcn", "cdo", "climate data online"), "GHCND"),
        (("coastwatch", "erddap", "erdatssta"), "erdATssta3day"),
        (("oisst", "optimum interpolation"), "oisst-avhrr-v02r01"),
    ]

    def _dataset_by_id(self, dataset_id: str) -> Optional[DatasetDescriptor]:
        for ds in self.datasets:
            if ds.dataset_id == dataset_id:
                return ds
        return None

    def _dataset_for_snapshot(
        self,
        snapshot,
        fetch_request: FetchRequest,
    ) -> Optional[DatasetDescriptor]:
        """
        Resolve which of the four bundled NOAA products applies, preferring
        an explicit signal from the snapshot (name/url/source_id) over
        generic variable-similarity scoring. See class-level comment above
        _SNAPSHOT_DATASET_HINTS for why this matters.
        """
        haystack = " ".join(
            str(x) for x in (
                getattr(snapshot, "name", "") if snapshot else "",
                getattr(snapshot, "source_id", "") if snapshot else "",
                getattr(snapshot, "url", "") if snapshot else "",
            )
        ).lower()

        if haystack.strip():
            for keywords, dataset_id in self._SNAPSHOT_DATASET_HINTS:
                if any(kw in haystack for kw in keywords):
                    pinned = self._dataset_by_id(dataset_id)
                    if pinned is not None:
                        return pinned

        # No unambiguous signal from the snapshot -- fall back to the
        # original variable-similarity matching.
        return self._best_dataset(snapshot, fetch_request)

    def _pick_dataset(
        self, fetch_request: FetchRequest, snapshot=None,
    ) -> Optional[DatasetDescriptor]:
        # CHANGED: was `self._best_dataset(None, fetch_request)`, silently
        # discarding whatever snapshot the caller had. Even though this
        # method isn't currently called elsewhere in this file, it's part
        # of the connector's internal API surface and should not carry a
        # latent version of the same bug fixed above.
        return self._dataset_for_snapshot(snapshot, fetch_request)

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
        snapshot,
        context=None,
    ) -> List[DatasetDescriptor]:
        """Search ERDDAP CoastWatch for datasets matching query."""
        # Derive search term from context, snapshot variables, or fall back to "ocean"
        _ctx = context if isinstance(context, dict) else (vars(context) if context and hasattr(context, "__dict__") else {})
        _snap_vars = list(getattr(snapshot, "variables_available", None) or []) if snapshot else []
        keywords = _ctx.get("keywords") or _ctx.get("variables") or _snap_vars
        query = " ".join(keywords) if keywords else "ocean"
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
        dataset = self._dataset_for_snapshot(snapshot, fetch_request)
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

        # Per-dataset real provider metadata enrichment. Fields that cannot be
        # determined for a given dataset/provider combination are left as None.
        rich: dict = {}
        if dataset.dataset_id == "erdATssta3day":
            info = _erddap_info(_ERDDAP_CW, "erdATssta3day")
            ga = info.get("global_attrs", {})
            lat_min, lat_max = ga.get("geospatial_lat_min"), ga.get("geospatial_lat_max")
            lon_min, lon_max = ga.get("geospatial_lon_min"), ga.get("geospatial_lon_max")
            bbox = None
            if all(v is not None for v in (lat_min, lat_max, lon_min, lon_max)):
                try:
                    bbox = [float(lon_min), float(lat_min), float(lon_max), float(lat_max)]
                except (TypeError, ValueError):
                    bbox = None
            rich = dict(
                dataset_name=ga.get("title"),
                provider=ga.get("institution") or "NOAA CoastWatch",
                description=ga.get("summary"),
                variable_units=info.get("var_units") or None,
                spatial_resolution=ga.get("geospatial_lat_resolution"),
                temporal_resolution=ga.get("time_coverage_resolution"),
                crs="EPSG:4326" if bbox else None,
                bounding_box=bbox,
                start_date=ga.get("time_coverage_start"),
                end_date=ga.get("time_coverage_end"),
                citation=ga.get("references"),
                keywords=[k.strip() for k in ga.get("keywords", "").split(",") if k.strip()] or None,
            )
        elif dataset.dataset_id == "GHCND":
            cdo = _cdo_dataset_metadata("GHCND", credentials)
            rich = dict(
                dataset_name=cdo.get("name"),
                provider="NOAA NCEI",
                description=(f"Global Historical Climatology Network - Daily; "
                             f"data coverage {cdo.get('datacoverage')}" if cdo.get("datacoverage") else None),
                start_date=cdo.get("mindate"),
                end_date=cdo.get("maxdate"),
                crs="EPSG:4326",
                citation="NOAA National Centers for Environmental Information, GHCN-Daily",
            )
        elif dataset.dataset_id == "oisst-avhrr-v02r01":
            cov = _oisst_coverage()
            rich = dict(
                dataset_name=dataset.dataset_name,
                provider="NOAA NCEI",
                description="Optimum Interpolation Sea Surface Temperature, AVHRR-only, v2.1, daily 0.25deg grid.",
                spatial_resolution="0.25 degree",
                temporal_resolution="Daily",
                crs="EPSG:4326",
                bounding_box=[-180.0, -90.0, 180.0, 90.0],
                start_date=cov.get("start_date"),
                end_date=cov.get("end_date"),
                citation="Huang et al. (2021), NOAA 1/4° Daily OISST v2.1, NCEI, doi:10.25921/RE9P-PT57",
            )
        elif dataset.dataset_id == "gfs-0p25":
            rich = dict(
                dataset_name=dataset.dataset_name,
                provider="NOAA NCEP/EMC",
                description="Global Forecast System, 0.25 degree operational forecast model output.",
                spatial_resolution="0.25 degree",
                temporal_resolution="3-hourly forecast steps",
                crs="EPSG:4326",
                bounding_box=[-180.0, -90.0, 180.0, 90.0],
                update_frequency="4x daily (00/06/12/18 UTC cycles)",
            )
        rich = {k: v for k, v in rich.items() if v not in (None, "", [], {})}

        # CHANGED: dataset.download_endpoint for gfs-0p25 is the bare NOMADS
        # directory listing (_NOMADS_GFS), which is an HTML index page, not
        # a data file. Falling back to it when the live scrape fails meant
        # Agent 4 would silently attempt to download an HTML page as GRIB2
        # and burn retries before failing -- exactly the "HTTP content type
        # indicates HTML" error seen in practice. GFS must never use that
        # generic fallback: if the live resolve fails, this is genuinely
        # unavailable right now (NOMADS layout changed, or a transient
        # scrape failure) and callers should move to the next source
        # immediately rather than retry a URL that was never going to work.
        if dataset.dataset_id == "gfs-0p25":
            resolved_download_endpoint = url  # None if the live scrape failed -- do not fall back
        else:
            resolved_download_endpoint = url or dataset.download_endpoint

        return DatasetMetadata(
            source_id=snapshot.source_id,
            dataset_id=dataset.dataset_id,
            collection=dataset.collection_name,
            product=dataset.dataset_name,
            download_endpoint=resolved_download_endpoint,
            api_endpoint=dataset.api_endpoint,
            metadata_endpoint=dataset.metadata_endpoint,
            file_size_bytes=size,
            variables=list(dataset.supported_variables),
            spatial_coverage=str(rich.get("bounding_box")) if rich.get("bounding_box") else dataset.spatial_coverage,
            temporal_coverage=dataset.temporal_coverage,
            file_format=", ".join(dataset.supported_formats),
            content_type="application/x-netcdf" if "NetCDF" in dataset.supported_formats else "application/octet-stream",
            license="NOAA open data (public domain)",
            retrieval_method="NOAA directory listing / ERDDAP info / CDO API / NOMADS",
            unavailable_reason=(
                "" if resolved_download_endpoint
                else (
                    "Could not resolve the current GFS forecast run/file from NOMADS "
                    "(the directory listing may have changed layout, or the scrape "
                    "timed out). Not falling back to the bare directory URL since "
                    "that returns an HTML index page, not GRIB2 data."
                    if dataset.dataset_id == "gfs-0p25"
                    else "Could not resolve download URL. CDO token may be required."
                )
            ),
            authentication_required=dataset.authentication_required,
            **rich,
        )

    def probe_size(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> SizeEstimate:
        meta = self.probe_metadata(snapshot, fetch_request, credentials)

        # Priority 1: HEAD Content-Length (exact)
        if meta.file_size_bytes:
            return SizeEstimate(
                source_id=snapshot.source_id,
                estimated_bytes=meta.file_size_bytes,
                is_exact=True,
                method="HEAD Content-Length",
                human_readable=format_bytes(meta.file_size_bytes),
            )

        # Priority 2: Dataset-specific estimation fallbacks
        dataset = self._dataset_for_snapshot(snapshot, fetch_request)
        if dataset is not None:
            est = self._estimate_size_from_dataset(dataset, fetch_request)
            if est is not None:
                return SizeEstimate(
                    source_id=snapshot.source_id,
                    estimated_bytes=float(est["bytes"]),
                    is_exact=False,
                    method=est["method"],
                    human_readable=format_bytes(est["bytes"]),
                )

        return SizeEstimate(
            source_id=snapshot.source_id,
            method="NOAA size unavailable (streaming or auth-gated endpoint)",
            human_readable="Unknown",
        )

    def _estimate_size_from_dataset(
        self,
        dataset: "DatasetDescriptor",
        fetch_request: FetchRequest,
    ) -> Optional[dict]:
        """
        Return a best-effort {'bytes': int, 'method': str} estimate using
        dataset-specific knowledge when HEAD Content-Length is unavailable.
        """
        did = dataset.dataset_id

        # ── erdATssta3day (ERDDAP griddap NetCDF) ─────────────────────────────
        # Grid: global 0.1° SST, 3-day composite. Rough uncompressed size:
        # lat(1801) × lon(3601) × 1 var × 4 bytes ≈ 25 MB per file.
        if did == "erdATssta3day":
            bbox = fetch_request.bounding_box
            if bbox:
                lat_pts = max(1, int((bbox[3] - bbox[1]) / 0.1))
                lon_pts = max(1, int((bbox[2] - bbox[0]) / 0.1))
            else:
                lat_pts, lon_pts = 1801, 3601  # global default
            n_vars  = max(1, len(fetch_request.variables or ["sst"]))
            n_times = 1  # 3-day composite → single time step per file
            est = lat_pts * lon_pts * n_times * n_vars * 4
            return {"bytes": est, "method": "ERDDAP griddap grid estimate (lat×lon×vars×4B)"}

        # ── OISST NetCDF ───────────────────────────────────────────────────────
        # Daily 0.25° global: 720 × 1440 × 4 vars × 4 bytes ≈ 16 MB/day
        if did == "oisst-avhrr-v02r01":
            bbox = fetch_request.bounding_box
            if bbox:
                lat_pts = max(1, int((bbox[3] - bbox[1]) / 0.25))
                lon_pts = max(1, int((bbox[2] - bbox[0]) / 0.25))
            else:
                lat_pts, lon_pts = 720, 1440
            n_vars = max(1, len(fetch_request.variables or ["sst"]))
            n_days = 1
            if fetch_request.time_range and len(fetch_request.time_range) == 2:
                try:
                    from datetime import datetime as _dt
                    n_days = max(1, (_dt.fromisoformat(str(fetch_request.time_range[1])[:10])
                                     - _dt.fromisoformat(str(fetch_request.time_range[0])[:10])).days + 1)
                except Exception:
                    pass
            est = lat_pts * lon_pts * n_vars * n_days * 4
            return {"bytes": est, "method": "OISST grid estimate (lat×lon×vars×days×4B)"}

        # ── GFS GRIB2 ─────────────────────────────────────────────────────────
        # 0.25° global, ~200 variables per cycle, each ~1 MB → full file ≈ 500 MB;
        # subset by requested variables.
        if did == "gfs-0p25":
            n_vars = len(fetch_request.variables or []) or 5  # default if unspecified
            est    = n_vars * 1_048_576  # ~1 MB per variable in GRIB2
            return {"bytes": est, "method": "GFS GRIB2 estimate (n_vars × 1 MB/variable)"}

        # ── GHCND (CSV via CDO API) ────────────────────────────────────────────
        # Tabular: roughly 200 bytes per station-day row.
        if did == "GHCND":
            n_days = 365  # default 1-year request
            if fetch_request.time_range and len(fetch_request.time_range) == 2:
                try:
                    from datetime import datetime as _dt
                    n_days = max(1, (_dt.fromisoformat(str(fetch_request.time_range[1])[:10])
                                     - _dt.fromisoformat(str(fetch_request.time_range[0])[:10])).days + 1)
                except Exception:
                    pass
            est = n_days * 50 * 200  # assume ~50 stations, 200 bytes/row
            return {"bytes": est, "method": "GHCND CSV estimate (stations×days×200B/row)"}

        return None

    def resolve_download_asset(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> Optional[str]:
        dataset = self._dataset_for_snapshot(snapshot, fetch_request)
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
        dataset = self._dataset_for_snapshot(snapshot, fetch_request)
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
