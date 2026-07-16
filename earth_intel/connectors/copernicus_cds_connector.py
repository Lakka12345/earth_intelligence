"""
Copernicus Climate Data Store (CDS) connector.

Provider APIs used:
  cdsapi (official SDK)     pip install cdsapi
  CDS REST API              https://cds.climate.copernicus.eu/api/v2/

Implements:
  discover_datasets     → CDS catalogue query
  probe_metadata        → live CDS metadata + job dry-run size estimation
  probe_size            → estimated via CDS catalogue
  resolve_download_asset→ submits CDS request, returns result path
  fetch_subset          → cdsapi with variable/area/time filters
  fetch_full            → cdsapi retrieve
  validate_download     → GRIB / NetCDF / error-file rejection
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional

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

_CDS_API_BASE = "https://cds.climate.copernicus.eu/api/v2"
_CDS_CATALOGUE = "https://cds.climate.copernicus.eu/api/v2/resources"

_HTML_SIGNATURES = (b"<!DOCTYPE", b"<html", b"<HTML", b"<head")

# Variable name aliases: user-friendly → CDS internal
_ERA5_VAR_ALIASES: Dict[str, str] = {
    "temperature":       "2m_temperature",
    "2m temperature":    "2m_temperature",
    "wind":              "10m_u_component_of_wind",
    "u wind":            "10m_u_component_of_wind",
    "v wind":            "10m_v_component_of_wind",
    "precipitation":     "total_precipitation",
    "pressure":          "surface_pressure",
    "evaporation":       "evaporation",
    "soil temperature":  "soil_temperature_level_1",
    "runoff":            "runoff",
}


def _normalise_era5_vars(variables: Optional[List[str]]) -> List[str]:
    if not variables:
        return ["2m_temperature"]
    out = []
    for v in variables:
        out.append(_ERA5_VAR_ALIASES.get(v.lower(), v.replace(" ", "_")))
    return out or ["2m_temperature"]


def _cdsapi_available() -> bool:
    try:
        import cdsapi  # noqa: F401
        return True
    except ImportError:
        return False


def _build_era5_request(
    dataset_id: str,
    variables: Optional[List[str]],
    bbox=None,
    time_range=None,
    fmt: str = "netcdf",
) -> Dict[str, Any]:
    """Build a cdsapi-compatible retrieve request dict."""
    import datetime
    vars_normalised = _normalise_era5_vars(variables)

    # Default temporal selection: last full year or from time_range
    now = datetime.datetime.utcnow()
    if time_range and time_range[0]:
        import re
        m = re.match(r"(\d{4})-?(\d{2})?-?(\d{2})?", time_range[0])
        year  = m.group(1) if m else str(now.year - 1)
        month = m.group(2) if (m and m.group(2)) else "01"
        day   = m.group(3) if (m and m.group(3)) else "01"
    else:
        year  = str(now.year - 1)
        month = "01"
        day   = "01"

    req: Dict[str, Any] = {
        "product_type": "reanalysis",
        "variable":     vars_normalised,
        "year":         year,
        "month":        month,
        "day":          day,
        "time":         ["00:00", "06:00", "12:00", "18:00"],
        "format":       fmt,
    }

    if bbox and len(bbox) == 4:
        west, south, east, north = bbox
        # CDS area: [N, W, S, E]
        req["area"] = [north, west, south, east]

    return req


def _is_html_content(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(512)
        return any(sig in chunk for sig in _HTML_SIGNATURES)
    except Exception:
        return False


def _is_error_json(path: str) -> bool:
    """Detect CDS error JSON responses."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(256).decode("utf-8", errors="ignore")
        return "traceback" in chunk.lower() or '"error"' in chunk.lower()
    except Exception:
        return False


# ── connector ──────────────────────────────────────────────────────────────────

class CopernicusCDSConnector(StaticDatasetConnector):
    descriptor = ConnectorDescriptor(
        connector_id="copernicus_cds",
        provider_name="Copernicus Climate Data Store",
        connector_type=ConnectorType.official_sdk,
        supported_access_types=(AccessType.public, AccessType.user_credentials_required),
        supported_dataset_types=(DatasetType.gridded, DatasetType.time_series),
        supported_authentication=(AuthenticationType.api_key, AuthenticationType.user_credentials),
        capabilities=(
            CapabilityFlags.supports_metadata
            | CapabilityFlags.supports_api
            | CapabilityFlags.supports_dataset_search
            | CapabilityFlags.supports_download
            | CapabilityFlags.supports_subsetting
        ),
        priority=25,
    )
    provider_keywords = ("copernicus", "copernicus climate", "cds", "era5", "ecmwf")
    api_keywords = ("cdsapi", "cds", "era5")

    datasets = [
        DatasetDescriptor(
            provider="Copernicus Climate Data Store",
            dataset_name="ERA5 hourly data on single levels",
            collection_name="ERA5",
            dataset_id="reanalysis-era5-single-levels",
            doi="10.24381/cds.adbb2d47",
            api_endpoint=_CDS_API_BASE,
            metadata_endpoint="https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels",
            download_endpoint="cdsapi:reanalysis-era5-single-levels",
            supported_variables=["2m temperature", "temperature", "wind", "precipitation",
                                  "surface pressure", "evaporation"],
            temporal_coverage="1940-present",
            spatial_coverage="Global",
            supported_formats=["GRIB", "NetCDF"],
            authentication_required=True,
            access_notes="Requires CDS account and ~/.cdsapirc or credentials object.",
        ),
        DatasetDescriptor(
            provider="Copernicus Climate Data Store",
            dataset_name="ERA5-Land hourly data",
            collection_name="ERA5-Land",
            dataset_id="reanalysis-era5-land",
            doi="10.24381/cds.e2161bac",
            api_endpoint=_CDS_API_BASE,
            metadata_endpoint="https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land",
            download_endpoint="cdsapi:reanalysis-era5-land",
            supported_variables=["2m temperature", "soil temperature", "runoff",
                                  "precipitation", "evaporation"],
            temporal_coverage="1950-present",
            spatial_coverage="Global land",
            supported_formats=["GRIB", "NetCDF"],
            authentication_required=True,
            access_notes="Requires CDS account and ~/.cdsapirc or credentials object.",
        ),
    ]

    # ── helpers ────────────────────────────────────────────────────────────────

    def _cds_credentials(self, credentials: Optional[Credentials]) -> Optional[Dict[str, str]]:
        """Return dict with url+key for cdsapi.Client, or None if unavailable."""
        if credentials:
            if credentials.api_key:
                return {
                    "url": _CDS_API_BASE,
                    "key": credentials.api_key,
                }
            if credentials.username and credentials.password:
                # Some older CDS setups use UID:key notation
                return {
                    "url": _CDS_API_BASE,
                    "key": f"{credentials.username}:{credentials.password}",
                }
        # Fall back to ~/.cdsapirc if present
        rc = os.path.expanduser("~/.cdsapirc")
        if os.path.exists(rc):
            return {}  # cdsapi reads rc file automatically
        return None

    def _catalogue_metadata(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Fetch live metadata from CDS catalogue API."""
        try:
            r = requests.get(
                f"{_CDS_CATALOGUE}/{dataset_id}",
                timeout=20,
                headers={"Accept": "application/json"},
                verify=False,
            )
            if r.ok:
                return r.json()
        except Exception:
            pass
        return None

    def _estimate_size_from_request(self, dataset_id: str, req: Dict[str, Any]) -> Optional[int]:
        """
        Rough size estimate based on request parameters.
        ERA5: ~1.5 MB/variable/day at 0.25deg global; scales with area.
        """
        nvars = len(req.get("variable", ["2m_temperature"]))
        ntimes = len(req.get("time", ["00:00", "06:00", "12:00", "18:00"]))
        fmt = req.get("format", "netcdf")

        area = req.get("area")
        if area:
            n, w, s, e = area
            lat_frac = (n - s) / 180.0
            lon_frac = (e - w) / 360.0
            area_frac = lat_frac * lon_frac
        else:
            area_frac = 1.0

        # ~300 KB per variable per 4-time-step slice at full global NetCDF
        base = 300_000 * nvars * (ntimes / 4) * area_frac
        if fmt == "grib":
            base *= 0.6  # GRIB is more compact
        return int(base)

    # ── public interface ───────────────────────────────────────────────────────

    def discover_datasets(
        self,
        snapshot,
        context=None,
    ) -> List[DatasetDescriptor]:
        """
        Query CDS catalogue for datasets matching the query keyword.
        Falls back to static catalogue on failure.
        """
        _ctx = context if isinstance(context, dict) else (vars(context) if context and hasattr(context, "__dict__") else {})
        _snap_vars = list(getattr(snapshot, "variables_available", None) or []) if snapshot else []
        keywords = _ctx.get("keywords") or _ctx.get("variables") or _snap_vars
        query = " ".join(keywords) if keywords else "reanalysis"
        try:
            r = requests.get(
                _CDS_CATALOGUE,
                params={"q": query, "limit": 10},
                timeout=20,
                headers={"Accept": "application/json"},
                verify=False,
            )
            if r.ok:
                items = r.json() if isinstance(r.json(), list) else r.json().get("results", [])
                out = []
                for item in items:
                    did = item.get("id", "")
                    out.append(DatasetDescriptor(
                        provider="Copernicus Climate Data Store",
                        dataset_name=item.get("title", did),
                        collection_name=item.get("type", "reanalysis"),
                        dataset_id=did,
                        api_endpoint=_CDS_API_BASE,
                        metadata_endpoint=f"https://cds.climate.copernicus.eu/datasets/{did}",
                        download_endpoint=f"cdsapi:{did}",
                        supported_variables=item.get("variables", []),
                        temporal_coverage=item.get("temporal_coverage", ""),
                        spatial_coverage="Global",
                        supported_formats=["GRIB", "NetCDF"],
                        authentication_required=True,
                    ))
                if out:
                    return out
        except Exception:
            pass
        return self.datasets

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
                retrieval_method="CDS static catalog",
                unavailable_reason="No matching CDS dataset found.",
            )

        cat_meta = self._catalogue_metadata(dataset.dataset_id) or {}
        req = _build_era5_request(
            dataset.dataset_id,
            list(fetch_request.variables or dataset.supported_variables),
            fetch_request.bounding_box,
            fetch_request.time_range,
        )
        size = self._estimate_size_from_request(dataset.dataset_id, req)

        # The CDS catalogue record's shape varies by dataset, so read
        # defensively across the field names Copernicus has used
        # (title/abstract vs description-block list, licence_list vs licence_id, etc.)
        title = cat_meta.get("title")
        abstract = cat_meta.get("abstract")
        if not abstract:
            for block in cat_meta.get("description", []) or []:
                if isinstance(block, dict) and block.get("id") in ("abstract", "summary", None):
                    abstract = block.get("content")
                    if abstract:
                        break
        licence_list = cat_meta.get("licence_list") or cat_meta.get("licences")
        licence = (
            cat_meta.get("licence_id")
            or (licence_list[0].get("title") if licence_list and isinstance(licence_list[0], dict) else None)
            or (licence_list[0] if licence_list and isinstance(licence_list[0], str) else None)
        )
        keywords = cat_meta.get("keywords")
        var_catalogue = cat_meta.get("variables")
        var_units = None
        if isinstance(var_catalogue, dict):
            var_units = {k: v.get("units") for k, v in var_catalogue.items()
                         if isinstance(v, dict) and v.get("units")} or None
        extent = cat_meta.get("extent") or cat_meta.get("bbox")
        update_freq = cat_meta.get("update_frequency") or cat_meta.get("temporal_update")
        version = cat_meta.get("version") or cat_meta.get("update_date")
        doi = cat_meta.get("doi")

        return DatasetMetadata(
            source_id=snapshot.source_id,
            dataset_id=dataset.dataset_id,
            collection=dataset.collection_name,
            product=dataset.dataset_name,
            download_endpoint=f"cdsapi:{dataset.dataset_id}",
            api_endpoint=_CDS_API_BASE,
            metadata_endpoint=dataset.metadata_endpoint,
            file_size_bytes=size,
            variables=list(fetch_request.variables or dataset.supported_variables),
            spatial_coverage=cat_meta.get("spatial_coverage", dataset.spatial_coverage),
            temporal_coverage=cat_meta.get("temporal_coverage", dataset.temporal_coverage),
            file_format="NetCDF",
            content_type="application/x-netcdf",
            license=licence or "Copernicus licence to use Copernicus products",
            retrieval_method="cdsapi retrieve" if _cdsapi_available() else "CDS REST API",
            unavailable_reason="" if self._cds_credentials(credentials) else
                "No CDS credentials found. Provide api_key or configure ~/.cdsapirc.",
            dataset_name=title or dataset.dataset_name,
            provider="Copernicus Climate Data Store (C3S)",
            description=abstract,
            variable_units=var_units,
            bounding_box=list(extent) if isinstance(extent, (list, tuple)) and len(extent) == 4 else None,
            crs="EPSG:4326",
            citation=(f"https://doi.org/{doi}" if doi else None),
            keywords=keywords,
            update_frequency=update_freq,
            version=str(version) if version else None,
            authentication_required=True,
        )

    def probe_size(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> SizeEstimate:
        dataset = self._best_dataset(snapshot, fetch_request)
        if dataset is None:
            return SizeEstimate(source_id=snapshot.source_id, method="No dataset matched")
        req = _build_era5_request(
            dataset.dataset_id,
            list(fetch_request.variables or dataset.supported_variables),
            fetch_request.bounding_box,
            fetch_request.time_range,
        )
        size = self._estimate_size_from_request(dataset.dataset_id, req)
        return SizeEstimate(
            source_id=snapshot.source_id,
            estimated_bytes=size,
            is_exact=False,
            method="CDS payload estimation (vars × time-steps × area fraction)",
            human_readable=format_bytes(size) if size else "Unknown",
        )

    def resolve_download_asset(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> Optional[str]:
        """
        CDS datasets require a retrieve job; there is no static pre-signed URL.
        Return None to indicate that fetch_full must be called.
        """
        return None  # CDS is job-based, not URL-based

    def fetch_subset(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> str:
        """
        CDS natively supports server-side subsetting via area + variable selection
        in the retrieve request. fetch_subset delegates to fetch_full with the
        full request (which already encodes bbox and variable constraints).
        """
        return self.fetch_full(snapshot, fetch_request, credentials)

    def fetch_full(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> str:
        dataset = self._best_dataset(snapshot, fetch_request)
        if dataset is None:
            raise RuntimeError("CDS connector: no dataset matched.")

        cds_creds = self._cds_credentials(credentials)
        if cds_creds is None:
            raise RuntimeError(
                "CDS connector: no credentials available. "
                "Provide api_key or create ~/.cdsapirc. "
                "Register at https://cds.climate.copernicus.eu/"
            )

        req = _build_era5_request(
            dataset.dataset_id,
            list(fetch_request.variables or dataset.supported_variables),
            fetch_request.bounding_box,
            fetch_request.time_range,
            fmt="netcdf",
        )

        dest = fetch_request.dest_path or os.path.join(
            tempfile.gettempdir(), f"{dataset.dataset_id}.nc"
        )

        if _cdsapi_available():
            return self._fetch_via_cdsapi(dataset.dataset_id, req, dest, cds_creds)
        else:
            return self._fetch_via_rest(dataset.dataset_id, req, dest, cds_creds)

    def _fetch_via_cdsapi(
        self,
        dataset_id: str,
        req: Dict[str, Any],
        dest: str,
        creds: Dict[str, str],
    ) -> str:
        """Use the official cdsapi SDK to retrieve data."""
        import cdsapi
        client_kwargs: Dict[str, Any] = {"quiet": True, "progress": False}
        if creds:  # if empty dict, cdsapi reads ~/.cdsapirc
            client_kwargs["url"] = creds.get("url", _CDS_API_BASE)
            client_kwargs["key"] = creds["key"]
        c = cdsapi.Client(**client_kwargs)
        c.retrieve(dataset_id, req, dest)
        self.validate_download(dest)
        return dest

    def _fetch_via_rest(
        self,
        dataset_id: str,
        req: Dict[str, Any],
        dest: str,
        creds: Dict[str, str],
    ) -> str:
        """
        Fall-back: use CDS REST API directly.
        POST a retrieve job → poll → download result.
        """
        import time
        import json as _json

        api_key = creds.get("key", "")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Submit job
        r = requests.post(
            f"{_CDS_API_BASE}/tasks/retrieve",
            headers=headers,
            json={"dataset": dataset_id, "inputs": req},
            timeout=60,
            verify=False,
        )
        if not r.ok:
            raise RuntimeError(
                f"CDS REST submit failed: {r.status_code} {r.text[:200]}"
            )
        job = r.json()
        job_id = job.get("request_id") or job.get("job_id")
        if not job_id:
            raise RuntimeError(f"CDS REST: no job_id in response: {job}")

        # Poll
        for _ in range(120):  # up to ~20 minutes
            time.sleep(10)
            s = requests.get(
                f"{_CDS_API_BASE}/tasks/{job_id}",
                headers=headers,
                timeout=30,
                verify=False,
            )
            if not s.ok:
                continue
            state = s.json()
            status = state.get("state", "")
            if status == "completed":
                result_url = state.get("result", {}).get("href")
                if not result_url:
                    raise RuntimeError("CDS REST: job completed but no download URL.")
                dl = requests.get(result_url, headers=headers, stream=True, timeout=600, verify=False)
                os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
                with open(dest, "wb") as f:
                    for chunk in dl.iter_content(chunk_size=65536):
                        f.write(chunk)
                self.validate_download(dest)
                return dest
            if status in ("failed", "error"):
                raise RuntimeError(f"CDS REST job failed: {state.get('error', state)}")

        raise RuntimeError("CDS REST: job timed out after 20 minutes.")

    def validate_download(self, path: str) -> None:
        if not path or not os.path.exists(path):
            raise RuntimeError(f"CDS: downloaded file not found: {path}")
        size = os.path.getsize(path)
        if size == 0:
            raise RuntimeError(f"CDS: downloaded file is empty: {path}")
        if _is_html_content(path):
            raise RuntimeError(
                f"CDS download returned HTML (login/error page): {path}. "
                "Check CDS credentials."
            )
        if _is_error_json(path):
            with open(path) as f:
                content = f.read(500)
            raise RuntimeError(f"CDS returned error JSON: {content}")
        with open(path, "rb") as f:
            header = f.read(4)
        # NetCDF3: CDF\x01; HDF5 (NetCDF4/GRIB2 container): \x89HDF; GRIB2: GRIB
        valid = (
            header[:3] == b"CDF"
            or header[:4] == b"\x89HDF"
            or header[:4] == b"GRIB"
        )
        if not valid and size < 1024:
            raise RuntimeError(
                f"CDS: downloaded file does not appear to be GRIB or NetCDF "
                f"(header={header!r}, size={size}): {path}"
            )


register_connector(CopernicusCDSConnector)
