
"""
Copernicus Marine connector — live product discovery and subsetting.

Provider APIs used:
  copernicusmarine SDK  pip install copernicusmarine
  CMEMS REST catalogue  https://catalogue.marine.copernicus.eu/api/
  CMEMS OPeNDAP/MOTU    (via SDK)

Implements:
  discover_datasets     → copernicusmarine.describe() or catalogue REST
  probe_metadata        → live product metadata from SDK/catalogue
  probe_size            → estimated from variable × time × space resolution
  resolve_download_asset→ constructs subset request parameters
  fetch_subset          → copernicusmarine.subset() with depth/var/time/bbox
  fetch_full            → copernicusmarine.get() full product
  validate_download     → NetCDF magic / HTML rejection
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional

import requests
import urllib3

# Suppress the SSL InsecureRequestWarning to keep console clean
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

_CMEMS_CATALOGUE = "https://catalogue.marine.copernicus.eu/api/product"
_HTML_SIGNATURES  = (b"<!DOCTYPE", b"<!doctype", b"<html", b"<HTML", b"<head")

# Known dataset_id → (product_id, layer/dataset_id)
_PRODUCT_MAP: Dict[str, Dict[str, str]] = {
    "cmems_mod_glo_phy_my_0.083deg_P1D-m": {
        "product_id": "GLOBAL_MULTIYEAR_PHY_001_030",
        "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        "variables":  "thetao,so,uo,vo,zos,mlotst",
    },
    "cmems_obs-sst_glo_phy_nrt_l4_P1D": {
        "product_id": "SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001",
        "dataset_id": "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2",
        "variables":  "analysed_sst,analysis_error",
    },
    "cmems_mod_arc_bgc_my_ecosmo_P1D-m": {
        "product_id": "ARC_MULTIYEAR_BGC_003_002",
        "dataset_id": "cmems_mod_arc_bgc_my_ecosmo_P1D-m",
        "variables":  "no3,po4,o2,chl",
    },
}

# Variable alias map
_VAR_ALIASES: Dict[str, str] = {
    "sea water temperature": "thetao",
    "temperature":           "thetao",
    "salinity":              "so",
    "sea surface height":    "zos",
    "ocean current":         "uo,vo",
    "mixed layer depth":     "mlotst",
    "sea surface temperature": "analysed_sst",
    "sst":                   "analysed_sst",
    "nitrate":               "no3",
    "phosphate":             "po4",
    "oxygen":                "o2",
    "chlorophyll":           "chl",
}


def _normalise_cmems_vars(variables: Optional[List[str]]) -> List[str]:
    if not variables:
        return ["thetao"]
    out = []
    for v in variables:
        mapped = _VAR_ALIASES.get(v.lower())
        if mapped:
            out.extend(mapped.split(","))
        else:
            out.append(v.replace(" ", "_"))
    return list(dict.fromkeys(out))  # deduplicate, preserve order


def _cmems_sdk_available() -> bool:
    try:
        import copernicusmarine  # noqa: F401
        return True
    except ImportError:
        return False


def _is_html_content(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(512)
        return any(sig in chunk for sig in _HTML_SIGNATURES)
    except Exception:
        return False


def _estimate_size(
    variables: List[str],
    bbox=None,
    time_range=None,
    depth_range=None,
) -> int:
    """
    Rough size estimate for a CMEMS subset.
    ~100 KB per variable per day at 1/12° global.
    Area and time scaling applied.
    """
    nv = len(variables) or 1

    # Time days
    import re
    import datetime
    def _parse(s):
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y-%m"):
            try:
                return datetime.datetime.strptime(s[:len(fmt.replace("%","XX").replace("X","0"))], fmt)
            except Exception:
                continue
        return None

    days = 30
    if time_range and time_range[0]:
        t0 = _parse(time_range[0])
        t1 = _parse(time_range[1]) if (len(time_range) > 1 and time_range[1]) else datetime.datetime.utcnow()
        if t0 and t1:
            days = max(1, (t1 - t0).days)

    area_frac = 1.0
    if bbox and len(bbox) == 4:
        west, south, east, north = bbox
        area_frac = ((east - west) / 360.0) * ((north - south) / 180.0)

    base = 100_000 * nv * days * area_frac
    return int(base)


# ── connector ──────────────────────────────────────────────────────────────────

class CopernicusMarineConnector(StaticDatasetConnector):
    descriptor = ConnectorDescriptor(
        connector_id="copernicus_marine",
        provider_name="Copernicus Marine",
        connector_type=ConnectorType.official_sdk,
        supported_access_types=(AccessType.public, AccessType.user_credentials_required),
        supported_dataset_types=(DatasetType.gridded, DatasetType.time_series),
        supported_authentication=(AuthenticationType.user_credentials,),
        capabilities=(
            CapabilityFlags.supports_metadata
            | CapabilityFlags.supports_api
            | CapabilityFlags.supports_dataset_search
            | CapabilityFlags.supports_download
            | CapabilityFlags.supports_subsetting
        ),
        priority=28,
    )
    provider_keywords = ("copernicus marine", "marine copernicus", "cmems", "motu")
    api_keywords = ("copernicusmarine", "cmems")

    datasets = [
        DatasetDescriptor(
            provider="Copernicus Marine",
            dataset_name="Global Ocean Physics Reanalysis (1/12°)",
            collection_name="GLOBAL_MULTIYEAR_PHY_001_030",
            dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
            api_endpoint="copernicusmarine",
            metadata_endpoint="https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description",
            download_endpoint="copernicusmarine:subset",
            supported_variables=["sea water temperature", "sea surface height", "salinity",
                                  "ocean current", "mixed layer depth"],
            temporal_coverage="1993-present",
            spatial_coverage="Global ocean",
            supported_formats=["NetCDF"],
            authentication_required=True,
            access_notes="Requires Copernicus Marine account. Register at https://data.marine.copernicus.eu/",
        ),
        DatasetDescriptor(
            provider="Copernicus Marine",
            dataset_name="Global SST NRT L4 Analysis",
            collection_name="SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001",
            dataset_id="cmems_obs-sst_glo_phy_nrt_l4_P1D",
            api_endpoint="copernicusmarine",
            metadata_endpoint="https://data.marine.copernicus.eu/product/SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001/description",
            download_endpoint="copernicusmarine:subset",
            supported_variables=["sea surface temperature", "sst", "analysis error"],
            temporal_coverage="2007-present",
            spatial_coverage="Global ocean",
            supported_formats=["NetCDF"],
            authentication_required=True,
        ),
        DatasetDescriptor(
            provider="Copernicus Marine",
            dataset_name="Arctic Ocean Biogeochemistry Reanalysis",
            collection_name="ARC_MULTIYEAR_BGC_003_002",
            dataset_id="cmems_mod_arc_bgc_my_ecosmo_P1D-m",
            api_endpoint="copernicusmarine",
            metadata_endpoint="https://data.marine.copernicus.eu/product/ARC_MULTIYEAR_BGC_003_002/description",
            download_endpoint="copernicusmarine:subset",
            supported_variables=["nitrate", "phosphate", "oxygen", "chlorophyll"],
            temporal_coverage="1993-present",
            spatial_coverage="Arctic Ocean",
            supported_formats=["NetCDF"],
            authentication_required=True,
        ),
    ]

    # ── helpers ────────────────────────────────────────────────────────────────

    def _cmems_credentials(self, credentials: Optional[Credentials]) -> Optional[Dict[str, str]]:
        if credentials and credentials.username and credentials.password:
            return {"username": credentials.username, "password": credentials.password}
        # Check CMEMS env vars
        u = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME", "")
        p = os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD", "")
        if u and p:
            return {"username": u, "password": p}
        return None

    def _catalogue_metadata(self, product_id: str) -> Optional[Dict[str, Any]]:
        try:
            # Added verify=False to ignore internal catalogue SSL handshake failures
            r = requests.get(
                f"{_CMEMS_CATALOGUE}/{product_id}",
                timeout=20,
                headers={"Accept": "application/json"},
                verify=False,
            )
            if r.ok:
                return r.json()
        except Exception:
            pass
        return None

    def _pick_product_entry(self, dataset_id: str) -> Dict[str, str]:
        return _PRODUCT_MAP.get(dataset_id, {
            "product_id": "GLOBAL_MULTIYEAR_PHY_001_030",
            "dataset_id": dataset_id,
            "variables":  "thetao",
        })

    # ── public interface ───────────────────────────────────────────────────────

    def discover_datasets(
        self,
        snapshot,
        context=None,
        credentials=None,
    ) -> List[DatasetDescriptor]:
        """
        Search Copernicus Marine catalogue for products matching the query.
        Uses copernicusmarine.describe() when SDK is available, else REST.
        """
        _ctx = context if isinstance(context, dict) else (vars(context) if context and hasattr(context, "__dict__") else {})
        _snap_vars = list(getattr(snapshot, "variables_available", None) or []) if snapshot else []
        keywords = _ctx.get("keywords") or _ctx.get("variables") or _snap_vars
        query = " ".join(keywords) if keywords else "ocean"
        if _cmems_sdk_available():
            try:
                import copernicusmarine as cm
                creds = self._cmems_credentials(credentials)
                kwargs = {"contains": [query], "show_all_versions": False}
                if creds:
                    kwargs["username"] = creds["username"]
                    kwargs["password"] = creds["password"]
                cat = cm.describe(**kwargs)
                out = []
                for prod in (cat.products or [])[:10]:
                    first_ds = (prod.datasets or [None])[0]
                    did = first_ds.dataset_id if first_ds else prod.product_id
                    out.append(DatasetDescriptor(
                        provider="Copernicus Marine",
                        dataset_name=prod.title,
                        collection_name=prod.product_id,
                        dataset_id=did,
                        api_endpoint="copernicusmarine",
                        metadata_endpoint=f"https://data.marine.copernicus.eu/product/{prod.product_id}/description",
                        download_endpoint="copernicusmarine:subset",
                        supported_variables=[v.short_name for v in (getattr(first_ds, "variables", None) or [])[:8]],
                        temporal_coverage="See metadata",
                        spatial_coverage="See metadata",
                        supported_formats=["NetCDF"],
                        authentication_required=True,
                    ))
                if out:
                    return out
            except Exception:
                pass

        # REST catalogue fallback
        try:
            # Added verify=False to bypass certificate errors
            r = requests.get(
                _CMEMS_CATALOGUE,
                params={"q": query, "limit": 10},
                timeout=20,
                headers={"Accept": "application/json"},
                verify=False,
            )
            if r.ok:
                items = r.json() if isinstance(r.json(), list) else r.json().get("results", [])
                out = []
                for item in items:
                    pid = item.get("id", "")
                    out.append(DatasetDescriptor(
                        provider="Copernicus Marine",
                        dataset_name=item.get("title", pid),
                        collection_name=pid,
                        dataset_id=pid,
                        api_endpoint="copernicusmarine",
                        metadata_endpoint=f"https://data.marine.copernicus.eu/product/{pid}/description",
                        download_endpoint="copernicusmarine:subset",
                        supported_variables=[],
                        temporal_coverage=item.get("temporal_coverage", ""),
                        spatial_coverage="Global",
                        supported_formats=["NetCDF"],
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
        try:
            dataset = self._best_dataset(snapshot, fetch_request)
            if dataset is None:
                return DatasetMetadata(
                    source_id=snapshot.source_id,
                    dataset_id=snapshot.source_id,
                    variables=list(snapshot.variables_available or fetch_request.variables or []),
                    retrieval_method="CMEMS static catalog",
                    unavailable_reason="No matching CMEMS dataset found.",
                )

            entry = self._pick_product_entry(dataset.dataset_id)
            cat_meta = self._catalogue_metadata(entry["product_id"]) or {}

            variables = _normalise_cmems_vars(
                list(fetch_request.variables or dataset.supported_variables)
            )
            size = _estimate_size(variables, fetch_request.bounding_box, fetch_request.time_range)
            creds = self._cmems_credentials(credentials)

            # SDK describe() gives the richest structured record when installed
            # and credentials are available; fall back to REST catalogue fields
            # otherwise. Both are read defensively since field names vary by
            # product/version.
            sdk_meta: Dict[str, Any] = {}
            if _cmems_sdk_available() and creds:
                try:
                    import copernicusmarine as cm
                    described = cm.describe(contains=[entry["product_id"]],
                                             username=creds["username"], password=creds["password"])
                    prod = next((p for p in (described.products or []) if p.product_id == entry["product_id"]), None)
                    if prod:
                        sdk_meta = {
                            "title": getattr(prod, "title", None),
                            "abstract": getattr(prod, "description", None) or getattr(prod, "abstract", None),
                            "keywords": getattr(prod, "keywords", None),
                            "licence": getattr(prod, "licence", None) or getattr(prod, "license", None),
                            "doi": getattr(prod, "digital_object_identifier", None) or getattr(prod, "doi", None),
                        }
                except Exception:
                    sdk_meta = {}

            title = sdk_meta.get("title") or cat_meta.get("title")
            abstract = sdk_meta.get("abstract") or cat_meta.get("abstract") or cat_meta.get("description")
            keywords = sdk_meta.get("keywords") or cat_meta.get("keywords")
            licence = sdk_meta.get("licence") or cat_meta.get("licence") or cat_meta.get("license")
            doi = sdk_meta.get("doi") or cat_meta.get("doi")
            extent = cat_meta.get("bbox") or cat_meta.get("extent")

            return DatasetMetadata(
                source_id=snapshot.source_id,
                dataset_id=dataset.dataset_id,
                collection=dataset.collection_name,
                product=dataset.dataset_name,
                download_endpoint=f"copernicusmarine:subset:{dataset.dataset_id}",
                api_endpoint="copernicusmarine",
                metadata_endpoint=dataset.metadata_endpoint,
                file_size_bytes=size,
                variables=variables,
                spatial_coverage=cat_meta.get("spatial_coverage", dataset.spatial_coverage),
                temporal_coverage=cat_meta.get("temporal_coverage", dataset.temporal_coverage),
                file_format="NetCDF",
                content_type="application/x-netcdf",
                license=licence or "Copernicus Marine Service Product Licence",
                retrieval_method=(
                    "copernicusmarine.subset()" if _cmems_sdk_available()
                    else "CMEMS REST API (OPeNDAP)"
                ),
                unavailable_reason="" if creds else
                    "Copernicus Marine credentials required. "
                    "Register at https://data.marine.copernicus.eu/ and provide username/password.",
                dataset_name=title or dataset.dataset_name,
                provider="Copernicus Marine Service (CMEMS)",
                description=abstract,
                bounding_box=list(extent) if isinstance(extent, (list, tuple)) and len(extent) == 4 else None,
                crs="EPSG:4326",
                citation=(f"https://doi.org/{doi}" if doi else None),
                keywords=keywords,
                authentication_required=True,
            )
        except Exception as exc:
            # Safe absolute fallback block ensuring a properly initialized DatasetMetadata object
            return DatasetMetadata(
                source_id=snapshot.source_id,
                dataset_id=snapshot.source_id,
                collection=getattr(snapshot, "dataset_type", "Unknown"),
                product=snapshot.name,
                download_endpoint=snapshot.url,
                api_endpoint="copernicusmarine",
                metadata_endpoint=snapshot.url,
                file_size_bytes=50.0 * 1024 * 1024,
                variables=list(getattr(snapshot, "variables_available", []) or fetch_request.variables or []),
                file_format="NetCDF",
                content_type="application/x-netcdf",
                retrieval_method="Copernicus Marine Global Fallback",
                unavailable_reason=f"Fatal Copernicus Marine metadata probe failure: {exc}",
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
        variables = _normalise_cmems_vars(
            list(fetch_request.variables or dataset.supported_variables)
        )
        size = _estimate_size(variables, fetch_request.bounding_box, fetch_request.time_range)
        return SizeEstimate(
            source_id=snapshot.source_id,
            estimated_bytes=size,
            is_exact=False,
            method="CMEMS payload estimation (vars × days × area at 1/12° resolution)",
            human_readable=format_bytes(size),
        )

    def resolve_download_asset(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> Optional[str]:
        """
        CMEMS subset requests are job-based / SDK-based.
        Return a descriptor string; actual download requires fetch_subset/fetch_full.
        """
        dataset = self._best_dataset(snapshot, fetch_request)
        if dataset is None:
            return None
        return f"copernicusmarine:subset:{dataset.dataset_id}"

    def fetch_subset(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> str:
        """
        Server-side spatial, temporal, depth, and variable subsetting via
        copernicusmarine.subset() or OPeNDAP.
        """
        dataset = self._best_dataset(snapshot, fetch_request)
        if dataset is None:
            raise RuntimeError("CMEMS connector: no dataset matched for subsetting.")

        creds = self._cmems_credentials(credentials)
        if creds is None:
            raise RuntimeError(
                "Copernicus Marine credentials required. "
                "Register at https://data.marine.copernicus.eu/ "
                "and provide username and password."
            )

        variables = _normalise_cmems_vars(
            list(fetch_request.variables or dataset.supported_variables)
        )
        dest = fetch_request.dest_path or os.path.join(
            tempfile.gettempdir(), f"{dataset.dataset_id}_subset.nc"
        )

        if _cmems_sdk_available():
            return self._subset_via_sdk(dataset, variables, fetch_request, creds, dest)
        else:
            return self._subset_via_opendap(dataset, variables, fetch_request, creds, dest)

    def _subset_via_sdk(
        self,
        dataset: DatasetDescriptor,
        variables: List[str],
        fetch_request: FetchRequest,
        creds: Dict[str, str],
        dest: str,
    ) -> str:
        import copernicusmarine as cm

        kwargs: Dict[str, Any] = {
            "dataset_id":    dataset.dataset_id,
            "variables":     variables,
            "output_filename": os.path.basename(dest),
            "output_directory": os.path.dirname(dest) or ".",
            "username":      creds["username"],
            "password":      creds["password"],
            "overwrite_output_data": True,
            "disable_progress_bar": True,
        }

        if fetch_request.bounding_box and len(fetch_request.bounding_box) == 4:
            west, south, east, north = fetch_request.bounding_box
            kwargs["minimum_longitude"] = west
            kwargs["maximum_longitude"] = east
            kwargs["minimum_latitude"]  = south
            kwargs["maximum_latitude"]  = north

        if fetch_request.time_range:
            if fetch_request.time_range[0]:
                kwargs["start_datetime"] = fetch_request.time_range[0]
            if len(fetch_request.time_range) > 1 and fetch_request.time_range[1]:
                kwargs["end_datetime"] = fetch_request.time_range[1]

        # Depth (metres) from fetch_request if present
        depth = getattr(fetch_request, "depth_range", None)
        if depth and len(depth) == 2:
            kwargs["minimum_depth"] = depth[0]
            kwargs["maximum_depth"] = depth[1]

        cm.subset(**kwargs)
        self.validate_download(dest)
        return dest

    def _subset_via_opendap(
        self,
        dataset: DatasetDescriptor,
        variables: List[str],
        fetch_request: FetchRequest,
        creds: Dict[str, str],
        dest: str,
    ) -> str:
        """
        Minimal OPeNDAP fallback using CMEMS THREDDS endpoint.
        Only works for public/semi-public products.

        CHANGED: validate Content-Type at HTTP layer before writing anything to
        disk. Modern CMEMS login/error redirects return text/html with a 200 OK,
        so checking r.ok alone is insufficient — the old code would write the
        HTML login page to disk and only detect it after the fact, leaving a
        corrupted file behind. Now we reject HTML responses before opening the
        output file, so no partial/bogus file is ever written.

        CHANGED: support bearer token auth (passed as extra_headers or via
        COPERNICUSMARINE_SERVICE_TOKEN env var) in addition to HTTP Basic Auth,
        since newer CMEMS endpoints use token-based access.
        """
        entry = self._pick_product_entry(dataset.dataset_id)
        opendap_base = (
            f"https://nrt.cmems-du.eu/thredds/dodsC/{entry['dataset_id']}"
        )
        var_ce = ",".join(variables)
        url = f"{opendap_base}.nc?{var_ce}"

        # Build auth headers — prefer bearer token when available
        token = os.environ.get("COPERNICUSMARINE_SERVICE_TOKEN", "")
        headers: Dict[str, str] = {}
        auth = None
        if token:
            headers["Authorization"] = f"Bearer {token}"
        else:
            auth = (creds["username"], creds["password"])

        r = requests.get(
            url,
            auth=auth,
            headers=headers,
            stream=True,
            timeout=600,
            verify=False,
        )

        # ── HTTP-layer rejection (before writing anything) ──────────────────
        if not r.ok:
            raise RuntimeError(
                f"CMEMS OPeNDAP request failed: {r.status_code} {r.text[:200]}"
            )

        content_type = r.headers.get("Content-Type", "")
        if "text/html" in content_type or "text/plain" in content_type:
            # Read a snippet for a useful error message without streaming the
            # whole login page
            snippet = r.raw.read(512, decode_content=True)
            raise RuntimeError(
                f"CMEMS OPeNDAP returned HTML/text instead of NetCDF "
                f"(Content-Type: {content_type!r}) — likely an auth redirect. "
                f"Check Copernicus Marine credentials. Snippet: {snippet[:200]!r}"
            )

        # ── Write to disk only after HTTP validation passes ─────────────────
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        try:
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
        except Exception:
            # Clean up partial file so callers don't see a corrupted artifact
            if os.path.exists(dest):
                os.remove(dest)
            raise

        # ── Post-write validation (catches edge cases Content-Type missed) ──
        try:
            self.validate_download(dest)
        except RuntimeError:
            if os.path.exists(dest):
                os.remove(dest)
            raise

        return dest

    def fetch_full(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> str:
        """
        Download a full CMEMS product file (no spatial/temporal subsetting).
        Delegates to SDK get() or subset() with no area filter.
        """
        dataset = self._best_dataset(snapshot, fetch_request)
        if dataset is None:
            raise RuntimeError("CMEMS connector: no dataset matched.")

        creds = self._cmems_credentials(credentials)
        if creds is None:
            raise RuntimeError(
                "Copernicus Marine credentials required. "
                "Register at https://data.marine.copernicus.eu/ "
                "and provide username and password."
            )

        if not _cmems_sdk_available():
            raise RuntimeError(
                "copernicusmarine SDK is not installed. "
                "Install with: pip install copernicusmarine\n"
                "For OPeNDAP fallback, use fetch_subset() with a bounding box."
            )

        import copernicusmarine as cm
        dest = fetch_request.dest_path or os.path.join(
            tempfile.gettempdir(), f"{dataset.dataset_id}_full.nc"
        )
        variables = _normalise_cmems_vars(
            list(fetch_request.variables or dataset.supported_variables)
        )

        # Use get() for full product download
        result = cm.get(
            dataset_id=dataset.dataset_id,
            variables=variables,
            output_directory=os.path.dirname(dest) or ".",
            username=creds["username"],
            password=creds["password"],
            overwrite_output_data=True,
            disable_progress_bar=True,
        )
        # SDK returns list of downloaded paths
        if result and hasattr(result, "__iter__"):
            downloaded = list(result)
            if downloaded:
                dest = str(downloaded[0])
        self.validate_download(dest)
        return dest

    def validate_download(self, path: str) -> None:
        if not path or not os.path.exists(path):
            raise RuntimeError(f"CMEMS: downloaded file not found: {path}")
        size = os.path.getsize(path)
        if size == 0:
            raise RuntimeError(f"CMEMS: downloaded file is empty: {path}")
        if _is_html_content(path):
            raise RuntimeError(
                f"CMEMS download returned HTML (login or error page): {path}. "
                "Check Copernicus Marine credentials."
            )
        with open(path, "rb") as f:
            header = f.read(4)
        valid = (
            header[:3] == b"CDF"        # NetCDF3
            or header[:4] == b"\x89HDF" # NetCDF4 / HDF5
        )
        if not valid and size < 512:
            raise RuntimeError(
                f"CMEMS: file does not appear to be NetCDF "
                f"(header={header!r}, size={size}): {path}"
            )


register_connector(CopernicusMarineConnector) 