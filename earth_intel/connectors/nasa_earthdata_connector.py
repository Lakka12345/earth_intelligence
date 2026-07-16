"""
NASA EarthData connector — live CMR granule resolution.

Implements full provider connector interface:
  discover_datasets  → CMR collection search
  probe_metadata     → CMR granule metadata
  probe_size         → HEAD / CMR granule size
  resolve_download_asset → live granule URL
  fetch_subset       → OPeNDAP or local subset
  fetch_full         → streaming download of actual granule
  validate_download  → MIME / extension / HTML rejection

Provider APIs used:
  https://cmr.earthdata.nasa.gov/search/collections.json
  https://cmr.earthdata.nasa.gov/search/granules.json
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

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

_CMR = "https://cmr.earthdata.nasa.gov/search"
_GISTEMP_CSV = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"

# short_name → (daac_provider, short_name)
_CMR_DATASETS: dict[str, Tuple[str, str]] = {
    "GISTEMP-v4":   ("GISS",       "GISTEMP"),
    "M2T1NXSLV":   ("GES_DISC",   "M2T1NXSLV"),
    "MOD11A1":      ("LPDAAC_ECS", "MOD11A1"),
    "MYD11A1":      ("LPDAAC_ECS", "MYD11A1"),
    "GPM_3IMERGDF": ("GES_DISC",   "GPM_3IMERGDF"),
    "TRMM_3B42":    ("GES_DISC",   "TRMM_3B42"),
    "MODIS_Terra_ChlA": ("OB_DAAC", "MODISA_L3m_CHL"),
}

_PREFERRED_FORMATS = {"application/x-netcdf", "application/x-hdf5",
                      "image/tiff", "application/x-hdf"}
_HTML_SIGNATURES   = (b"<!DOCTYPE", b"<html", b"<HTML", b"<head", b"<body")
_BAD_MIMES         = {"text/html", "application/xhtml+xml", "text/plain"}


# ── helpers ────────────────────────────────────────────────────────────────────

def _cmr_headers(credentials: Optional[Credentials] = None) -> dict:
    h = {"Accept": "application/json"}
    if credentials and credentials.token:
        h["Authorization"] = f"Bearer {credentials.token}"
    if credentials and credentials.headers:
        h.update(credentials.headers)
    return h


def _cmr_cookies(credentials: Optional[Credentials] = None) -> Optional[dict]:
    """Cookies obtained from a completed browser EarthData Login, if any."""
    if credentials and credentials.cookies:
        return credentials.cookies
    return None


def _bbox_param(bb) -> Optional[str]:
    if bb and len(bb) == 4:
        return ",".join(str(x) for x in bb)
    return None


def _temporal_param(tr) -> Optional[str]:
    if not tr:
        return None
    start = tr[0] or ""
    end   = tr[1] if len(tr) > 1 else ""
    return f"{start},{end}"


def _estimate_size_head(url: str, credentials: Optional[Credentials] = None) -> Optional[int]:
    try:
        h = _cmr_headers(credentials)
        h.pop("Accept", None)
        r = requests.head(url, headers=h, cookies=_cmr_cookies(credentials), timeout=15, allow_redirects=True, verify=False)
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


# ── connector ──────────────────────────────────────────────────────────────────

class NASAEarthDataConnector(StaticDatasetConnector):
    descriptor = ConnectorDescriptor(
        connector_id="nasa_earthdata",
        provider_name="NASA EarthData",
        connector_type=ConnectorType.official_sdk,
        supported_access_types=(AccessType.public, AccessType.user_credentials_required),
        supported_dataset_types=(DatasetType.gridded, DatasetType.raster, DatasetType.time_series),
        supported_authentication=(AuthenticationType.none, AuthenticationType.user_credentials),
        capabilities=(
            CapabilityFlags.supports_metadata
            | CapabilityFlags.supports_api
            | CapabilityFlags.supports_dataset_search
            | CapabilityFlags.supports_download
            | CapabilityFlags.supports_subsetting
            | CapabilityFlags.requires_browser_auth
        ),
        priority=20,
    )
    provider_keywords = ("nasa", "earthdata", "ges disc", "podaac", "giss", "modis", "viirs",
                         "merra", "gpm", "trmm")
    api_keywords = ("earthaccess", "cmr", "opendap", "nasa")

    datasets = [
        DatasetDescriptor(
            provider="NASA EarthData",
            dataset_name="NASA GISTEMP v4 Surface Temperature Analysis",
            collection_name="GISS Surface Temperature Analysis",
            dataset_id="GISTEMP-v4",
            doi="10.5281/zenodo.594034",
            api_endpoint=_CMR,
            metadata_endpoint="https://data.giss.nasa.gov/gistemp/",
            download_endpoint=_GISTEMP_CSV,
            supported_variables=["temperature anomaly", "surface temperature", "global temperature"],
            temporal_coverage="1880-present",
            spatial_coverage="Global",
            supported_formats=["CSV", "NetCDF"],
            authentication_required=False,
            source_url="https://data.giss.nasa.gov/gistemp/",
        ),
        DatasetDescriptor(
            provider="NASA EarthData",
            dataset_name="MERRA-2 Reanalysis (Hourly, Single Level)",
            collection_name="NASA GES DISC MERRA-2",
            dataset_id="M2T1NXSLV",
            api_endpoint=_CMR,
            metadata_endpoint="https://disc.gsfc.nasa.gov/datasets/M2T1NXSLV_5.12.4/summary",
            download_endpoint="https://disc.gsfc.nasa.gov/",
            supported_variables=["temperature", "wind", "humidity", "pressure", "precipitation"],
            temporal_coverage="1980-present",
            spatial_coverage="Global",
            supported_formats=["NetCDF", "OPeNDAP"],
            authentication_required=True,
        ),
        DatasetDescriptor(
            provider="NASA EarthData",
            dataset_name="MODIS Terra Land Surface Temperature (MOD11A1)",
            collection_name="MODIS Terra LST",
            dataset_id="MOD11A1",
            api_endpoint=_CMR,
            metadata_endpoint="https://lpdaac.usgs.gov/products/mod11a1v061/",
            download_endpoint="https://e4ftl01.cr.usgs.gov/MOLT/MOD11A1.061/",
            supported_variables=["land surface temperature", "LST", "emissivity"],
            temporal_coverage="2000-present",
            spatial_coverage="Global land",
            supported_formats=["HDF", "GeoTIFF"],
            authentication_required=True,
        ),
        DatasetDescriptor(
            provider="NASA EarthData",
            dataset_name="GPM IMERG Daily Precipitation",
            collection_name="GPM IMERG",
            dataset_id="GPM_3IMERGDF",
            api_endpoint=_CMR,
            metadata_endpoint="https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGDF_07/summary",
            download_endpoint="https://disc.gsfc.nasa.gov/",
            supported_variables=["precipitation", "rain rate", "rainfall"],
            temporal_coverage="2000-present",
            spatial_coverage="Global (60°S–60°N)",
            supported_formats=["NetCDF", "HDF5"],
            authentication_required=True,
        ),
    ]

    # ── internal helpers ───────────────────────────────────────────────────────

    def _cmr_collection_metadata(self, dataset_id: str) -> dict:
        """Fetch the CMR collection record for a dataset (description, keywords, DOI, license)."""
        provider, short_name = self._cmr_short_name(dataset_id)
        params = {"short_name": short_name, "page_size": 1}
        if provider:
            params["provider"] = provider
        try:
            r = requests.get(f"{_CMR}/collections.json", params=params, timeout=20, verify=False)
            if not r.ok:
                return {}
            entries = r.json().get("feed", {}).get("entry", [])
            return entries[0] if entries else {}
        except Exception:
            return {}

    @staticmethod
    def _granule_spatiotemporal(granule: dict) -> dict:
        """Extract bbox / start / end date from a CMR granule entry, where present."""
        out: dict = {}
        boxes = granule.get("boxes")
        if boxes:
            try:
                # CMR "boxes" are "south west north east" strings
                s, w, n, e = (float(x) for x in boxes[0].split())
                out["bbox"] = [w, s, e, n]
            except Exception:
                pass
        out["start_date"] = granule.get("time_start")
        out["end_date"] = granule.get("time_end")
        return out

    def _cmr_short_name(self, dataset_id: str) -> Tuple[str, str]:
        return _CMR_DATASETS.get(dataset_id, ("", dataset_id))

    def _cmr_collection_search(self, keyword: str, limit: int = 5) -> List[dict]:
        """Search CMR collections by free-text keyword."""
        try:
            r = requests.get(
                f"{_CMR}/collections.json",
                params={"keyword": keyword, "page_size": limit,
                        "sort_key": "-score", "has_granules": True},
                timeout=20,
                verify=False,
            )
            if not r.ok:
                return []
            entries = r.json().get("feed", {}).get("entry", [])
            return entries
        except Exception:
            return []

    def _cmr_granule_search(
        self,
        short_name: str,
        provider: str,
        bbox=None,
        time_range=None,
        limit: int = 1,
        credentials: Optional[Credentials] = None,
    ) -> List[dict]:
        """Return CMR granule entries for a given collection."""
        params: dict = {
            "short_name": short_name,
            "page_size": limit,
            "sort_key": "-start_date",
            "online_only": True,
        }
        if provider:
            params["provider"] = provider
        bb = _bbox_param(bbox)
        if bb:
            params["bounding_box"] = bb
        tp = _temporal_param(time_range)
        if tp:
            params["temporal[]"] = tp

        try:
            r = requests.get(
                f"{_CMR}/granules.json",
                params=params,
                headers=_cmr_headers(credentials),
                timeout=25,
                verify=False,
            )
            if not r.ok:
                return []
            return r.json().get("feed", {}).get("entry", [])
        except Exception:
            return []

    def _pick_granule_url(self, granule: dict) -> Optional[str]:
        """
        Pick the best downloadable link from a CMR granule entry.
        Preference order: NetCDF / HDF5 > GeoTIFF > any non-browse link.
        """
        links = granule.get("links", [])
        # Filter to actual data links (not browse images, not documentation)
        data_links = [
            lk for lk in links
            if lk.get("rel", "").endswith("/data#") or lk.get("type", "") in _PREFERRED_FORMATS
            or (lk.get("href", "").endswith((".nc", ".nc4", ".hdf", ".h5", ".hdf5", ".tif", ".tiff")))
        ]
        # De-prioritise browse / metadata
        non_browse = [lk for lk in data_links if "browse" not in lk.get("href", "").lower()]
        candidates = non_browse or data_links
        if not candidates:
            # Fall back to any link that looks like a file
            candidates = [
                lk for lk in links
                if any(lk.get("href", "").endswith(ext)
                       for ext in (".nc", ".nc4", ".hdf", ".h5", ".hdf5",
                                   ".tif", ".tiff", ".csv", ".gz"))
            ]
        return candidates[0]["href"] if candidates else None

    def _granule_size_bytes(self, granule: dict) -> Optional[int]:
        """Extract file size from CMR granule metadata."""
        # granule-size is in MB in CMR
        sz = granule.get("granule_size")
        if sz:
            try:
                return int(float(sz) * 1024 * 1024)
            except (ValueError, TypeError):
                pass
        return None

    def _resolve_live_granule(
        self,
        dataset_id: str,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> Tuple[Optional[str], Optional[int]]:
        """
        Return (download_url, size_bytes) for the best matching CMR granule.
        """
        provider, short_name = self._cmr_short_name(dataset_id)
        granules = self._cmr_granule_search(
            short_name,
            provider,
            bbox=fetch_request.bounding_box,
            time_range=fetch_request.time_range,
            limit=5,
            credentials=credentials,
        )
        if not granules:
            return None, None

        for g in granules:
            url = self._pick_granule_url(g)
            if url:
                size = self._granule_size_bytes(g) or _estimate_size_head(url, credentials)
                return url, size

        return None, None

    # ── public interface ───────────────────────────────────────────────────────

    def discover_datasets(
        self,
        snapshot,
        context=None,
    ) -> List[DatasetDescriptor]:
        """Search CMR collections and return DatasetDescriptors."""
        _ctx = context if isinstance(context, dict) else (vars(context) if context and hasattr(context, "__dict__") else {})
        _snap_vars = list(getattr(snapshot, "variables_available", None) or []) if snapshot else []
        keywords = _ctx.get("keywords") or _ctx.get("variables") or _snap_vars
        query = " ".join(keywords) if keywords else "earth observation"
        entries = self._cmr_collection_search(query, limit=10)
        results = []
        for e in entries:
            short = e.get("short_name", "")
            results.append(DatasetDescriptor(
                provider="NASA EarthData",
                dataset_name=e.get("dataset_id", short),
                collection_name=e.get("archive_center", ""),
                dataset_id=short,
                api_endpoint=_CMR,
                metadata_endpoint=f"https://cmr.earthdata.nasa.gov/search/concepts/{e.get('id','')}.html",
                download_endpoint="",
                supported_variables=e.get("science_keywords_flat", "").split(" > ")[-3:],
                temporal_coverage=str(e.get("time_start", "")) + " – " + str(e.get("time_end", "present")),
                spatial_coverage="Global",
                supported_formats=["NetCDF", "HDF5"],
                authentication_required=True,
            ))
        # Supplement with static catalogue
        return results or self.datasets

    def probe_metadata(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> DatasetMetadata:
        dataset = self._best_dataset(snapshot, fetch_request)

        # ── GISTEMP: public direct CSV ─────────────────────────────────────────
        if dataset and dataset.dataset_id == "GISTEMP-v4":
            size = _estimate_size_head(_GISTEMP_CSV)
            return DatasetMetadata(
                source_id=snapshot.source_id,
                dataset_id="GISTEMP-v4",
                collection="GISS Surface Temperature Analysis",
                product="NASA GISTEMP v4",
                download_endpoint=_GISTEMP_CSV,
                api_endpoint=_CMR,
                metadata_endpoint="https://data.giss.nasa.gov/gistemp/",
                file_size_bytes=size,
                variables=["temperature anomaly", "global temperature"],
                spatial_coverage="Global",
                temporal_coverage="1880-present",
                file_format="CSV",
                content_type="text/csv",
                license="Public domain (NASA)",
                retrieval_method="Direct GISS endpoint",
                dataset_name="NASA GISTEMP v4 Surface Temperature Analysis",
                provider="NASA GISS",
                description="Combined land-surface air and sea-surface water temperature anomalies (GISTEMP v4).",
                spatial_resolution="2x2 degree grid (gridded product); station-based for tabular CSV",
                temporal_resolution="Monthly",
                crs="EPSG:4326",
                bounding_box=[-180.0, -90.0, 180.0, 90.0],
                start_date="1880-01-01",
                citation="GISTEMP Team, 2024: GISS Surface Temperature Analysis (GISTEMP), version 4. NASA GISS.",
                keywords=["temperature anomaly", "climate", "GISTEMP", "NASA GISS"],
                update_frequency="Monthly",
                authentication_required=False,
            )

        # ── CMR granule resolution ─────────────────────────────────────────────
        if dataset is None:
            return DatasetMetadata(
                source_id=snapshot.source_id,
                dataset_id=snapshot.source_id,
                variables=list(snapshot.variables_available or fetch_request.variables or []),
                retrieval_method="NASA EarthData static catalog",
                unavailable_reason="No matching dataset descriptor found.",
            )

        url, size = self._resolve_live_granule(dataset.dataset_id, fetch_request, credentials)

        # Pull the actual granule entry again (cheap, cached by CMR) purely to
        # extract bounding box / start / end date rather than re-deriving them.
        provider, short_name = self._cmr_short_name(dataset.dataset_id)
        granules = self._cmr_granule_search(
            short_name, provider, bbox=fetch_request.bounding_box,
            time_range=fetch_request.time_range, limit=1, credentials=credentials,
        )
        st = self._granule_spatiotemporal(granules[0]) if granules else {}

        col = self._cmr_collection_metadata(dataset.dataset_id)
        col_summary = col.get("summary") or None
        col_doi = ((col.get("associated_dois") or [{}])[0].get("doi") if col.get("associated_dois") else None) or col.get("doi")

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
            spatial_coverage=str(st["bbox"]) if st.get("bbox") else dataset.spatial_coverage,
            temporal_coverage=dataset.temporal_coverage,
            file_format=", ".join(dataset.supported_formats),
            content_type="application/x-netcdf",
            license="NASA open data",
            retrieval_method="NASA CMR granule search + CMR collection metadata",
            unavailable_reason="" if url else (
                "CMR granule search returned no results. "
                "EarthData Login credentials may be required for this dataset."
            ),
            dataset_name=dataset.dataset_name,
            provider="NASA " + (col.get("archive_center") or provider or "EarthData"),
            description=col_summary,
            bounding_box=st.get("bbox"),
            crs="EPSG:4326",
            start_date=st.get("start_date"),
            end_date=st.get("end_date"),
            citation=(f"https://doi.org/{col_doi}" if col_doi else None),
            keywords=(col.get("science_keywords_flat", "").split(" > ") if col.get("science_keywords_flat") else None) or None,
            version=col.get("version_id"),
            authentication_required=dataset.authentication_required,
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

        if dataset.dataset_id == "GISTEMP-v4":
            size = _estimate_size_head(_GISTEMP_CSV)
            if size:
                return SizeEstimate(
                    source_id=snapshot.source_id,
                    estimated_bytes=size,
                    is_exact=True,
                    method="HEAD Content-Length on GISS CSV",
                    human_readable=format_bytes(size),
                )

        url, size = self._resolve_live_granule(dataset.dataset_id, fetch_request, credentials)
        if size:
            return SizeEstimate(
                source_id=snapshot.source_id,
                estimated_bytes=size,
                is_exact=False,
                method="CMR granule_size metadata or HEAD",
                human_readable=format_bytes(size),
            )
        return SizeEstimate(
            source_id=snapshot.source_id,
            method="CMR returned no size information",
        )

    def resolve_download_asset(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> Optional[str]:
        """Return the best real downloadable asset URL."""
        dataset = self._best_dataset(snapshot, fetch_request)
        if dataset and dataset.dataset_id == "GISTEMP-v4":
            return _GISTEMP_CSV
        if dataset:
            url, _ = self._resolve_live_granule(dataset.dataset_id, fetch_request, credentials)
            return url
        return None

    def fetch_subset(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> str:
        """
        Subset via OPeNDAP constraint expression when the dataset supports it,
        otherwise download full and apply local xarray subset.
        """
        dataset = self._best_dataset(snapshot, fetch_request)
        url, _ = self._resolve_live_granule(
            dataset.dataset_id if dataset else snapshot.source_id,
            fetch_request,
            credentials,
        )
        if not url:
            raise RuntimeError("NASA EarthData: no granule URL resolved for subsetting.")

        # OPeNDAP: if URL ends in .nc/.nc4 we can append constraint expressions
        if url.endswith((".nc", ".nc4")) and fetch_request.variables:
            ce = ",".join(v.replace(" ", "_") for v in fetch_request.variables)
            opendap_url = f"{url}.nc?{ce}"
            try:
                r = requests.head(opendap_url, timeout=10, verify=False)
                if r.ok and "html" not in r.headers.get("content-type", ""):
                    url = opendap_url
            except Exception:
                pass  # fall through to full download

        return self.fetch_full(snapshot, fetch_request, credentials, _override_url=url)

    def fetch_full(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
        _override_url: Optional[str] = None,
    ) -> str:
        from agents.agent4_download_engine import DownloadEngine, DownloadTask

        meta = self.probe_metadata(snapshot, fetch_request, credentials)
        url = _override_url or meta.download_endpoint
        if not url or not url.startswith("http"):
            raise RuntimeError(
                f"NASA EarthData: no downloadable URL for {snapshot.name}. "
                "CMR returned no granule links. "
                "Ensure EarthData Login credentials are provided."
            )
        task = DownloadTask(
            url=url,
            dest_path=fetch_request.dest_path,
            expected_size=meta.file_size_bytes,
            source_id=snapshot.source_id,
            provider=snapshot.name,
            connector_id=self.name,
            protocol=self.descriptor.connector_type.value,
        )
        result = DownloadEngine().download_one(task, credentials)
        fetch_request.metadata["download_result"] = result
        if not result.success:
            raise RuntimeError(result.error or "NASA EarthData download failed validation.")
        self.validate_download(result.dest_path)
        return result.dest_path

    def validate_download(self, path: str) -> None:
        """
        Reject HTML pages, empty files, login redirects, error pages.
        Raises RuntimeError if the downloaded file is invalid.
        """
        if not path or not os.path.exists(path):
            raise RuntimeError(f"Downloaded file not found: {path}")
        size = os.path.getsize(path)
        if size == 0:
            raise RuntimeError(f"Downloaded file is empty: {path}")
        if _is_html_content(path):
            raise RuntimeError(
                f"Downloaded file is an HTML page (login redirect or landing page), not a dataset: {path}. "
                "Provide EarthData Login credentials."
            )
        # Accept NetCDF magic bytes or any non-HTML binary
        with open(path, "rb") as f:
            header = f.read(4)
        # NetCDF3 magic: CDF\x01 or CDF\x02; HDF5: \x89HDF
        valid_magic = (
            header[:3] == b"CDF"
            or header[:4] == b"\x89HDF"
            or header[:4] == b"PK\x03\x04"   # zip
            or size > 1024                    # any reasonably-sized binary
        )
        if not valid_magic and size < 512:
            raise RuntimeError(f"Downloaded file appears invalid (too small, {size} bytes): {path}")


register_connector(NASAEarthDataConnector)
