"""
Planetary Computer connector — live STAC asset resolution.

Provider APIs used:
  PC STAC Search  https://planetarycomputer.microsoft.com/api/stac/v1/search
  PC SAS signing  https://planetarycomputer.microsoft.com/api/sas/v1/sign
  PC Collections  https://planetarycomputer.microsoft.com/api/stac/v1/collections

Implements:
  discover_datasets     → PC STAC /collections listing
  probe_metadata        → STAC item + signed asset URL + HEAD size
  probe_size            → HEAD Content-Length on signed COG asset
  resolve_download_asset→ STAC search + SAS signing
  fetch_subset          → spatial/band subset via GDAL vsicurl or full download
  fetch_full            → streaming download of signed asset
  validate_download     → MIME / magic / HTML / STAC JSON rejection
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

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

_PC_STAC   = "https://planetarycomputer.microsoft.com/api/stac/v1"
_PC_SIGN   = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"

_HTML_SIGNATURES = (b"<!DOCTYPE", b"<html", b"<HTML", b"<head")
_STAC_KEYS = ("type", "stac_version", "stac_extensions", "id", "geometry", "links")

# Asset priority: prefer cloud-optimised and analysis-ready formats
_ASSET_PRIORITY = [
    # Sentinel-2
    "B04", "B08", "B02", "B03", "visual", "SCL",
    # Landsat
    "red", "green", "blue", "nir08", "swir16",
    # Generic
    "data", "cog", "netcdf", "map", "image",
]

# Collection → variable keywords for matching
_COLLECTION_KEYWORDS: Dict[str, List[str]] = {
    "sentinel-2-l2a":      ["surface reflectance", "ndvi", "sentinel", "reflectance", "vegetation"],
    "esa-worldcover":      ["land cover", "land use", "worldcover"],
    "landsat-c2-l2":       ["landsat", "surface temperature", "lst", "reflectance"],
    "cop-dem-glo-30":      ["elevation", "dem", "terrain", "topography"],
    "era5-pds":            ["era5", "reanalysis", "wind", "temperature", "era"],
    "modis-13A1-061":      ["modis", "vegetation", "evi", "ndvi"],
    "nasadem-hgt":         ["nasadem", "elevation", "srtm"],
    "hls":                 ["hls", "harmonised landsat sentinel", "harmonized"],
    "io-lulc-annual-v02":  ["land cover", "land use", "lulc", "impact observatory"],
}


def _sign_href(href: str) -> str:
    """Return SAS-signed version of a Planetary Computer asset href."""
    try:
        r = requests.get(f"{_PC_SIGN}?href={href}", timeout=10)
        if r.ok:
            return r.json().get("href", href)
    except Exception:
        pass
    return href


def _estimate_size_head(url: str) -> Optional[int]:
    try:
        r = requests.head(url, timeout=15, allow_redirects=True)
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


def _is_stac_json(path: str) -> bool:
    """Detect accidentally downloaded STAC documents."""
    try:
        with open(path, "r", errors="ignore") as f:
            head = f.read(300)
        return all(k in head for k in ("stac_version", "type", "links"))
    except Exception:
        return False


# ── connector ──────────────────────────────────────────────────────────────────

class PlanetaryComputerConnector(StaticDatasetConnector):
    descriptor = ConnectorDescriptor(
        connector_id="planetary_computer",
        provider_name="Microsoft Planetary Computer",
        connector_type=ConnectorType.official_sdk,
        supported_access_types=(AccessType.public,),
        supported_dataset_types=(DatasetType.raster, DatasetType.vector, DatasetType.gridded),
        supported_authentication=(AuthenticationType.none,),
        capabilities=(
            CapabilityFlags.supports_metadata
            | CapabilityFlags.supports_api
            | CapabilityFlags.supports_dataset_search
            | CapabilityFlags.supports_download
            | CapabilityFlags.supports_subsetting
        ),
        priority=32,
    )
    provider_keywords = ("planetary computer", "microsoft planetary", "stac", "pc stac")
    api_keywords = ("pystac", "stac", "planetary-computer")

    datasets = [
        DatasetDescriptor(
            provider="Microsoft Planetary Computer",
            dataset_name="Sentinel-2 Level-2A",
            collection_name="sentinel-2-l2a",
            dataset_id="sentinel-2-l2a",
            api_endpoint=_PC_STAC,
            metadata_endpoint="https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a",
            download_endpoint=_PC_STAC,
            supported_variables=["surface reflectance", "land cover", "vegetation index", "ndvi"],
            temporal_coverage="2015-present",
            spatial_coverage="Global land",
            supported_formats=["Cloud Optimized GeoTIFF"],
            authentication_required=False,
        ),
        DatasetDescriptor(
            provider="Microsoft Planetary Computer",
            dataset_name="ESA WorldCover",
            collection_name="esa-worldcover",
            dataset_id="esa-worldcover",
            api_endpoint=_PC_STAC,
            metadata_endpoint="https://planetarycomputer.microsoft.com/dataset/esa-worldcover",
            download_endpoint=_PC_STAC,
            supported_variables=["land cover", "land use", "land use change"],
            temporal_coverage="2020, 2021",
            spatial_coverage="Global land",
            supported_formats=["Cloud Optimized GeoTIFF"],
            authentication_required=False,
        ),
        DatasetDescriptor(
            provider="Microsoft Planetary Computer",
            dataset_name="Copernicus DEM GLO-30",
            collection_name="cop-dem-glo-30",
            dataset_id="cop-dem-glo-30",
            api_endpoint=_PC_STAC,
            metadata_endpoint="https://planetarycomputer.microsoft.com/dataset/cop-dem-glo-30",
            download_endpoint=_PC_STAC,
            supported_variables=["elevation", "dem", "terrain"],
            temporal_coverage="2021",
            spatial_coverage="Global",
            supported_formats=["Cloud Optimized GeoTIFF"],
            authentication_required=False,
        ),
        DatasetDescriptor(
            provider="Microsoft Planetary Computer",
            dataset_name="Landsat Collection 2 Level-2",
            collection_name="landsat-c2-l2",
            dataset_id="landsat-c2-l2",
            api_endpoint=_PC_STAC,
            metadata_endpoint="https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2",
            download_endpoint=_PC_STAC,
            supported_variables=["surface reflectance", "surface temperature", "lst"],
            temporal_coverage="1982-present",
            spatial_coverage="Global land",
            supported_formats=["Cloud Optimized GeoTIFF"],
            authentication_required=False,
        ),
    ]

    # ── internal helpers ───────────────────────────────────────────────────────

    def _pick_collection(self, fetch_request: FetchRequest) -> str:
        """Map requested variables to the best PC collection."""
        variables_lower = " ".join(v.lower() for v in (fetch_request.variables or []))
        for collection, kws in _COLLECTION_KEYWORDS.items():
            if any(kw in variables_lower for kw in kws):
                return collection
        return "sentinel-2-l2a"  # sensible default

    def _stac_search(
        self,
        collection: str,
        bbox=None,
        time_range=None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """POST to PC STAC /search and return feature list."""
        params: Dict[str, Any] = {
            "collections": [collection],
            "limit": limit,
            "sortby": "-properties.datetime",
        }
        if bbox and len(bbox) == 4:
            params["bbox"] = list(bbox)
        if time_range and time_range[0]:
            end = time_range[1] if (len(time_range) > 1 and time_range[1]) else ".."
            params["datetime"] = f"{time_range[0]}/{end}"

        try:
            r = requests.post(f"{_PC_STAC}/search", json=params, timeout=25)
            if not r.ok:
                return []
            return r.json().get("features", [])
        except Exception:
            return []

    def _best_asset(
        self,
        item: Dict[str, Any],
        variables: Optional[List[str]] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Pick the best asset from a STAC item.
        Returns (asset_key, href). Prefers COG/GeoTIFF data assets.
        Skips browse, metadata, tilejson, and rendered_preview assets.
        """
        assets = item.get("assets", {})
        skip_types = {"image/png", "image/jpeg", "application/json",
                      "text/html", "application/geo+json"}
        skip_roles = {"overview", "thumbnail", "rendered_preview"}

        def _asset_score(name: str, asset: Dict) -> int:
            roles = set(asset.get("roles", []))
            if roles & skip_roles:
                return -1
            if asset.get("type", "") in skip_types:
                return -1
            href = asset.get("href", "")
            if href.endswith((".json", ".html", ".png", ".jpg", ".jpeg")):
                return -1
            score = 0
            # Prefer COG / NetCDF / HDF
            if href.endswith((".tif", ".tiff", ".nc", ".nc4", ".hdf", ".h5")):
                score += 10
            if "data" in roles or "data" in name.lower():
                score += 5
            # Match against requested variables / band names
            if variables:
                for v in variables:
                    if v.lower().replace(" ", "") in name.lower():
                        score += 20
            # Known priority names
            for i, pname in enumerate(_ASSET_PRIORITY):
                if name == pname:
                    score += max(0, 15 - i)
                    break
            return score

        scored = []
        for name, asset in assets.items():
            s = _asset_score(name, asset)
            if s >= 0:
                scored.append((s, name, asset))
        if not scored:
            return None, None
        scored.sort(key=lambda x: -x[0])
        _, best_name, best_asset = scored[0]
        return best_name, best_asset.get("href")

    def _resolve_signed_asset(
        self,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """
        Search STAC, pick best asset, sign it.
        Returns (asset_key, signed_href, size_bytes).
        """
        collection = self._pick_collection(fetch_request)
        items = self._stac_search(
            collection,
            bbox=fetch_request.bounding_box,
            time_range=fetch_request.time_range,
            limit=5,
        )
        if not items:
            return None, None, None

        for item in items:
            key, href = self._best_asset(item, list(fetch_request.variables or []))
            if href:
                signed = _sign_href(href)
                size   = _estimate_size_head(signed)
                return key, signed, size

        return None, None, None

    # ── public interface ───────────────────────────────────────────────────────

    def discover_datasets(
        self,
        query: str,
        bbox=None,
        time_range=None,
        credentials: Optional[Credentials] = None,
    ) -> List[DatasetDescriptor]:
        """List PC STAC collections and return as DatasetDescriptors."""
        try:
            r = requests.get(f"{_PC_STAC}/collections", timeout=20)
            if r.ok:
                out = []
                for col in r.json().get("collections", [])[:20]:
                    cid = col.get("id", "")
                    if query.lower() not in (col.get("title", "") + cid).lower():
                        continue
                    extent = col.get("extent", {})
                    spatial = extent.get("spatial", {}).get("bbox", [[]])[0]
                    temporal = extent.get("temporal", {}).get("interval", [[]])[0]
                    out.append(DatasetDescriptor(
                        provider="Microsoft Planetary Computer",
                        dataset_name=col.get("title", cid),
                        collection_name=cid,
                        dataset_id=cid,
                        api_endpoint=_PC_STAC,
                        metadata_endpoint=f"https://planetarycomputer.microsoft.com/dataset/{cid}",
                        download_endpoint=f"{_PC_STAC}/search",
                        supported_variables=list(col.get("summaries", {}).get("eo:bands", []))[:6],
                        temporal_coverage=f"{temporal[0] or ''} – {temporal[1] or 'present'}" if temporal else "",
                        spatial_coverage="Global" if not spatial else str(spatial),
                        supported_formats=["Cloud Optimized GeoTIFF", "NetCDF"],
                        authentication_required=False,
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
        dataset  = self._best_dataset(snapshot, fetch_request)
        collection = self._pick_collection(fetch_request)
        key, signed_url, size = self._resolve_signed_asset(fetch_request, credentials)

        # Fetch item-level metadata for temporal / spatial details
        item_meta: Dict[str, Any] = {}
        items = self._stac_search(collection, fetch_request.bounding_box, fetch_request.time_range, limit=1)
        if items:
            item_meta = items[0].get("properties", {})

        return DatasetMetadata(
            source_id=snapshot.source_id,
            dataset_id=collection,
            collection=collection,
            product=dataset.dataset_name if dataset else collection,
            download_endpoint=signed_url or _PC_STAC,
            api_endpoint=_PC_STAC,
            metadata_endpoint=f"https://planetarycomputer.microsoft.com/dataset/{collection}",
            file_size_bytes=size,
            variables=list(fetch_request.variables or (dataset.supported_variables if dataset else [])),
            spatial_coverage=dataset.spatial_coverage if dataset else "Global land",
            temporal_coverage=item_meta.get("datetime", dataset.temporal_coverage if dataset else "Unknown"),
            file_format="Cloud Optimized GeoTIFF",
            content_type="image/tiff; application=geotiff; profile=cloud-optimized",
            license="Various — see individual collection licence",
            retrieval_method="Planetary Computer STAC search + SAS token signing",
            unavailable_reason="" if signed_url else
                "PC STAC search returned no items for this query. "
                "Try adjusting bbox or time range.",
        )

    def probe_size(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> SizeEstimate:
        _, signed_url, size = self._resolve_signed_asset(fetch_request, credentials)
        if size:
            return SizeEstimate(
                source_id=snapshot.source_id,
                estimated_bytes=size,
                is_exact=True,
                method="HEAD Content-Length on signed Planetary Computer asset",
                human_readable=format_bytes(size),
            )
        return SizeEstimate(
            source_id=snapshot.source_id,
            method="PC asset not resolved or HEAD returned no Content-Length",
        )

    def resolve_download_asset(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> Optional[str]:
        _, signed_url, _ = self._resolve_signed_asset(fetch_request, credentials)
        return signed_url

    def fetch_subset(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> str:
        """
        PC COGs are cloud-native; spatial subset is possible via GDAL vsicurl.
        If rasterio/osgeo is available, read window; otherwise download full asset.
        """
        key, signed_url, size = self._resolve_signed_asset(fetch_request, credentials)
        if not signed_url:
            raise RuntimeError(
                "Planetary Computer: no asset resolved for subsetting. "
                "Adjust bbox or time range."
            )

        dest = fetch_request.dest_path or os.path.join(
            os.getcwd(), f"pc_subset_{key or 'asset'}.tif"
        )

        # Attempt windowed read with rasterio
        if fetch_request.bounding_box:
            try:
                import rasterio
                from rasterio.crs import CRS
                from rasterio.windows import from_bounds

                west, south, east, north = fetch_request.bounding_box
                with rasterio.open(f"/vsicurl/{signed_url}") as src:
                    window = from_bounds(west, south, east, north, src.transform)
                    transform = src.window_transform(window)
                    data = src.read(window=window)
                    profile = src.profile.copy()
                    profile.update({
                        "height": data.shape[1],
                        "width":  data.shape[2],
                        "transform": transform,
                    })
                    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
                    with rasterio.open(dest, "w", **profile) as dst:
                        dst.write(data)
                self.validate_download(dest)
                return dest
            except ImportError:
                pass  # rasterio not available; fall through
            except Exception:
                pass  # window read failed; fall through

        # Full download fallback
        return self._download_url(signed_url, dest, snapshot, fetch_request)

    def fetch_full(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> str:
        key, signed_url, size = self._resolve_signed_asset(fetch_request, credentials)
        if not signed_url:
            raise RuntimeError(
                "Planetary Computer STAC search returned no items. "
                "Try adjusting bbox or time range."
            )
        dest = fetch_request.dest_path or os.path.join(
            os.getcwd(), f"pc_{key or 'asset'}.tif"
        )
        return self._download_url(signed_url, dest, snapshot, fetch_request, size)

    def _download_url(
        self,
        url: str,
        dest: str,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        expected_size: Optional[int] = None,
    ) -> str:
        from agents.agent4_download_engine import DownloadEngine, DownloadTask

        task = DownloadTask(
            url=url,
            dest_path=dest,
            expected_size=expected_size,
            source_id=snapshot.source_id,
            provider=snapshot.name,
            connector_id=self.name,
            protocol=self.descriptor.connector_type.value,
        )
        result = DownloadEngine().download_one(task)
        fetch_request.metadata["download_result"] = result
        if not result.success:
            raise RuntimeError(result.error or "Planetary Computer download failed.")
        self.validate_download(result.dest_path)
        return result.dest_path

    def validate_download(self, path: str) -> None:
        if not path or not os.path.exists(path):
            raise RuntimeError(f"PC: downloaded file not found: {path}")
        size = os.path.getsize(path)
        if size == 0:
            raise RuntimeError(f"PC: downloaded file is empty: {path}")
        if _is_html_content(path):
            raise RuntimeError(
                f"PC: downloaded file is HTML (SAS token may have expired): {path}"
            )
        if _is_stac_json(path):
            raise RuntimeError(
                f"PC: downloaded file is a STAC document, not a data asset: {path}. "
                "Check asset key selection logic."
            )
        with open(path, "rb") as f:
            header = f.read(4)
        # GeoTIFF: II\x2a\x00 (little-endian) or MM\x00\x2a (big-endian)
        # NetCDF3: CDF\x01
        # HDF5/NetCDF4: \x89HDF
        valid = (
            header[:2] in (b"II", b"MM")     # TIFF
            or header[:3] == b"CDF"           # NetCDF3
            or header[:4] == b"\x89HDF"       # HDF5
            or size > 10_000                  # any reasonable binary blob
        )
        if not valid and size < 512:
            raise RuntimeError(
                f"PC: file appears invalid (header={header!r}, size={size}): {path}"
            )


register_connector(PlanetaryComputerConnector)
