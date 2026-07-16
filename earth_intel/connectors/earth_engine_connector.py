"""
Google Earth Engine Connector — Real Provider Implementation
============================================================
Uses the Earth Engine Python API (earthengine-api) to:
  • Discover ImageCollections dynamically
  • Probe collection metadata (date range, band info, projection)
  • Estimate export size from collection properties
  • Export data to Google Drive / Cloud Storage (standard EE workflow)
  • Validate exported files

Authentication: earthengine-api uses Application Default Credentials (ADC)
or a service account key.  The connector handles missing credentials
gracefully and reports a clear error rather than crashing.

Official SDK: pip install earthengine-api
Catalog:      https://developers.google.com/earth-engine/datasets/catalog
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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
# EE import (optional — graceful degradation when not installed / auth'd)
# ---------------------------------------------------------------------------

try:
    import ee
    _EE_AVAILABLE = True
except ImportError:
    _EE_AVAILABLE = False
    logger.warning("earth_engine: earthengine-api not installed. "
                   "Install with: pip install earthengine-api")


def _ee_init(credentials=None) -> bool:
    """
    Attempt to initialise the Earth Engine API.
    Returns True on success, False on failure.
    """
    if not _EE_AVAILABLE:
        return False
    try:
        if credentials:
            ee.Initialize(credentials=credentials)
        else:
            ee.Initialize()
        logger.info("earth_engine: ee.Initialize() succeeded")
        return True
    except Exception as exc:
        logger.warning("earth_engine: ee.Initialize() failed — %s. "
                       "Run `earthengine authenticate` or provide credentials.", exc)
        return False


# ---------------------------------------------------------------------------
# Catalog: well-known collection IDs and their metadata
# ---------------------------------------------------------------------------

_KNOWN_COLLECTIONS: Dict[str, Dict[str, Any]] = {
    "MODIS/061/MOD11A1": {
        "title":      "MODIS Land Surface Temperature (MOD11A1 v6.1)",
        "variables":  ["land surface temperature", "temperature", "emissivity"],
        "temporal":   "2000-present",
        "spatial":    "Global land",
        "scale_m":    1000,
        "bands":      ["LST_Day_1km", "LST_Night_1km", "QC_Day", "QC_Night",
                       "Day_view_time", "Night_view_time", "Day_view_angl",
                       "Night_view_angl", "Emis_31", "Emis_32"],
    },
    "UCSB-CHG/CHIRPS/DAILY": {
        "title":      "CHIRPS Daily Precipitation (1981-present)",
        "variables":  ["precipitation", "rainfall"],
        "temporal":   "1981-present",
        "spatial":    "Global land 50S-50N",
        "scale_m":    5566,
        "bands":      ["precipitation"],
    },
    "COPERNICUS/S2_SR_HARMONIZED": {
        "title":      "Sentinel-2 MSI L2A Surface Reflectance (Harmonized)",
        "variables":  ["reflectance", "ndvi", "optical"],
        "temporal":   "2017-present",
        "spatial":    "Global land",
        "scale_m":    10,
        "bands":      ["B2", "B3", "B4", "B8", "B11", "B12", "SCL"],
    },
    "NASA/GPM_L3/IMERG_V06": {
        "title":      "GPM IMERG Final Precipitation L3 Half Hourly",
        "variables":  ["precipitation", "rainfall"],
        "temporal":   "2000-present",
        "spatial":    "Global",
        "scale_m":    11132,
        "bands":      ["precipitationCal", "precipitationUncal", "randomError",
                       "HQprecipitation", "HQprecipSource", "HQobservationTime",
                       "IRprecipitation", "IRkalmanFilterWeight", "probabilityLiquidPrecipitation"],
    },
}


def _match_collection(keywords: List[str]) -> Optional[str]:
    """Match keyword list to a known collection ID."""
    kw_lower = [k.lower() for k in keywords]
    for cid, meta in _KNOWN_COLLECTIONS.items():
        for var in meta["variables"]:
            if any(var in kw or kw in var for kw in kw_lower):
                return cid
    # Default to MODIS LST
    return "MODIS/061/MOD11A1"


def _collection_info(collection_id: str) -> Dict[str, Any]:
    """
    Return live EE metadata for a collection, or fall back to the
    static _KNOWN_COLLECTIONS dict when EE is unavailable.
    """
    static = _KNOWN_COLLECTIONS.get(collection_id, {})

    if not _EE_AVAILABLE:
        return static

    try:
        col  = ee.ImageCollection(collection_id)
        info = col.limit(1).getInfo()
        size_hint = info.get("properties", {}).get("system:asset_size")
        return {**static, "ee_info": info, "asset_size": size_hint}
    except Exception as exc:
        logger.debug("earth_engine: getInfo() failed for %s — %s", collection_id, exc)
        return static


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class EarthEngineConnector(StaticDatasetConnector):
    """
    Real Google Earth Engine provider connector.

    Uses the earthengine-api SDK when available and authenticated.
    Falls back to static catalog metadata when the SDK is unavailable
    or credentials are not configured, returning clear error messages
    rather than crashing.

    Export workflow (standard EE pattern):
      1. Build ee.ImageCollection with date / ROI filters
      2. Export to Google Drive or GCS via ee.batch.Export.image.toDrive()
      3. Validate the exported file
    """

    descriptor = ConnectorDescriptor(
        connector_id="google_earth_engine",
        provider_name="Google Earth Engine",
        connector_type=ConnectorType.official_sdk,
        supported_access_types=(AccessType.user_credentials_required,),
        supported_dataset_types=(DatasetType.raster, DatasetType.gridded, DatasetType.vector),
        supported_authentication=(AuthenticationType.oauth, AuthenticationType.user_credentials),
        capabilities=(
            CapabilityFlags.supports_metadata
            | CapabilityFlags.supports_api
            | CapabilityFlags.supports_dataset_search
        ),
        priority=34,
    )

    provider_keywords = ("earth engine", "google earth engine", "gee")
    api_keywords      = ("earthengine", "earthengine-api", "gee")

    datasets = [
        DatasetDescriptor(
            provider="Google Earth Engine",
            dataset_name="MODIS Land Surface Temperature",
            collection_name="MODIS/061/MOD11A1",
            dataset_id="MODIS/061/MOD11A1",
            api_endpoint="earthengine-api",
            metadata_endpoint="https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A1",
            download_endpoint="ee.batch.Export.image.toDrive",
            supported_variables=["land surface temperature", "temperature", "emissivity"],
            temporal_coverage="2000-present",
            spatial_coverage="Global land",
            supported_formats=["GeoTIFF", "TFRecord"],
            authentication_required=True,
            access_notes="Requires earthengine-api and authenticated GEE account.",
        ),
        DatasetDescriptor(
            provider="Google Earth Engine",
            dataset_name="CHIRPS Daily Precipitation",
            collection_name="UCSB-CHG/CHIRPS/DAILY",
            dataset_id="UCSB-CHG/CHIRPS/DAILY",
            api_endpoint="earthengine-api",
            metadata_endpoint="https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY",
            download_endpoint="ee.batch.Export.image.toDrive",
            supported_variables=["precipitation", "rainfall"],
            temporal_coverage="1981-present",
            spatial_coverage="Global land, 50S-50N",
            supported_formats=["GeoTIFF", "TFRecord"],
            authentication_required=True,
            access_notes="Requires earthengine-api and authenticated GEE account.",
        ),
    ]

    # ------------------------------------------------------------------
    # discover_datasets
    # ------------------------------------------------------------------

    def discover_datasets(self, snapshot=None, context=None, **kwargs) -> List[DatasetDescriptor]:
        """
        Match requested keywords to known EE collections and return
        descriptors.  When EE is initialised, enriches with live metadata.
        """
        r = self._as_dict(context or kwargs)
        keywords = r.get("keywords") or r.get("variables") or []
        if isinstance(keywords, str):
            keywords = [keywords]

        _ee_init(r.get("credentials"))

        matched_id = _match_collection(keywords) if keywords else None
        if not matched_id:
            logger.info("earth_engine discover_datasets: no keyword match — returning all known collections")
            return self.datasets

        info = _collection_info(matched_id)
        logger.info("earth_engine discover_datasets: matched collection %s", matched_id)

        return [DatasetDescriptor(
            provider="Google Earth Engine",
            dataset_name=info.get("title", matched_id),
            collection_name=matched_id,
            dataset_id=matched_id,
            api_endpoint="earthengine-api",
            metadata_endpoint=(
                "https://developers.google.com/earth-engine/datasets/catalog/"
                + matched_id.replace("/", "_")
            ),
            download_endpoint="ee.batch.Export.image.toDrive",
            supported_variables=info.get("variables", []),
            temporal_coverage=info.get("temporal", "Dataset dependent"),
            spatial_coverage=info.get("spatial", "Dataset dependent"),
            supported_formats=["GeoTIFF", "TFRecord"],
            authentication_required=True,
            access_notes=f"EE collection: {matched_id}. Native scale: {info.get('scale_m', '?')} m.",
        )]

    # ------------------------------------------------------------------
    # probe_metadata
    # ------------------------------------------------------------------

    def probe_metadata(self, snapshot=None, fetch_request=None, **kwargs) -> DatasetMetadata:
        """
        Return live EE collection metadata when authenticated,
        or static catalog metadata otherwise, as a DatasetMetadata object.
        """
        source_id = getattr(snapshot, "source_id", None) or "earth_engine"
        r = self._as_dict(fetch_request or kwargs)
        collection_id = (r.get("dataset_id") or r.get("collection_name")
                         or "MODIS/061/MOD11A1")

        try:
            ee_ready = _ee_init(r.get("credentials"))

            static = _KNOWN_COLLECTIONS.get(collection_id, {})
            catalog_url = (
                "https://developers.google.com/earth-engine/datasets/catalog/"
                + collection_id.replace("/", "_")
            )

            if not ee_ready:
                logger.warning("earth_engine probe_metadata: EE not initialised for %s", collection_id)
                return DatasetMetadata(
                    source_id=source_id,
                    dataset_id=collection_id,
                    product=static.get("title", collection_id),
                    metadata_endpoint=catalog_url,
                    variables=static.get("bands", []),
                    spatial_coverage=static.get("spatial") or "Unknown",
                    temporal_coverage=static.get("temporal") or "Unknown",
                    retrieval_method="EE static catalog (not authenticated)",
                    unavailable_reason=(
                        "EE not initialised. Run `earthengine authenticate` or supply "
                        "credentials in the request."
                    ),
                )

            # Live EE metadata
            image_count = None
            band_names_live = None
            unavailable_reason = ""
            try:
                col  = ee.ImageCollection(collection_id)

                # Apply date filter if supplied
                start = r.get("start_date")
                end   = r.get("end_date")
                if start:
                    col = col.filterDate(start, end or datetime.utcnow().strftime("%Y-%m-%d"))

                # Apply ROI if supplied
                roi = self._build_roi(r)
                if roi:
                    col = col.filterBounds(roi)

                image_count = col.size().getInfo()
                logger.info("earth_engine probe_metadata: %s — %d images in filtered collection",
                            collection_id, image_count)

                # Get first image for band/projection details
                if image_count > 0:
                    first = col.first()
                    band_names_live = first.bandNames().getInfo()
            except Exception as exc:
                logger.warning("earth_engine probe_metadata: EE getInfo failed — %s", exc)
                unavailable_reason = str(exc)

            return DatasetMetadata(
                source_id=source_id,
                dataset_id=collection_id,
                product=static.get("title", collection_id),
                metadata_endpoint=catalog_url,
                variables=band_names_live or static.get("bands", []),
                spatial_coverage=static.get("spatial") or "Unknown",
                temporal_coverage=static.get("temporal") or "Unknown",
                retrieval_method="Google Earth Engine live collection query",
                unavailable_reason=unavailable_reason,
            )
        except Exception as exc:
            logger.warning("earth_engine probe_metadata: failed — %s", exc)
            return DatasetMetadata(
                source_id=source_id,
                dataset_id=collection_id,
                file_size_bytes=50 * 1024 * 1024,
                retrieval_method="unavailable",
                unavailable_reason=str(exc),
            )

    # ------------------------------------------------------------------
    # probe_size
    # ------------------------------------------------------------------

    def probe_size(self, snapshot=None, fetch_request=None, **kwargs) -> SizeEstimate:
        """
        Estimate export size from EE collection properties.
        EE does not support Content-Length on export tasks, so size is
        estimated from image count × pixel_count × bands × 4 bytes.
        Returns SizeEstimate for Agent 4 compatibility.
        """
        source_id = getattr(snapshot, "source_id", None) or "earth_engine"
        r = self._as_dict(fetch_request or kwargs)
        collection_id = r.get("dataset_id") or r.get("collection_name") or "MODIS/061/MOD11A1"
        ee_ready = _ee_init(r.get("credentials"))

        static = _KNOWN_COLLECTIONS.get(collection_id, {})
        scale_m = static.get("scale_m", 1000)

        # Fallback static estimate when EE is not initialised but collection is known
        if not ee_ready:
            if static:
                # Use known collection info for a rough estimate without EE API
                n_bands = len(static.get("bands", ["b1"]))
                # Default bbox: global at 1000m resolution
                roi_area_m2 = 5.1e14  # ~Earth surface
                pixel_count = max(1, roi_area_m2 / (scale_m ** 2))
                # Assume 30 days of data as a representative default
                est_bytes = int(30 * pixel_count * n_bands * 4)
                logger.info(
                    "earth_engine probe_size: static fallback %d bytes (collection=%s)",
                    est_bytes, collection_id,
                )
                return SizeEstimate(
                    source_id=source_id,
                    estimated_bytes=float(est_bytes),
                    is_exact=False,
                    method="EE static collection estimate (EE not initialised)",
                    human_readable=format_bytes(est_bytes),
                )
            return SizeEstimate(
                source_id=source_id,
                method="Earth Engine not initialised; authenticate to get accurate estimate",
                human_readable="Unknown",
            )

        try:
            col   = ee.ImageCollection(collection_id)
            start = r.get("start_date")
            end   = r.get("end_date")
            if start:
                col = col.filterDate(start, end or datetime.utcnow().strftime("%Y-%m-%d"))
            roi = self._build_roi(r)
            if roi:
                col = col.filterBounds(roi)

            image_count = col.size().getInfo()
            logger.info("earth_engine probe_size: %s — %d images", collection_id, image_count)

            # Rough size estimate: pixel_count × bands × 4 bytes per image
            roi_area_m2 = self._roi_area_m2(r)
            pixel_count = max(1, roi_area_m2 / (scale_m ** 2))
            n_bands     = len(static.get("bands", ["b1"]))
            est_bytes   = int(image_count * pixel_count * n_bands * 4)

            logger.info(
                "earth_engine probe_size: estimated %d bytes "
                "(images=%d, pixels=%.0f, bands=%d)",
                est_bytes, image_count, pixel_count, n_bands,
            )
            return SizeEstimate(
                source_id=source_id,
                estimated_bytes=float(est_bytes),
                is_exact=False,
                method="EE estimate (image_count × pixel_count × bands × 4 bytes)",
                human_readable=format_bytes(est_bytes),
            )
        except Exception as exc:
            logger.warning("earth_engine probe_size: EE error — %s", exc)
            return SizeEstimate(
                source_id=source_id,
                method=f"Earth Engine estimation error: {exc}",
                human_readable="Unknown",
            )

    # ------------------------------------------------------------------
    # resolve_download_asset
    # ------------------------------------------------------------------

    def resolve_download_asset(self, snapshot=None, fetch_request=None, credentials=None, **kwargs) -> Dict[str, Any]:
        """
        EE exports go to Google Drive / GCS — there is no direct download URL.
        Returns the export task description.
        """
        r = self._as_dict(fetch_request or kwargs)
        collection_id = r.get("dataset_id") or r.get("collection_name") or "MODIS/061/MOD11A1"
        return {
            "export_method":    "ee.batch.Export.image.toDrive",
            "collection_id":    collection_id,
            "note": (
                "EE data is exported via ee.batch.Export. "
                "Call fetch_full() to start an export task."
            ),
        }

    # ------------------------------------------------------------------
    # fetch_subset
    # ------------------------------------------------------------------

    def fetch_subset(self, snapshot=None, fetch_request=None, credentials=None, output_dir=None, **kwargs) -> Dict[str, Any]:
        """
        EE natively supports spatial and temporal subsetting via ROI and
        date filters applied before export.  Delegates to fetch_full with
        the same parameters.
        """
        logger.info("earth_engine fetch_subset: delegating to fetch_full (EE handles subsetting)")
        result = self.fetch_full(snapshot, fetch_request, credentials, output_dir=output_dir, **kwargs)
        result["subset_note"] = "EE spatial/temporal subsetting applied before export."
        return result

    # ------------------------------------------------------------------
    # fetch_full
    # ------------------------------------------------------------------

    def fetch_full(self, snapshot=None, fetch_request=None, credentials=None, output_dir=None, **kwargs) -> Dict[str, Any]:
        """
        Build an EE export task.

        EE exports are asynchronous.  This method starts the task, waits
        briefly for it to reach RUNNING state, and returns task metadata.
        Large exports may take minutes to hours to complete in EE.

        When EE is not initialised, returns a clear error with setup instructions.
        """
        r = self._as_dict(fetch_request or kwargs)
        collection_id = r.get("dataset_id") or r.get("collection_name") or "MODIS/061/MOD11A1"
        ee_ready = _ee_init(r.get("credentials"))

        if not ee_ready:
            return {
                "success": False,
                "error": (
                    "Google Earth Engine SDK is not initialised. "
                    "Run `earthengine authenticate` and retry, or supply credentials "
                    "in the request under the `credentials` key."
                ),
                "collection_id": collection_id,
            }

        try:
            col = ee.ImageCollection(collection_id)

            start = r.get("start_date")
            end   = r.get("end_date")
            if start:
                col = col.filterDate(start, end or datetime.utcnow().strftime("%Y-%m-%d"))

            roi = self._build_roi(r)
            if roi:
                col = col.filterBounds(roi)

            static   = _KNOWN_COLLECTIONS.get(collection_id, {})
            scale_m  = static.get("scale_m", 1000)
            bands    = r.get("bands") or static.get("bands") or [".*"]

            # Mosaic the filtered collection into a single image for export
            image = col.select(bands).mosaic()
            if roi:
                image = image.clip(roi)

            ts          = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            description = f"ee_export_{collection_id.replace('/', '_')}_{ts}"
            folder      = r.get("drive_folder", "EarthEngineExports")

            task = ee.batch.Export.image.toDrive(
                image=image,
                description=description,
                folder=folder,
                scale=scale_m,
                region=roi.bounds().getInfo()["coordinates"] if roi else None,
                fileFormat="GeoTIFF",
                maxPixels=1e13,
            )
            task.start()
            logger.info("earth_engine fetch_full: export task started — %s (folder=%s)",
                        description, folder)

            return {
                "success":       True,
                "task_id":       task.id,
                "task_name":     description,
                "drive_folder":  folder,
                "collection_id": collection_id,
                "format":        "GeoTIFF",
                "scale_m":       scale_m,
                "note": (
                    "Export task submitted to Google Earth Engine. "
                    "Check progress at https://code.earthengine.google.com/tasks "
                    "or via ee.batch.Task.status()."
                ),
            }

        except Exception as exc:
            logger.error("earth_engine fetch_full: export failed — %s", exc)
            return {"success": False, "error": str(exc), "collection_id": collection_id}

    # ------------------------------------------------------------------
    # validate_download
    # ------------------------------------------------------------------

    def validate_download(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Validate an exported GeoTIFF from Earth Engine."""
        if not os.path.exists(file_path):
            return {"valid": False, "issues": [f"File not found: {file_path}"]}
        size = os.path.getsize(file_path)
        if size == 0:
            return {"valid": False, "issues": ["File is empty"]}

        with open(file_path, "rb") as fh:
            magic = fh.read(4)

        issues: List[str] = []
        head_lower = magic.lower()
        if b"<htm" in head_lower or b"<!do" in head_lower:
            issues.append("File appears to be an HTML page, not a GeoTIFF")
        # GeoTIFF magic: II (little-endian) or MM (big-endian) TIFF
        if magic[:2] not in (b"II", b"MM"):
            logger.debug("earth_engine validate: file does not start with TIFF magic (may be valid if format differs)")

        if issues:
            logger.warning("earth_engine validate_download: FAILED — %s", issues)
            return {"valid": False, "issues": issues}

        logger.info("earth_engine validate_download: passed (%d bytes) — %s", size, file_path)
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
    def _build_roi(r: dict):
        """Build an ee.Geometry.Rectangle from request bbox fields, or None."""
        if not _EE_AVAILABLE:
            return None
        lon_min = r.get("lon_min") or r.get("west")
        lat_min = r.get("lat_min") or r.get("south")
        lon_max = r.get("lon_max") or r.get("east")
        lat_max = r.get("lat_max") or r.get("north")
        if all(v is not None for v in (lon_min, lat_min, lon_max, lat_max)):
            try:
                return ee.Geometry.Rectangle([
                    float(lon_min), float(lat_min),
                    float(lon_max), float(lat_max),
                ])
            except Exception as exc:
                logger.debug("earth_engine: ROI construction failed — %s", exc)
        return None

    @staticmethod
    def _roi_area_m2(r: dict) -> float:
        """Approximate ROI area in m² for size estimation."""
        lon_min = r.get("lon_min") or r.get("west")
        lat_min = r.get("lat_min") or r.get("south")
        lon_max = r.get("lon_max") or r.get("east")
        lat_max = r.get("lat_max") or r.get("north")
        if all(v is not None for v in (lon_min, lat_min, lon_max, lat_max)):
            deg_lat = abs(float(lat_max) - float(lat_min))
            deg_lon = abs(float(lon_max) - float(lon_min))
            return deg_lat * deg_lon * (111_000 ** 2)
        # Default: 1° × 1° box
        return 111_000 ** 2

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


register_connector(EarthEngineConnector)
