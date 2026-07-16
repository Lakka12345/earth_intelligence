"""
Reusable protocol connectors for Agent 4 — production revision.

Changes from v1:
  • Pass ConnectorDiagnostics through every resolver call
  • Use structured ConnectorError types instead of bare RuntimeError
  • Use _session() pool (connection reuse, keep-alive)
  • estimate_size_full() instead of bare estimate_size_from_url()
  • Extended download safety checks via validate_downloaded_file()
  • Multi-candidate ranking passed through to asset resolver
  • All public interfaces remain backward compatible
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from agents.agent4_asset_resolver import (
    ConnectorDiagnostics,
    ConnectorError,
    AssetNotFoundError,
    DatasetNotFoundError,
    NetworkError,
    ValidationError,
    _session,
    _is_data_url,
    _is_rejected_url,
    estimate_size_from_url,
    estimate_size_full,
    resolve_ckan_asset,
    resolve_erddap_asset,
    resolve_generic_asset,
    resolve_opendap_asset,
    resolve_stac_asset,
    resolve_thredds_asset,
    validate_downloaded_file,
)
from agents.agent4_download_engine import DownloadEngine, DownloadTask
from connectors.base_connector import BaseConnector, ConnectorDescriptor, ConnectorMatch, Credentials, FetchRequest
from connectors.connector_registry import register_connector
from connectors.connector_types import (
    AccessType, AuthenticationType, CapabilityFlags, ConnectorType, DatasetType,
)
from models.agent4_schemas import DatasetDescriptor, DatasetMetadata, SizeEstimate, format_bytes
from models.website_analysis_schemas import SourceSnapshot

DATA_EXTENSIONS = (".nc", ".nc4", ".grib", ".grb", ".tif", ".tiff", ".csv",
                   ".json", ".geojson", ".parquet", ".zarr")
HTML_TYPES = ("text/html", "application/xhtml")


def _text(snapshot: SourceSnapshot) -> str:
    return " ".join(
        str(part or "").lower()
        for part in (snapshot.name, snapshot.url, snapshot.api_type, snapshot.dataset_type)
    )


def _has_any(snapshot: SourceSnapshot, terms: Iterable[str]) -> bool:
    blob = _text(snapshot)
    return any(term.lower() in blob for term in terms)


def _extension_format(url: str) -> str:
    path = urlparse(url).path.lower()
    for ext in DATA_EXTENSIONS:
        if path.endswith(ext):
            return ext.lstrip(".").upper()
    return "Unknown"


def _safe_get_json(session, url: str, params: Optional[dict] = None, timeout: int = 15):
    """GET a URL and return parsed JSON, or None on any failure. Never raises."""
    try:
        r = session.get(url, params=params, timeout=timeout,
                         headers={"Accept": "application/json"})
        if r.ok:
            try:
                return r.json()
            except Exception:
                return None
    except Exception:
        return None
    return None


def _safe_get_text(session, url: str, params: Optional[dict] = None, timeout: int = 15) -> Optional[str]:
    """GET a URL and return raw text, or None on any failure. Never raises."""
    try:
        r = session.get(url, params=params, timeout=timeout)
        if r.ok:
            return r.text
    except Exception:
        return None
    return None


def _first_non_empty(*values):
    for v in values:
        if v not in (None, "", [], {}):
            return v
    return None


@dataclass(frozen=True)
class ProtocolSpec:
    connector_id: str
    provider_name: str
    connector_type: ConnectorType
    keywords: tuple
    priority: int
    formats: tuple
    supports_subsetting: bool = False
    auth: tuple = (
        AuthenticationType.none,
        AuthenticationType.api_key,
        AuthenticationType.user_credentials,
    )


class ProtocolConnector(BaseConnector):
    spec: ProtocolSpec

    @property
    def descriptor(self) -> ConnectorDescriptor:  # type: ignore[override]
        caps = (
            CapabilityFlags.supports_metadata
            | CapabilityFlags.supports_api
            | CapabilityFlags.supports_dataset_search
            | CapabilityFlags.supports_download
            | CapabilityFlags.supports_resume
            | CapabilityFlags.supports_checksum
            | CapabilityFlags.supports_parallel_download
        )
        if self.spec.supports_subsetting:
            caps |= CapabilityFlags.supports_subsetting
        return ConnectorDescriptor(
            connector_id=self.spec.connector_id,
            provider_name=self.spec.provider_name,
            connector_type=self.spec.connector_type,
            supported_access_types=(
                AccessType.public, AccessType.user_credentials_required, AccessType.unknown
            ),
            supported_dataset_types=(
                DatasetType.gridded, DatasetType.raster, DatasetType.vector,
                DatasetType.tabular, DatasetType.time_series, DatasetType.unknown,
            ),
            supported_authentication=self.spec.auth,
            capabilities=caps,
            priority=self.spec.priority,
        )

    def can_handle(self, snapshot: SourceSnapshot) -> bool:
        return _has_any(snapshot, self.spec.keywords)

    def match_score(self, snapshot: SourceSnapshot, context=None) -> ConnectorMatch:
        if not self.can_handle(snapshot):
            return ConnectorMatch(0, "Protocol keywords did not match.")
        score = max(2, 800 - self.spec.priority)
        api_type = str(getattr(snapshot, "api_type", "") or "").lower()
        if any(term in api_type for term in self.spec.keywords):
            score += 75
        return ConnectorMatch(score, f"{self.spec.provider_name} protocol match.")

    def discover_datasets(self, snapshot: SourceSnapshot, context=None) -> List[DatasetDescriptor]:
        variables = list(
            getattr(snapshot, "variables_available", [])
            or (context or {}).get("variables")
            or []
        )
        fmt = _extension_format(snapshot.url)
        formats = list(self.spec.formats if fmt == "Unknown" else (fmt,))
        return [
            DatasetDescriptor(
                provider=snapshot.name,
                dataset_name=snapshot.name,
                collection_name=snapshot.dataset_type,
                dataset_id=snapshot.source_id,
                api_endpoint=snapshot.url,
                metadata_endpoint=snapshot.url,
                download_endpoint=snapshot.url,
                supported_variables=variables,
                supported_formats=formats,
                authentication_required=False,
                source_url=snapshot.url,
                access_notes=f"Resolved through {self.spec.provider_name} connector.",
            )
        ]

    def _make_diagnostics(self) -> ConnectorDiagnostics:
        return ConnectorDiagnostics(
            provider=self.spec.provider_name,
            protocol=self.spec.connector_type.value,
            subset_capable=self.spec.supports_subsetting,
        )

    def _resolve_asset_url(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
        diagnostics: Optional[ConnectorDiagnostics] = None,
    ) -> Optional[str]:
        """Return the actual downloadable URL, or None if resolution fails."""
        return resolve_generic_asset(snapshot.url, credentials, diagnostics=diagnostics)

    def probe_metadata(self, snapshot: SourceSnapshot, fetch_request: FetchRequest) -> DatasetMetadata:
        diag = self._make_diagnostics()
        resolved = None
        try:
            resolved = self._resolve_asset_url(snapshot, fetch_request, diagnostics=diag)
        except ConnectorError as exc:
            diag.log_error(exc)

        probe_url = resolved or snapshot.url
        size_result = estimate_size_full(probe_url)

        content_encoding = None
        try:
            s = _session()
            s.verify = False  # SSL verification disabled
            resp = s.head(probe_url, allow_redirects=True, timeout=15)
            content_type = resp.headers.get("Content-Type")
            content_encoding = resp.headers.get("Content-Encoding")
            checksum = (
                resp.headers.get("ETag")
                or resp.headers.get("Content-MD5")
                or resp.headers.get("x-amz-checksum-sha256")
            )
            ok = resp.ok
        except Exception:
            content_type = None
            checksum = None
            ok = False

        unavail = ""
        if not resolved:
            unavail = "; ".join(str(e) for e in diag.errors) or f"Asset resolution returned no URL from {snapshot.url}"

        # Base protocol connector only has generic HTTP signals available.
        # Fields it genuinely cannot determine (description, license, bbox,
        # dates, etc.) are left as None rather than a placeholder string;
        # protocol-aware subclasses (STAC/ERDDAP/CKAN/THREDDS/OGC) override
        # this method to pull those from the provider's real metadata API.
        return DatasetMetadata(
            source_id=snapshot.source_id,
            dataset_id=snapshot.source_id,
            collection=snapshot.dataset_type,
            product=snapshot.name,
            download_endpoint=probe_url,
            api_endpoint=snapshot.url,
            metadata_endpoint=snapshot.url,
            file_size_bytes=size_result.bytes,
            variables=list(getattr(snapshot, "variables_available", []) or fetch_request.variables or []),
            spatial_coverage="Unknown",
            temporal_coverage="Unknown",
            file_format=_extension_format(probe_url) if not content_type else content_type,
            checksum=checksum,
            content_type=content_type,
            retrieval_method=f"{self.spec.provider_name} asset resolution (confidence {size_result.confidence:.0%})",
            unavailable_reason=unavail,
            dataset_name=snapshot.name or None,
            provider=self.spec.provider_name,
            compression=content_encoding,
            authentication_required=(AuthenticationType.none not in self.spec.auth),
        )

    def probe_size(self, snapshot: SourceSnapshot, fetch_request: FetchRequest) -> SizeEstimate:
        diag = self._make_diagnostics()
        resolved = None
        try:
            resolved = self._resolve_asset_url(snapshot, fetch_request, diagnostics=diag)
        except ConnectorError as exc:
            diag.log_error(exc)

        probe_url = resolved or snapshot.url
        size_result = estimate_size_full(probe_url)

        if size_result:
            return SizeEstimate(
                source_id=snapshot.source_id,
                estimated_bytes=size_result.bytes,
                is_exact=size_result.confidence >= 0.9,
                method=f"{self.spec.provider_name}: {size_result.method} (confidence {size_result.confidence:.0%})",
                human_readable=size_result.human_readable,
            )
        return SizeEstimate(
            source_id=snapshot.source_id,
            method=f"{self.spec.provider_name} size unavailable",
            human_readable="Unknown",
        )

    def fetch_full(
        self,
        snapshot: SourceSnapshot,
        fetch_request: FetchRequest,
        credentials: Optional[Credentials] = None,
    ) -> str:
        diag = self._make_diagnostics()
        resolved = None
        try:
            resolved = self._resolve_asset_url(snapshot, fetch_request, credentials, diag)
        except ConnectorError as exc:
            diag.log_error(exc)
            raise RuntimeError(
                f"{self.spec.provider_name} asset resolution failed: {exc}\n{diag.summary()}"
            ) from exc

        if resolved is None:
            raise RuntimeError(
                f"{self.spec.provider_name} could not locate a downloadable dataset URL.\n"
                f"Source URL was: {snapshot.url}\n"
                f"Diagnostics:\n{diag.summary()}"
            )

        if _is_rejected_url(resolved):
            raise RuntimeError(f"Resolved URL has a rejected file extension: {resolved}")

        task = DownloadTask(
            url=resolved,
            dest_path=fetch_request.dest_path,
            expected_size=fetch_request.metadata.get("expected_size"),
            expected_content_type=fetch_request.metadata.get("expected_content_type"),
            expected_format=fetch_request.metadata.get("expected_format"),
            checksum=fetch_request.metadata.get("checksum"),
            source_id=snapshot.source_id,
            provider=snapshot.name,
            connector_id=self.name,
            protocol=self.descriptor.connector_type.value,
        )
        result = DownloadEngine().download_one(task, credentials)
        fetch_request.metadata["download_result"] = result
        fetch_request.metadata["diagnostics"] = diag

        if result.success and os.path.exists(fetch_request.dest_path):
            valid, reason = validate_downloaded_file(fetch_request.dest_path)
            if not valid:
                diag.validation_result = f"failed: {reason}"
                raise RuntimeError(f"Download validation failed: {reason}")
            diag.validation_result = "passed"

        if not result.success:
            diag.validation_result = "failed"
            raise RuntimeError(
                f"{self.spec.provider_name} download failed: {result.error}\n{diag.summary()}"
            )

        return result.dest_path


# ── Concrete protocol connectors ──────────────────────────────────────────────

class STACConnector(ProtocolConnector):
    spec = ProtocolSpec(
        "stac", "STAC", ConnectorType.stac,
        ("stac", "/collections", "/search", "pystac"),
        100, ("STAC", "COG", "GeoTIFF", "JSON"),
    )

    def _resolve_asset_url(self, snapshot, fetch_request, credentials=None, diagnostics=None):
        diag = diagnostics or self._make_diagnostics()
        try:
            return resolve_stac_asset(
                snapshot.url,
                fetch_request.variables,
                bbox=fetch_request.bounding_box,
                time_range=fetch_request.time_range,
                credentials=credentials,
                diagnostics=diag,
            )
        except (DatasetNotFoundError, AssetNotFoundError) as exc:
            diag.log_error(exc)
            return None

    def probe_metadata(self, snapshot, fetch_request, credentials=None):
        diag = self._make_diagnostics()
        resolved = None
        try:
            resolved = self._resolve_asset_url(snapshot, fetch_request, credentials, diag)
        except ConnectorError as exc:
            diag.log_error(exc)

        base = snapshot.url.rstrip("/")
        s = _session(credentials)
        s.verify = False  # SSL verification disabled

        item = None
        try:
            r = s.post(f"{base}/search", json={
                "limit": 1,
                "bbox": list(fetch_request.bounding_box) if fetch_request.bounding_box else None,
                "datetime": (
                    f"{fetch_request.time_range[0]}/{fetch_request.time_range[1] if len(fetch_request.time_range) > 1 else '..'}"
                    if fetch_request.time_range else None
                ),
            }, timeout=20)
            if r.ok:
                feats = r.json().get("features", [])
                if feats:
                    item = feats[0]
        except Exception:
            pass

        props = (item or {}).get("properties", {}) if item else {}
        collection_id = (item or {}).get("collection") if item else None
        col_meta = _safe_get_json(s, f"{base}/collections/{collection_id}") if collection_id else None

        extent = (col_meta or {}).get("extent", {})
        spatial_bbox = extent.get("spatial", {}).get("bbox", [None])[0]
        temporal_interval = extent.get("temporal", {}).get("interval", [None])[0]

        bands = props.get("eo:bands") or (col_meta or {}).get("summaries", {}).get("eo:bands") or []
        band_names = [b.get("common_name") or b.get("name") for b in bands if isinstance(b, dict)]
        units = {b.get("name"): b.get("unit") for b in bands if isinstance(b, dict) and b.get("unit")}

        keywords = (col_meta or {}).get("keywords")
        providers = (col_meta or {}).get("providers", [])
        provider_name = next((p.get("name") for p in providers if "producer" in (p.get("roles") or []) or "host" in (p.get("roles") or [])), None) \
            or (providers[0].get("name") if providers else None) or self.spec.provider_name

        license_val = (col_meta or {}).get("license")
        citation = props.get("sci:citation") or (col_meta or {}).get("sci:citation")
        doi = props.get("sci:doi") or (col_meta or {}).get("sci:doi")
        if doi and not citation:
            citation = f"https://doi.org/{doi}"

        size_result = estimate_size_full(resolved) if resolved else None

        return DatasetMetadata(
            source_id=snapshot.source_id,
            dataset_id=collection_id or snapshot.source_id,
            collection=collection_id or snapshot.dataset_type,
            product=(col_meta or {}).get("title") or snapshot.name,
            download_endpoint=resolved,
            api_endpoint=base,
            metadata_endpoint=f"{base}/collections/{collection_id}" if collection_id else base,
            file_size_bytes=size_result.bytes if size_result else None,
            variables=band_names or list(fetch_request.variables or []),
            spatial_coverage=str(spatial_bbox) if spatial_bbox else "Unknown",
            temporal_coverage=(
                f"{temporal_interval[0] or ''} – {temporal_interval[1] or 'present'}"
                if temporal_interval else "Unknown"
            ),
            file_format=_extension_format(resolved) if resolved else "STAC asset",
            retrieval_method="STAC /search item + /collections metadata",
            unavailable_reason="" if resolved else f"STAC search returned no matching item for {snapshot.url}",
            dataset_name=(col_meta or {}).get("title") or None,
            provider=provider_name,
            description=(col_meta or {}).get("description"),
            variable_units=units or None,
            spatial_resolution=(
                f"{props.get('gsd')} m" if props.get("gsd") else
                (f"{(col_meta or {}).get('summaries', {}).get('gsd', [None])[0]} m"
                 if (col_meta or {}).get("summaries", {}).get("gsd") else None)
            ),
            crs=(
                f"EPSG:{props.get('proj:epsg')}" if props.get("proj:epsg")
                else (f"EPSG:{(col_meta or {}).get('summaries', {}).get('proj:epsg', [None])[0]}"
                      if (col_meta or {}).get("summaries", {}).get("proj:epsg") else None)
            ),
            bounding_box=list(spatial_bbox) if spatial_bbox else None,
            start_date=str(temporal_interval[0]) if temporal_interval and temporal_interval[0] else None,
            end_date=str(temporal_interval[1]) if temporal_interval and len(temporal_interval) > 1 and temporal_interval[1] else None,
            license=license_val,
            citation=citation,
            keywords=keywords,
            version=(col_meta or {}).get("stac_version"),
            authentication_required=None,
        )

    def probe_size(self, snapshot, fetch_request):
        diag = self._make_diagnostics()
        # Try STAC item file:size first via parallel search
        from agents.agent4_asset_resolver import _session as _get_session, estimate_size_from_stac_item
        s = _get_session()
        s.verify = False  # SSL verification disabled
        try:
            base = snapshot.url.rstrip("/")
            r = s.post(f"{base}/search", json={"limit": 1}, timeout=15)
            if r.ok:
                features = r.json().get("features", [])
                if features:
                    sz = estimate_size_from_stac_item(features[0])
                    if sz:
                        return SizeEstimate(
                            source_id=snapshot.source_id,
                            estimated_bytes=sz,
                            is_exact=True,
                            method="STAC item file:size (confidence 90%)",
                            human_readable=format_bytes(sz),
                        )
        except Exception:
            pass
        return super().probe_size(snapshot, fetch_request)


class OGCAPIConnector(ProtocolConnector):
    spec = ProtocolSpec(
        "ogc_api", "OGC API", ConnectorType.ogc_api,
        ("ogc", "wms", "wfs", "wcs", "api/collections", "collections/"),
        150, ("JSON", "GeoJSON", "Coverage", "Map"), True,
    )

    def _resolve_asset_url(self, snapshot, fetch_request, credentials=None, diagnostics=None):
        diag = diagnostics or self._make_diagnostics()
        url = snapshot.url.rstrip("/")
        s = _session(credentials)
        s.verify = False  # SSL verification disabled
        try:
            r = s.get(f"{url}/collections", timeout=15)
            if r.ok and "json" in (r.headers.get("Content-Type") or ""):
                from agents.agent4_asset_resolver import score_dataset_match
                cols = r.json().get("collections", [])
                if cols:
                    scored = [(score_dataset_match(c, fetch_request.variables), c) for c in cols]
                    scored.sort(key=lambda x: x[0], reverse=True)
                    col_id = scored[0][1].get("id", "")
                    diag.dataset_selected = col_id
                    items_url = f"{url}/collections/{col_id}/items"
                    r2 = s.get(items_url, params={"f": "json", "limit": 1}, timeout=15)
                    if r2.ok:
                        features = r2.json().get("features", [])
                        if features:
                            for link in features[0].get("links", []):
                                if link.get("rel") in ("enclosure", "data", "download"):
                                    href = link.get("href")
                                    if href:
                                        diag.download_endpoint = href
                                        return href
                            diag.download_endpoint = items_url
                            return items_url
        except Exception as exc:
            diag.log_error(NetworkError(f"OGC API collection browse failed: {exc}", url=url))
        return resolve_generic_asset(snapshot.url, credentials, diagnostics=diag)

    def probe_metadata(self, snapshot, fetch_request, credentials=None):
        diag = self._make_diagnostics()
        resolved = None
        try:
            resolved = self._resolve_asset_url(snapshot, fetch_request, credentials, diag)
        except ConnectorError as exc:
            diag.log_error(exc)

        base = snapshot.url.rstrip("/")
        s = _session(credentials)
        s.verify = False  # SSL verification disabled
        col_id = diag.dataset_selected
        col_meta = _safe_get_json(s, f"{base}/collections/{col_id}") if col_id else None

        extent = (col_meta or {}).get("extent", {})
        spatial_bbox = extent.get("spatial", {}).get("bbox", [None])[0]
        temporal_interval = extent.get("temporal", {}).get("interval", [None])[0]
        crs_list = (col_meta or {}).get("crs") or extent.get("spatial", {}).get("crs")

        size_result = estimate_size_full(resolved) if resolved else None

        return DatasetMetadata(
            source_id=snapshot.source_id,
            dataset_id=col_id or snapshot.source_id,
            collection=col_id,
            product=(col_meta or {}).get("title") or snapshot.name,
            download_endpoint=resolved,
            api_endpoint=base,
            metadata_endpoint=f"{base}/collections/{col_id}" if col_id else base,
            file_size_bytes=size_result.bytes if size_result else None,
            variables=list(fetch_request.variables or []),
            spatial_coverage=str(spatial_bbox) if spatial_bbox else "Unknown",
            temporal_coverage=(
                f"{temporal_interval[0] or ''} – {temporal_interval[1] or 'present'}"
                if temporal_interval else "Unknown"
            ),
            file_format=_extension_format(resolved) if resolved else "GeoJSON/Coverage (OGC API)",
            retrieval_method="OGC API Features/Coverages /collections metadata",
            unavailable_reason="" if resolved else f"OGC API browse failed for {snapshot.url}",
            dataset_name=(col_meta or {}).get("title"),
            provider=self.spec.provider_name,
            description=(col_meta or {}).get("description"),
            bounding_box=list(spatial_bbox) if spatial_bbox else None,
            crs=(crs_list[0] if isinstance(crs_list, list) and crs_list else (crs_list if isinstance(crs_list, str) else None)),
            start_date=str(temporal_interval[0]) if temporal_interval and temporal_interval[0] else None,
            end_date=str(temporal_interval[1]) if temporal_interval and len(temporal_interval) > 1 and temporal_interval[1] else None,
            license=(col_meta or {}).get("license"),
            keywords=(col_meta or {}).get("keywords"),
        )


class THREDDSConnector(ProtocolConnector):
    spec = ProtocolSpec(
        "thredds", "THREDDS", ConnectorType.thredds,
        ("thredds", "catalog.xml", "dodsC", "fileServer"),
        200, ("NetCDF", "OPeNDAP", "NcML"),
    )

    def _resolve_asset_url(self, snapshot, fetch_request, credentials=None, diagnostics=None):
        diag = diagnostics or self._make_diagnostics()
        url = resolve_thredds_asset(snapshot.url, fetch_request.variables, credentials, diag)
        if url:
            return url
        if "fileServer" in snapshot.url or any(snapshot.url.endswith(e) for e in DATA_EXTENSIONS):
            return snapshot.url
        return None

    @staticmethod
    def _catalog_xml_url(url: str) -> str:
        if url.endswith("catalog.xml"):
            return url
        if "/catalog.html" in url:
            return url.replace("/catalog.html", "/catalog.xml")
        base = re.sub(r"/(fileServer|dodsC)/", "/catalog/", url)
        return base.rsplit("/", 1)[0] + "/catalog.xml"

    def _parse_thredds_catalog(self, url: str, credentials=None) -> dict:
        """Parse a THREDDS catalog.xml for dataset name/documentation/coverage. Never raises."""
        s = _session(credentials)
        s.verify = False  # SSL verification disabled
        xml_text = _safe_get_text(s, self._catalog_xml_url(url))
        if not xml_text:
            return {}
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_text)
            ns = {"t": "http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"}

            def _find(tag, node=root):
                return node.find(f".//t:{tag}", ns)

            ds = _find("dataset")
            name = ds.get("name") if ds is not None else None
            doc = _find("documentation")
            description = doc.text.strip() if doc is not None and doc.text else None

            geo = _find("geospatialCoverage")
            bbox = None
            if geo is not None:
                n_el = geo.find("t:northsouth", ns)
                e_el = geo.find("t:eastwest", ns)
                if n_el is not None and e_el is not None:
                    try:
                        n_start = float(n_el.find("t:start", ns).text)
                        n_size = float(n_el.find("t:size", ns).text)
                        e_start = float(e_el.find("t:start", ns).text)
                        e_size = float(e_el.find("t:size", ns).text)
                        bbox = [e_start, n_start, e_start + e_size, n_start + n_size]
                    except Exception:
                        bbox = None

            time_cov = _find("timeCoverage")
            start_date = end_date = None
            if time_cov is not None:
                start_el = time_cov.find("t:start", ns)
                end_el = time_cov.find("t:end", ns)
                start_date = start_el.text.strip() if start_el is not None and start_el.text else None
                end_date = end_el.text.strip() if end_el is not None and end_el.text else None

            return {
                "name": name, "description": description, "bbox": bbox,
                "start_date": start_date, "end_date": end_date,
            }
        except Exception:
            return {}

    def probe_metadata(self, snapshot, fetch_request, credentials=None):
        diag = self._make_diagnostics()
        resolved = self._resolve_asset_url(snapshot, fetch_request, credentials, diag)
        cat = self._parse_thredds_catalog(snapshot.url, credentials)
        size_result = estimate_size_full(resolved) if resolved else None

        return DatasetMetadata(
            source_id=snapshot.source_id,
            dataset_id=snapshot.source_id,
            product=cat.get("name") or snapshot.name,
            download_endpoint=resolved,
            api_endpoint=snapshot.url,
            metadata_endpoint=self._catalog_xml_url(snapshot.url),
            file_size_bytes=size_result.bytes if size_result else None,
            variables=list(fetch_request.variables or []),
            spatial_coverage=str(cat["bbox"]) if cat.get("bbox") else "Unknown",
            temporal_coverage=(
                f"{cat.get('start_date', '')} – {cat.get('end_date', 'present')}"
                if cat.get("start_date") else "Unknown"
            ),
            file_format=_extension_format(resolved) if resolved else "NetCDF (THREDDS)",
            retrieval_method="THREDDS catalog.xml parsing",
            unavailable_reason="" if resolved else f"THREDDS asset resolution failed for {snapshot.url}",
            dataset_name=cat.get("name"),
            provider=self.spec.provider_name,
            description=cat.get("description"),
            bounding_box=cat.get("bbox"),
            crs="EPSG:4326" if cat.get("bbox") else None,
            start_date=cat.get("start_date"),
            end_date=cat.get("end_date"),
        )

    def fetch_subset(self, snapshot, fetch_request, credentials=None):
        diag = self._make_diagnostics()
        opendap_url = re.sub(r"/fileServer/", "/dodsC/", snapshot.url)
        opendap_url = re.sub(r"/catalog\.xml$", "", opendap_url)
        resolved = resolve_opendap_asset(
            opendap_url, fetch_request.variables,
            fetch_request.bounding_box, fetch_request.time_range, credentials, diag,
        )
        if resolved is None:
            raise NotImplementedError("THREDDS OPeNDAP subset resolution failed.")
        task = DownloadTask(
            url=resolved, dest_path=fetch_request.dest_path,
            source_id=snapshot.source_id, provider=snapshot.name,
            connector_id=self.name, protocol="opendap",
        )
        result = DownloadEngine().download_one(task, credentials)
        fetch_request.metadata["diagnostics"] = diag
        if not result.success:
            raise RuntimeError(f"THREDDS OPeNDAP subset failed: {result.error}\n{diag.summary()}")
        return result.dest_path


class ERDDAPConnector(ProtocolConnector):
    spec = ProtocolSpec(
        "erddap", "ERDDAP", ConnectorType.erddap,
        ("erddap", "tabledap", "griddap"),
        210, ("NetCDF", "CSV", "JSON"), True,
    )

    def _resolve_asset_url(self, snapshot, fetch_request, credentials=None, diagnostics=None):
        diag = diagnostics or self._make_diagnostics()
        try:
            return resolve_erddap_asset(
                snapshot.url, fetch_request.variables,
                fetch_request.bounding_box, fetch_request.time_range, credentials, diag,
            )
        except ConnectorError as exc:
            diag.log_error(exc)
            return None

    def fetch_subset(self, snapshot, fetch_request, credentials=None):
        diag = self._make_diagnostics()
        try:
            url = resolve_erddap_asset(
                snapshot.url, fetch_request.variables,
                fetch_request.bounding_box, fetch_request.time_range, credentials, diag,
            )
        except ConnectorError as exc:
            raise NotImplementedError(f"ERDDAP subset query failed: {exc}")

        if not url:
            raise NotImplementedError("ERDDAP subset query could not be constructed.")
        os.makedirs(os.path.dirname(fetch_request.dest_path) or ".", exist_ok=True)
        s = _session(credentials)
        s.verify = False  # SSL verification disabled
        try:
            with s.get(url, stream=True, timeout=60) as resp:
                ct = (resp.headers.get("Content-Type") or "").lower()
                if "text/html" in ct:
                    raise RuntimeError(
                        f"ERDDAP returned HTML — likely invalid variable names or dataset ID.\n"
                        f"{diag.summary()}"
                    )
                resp.raise_for_status()
                with open(fetch_request.dest_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"ERDDAP subset download failed: {exc}\n{diag.summary()}") from exc

        fetch_request.metadata["diagnostics"] = diag
        valid, reason = validate_downloaded_file(fetch_request.dest_path)
        if not valid:
            raise RuntimeError(f"ERDDAP download validation failed: {reason}")
        return fetch_request.dest_path

    @staticmethod
    def _erddap_base_and_id(url: str) -> Tuple[str, Optional[str]]:
        """Split an ERDDAP griddap/tabledap/info URL into (server_base, dataset_id)."""
        m = re.search(r"(https?://[^/]+/erddap)/(?:griddap|tabledap|info)/([^/.?]+)", url)
        if m:
            return m.group(1), m.group(2)
        m2 = re.match(r"(https?://[^/]+/erddap)", url)
        return (m2.group(1) if m2 else url.rstrip("/")), None

    def _erddap_info(self, url: str, credentials=None) -> dict:
        """
        Fetch ERDDAP's info/{id}/index.json for a dataset and reduce it to a
        dict of global attributes + a {variable: unit} map. Never raises.
        """
        base, dataset_id = self._erddap_base_and_id(url)
        if not dataset_id:
            return {}
        s = _session(credentials)
        s.verify = False  # SSL verification disabled
        data = _safe_get_json(s, f"{base}/info/{dataset_id}/index.json")
        if not data:
            return {}
        rows = data.get("table", {}).get("rows", [])
        cols = data.get("table", {}).get("columnNames", [])
        try:
            ri, vi, ai, vali = (cols.index("Row Type"), cols.index("Variable Name"),
                                cols.index("Attribute Name"), cols.index("Value"))
        except ValueError:
            return {}

        global_attrs: Dict[str, str] = {}
        var_units: Dict[str, str] = {}
        variables: List[str] = []
        for row in rows:
            row_type, var_name, attr_name, value = row[ri], row[vi], row[ai], row[vali]
            if row_type == "attribute" and var_name == "NC_GLOBAL":
                global_attrs[attr_name] = value
            elif row_type == "variable":
                if var_name and var_name not in variables:
                    variables.append(var_name)
            elif row_type == "attribute" and attr_name == "units" and var_name:
                var_units[var_name] = value
        return {
            "global_attrs": global_attrs,
            "var_units": var_units,
            "variables": variables,
            "dataset_id": dataset_id,
            "base": base,
        }

    def probe_metadata(self, snapshot, fetch_request, credentials=None):
        diag = self._make_diagnostics()
        try:
            resolved = resolve_erddap_asset(
                snapshot.url, fetch_request.variables,
                fetch_request.bounding_box, fetch_request.time_range, credentials, diag,
            )
        except ConnectorError as exc:
            diag.log_error(exc)
            resolved = None

        info = self._erddap_info(snapshot.url, credentials)
        ga = info.get("global_attrs", {})
        var_units = info.get("var_units", {})
        variables = info.get("variables") or list(fetch_request.variables or [])

        lat_min = ga.get("geospatial_lat_min")
        lat_max = ga.get("geospatial_lat_max")
        lon_min = ga.get("geospatial_lon_min")
        lon_max = ga.get("geospatial_lon_max")
        bbox = None
        if all(v is not None for v in (lat_min, lat_max, lon_min, lon_max)):
            try:
                bbox = [float(lon_min), float(lat_min), float(lon_max), float(lat_max)]
            except (TypeError, ValueError):
                bbox = None

        size_result = estimate_size_full(resolved) if resolved else None

        return DatasetMetadata(
            source_id=snapshot.source_id,
            dataset_id=info.get("dataset_id") or snapshot.source_id,
            collection=info.get("dataset_id"),
            product=ga.get("title") or snapshot.name,
            download_endpoint=resolved,
            api_endpoint=info.get("base") or snapshot.url,
            metadata_endpoint=(f"{info['base']}/info/{info['dataset_id']}/index.json" if info.get("dataset_id") else snapshot.url),
            file_size_bytes=size_result.bytes if size_result else None,
            variables=variables,
            spatial_coverage=str(bbox) if bbox else "Unknown",
            temporal_coverage=(
                f"{ga.get('time_coverage_start', '')} – {ga.get('time_coverage_end', 'present')}"
                if ga.get("time_coverage_start") else "Unknown"
            ),
            file_format=_extension_format(resolved) if resolved else "NetCDF/CSV (ERDDAP)",
            retrieval_method="ERDDAP info/{dataset_id}/index.json",
            unavailable_reason="" if resolved else f"ERDDAP asset resolution failed for {snapshot.url}",
            dataset_name=ga.get("title"),
            provider=ga.get("institution") or self.spec.provider_name,
            description=ga.get("summary"),
            variable_units=var_units or None,
            spatial_resolution=ga.get("geospatial_lat_resolution"),
            temporal_resolution=ga.get("time_coverage_resolution"),
            crs="EPSG:4326" if bbox else None,
            bounding_box=bbox,
            start_date=ga.get("time_coverage_start"),
            end_date=ga.get("time_coverage_end"),
            license=ga.get("license"),
            citation=ga.get("references") or ga.get("citation"),
            keywords=[k.strip() for k in ga.get("keywords", "").split(",") if k.strip()] or None,
            update_frequency=ga.get("update_frequency") if ga.get("update_frequency") not in (None, "") else None,
            version=ga.get("version") if ga.get("version") not in (None, "") else None,
        )

    def probe_size(self, snapshot, fetch_request):
        diag = self._make_diagnostics()
        try:
            url = resolve_erddap_asset(
                snapshot.url, fetch_request.variables,
                fetch_request.bounding_box, fetch_request.time_range, diagnostics=diag,
            )
        except ConnectorError:
            url = None
        if url:
            size_result = estimate_size_full(url)
            if size_result:
                return SizeEstimate(
                    source_id=snapshot.source_id,
                    estimated_bytes=size_result.bytes,
                    is_exact=size_result.confidence >= 0.9,
                    method=f"ERDDAP: {size_result.method}",
                    human_readable=size_result.human_readable,
                )
        return SizeEstimate(source_id=snapshot.source_id, method="ERDDAP size unavailable")


class OPeNDAPConnector(ProtocolConnector):
    spec = ProtocolSpec(
        "opendap", "OPeNDAP", ConnectorType.opendap,
        ("opendap", "dods", "dodsC", ".dds", ".das"),
        220, ("NetCDF", "DAP2", "DAP4"), True,
    )

    def _resolve_asset_url(self, snapshot, fetch_request, credentials=None, diagnostics=None):
        diag = diagnostics or self._make_diagnostics()
        return resolve_opendap_asset(
            snapshot.url, fetch_request.variables,
            fetch_request.bounding_box, fetch_request.time_range, credentials, diag,
        )

    def fetch_subset(self, snapshot, fetch_request, credentials=None):
        diag = self._make_diagnostics()
        url = resolve_opendap_asset(
            snapshot.url, fetch_request.variables,
            fetch_request.bounding_box, fetch_request.time_range, credentials, diag,
        )
        if not url:
            raise NotImplementedError("OPeNDAP constraint expression could not be built.")
        task = DownloadTask(
            url=url, dest_path=fetch_request.dest_path,
            source_id=snapshot.source_id, provider=snapshot.name,
            connector_id=self.name, protocol="opendap",
        )
        result = DownloadEngine().download_one(task, credentials)
        fetch_request.metadata["diagnostics"] = diag
        if not result.success:
            raise RuntimeError(f"OPeNDAP download failed: {result.error}\n{diag.summary()}")
        return result.dest_path


class CKANConnector(ProtocolConnector):
    spec = ProtocolSpec(
        "ckan", "CKAN", ConnectorType.ckan,
        ("ckan", "package_show", "datastore_search", "/api/3/action"),
        260, ("JSON", "CSV", "Resource"),
    )

    def _resolve_asset_url(self, snapshot, fetch_request, credentials=None, diagnostics=None):
        diag = diagnostics or self._make_diagnostics()
        try:
            return resolve_ckan_asset(snapshot.url, fetch_request.variables, credentials, diag)
        except ConnectorError as exc:
            diag.log_error(exc)
            return None

    @staticmethod
    def _ckan_base_and_id(url: str) -> Tuple[str, Optional[str]]:
        m = re.search(r"(https?://[^/]+)/(?:dataset|api/3/action/package_show)/([^/?&]+)", url)
        if m:
            return m.group(1), m.group(2)
        m2 = re.match(r"(https?://[^/]+)", url)
        return (m2.group(1) if m2 else url.rstrip("/")), None

    def probe_metadata(self, snapshot, fetch_request, credentials=None):
        diag = self._make_diagnostics()
        resolved = None
        try:
            resolved = self._resolve_asset_url(snapshot, fetch_request, credentials, diag)
        except ConnectorError as exc:
            diag.log_error(exc)

        base, dataset_id = self._ckan_base_and_id(snapshot.url)
        pkg = None
        if dataset_id:
            s = _session(credentials)
            s.verify = False  # SSL verification disabled
            result = _safe_get_json(s, f"{base}/api/3/action/package_show", params={"id": dataset_id})
            if result and result.get("success"):
                pkg = result.get("result")

        size_result = estimate_size_full(resolved) if resolved else None

        if not pkg:
            return DatasetMetadata(
                source_id=snapshot.source_id,
                dataset_id=dataset_id or snapshot.source_id,
                download_endpoint=resolved,
                api_endpoint=f"{base}/api/3/action/package_show",
                metadata_endpoint=snapshot.url,
                file_size_bytes=size_result.bytes if size_result else None,
                variables=list(fetch_request.variables or []),
                retrieval_method="CKAN package_show (no result)",
                unavailable_reason="CKAN package_show returned no result for this dataset.",
                provider=self.spec.provider_name,
            )

        resources = pkg.get("resources", [])
        formats = sorted({r.get("format") for r in resources if r.get("format")})
        org = pkg.get("organization") or {}

        return DatasetMetadata(
            source_id=snapshot.source_id,
            dataset_id=pkg.get("id") or dataset_id or snapshot.source_id,
            collection=pkg.get("name"),
            product=pkg.get("title"),
            download_endpoint=resolved,
            api_endpoint=f"{base}/api/3/action/package_show",
            metadata_endpoint=f"{base}/dataset/{pkg.get('name', dataset_id)}",
            file_size_bytes=size_result.bytes if size_result else None,
            variables=list(fetch_request.variables or []),
            spatial_coverage="Unknown",
            temporal_coverage="Unknown",
            file_format=", ".join(formats) if formats else (_extension_format(resolved) if resolved else None),
            retrieval_method="CKAN package_show",
            unavailable_reason="" if resolved else "CKAN package found but no downloadable resource resolved.",
            dataset_name=pkg.get("title"),
            provider=org.get("title") or org.get("name") or self.spec.provider_name,
            description=pkg.get("notes"),
            number_of_files=len(resources) or None,
            license=pkg.get("license_title") or pkg.get("license_id"),
            keywords=[t.get("name") for t in pkg.get("tags", []) if t.get("name")] or None,
            update_frequency=pkg.get("frequency") if pkg.get("frequency") else None,
            version=pkg.get("version") or None,
            start_date=pkg.get("metadata_created"),
            end_date=pkg.get("metadata_modified"),
        )


class FTPConnector(ProtocolConnector):
    spec = ProtocolSpec(
        "ftp", "FTP", ConnectorType.ftp,
        ("ftp://",),
        300, ("Native",),
    )

    def _resolve_asset_url(self, snapshot, fetch_request, credentials=None, diagnostics=None):
        diag = diagnostics or self._make_diagnostics()
        url = snapshot.url
        if _is_data_url(url):
            return url
        try:
            import ftplib
            from urllib.parse import urlparse
            p = urlparse(url)
            ftp = ftplib.FTP(p.netloc, timeout=20)
            if credentials and credentials.username:
                ftp.login(credentials.username, credentials.password)
            else:
                ftp.login()
            files = ftp.nlst(p.path or "/")
            ftp.quit()
            data_files = [f for f in files if _is_data_url(f)]

            def _score(fname: str) -> int:
                fl = fname.lower()
                return sum(1 for v in fetch_request.variables if v.lower().replace(" ", "_") in fl)

            if data_files:
                best = max(data_files, key=_score)
                resolved = f"ftp://{p.netloc}{best}"
                diag.download_endpoint = resolved
                return resolved
        except Exception as exc:
            diag.log_error(NetworkError(f"FTP directory listing failed: {exc}", url=url))
        return url if _is_data_url(url) else None

    def fetch_full(self, snapshot, fetch_request, credentials=None):
        diag = self._make_diagnostics()
        url = self._resolve_asset_url(snapshot, fetch_request, credentials, diag)
        if not url:
            raise RuntimeError(
                f"FTP connector could not find a data file at {snapshot.url}\n{diag.summary()}"
            )
        import urllib.request
        os.makedirs(os.path.dirname(fetch_request.dest_path) or ".", exist_ok=True)
        if credentials and credentials.username:
            from urllib.parse import urlparse
            p = urlparse(url)
            auth_url = f"ftp://{credentials.username}:{credentials.password}@{p.netloc}{p.path}"
        else:
            auth_url = url
        urllib.request.urlretrieve(auth_url, fetch_request.dest_path)
        valid, reason = validate_downloaded_file(fetch_request.dest_path)
        if not valid:
            raise RuntimeError(f"FTP download validation failed: {reason}")
        return fetch_request.dest_path


class S3Connector(ProtocolConnector):
    spec = ProtocolSpec(
        "s3", "S3-compatible storage", ConnectorType.s3,
        ("s3://", "amazonaws.com", "s3."),
        320, ("Native", "Cloud Object"),
    )

    def _resolve_asset_url(self, snapshot, fetch_request, credentials=None, diagnostics=None):
        url = snapshot.url
        if url.startswith("s3://"):
            parts = url[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
            return f"https://{bucket}.s3.amazonaws.com/{key}"
        return resolve_generic_asset(url, credentials, diagnostics=diagnostics)


class GenericRESTConnector(ProtocolConnector):
    spec = ProtocolSpec(
        "generic_rest", "Generic REST API", ConnectorType.generic_rest,
        ("api", "rest", "json", "/v1/", "/v2/"),
        900, ("JSON", "CSV", "Native"),
    )

    def _resolve_asset_url(self, snapshot, fetch_request, credentials=None, diagnostics=None):
        if _is_data_url(snapshot.url):
            return snapshot.url
        return resolve_generic_asset(snapshot.url, credentials, diagnostics=diagnostics)


# ── Register all ──────────────────────────────────────────────────────────────

for _cls in (
    STACConnector,
    OGCAPIConnector,
    THREDDSConnector,
    ERDDAPConnector,
    OPeNDAPConnector,
    CKANConnector,
    FTPConnector,
    S3Connector,
    GenericRESTConnector,
):
    register_connector(_cls)
