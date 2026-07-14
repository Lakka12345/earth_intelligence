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
from typing import Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import requests

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

        try:
            s = _session()
            resp = s.head(probe_url, allow_redirects=True, timeout=15)
            content_type = resp.headers.get("Content-Type")
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

    def probe_size(self, snapshot, fetch_request):
        diag = self._make_diagnostics()
        # Try STAC item file:size first via parallel search
        from agents.agent4_asset_resolver import _session as _get_session, estimate_size_from_stac_item
        s = _get_session()
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
