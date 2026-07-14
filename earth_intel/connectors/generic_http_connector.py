"""
Generic HTTP fallback connector.

This is intentionally provider-agnostic. Specialised providers belong
in future connector files and should be selected by ConnectorFactory.
"""

from typing import Optional

import requests

from agents.agent4_download_engine import DownloadEngine, DownloadTask
from connectors.base_connector import BaseConnector, ConnectorDescriptor, Credentials, FetchRequest
from connectors.connector_registry import register_connector
from connectors.connector_types import (
    AccessType,
    AuthenticationType,
    CapabilityFlags,
    ConnectorType,
    DatasetType,
    RetryStrategy,
    ValidationStrategy,
)
from models.agent4_schemas import DatasetMetadata, SizeEstimate, format_bytes
from models.agent4_schemas import DatasetDescriptor
from models.website_analysis_schemas import SourceSnapshot

def _is_obvious_webpage(resp: requests.Response) -> bool:
    content_type = (resp.headers.get("Content-Type") or "").lower()
    disposition = (resp.headers.get("Content-Disposition") or "").lower()
    if "attachment" in disposition:
        return False
    return "text/html" in content_type


class GenericHTTPConnector(BaseConnector):
    descriptor = ConnectorDescriptor(
        connector_id="generic_http",
        provider_name="Generic HTTP",
        connector_type=ConnectorType.generic_http,
        supported_access_types=(AccessType.public, AccessType.unknown),
        supported_dataset_types=(DatasetType.unknown,),
        supported_authentication=(AuthenticationType.none, AuthenticationType.basic_auth, AuthenticationType.unknown),
        capabilities=(
            CapabilityFlags.supports_metadata
            | CapabilityFlags.supports_download
            | CapabilityFlags.supports_resume
            | CapabilityFlags.supports_checksum
            | CapabilityFlags.supports_parallel_download
        ),
        retry_strategy=RetryStrategy.exponential_backoff,
        validation_strategy=ValidationStrategy.basic_file,
        priority=10_000,
    )

    def can_handle(self, snapshot: SourceSnapshot) -> bool:
        return True

    def match_score(self, snapshot: SourceSnapshot, context=None):
        from connectors.base_connector import ConnectorMatch
        return ConnectorMatch(score=1, reason="Generic fallback connector.")

    def probe_size(self, snapshot: SourceSnapshot, fetch_request: FetchRequest) -> SizeEstimate:
        try:
            resp = requests.head(snapshot.url, allow_redirects=True, timeout=10)
            content_length = resp.headers.get("Content-Length")
            if content_length:
                size = float(content_length)
                return SizeEstimate(
                    source_id=snapshot.source_id,
                    estimated_bytes=size,
                    is_exact=True,
                    method="content-length header",
                    human_readable=format_bytes(size),
                )
        except Exception as exc:
            print(f"[GenericHTTPConnector] Size probe failed for {snapshot.source_id} (non-fatal): {exc}")
        return SizeEstimate(source_id=snapshot.source_id, method="unavailable")

    def discover_datasets(self, snapshot: SourceSnapshot, context=None):
        return [
            DatasetDescriptor(
                provider=snapshot.name,
                dataset_name=snapshot.name,
                collection_name=snapshot.dataset_type,
                dataset_id=snapshot.source_id,
                api_endpoint=snapshot.url,
                metadata_endpoint=snapshot.url,
                download_endpoint=snapshot.url,
                supported_variables=list(snapshot.variables_available or (context or {}).get("variables") or []),
                supported_formats=["Unknown"],
                authentication_required=False,
                source_url=snapshot.url,
                metadata_unavailable_reason="Generic HTTP fallback has no provider catalog; using source URL as a dataset-like descriptor.",
            )
        ]

    def probe_metadata(self, snapshot: SourceSnapshot, fetch_request: FetchRequest) -> DatasetMetadata:
        try:
            resp = requests.head(snapshot.url, allow_redirects=True, timeout=10)
            content_length = resp.headers.get("Content-Length")
            content_type = resp.headers.get("Content-Type")
            checksum = (
                resp.headers.get("ETag")
                or resp.headers.get("Content-MD5")
                or resp.headers.get("x-amz-checksum-sha256")
            )
            return DatasetMetadata(
                source_id=snapshot.source_id,
                dataset_id=snapshot.source_id,
                collection=snapshot.dataset_type,
                product=snapshot.name,
                download_endpoint=resp.url or snapshot.url,
                api_endpoint=snapshot.url,
                metadata_endpoint=snapshot.url,
                file_size_bytes=float(content_length) if content_length else None,
                variables=list(snapshot.variables_available or fetch_request.variables or []),
                file_format=content_type,
                checksum=checksum,
                content_type=content_type,
                retrieval_method="HTTP HEAD",
                unavailable_reason="" if resp.ok else f"Metadata HEAD returned HTTP {resp.status_code}.",
            )
        except Exception as exc:
            return DatasetMetadata(
                source_id=snapshot.source_id,
                dataset_id=snapshot.source_id,
                collection=snapshot.dataset_type,
                product=snapshot.name,
                download_endpoint=snapshot.url,
                api_endpoint=snapshot.url,
                metadata_endpoint=snapshot.url,
                variables=list(snapshot.variables_available or fetch_request.variables or []),
                retrieval_method="HTTP HEAD",
                unavailable_reason=f"HTTP metadata probe failed: {exc}",
            )

    def fetch_full(self, snapshot: SourceSnapshot, fetch_request: FetchRequest, credentials: Optional[Credentials] = None) -> str:
        task = DownloadTask(
            url=snapshot.url,
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
        if not result.success:
            raise RuntimeError(
                result.error
                or "Provider returned a response that failed dataset download validation."
            )
        return result.dest_path


register_connector(GenericHTTPConnector)
