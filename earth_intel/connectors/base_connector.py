"""
Abstract connector interface for Agent 4 retrieval providers.

This module intentionally defines hooks for future phases without
implementing provider-specific behaviour. Concrete connectors inherit
from BaseConnector and advertise their capabilities.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from connectors.connector_types import (
    AccessType,
    AuthenticationType,
    CapabilityFlags,
    ConnectorType,
    DatasetType,
    RetryStrategy,
    ValidationStrategy,
)
from models.website_analysis_schemas import SourceSnapshot


@dataclass
class Credentials:
    username: str = ""
    password: str = ""
    email: Optional[str] = None
    api_key: Optional[str] = None
    token: Optional[str] = None
    session_token: Optional[str] = None


@dataclass
class FetchRequest:
    variables: List[str]
    bounding_box: Optional[Tuple[float, float, float, float]] = None
    time_range: Optional[Tuple[str, str]] = None
    dest_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConnectorDescriptor:
    connector_id: str
    provider_name: str
    connector_type: ConnectorType
    supported_access_types: Tuple[AccessType, ...] = (AccessType.unknown,)
    supported_dataset_types: Tuple[DatasetType, ...] = (DatasetType.unknown,)
    supported_authentication: Tuple[AuthenticationType, ...] = (AuthenticationType.unknown,)
    capabilities: CapabilityFlags = CapabilityFlags.none
    retry_strategy: RetryStrategy = RetryStrategy.exponential_backoff
    validation_strategy: ValidationStrategy = ValidationStrategy.basic_file
    priority: int = 100


@dataclass(frozen=True)
class ConnectorMatch:
    score: int
    reason: str = ""


class BaseConnector(ABC):
    descriptor: ConnectorDescriptor

    @property
    def name(self) -> str:
        return self.descriptor.connector_id

    @property
    def capabilities(self) -> CapabilityFlags:
        return self.descriptor.capabilities

    @abstractmethod
    def can_handle(self, snapshot: SourceSnapshot) -> bool:
        """Return True when this connector can handle the source."""

    def match_score(self, snapshot: SourceSnapshot, context: Optional[Dict[str, Any]] = None) -> ConnectorMatch:
        if self.can_handle(snapshot):
            return ConnectorMatch(score=max(1, 100 - self.descriptor.priority), reason="Connector can handle source.")
        return ConnectorMatch(score=0, reason="Connector does not match source.")

    def probe_size(self, snapshot: SourceSnapshot, fetch_request: FetchRequest):
        from models.agent4_schemas import SizeEstimate
        return SizeEstimate(source_id=snapshot.source_id, method="unavailable")

    def probe_metadata(self, snapshot: SourceSnapshot, fetch_request: FetchRequest):
        from models.agent4_schemas import DatasetMetadata
        return DatasetMetadata(
            source_id=snapshot.source_id,
            dataset_id=snapshot.source_id,
            download_endpoint=snapshot.url,
            api_endpoint=snapshot.url,
            metadata_endpoint=snapshot.url,
            variables=list(getattr(snapshot, "variables_available", []) or fetch_request.variables or []),
            retrieval_method="unavailable",
            unavailable_reason=f"{self.name} connector does not expose metadata probing.",
        )

    def discover_datasets(self, snapshot: SourceSnapshot, context: Optional[Dict[str, Any]] = None):
        raise NotImplementedError(f"{self.name} connector does not implement dataset discovery yet.")

    def plan_download(self, snapshot: SourceSnapshot, fetch_request: FetchRequest):
        raise NotImplementedError(f"{self.name} connector does not implement download planning yet.")

    def fetch_subset(self, snapshot: SourceSnapshot, fetch_request: FetchRequest, credentials: Optional[Credentials] = None) -> str:
        raise NotImplementedError(f"{self.name} connector does not support server-side subsetting.")

    def fetch_full(self, snapshot: SourceSnapshot, fetch_request: FetchRequest, credentials: Optional[Credentials] = None) -> str:
        from agents.agent4_download_engine import DownloadEngine, DownloadTask

        task = DownloadTask(
            url=getattr(snapshot, "url", ""),
            dest_path=fetch_request.dest_path,
            expected_size=fetch_request.metadata.get("expected_size"),
            expected_content_type=fetch_request.metadata.get("expected_content_type"),
            expected_format=fetch_request.metadata.get("expected_format"),
            checksum=fetch_request.metadata.get("checksum"),
            source_id=getattr(snapshot, "source_id", ""),
            provider=getattr(snapshot, "name", ""),
            connector_id=self.name,
            protocol=self.descriptor.connector_type.value,
            allow_resume=CapabilityFlags.supports_resume in self.capabilities,
        )
        result = DownloadEngine().download_one(task, credentials)
        fetch_request.metadata["download_result"] = result
        if not result.success:
            raise RuntimeError(result.error or "Download failed validation.")
        return result.dest_path

    def supports_self_registration(self) -> bool:
        return CapabilityFlags.supports_registration in self.capabilities

    def self_register(self, snapshot: SourceSnapshot, real_email: str) -> Credentials:
        raise NotImplementedError(f"{self.name} connector does not support self-registration.")

    def login(self, snapshot: SourceSnapshot, credentials: Credentials):
        raise NotImplementedError(f"{self.name} connector does not implement authentication.")

    def validate_download(self, local_path: str, metadata=None):
        raise NotImplementedError(f"{self.name} connector does not implement provider validation.")

    def retry_policy(self):
        return self.descriptor.retry_strategy
