"""
Agent 4 core schemas.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SizeEstimate(BaseModel):
    source_id: str
    estimated_bytes: Optional[float] = None
    is_exact: bool = False
    method: str = "unavailable"   # "content-length header" | "catalog metadata" | "unavailable"
    human_readable: str = "Unknown"


class DatasetMetadata(BaseModel):
    source_id: str
    dataset_id: Optional[str] = None
    collection: Optional[str] = None
    product: Optional[str] = None
    download_endpoint: Optional[str] = None
    api_endpoint: Optional[str] = None
    metadata_endpoint: Optional[str] = None
    file_size_bytes: Optional[float] = None
    variables: List[str] = Field(default_factory=list)
    spatial_coverage: str = "Unknown"
    temporal_coverage: str = "Unknown"
    file_format: Optional[str] = None
    checksum: Optional[str] = None
    content_type: Optional[str] = None
    retrieval_method: str = "unavailable"
    unavailable_reason: str = ""


class DatasetDescriptor(BaseModel):
    provider: str
    dataset_name: str
    collection_name: Optional[str] = None
    dataset_id: str
    doi: Optional[str] = None
    api_endpoint: Optional[str] = None
    metadata_endpoint: Optional[str] = None
    download_endpoint: Optional[str] = None
    supported_variables: List[str] = Field(default_factory=list)
    temporal_coverage: str = "Unknown"
    spatial_coverage: str = "Unknown"
    supported_formats: List[str] = Field(default_factory=list)
    estimated_size_bytes: Optional[float] = None
    granule_count: Optional[int] = None
    checksum_url: Optional[str] = None
    authentication_required: bool = False
    access_notes: str = ""
    source_url: Optional[str] = None
    metadata_unavailable_reason: str = ""

    def to_metadata(self, source_id: str) -> DatasetMetadata:
        return DatasetMetadata(
            source_id=source_id,
            dataset_id=self.dataset_id,
            collection=self.collection_name,
            product=self.dataset_name,
            download_endpoint=self.download_endpoint,
            api_endpoint=self.api_endpoint,
            metadata_endpoint=self.metadata_endpoint,
            file_size_bytes=self.estimated_size_bytes,
            variables=list(self.supported_variables),
            spatial_coverage=self.spatial_coverage,
            temporal_coverage=self.temporal_coverage,
            file_format=", ".join(self.supported_formats) if self.supported_formats else None,
            retrieval_method="connector dataset discovery",
            unavailable_reason=self.metadata_unavailable_reason,
        )


def format_bytes(num_bytes: Optional[float]) -> str:
    if num_bytes is None:
        return "Unknown"
    n = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


class AccessDecisionType(str, Enum):
    free_access = "free_access"
    agent_self_registered = "agent_self_registered"
    user_provided_credentials = "user_provided_credentials"
    payment_redirect = "payment_redirect"          # user handles this manually
    skipped_declined = "skipped_declined"           # user said no, alternate substituted or gap accepted
    skipped_unresolved = "skipped_unresolved"       # could not resolve access, no alternate available


class SourceDecision(BaseModel):
    source_id: str
    decision: AccessDecisionType
    credentials_used: bool = False
    credentials_persisted: bool = False
    approved_size: Optional[SizeEstimate] = None
    dataset_metadata: Optional[DatasetMetadata] = None
    variables_expected: List[str] = Field(default_factory=list)
    notes: str = ""


class DownloadFormat(str, Enum):
    native = "native"       # whatever the source natively provides -- recommended default
    csv = "csv"
    netcdf = "netcdf"
    parquet = "parquet"
    geotiff = "geotiff"


class DownloadLocationMode(str, Enum):
    managed_project_folder = "managed_project_folder"
    custom_path = "custom_path"
    ask_each_time = "ask_each_time"


class FetchMethod(str, Enum):
    server_side_subset = "server_side_subset"          # best case: exact slice from the source
    full_download_then_trimmed = "full_download_then_trimmed"  # fallback: full file, trimmed locally, raw deleted
    full_download_untrimmed = "full_download_untrimmed"        # no subsetting or trimming support at all
    manual_user_download = "manual_user_download"               # paid/manual sources


class DownloadManifestEntry(BaseModel):
    source_id: str
    source_name: str
    local_path: Optional[str] = None
    format: DownloadFormat = DownloadFormat.native
    size_bytes: Optional[float] = None
    estimated_size_bytes: Optional[float] = None
    fetch_method: FetchMethod = FetchMethod.full_download_untrimmed
    variables_included: List[str] = Field(default_factory=list)
    dataset_metadata: Optional[DatasetMetadata] = None
    provider: Optional[str] = None
    connector_used: Optional[str] = None
    protocol_used: Optional[str] = None
    download_time_seconds: Optional[float] = None
    checksum_status: str = "not_checked"
    validation_status: str = "not_checked"
    success: bool = False
    error: Optional[str] = None
    retries_attempted: int = 0
    validation_notes: List[str] = Field(default_factory=list)


class Agent4Output(BaseModel):
    plan_source_ids: List[str] = Field(default_factory=list)
    source_decisions: List[SourceDecision] = Field(default_factory=list)
    manifest: List[DownloadManifestEntry] = Field(default_factory=list)
    total_size_bytes: float = 0.0
    actual_downloaded_bytes: float = 0.0
    covers_full_query: bool = False
    uncovered_variables: List[str] = Field(default_factory=list)
    retrieved_variables: List[str] = Field(default_factory=list)
    coverage_percent: float = 0.0
    coverage_table: List[Dict[str, str]] = Field(default_factory=list)
    download_location: Optional[str] = None
    send_to_agent5: bool = False
    notes: List[str] = Field(default_factory=list)

    @property
    def successful_download_count(self) -> int:
        return len([entry for entry in self.manifest if entry.success])
