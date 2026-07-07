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
    fetch_method: FetchMethod = FetchMethod.full_download_untrimmed
    variables_included: List[str] = Field(default_factory=list)
    success: bool = False
    error: Optional[str] = None


class Agent4Output(BaseModel):
    plan_source_ids: List[str] = Field(default_factory=list)
    source_decisions: List[SourceDecision] = Field(default_factory=list)
    manifest: List[DownloadManifestEntry] = Field(default_factory=list)
    total_size_bytes: float = 0.0
    covers_full_query: bool = False
    uncovered_variables: List[str] = Field(default_factory=list)
    download_location: Optional[str] = None
    send_to_agent5: bool = False
    notes: List[str] = Field(default_factory=list)
