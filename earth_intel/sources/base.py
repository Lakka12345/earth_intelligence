"""
Agent 4 Connectors — pluggable per-provider integration points.

WHY THIS EXISTS
  There is no generic way to "register on any website" or "log into
  any website" -- every provider's signup form, auth flow, and data API
  is different. Rather than pretending otherwise, Agent 4 is built
  around a connector registry (same pattern as retrieval_manager.py's
  OFFICIAL_API_PROVIDERS): a generic fallback that handles plain,
  anonymous HTTP downloads, plus named connectors added over time for
  specific providers that need real integration work.

WHAT A CONNECTOR MUST DO
  - can_handle(snapshot): does this connector know how to talk to this
    specific source?
  - probe_size(...): best-effort size estimate before download (never
    guess silently -- return None if genuinely unknown)
  - fetch_subset(...): pull exactly the requested variable/region/time
    slice, if the provider's API supports it. Raise NotImplementedError
    if it doesn't -- callers fall back to fetch_full + local trim.
  - fetch_full(...): download the whole resource.
  - self_register(...) / login(...): only implemented by connectors for
    providers where this is actually well-understood and safe to
    automate. The generic fallback does NOT implement these -- a
    source needing auth with no specific connector is routed to "ask
    the user" rather than guessed at.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from models.website_analysis_schemas import SourceSnapshot


@dataclass
class Credentials:
    username: str
    password: str
    email: Optional[str] = None


@dataclass
class FetchRequest:
    variables: list
    bounding_box: Optional[tuple] = None    # (min_lon, min_lat, max_lon, max_lat)
    time_range: Optional[tuple] = None       # (start_iso, end_iso)
    dest_path: str = ""


class BaseConnector(ABC):
    name: str = "base"

    @abstractmethod
    def can_handle(self, snapshot: SourceSnapshot) -> bool:
        ...

    def probe_size(self, snapshot: SourceSnapshot, fetch_request: FetchRequest):
        """Returns a SizeEstimate. Default: unavailable. Override when
        the provider's API exposes size info."""
        from models.agent4_schemas import SizeEstimate
        return SizeEstimate(source_id=snapshot.source_id, method="unavailable")

    def fetch_subset(self, snapshot: SourceSnapshot, fetch_request: FetchRequest, credentials: Optional[Credentials] = None) -> str:
        """Returns the local file path on success. Raise
        NotImplementedError if this provider/connector doesn't support
        server-side subsetting -- caller falls back to fetch_full."""
        raise NotImplementedError(f"{self.name} connector does not support server-side subsetting.")

    @abstractmethod
    def fetch_full(self, snapshot: SourceSnapshot, fetch_request: FetchRequest, credentials: Optional[Credentials] = None) -> str:
        ...

    def supports_self_registration(self) -> bool:
        return False

    def self_register(self, snapshot: SourceSnapshot, real_email: str) -> Credentials:
        raise NotImplementedError(f"{self.name} connector does not support self-registration.")

    def login(self, snapshot: SourceSnapshot, credentials: Credentials):
        """Returns an opaque session object connectors can pass to
        fetch_* methods. Default: no session needed."""
        return None
