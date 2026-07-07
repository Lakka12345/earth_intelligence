"""
Generic HTTP Connector — the universal fallback.

Handles anonymous (or already-authenticated-via-header) HTTP(S)
downloads. Does not support subsetting or self-registration -- it's
the safety net every other connector's absence falls back to, and it
is honest about that (Agent 4's orchestrator knows to expect a full
download + local trim from this connector, not an exact slice).
"""

import os
from typing import Optional

import requests

from agents.agent4_connectors.base import BaseConnector, Credentials, FetchRequest
from models.agent4_schemas import SizeEstimate, format_bytes
from models.website_analysis_schemas import SourceSnapshot

_CHUNK_SIZE = 1024 * 1024  # 1MB chunks -- streams instead of loading the whole file into memory


class GenericHTTPConnector(BaseConnector):
    name = "generic_http"

    def can_handle(self, snapshot: SourceSnapshot) -> bool:
        # Always the last-resort match -- registry tries named
        # connectors first, this one always returns True.
        return True

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

    def fetch_full(self, snapshot: SourceSnapshot, fetch_request: FetchRequest, credentials: Optional[Credentials] = None) -> str:
        auth = (credentials.username, credentials.password) if credentials else None
        os.makedirs(os.path.dirname(fetch_request.dest_path) or ".", exist_ok=True)

        with requests.get(snapshot.url, stream=True, auth=auth, timeout=30) as resp:
            resp.raise_for_status()
            with open(fetch_request.dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)

        return fetch_request.dest_path

    # fetch_subset intentionally not overridden -- this connector has
    # no way to know a provider-specific subsetting query syntax, so it
    # correctly raises NotImplementedError via the base class and the
    # orchestrator falls back to fetch_full + local trim.
