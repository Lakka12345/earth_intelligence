"""
ERDDAP Connector — a concrete, working example of the connector
pattern (not just the generic fallback).

ERDDAP's griddap endpoints support real server-side subsetting: a
query like

    {base}/griddap/{dataset_id}.nc?{variable}[(t0):(t1)][(lat0):(lat1)][(lon0):(lon1)]

returns ONLY that slice, not the whole dataset -- this is exactly the
"least storage possible, no accuracy compromise" mechanism described
in the design. This is a best-effort implementation: ERDDAP dimension
names/order can vary per dataset, so this covers the common
(time, latitude, longitude) ordering and is meant as the concrete
example to extend, not a guarantee for every ERDDAP dataset in
existence.
"""

import os
import re
from typing import Optional
from urllib.parse import urljoin

import requests

from agents.agent4_connectors.base import BaseConnector, Credentials, FetchRequest
from models.agent4_schemas import SizeEstimate, format_bytes
from models.website_analysis_schemas import SourceSnapshot

_CHUNK_SIZE = 1024 * 1024


class ERDDAPConnector(BaseConnector):
    name = "erddap"

    def can_handle(self, snapshot: SourceSnapshot) -> bool:
        return "erddap" in (snapshot.api_type or "").lower() or "erddap" in (snapshot.url or "").lower()

    def _dataset_id_from_url(self, url: str) -> Optional[str]:
        # ERDDAP griddap/tabledap URLs end in /griddap/{id} or /griddap/{id}.html etc.
        match = re.search(r"/(?:griddap|tabledap)/([^/.?]+)", url)
        return match.group(1) if match else None

    def _build_subset_url(self, snapshot: SourceSnapshot, fetch_request: FetchRequest) -> Optional[str]:
        dataset_id = self._dataset_id_from_url(snapshot.url)
        if not dataset_id or not fetch_request.variables:
            return None

        base = re.sub(r"/griddap/.*", "/griddap/", snapshot.url)
        if "/griddap/" not in snapshot.url:
            return None  # tabledap subsetting uses a different syntax, not covered by this example

        t0, t1 = fetch_request.time_range or ("", "")
        bbox = fetch_request.bounding_box
        lat_clause = f"[({bbox[1]}):({bbox[3]})]" if bbox else "[]"
        lon_clause = f"[({bbox[0]}):({bbox[2]})]" if bbox else "[]"
        time_clause = f"[({t0}):({t1})]" if t0 and t1 else "[]"

        var_clauses = "".join(f"{v}{time_clause}{lat_clause}{lon_clause}," for v in fetch_request.variables)
        var_clauses = var_clauses.rstrip(",")

        return f"{base}{dataset_id}.nc?{var_clauses}"

    def probe_size(self, snapshot: SourceSnapshot, fetch_request: FetchRequest) -> SizeEstimate:
        subset_url = self._build_subset_url(snapshot, fetch_request)
        if not subset_url:
            return SizeEstimate(source_id=snapshot.source_id, method="unavailable")
        try:
            resp = requests.head(subset_url, allow_redirects=True, timeout=15)
            content_length = resp.headers.get("Content-Length")
            if content_length:
                size = float(content_length)
                return SizeEstimate(
                    source_id=snapshot.source_id,
                    estimated_bytes=size,
                    is_exact=True,
                    method="ERDDAP subset HEAD request",
                    human_readable=format_bytes(size),
                )
        except Exception as exc:
            print(f"[ERDDAPConnector] Size probe failed for {snapshot.source_id} (non-fatal): {exc}")
        return SizeEstimate(source_id=snapshot.source_id, method="unavailable")

    def fetch_subset(self, snapshot: SourceSnapshot, fetch_request: FetchRequest, credentials: Optional[Credentials] = None) -> str:
        subset_url = self._build_subset_url(snapshot, fetch_request)
        if not subset_url:
            raise NotImplementedError(
                "Could not construct an ERDDAP subset query for this dataset "
                "(non-griddap endpoint, or dataset id/variables unresolved). "
                "Falling back to full download."
            )

        os.makedirs(os.path.dirname(fetch_request.dest_path) or ".", exist_ok=True)
        with requests.get(subset_url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(fetch_request.dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
        return fetch_request.dest_path

    def fetch_full(self, snapshot: SourceSnapshot, fetch_request: FetchRequest, credentials: Optional[Credentials] = None) -> str:
        # Full-dataset .nc download -- last resort if subsetting failed.
        dataset_id = self._dataset_id_from_url(snapshot.url)
        full_url = snapshot.url if dataset_id is None else re.sub(r"(/griddap/[^/.?]+).*", r"\1.nc", snapshot.url)
        os.makedirs(os.path.dirname(fetch_request.dest_path) or ".", exist_ok=True)
        with requests.get(full_url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(fetch_request.dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
        return fetch_request.dest_path
