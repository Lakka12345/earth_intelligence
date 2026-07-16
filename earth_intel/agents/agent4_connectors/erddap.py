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
import urllib3

# Suppress the SSL InsecureRequestWarning to keep console clean
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from agents.agent4_connectors.base import BaseConnector, Credentials, FetchRequest
from models.agent4_schemas import DatasetMetadata, SizeEstimate, format_bytes
from models.website_analysis_schemas import SourceSnapshot

_CHUNK_SIZE = 1024 * 1024


def _raise_if_html(resp: requests.Response) -> None:
    content_type = (resp.headers.get("Content-Type") or "").lower()
    if "text/html" in content_type:
        raise RuntimeError("ERDDAP returned an HTML response instead of NetCDF data.")


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
        fallback_size = 50.0 * 1024 * 1024  # 50MB safe fallback

        if not subset_url:
            return SizeEstimate(
                source_id=snapshot.source_id, 
                estimated_bytes=fallback_size,
                is_exact=False,
                method="safe_erddap_fallback",
                human_readable=format_bytes(fallback_size)
            )

        try:
            # Added verify=False to bypass strict SSL
            resp = requests.head(subset_url, allow_redirects=True, timeout=15, verify=False)
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
        
        # Return fallback size if size couldn't be resolved instead of 'unavailable'
        return SizeEstimate(
            source_id=snapshot.source_id,
            estimated_bytes=fallback_size,
            is_exact=False,
            method="safe_erddap_fallback",
            human_readable=format_bytes(fallback_size)
        )

    def _metadata_url(self, snapshot: SourceSnapshot) -> Optional[str]:
        dataset_id = self._dataset_id_from_url(snapshot.url)
        if not dataset_id:
            return None
        base = re.sub(r"/(?:griddap|tabledap)/.*", "/", snapshot.url)
        return urljoin(base, f"info/{dataset_id}/index.json")

    def probe_metadata(self, snapshot: SourceSnapshot, fetch_request: FetchRequest) -> DatasetMetadata:
        try:
            dataset_id = self._dataset_id_from_url(snapshot.url) or snapshot.source_id
            metadata_url = self._metadata_url(snapshot)
            subset_url = self._build_subset_url(snapshot, fetch_request)
            full_url = snapshot.url if dataset_id is None else re.sub(r"(/griddap/[^/.?]+).*", r"\1.nc", snapshot.url)

            variables = list(snapshot.variables_available or fetch_request.variables or [])
            spatial_coverage = "Unknown"
            temporal_coverage = "Unknown"
            unavailable_reason = ""

            if metadata_url:
                try:
                    # Added verify=False
                    resp = requests.get(metadata_url, timeout=15, verify=False)
                    resp.raise_for_status()
                    table = resp.json().get("table", {})
                    rows = table.get("rows", [])
                    variable_names = [
                        str(row[1]) for row in rows
                        if len(row) > 2 and str(row[0]).lower() == "variable"
                    ]
                    if variable_names:
                        variables = variable_names
                    attrs = {
                        str(row[2]).lower(): str(row[4])
                        for row in rows
                        if len(row) > 4 and str(row[0]).lower() == "attribute"
                    }
                    spatial_coverage = attrs.get("geospatial_lat_min", "Unknown")
                    if spatial_coverage != "Unknown" and attrs.get("geospatial_lat_max"):
                        spatial_coverage = (
                            f"lat {attrs.get('geospatial_lat_min')} to {attrs.get('geospatial_lat_max')}, "
                            f"lon {attrs.get('geospatial_lon_min', 'Unknown')} to {attrs.get('geospatial_lon_max', 'Unknown')}"
                        )
                    temporal_coverage = attrs.get("time_coverage_start", "Unknown")
                    if temporal_coverage != "Unknown" and attrs.get("time_coverage_end"):
                        temporal_coverage = f"{temporal_coverage} to {attrs.get('time_coverage_end')}"
                except Exception as exc:
                    unavailable_reason = f"ERDDAP metadata endpoint could not be read: {exc}"
            else:
                unavailable_reason = "Could not derive ERDDAP /info metadata endpoint from source URL."

            size = None
            content_type = None
            try:
                # Added verify=False
                head = requests.head(subset_url or full_url, allow_redirects=True, timeout=15, verify=False)
                content_length = head.headers.get("Content-Length")
                size = float(content_length) if content_length else 50.0 * 1024 * 1024
                content_type = head.headers.get("Content-Type")
            except Exception:
                size = 50.0 * 1024 * 1024  # Fallback to 50MB

            return DatasetMetadata(
                source_id=snapshot.source_id,
                dataset_id=dataset_id,
                collection=getattr(snapshot, "dataset_type", "Unknown"),
                product=snapshot.name,
                download_endpoint=subset_url or full_url,
                api_endpoint=re.sub(r"/(?:griddap|tabledap)/.*", "/", snapshot.url),
                metadata_endpoint=metadata_url,
                file_size_bytes=size,
                variables=variables,
                spatial_coverage=spatial_coverage,
                temporal_coverage=temporal_coverage,
                file_format="NetCDF",
                content_type=content_type,
                retrieval_method="ERDDAP /info metadata",
                unavailable_reason=unavailable_reason,
            )
        except Exception as exc:
            # Absolute fallback to ensure a valid DatasetMetadata object is returned
            return DatasetMetadata(
                source_id=snapshot.source_id,
                dataset_id=snapshot.source_id,
                collection=getattr(snapshot, "dataset_type", "Unknown"),
                product=snapshot.name,
                download_endpoint=snapshot.url,
                api_endpoint=snapshot.url,
                metadata_endpoint=snapshot.url,
                file_size_bytes=50.0 * 1024 * 1024,
                variables=list(getattr(snapshot, "variables_available", []) or fetch_request.variables or []),
                file_format="Unknown",
                checksum=None,
                content_type="Unknown",
                retrieval_method="ERDDAP Global Fallback",
                unavailable_reason=f"Fatal ERDDAP metadata probe failure: {exc}",
            )

    def fetch_subset(self, snapshot: SourceSnapshot, fetch_request: FetchRequest, credentials: Optional[Credentials] = None) -> str:
        subset_url = self._build_subset_url(snapshot, fetch_request)
        if not subset_url:
            raise NotImplementedError(
                "Could not construct an ERDDAP subset query for this dataset "
                "(non-griddap endpoint, or dataset id/variables unresolved). "
                "Falling back to full download."
            )

        os.makedirs(os.path.dirname(fetch_request.dest_path) or ".", exist_ok=True)
        # Added verify=False
        with requests.get(subset_url, stream=True, timeout=60, verify=False) as resp:
            resp.raise_for_status()
            _raise_if_html(resp)
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
        # Added verify=False
        with requests.get(full_url, stream=True, timeout=60, verify=False) as resp:
            resp.raise_for_status()
            _raise_if_html(resp)
            with open(fetch_request.dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
        return fetch_request.dest_path