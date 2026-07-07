"""
ERDDAP Discoverer — searches a list of known ERDDAP servers for datasets
matching the request variables.

ERDDAP exposes a machine-readable search API:
  GET /erddap/search/index.json?searchFor=<query>&page=1&itemsPerPage=20

We hit every server in ERDDAP_SERVERS, parse results, and return
CandidateSource objects.
"""

import requests
from typing import List

from discovery.base import BaseDiscoverer
from models.discovery_schemas import AccessType, APIType, CandidateSource, DownloadFormat
from models.retrieval_request import RetrievalRequest
from sources.knowledge_base import expand_variables


ERDDAP_SERVERS = [
    "https://coastwatch.pfeg.noaa.gov/erddap",
    "https://erddap.ioos.us/erddap",
    "https://erddap.sensors.ioos.us/erddap",
    "https://erddap.axiomdatascience.com/erddap",
    "https://erddap.secoora.org/erddap",
    "https://erddap.marine.ie/erddap",
    "https://erddap.incois.gov.in/erddap",       # INCOIS ERDDAP (if available)
]

PROBE_TIMEOUT = 6


class ERDDAPDiscoverer(BaseDiscoverer):

    def search(self, request: RetrievalRequest) -> List[CandidateSource]:
        # CHANGED: expand raw variable names into their scientific
        # synonyms (e.g. "harmful algal bloom" -> "chlorophyll-a",
        # "ocean colour") before building the search query, so ERDDAP's
        # full-text search sees the richer term set. request.variables
        # itself is never modified -- only this discoverer's local copy.
        variables = expand_variables([v.variable for v in request.variables])
        query = " ".join(variables[:3])          # keep search focused
        results = []

        for server_url in ERDDAP_SERVERS:
            try:
                results.extend(self._search_server(server_url, query, request))
            except Exception as exc:
                print(f"[ERDDAPDiscoverer] {server_url} failed: {exc}")

        print(f"[ERDDAPDiscoverer] Found {len(results)} dataset(s) across ERDDAP servers.")
        return results

    def _search_server(
        self,
        server_url: str,
        query: str,
        request: RetrievalRequest,
    ) -> List[CandidateSource]:
        search_url = (
            f"{server_url}/search/index.json"
            f"?searchFor={requests.utils.quote(query)}"
            f"&page=1&itemsPerPage=10"
        )
        resp = requests.get(
            search_url,
            timeout=PROBE_TIMEOUT,
            headers={"User-Agent": "EarthIntelligenceAgent/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()

        # ERDDAP search returns: {"table": {"columnNames": [...], "rows": [[...], ...]}}
        col_names = data.get("table", {}).get("columnNames", [])
        rows = data.get("table", {}).get("rows", [])

        if not col_names or not rows:
            return []

        # Locate useful columns
        try:
            title_idx = col_names.index("Title")
            dataset_id_idx = col_names.index("Dataset ID")
            summary_idx = col_names.index("Summary") if "Summary" in col_names else None
        except ValueError:
            return []

        candidates = []
        for row in rows[:8]:                     # cap per server
            try:
                dataset_id = row[dataset_id_idx]
                title = row[title_idx]
                summary = row[summary_idx] if summary_idx is not None else ""

                dataset_url = f"{server_url}/griddap/{dataset_id}"
                source_id = f"erddap_{dataset_id[:40].replace('/', '_').replace(' ', '_')}"

                candidate = CandidateSource(
                    source_id=source_id,
                    name=title[:100],
                    url=dataset_url,
                    dataset_type=self._infer_dataset_type(title, summary),
                    variables_available=self._extract_variables(title, summary, request),
                    spatial_coverage="Unknown — check ERDDAP metadata",
                    temporal_coverage="Unknown — check ERDDAP metadata",
                    temporal_resolution="Unknown",
                    spatial_resolution="Unknown",
                    available_formats=[DownloadFormat.netcdf, DownloadFormat.csv],
                    access_type=AccessType.free,
                    requires_login=False,
                    requires_payment=False,
                    api_type=APIType.erddap,
                    api_docs=f"{server_url}/info/{dataset_id}/index.html",
                    metadata_url=f"{server_url}/info/{dataset_id}/index.html",
                    description=summary[:300] if summary else title,
                    discovery_origin="erddap_discovery",
                    catalog_authority_score=0.75,
                    catalog_scientific_acceptance=0.72,
                    catalog_historical_reliability=0.70,
                )
                candidates.append(candidate)
            except Exception:
                continue

        return candidates

    def _infer_dataset_type(self, title: str, summary: str) -> str:
        text = (title + " " + summary).lower()
        if any(k in text for k in ["ocean", "sst", "sea surface", "salinity", "chlorophyll", "wave"]):
            return "ocean"
        if any(k in text for k in ["weather", "wind", "precipitation", "temperature", "atmosphere"]):
            return "weather"
        if any(k in text for k in ["cyclone", "storm", "flood", "surge"]):
            return "disaster"
        if any(k in text for k in ["land", "elevation", "dem", "boundary"]):
            return "gis"
        return "ocean"   # default for ERDDAP

    def _extract_variables(
        self,
        title: str,
        summary: str,
        request: RetrievalRequest,
    ) -> List[str]:
        """Returns variables from the request that seem to appear in this dataset."""
        text = (title + " " + summary).lower()
        matched = []
        for var in request.variables:
            if var.variable.lower() in text or var.variable.lower().replace("_", " ") in text:
                matched.append(var.variable)
        return matched if matched else ["unknown"]
