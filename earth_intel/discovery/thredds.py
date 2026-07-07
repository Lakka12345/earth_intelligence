"""
THREDDS Discoverer — searches THREDDS Data Servers.

THREDDS (Thematic Realtime Environmental Distributed Data Services)
exposes a catalog XML at /thredds/catalog.xml with nested datasets.
We search known THREDDS servers and match against request variables.
"""

import requests
import xml.etree.ElementTree as ET
from typing import List, Optional

from discovery.base import BaseDiscoverer
from models.discovery_schemas import AccessType, APIType, CandidateSource, DownloadFormat
from models.retrieval_request import RetrievalRequest
from sources.knowledge_base import expand_variables


THREDDS_SERVERS = [
    {
        "name": "NOAA NOMADS THREDDS",
        "catalog_url": "https://nomads.ncep.noaa.gov/thredds/catalog.xml",
        "base_url": "https://nomads.ncep.noaa.gov/thredds",
    },
    {
        "name": "Unidata THREDDS",
        "catalog_url": "https://thredds.ucar.edu/thredds/catalog.xml",
        "base_url": "https://thredds.ucar.edu/thredds",
    },
    {
        "name": "INCOIS THREDDS",
        "catalog_url": "https://incois.gov.in/thredds/catalog.xml",
        "base_url": "https://incois.gov.in/thredds",
    },
]

NS = {"thredds": "http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"}
PROBE_TIMEOUT = 8


class THREDDSDiscoverer(BaseDiscoverer):

    def search(self, request: RetrievalRequest) -> List[CandidateSource]:
        # CHANGED: expand to scientific synonyms before building the
        # keyword filter. This filter is actively used (see
        # _search_server below) to decide which catalog datasets are
        # even considered, so expansion here has real effect on recall
        # -- e.g. a "harmful algal bloom" request now also matches
        # THREDDS datasets named with "chlorophyll" or "ocean color".
        keywords = [
            t.replace("_", " ").lower()
            for t in expand_variables([v.variable for v in request.variables])[:4]
        ]
        candidates = []

        for server in THREDDS_SERVERS:
            try:
                results = self._search_server(server, keywords, request)
                candidates.extend(results)
            except Exception as exc:
                print(f"[THREDDSDiscoverer] {server['name']} failed: {exc}")

        print(f"[THREDDSDiscoverer] Found {len(candidates)} dataset(s) from THREDDS servers.")
        return candidates

    def _search_server(
        self,
        server: dict,
        keywords: List[str],
        request: RetrievalRequest,
    ) -> List[CandidateSource]:

        resp = requests.get(
            server["catalog_url"],
            timeout=PROBE_TIMEOUT,
            headers={"User-Agent": "EarthIntelligenceAgent/1.0"},
        )
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        candidates = []
        count = 0

        for dataset in root.iter():
            if count >= 6:
                break

            tag = dataset.tag.split("}")[-1] if "}" in dataset.tag else dataset.tag
            if tag != "dataset":
                continue

            name = dataset.get("name", "")
            if not name or len(name) < 3:
                continue

            name_lower = name.lower()
            if not any(kw in name_lower for kw in keywords):
                continue

            # Build OPeNDAP URL
            dataset_path = dataset.get("urlPath", "")
            if dataset_path:
                data_url = f"{server['base_url']}/dodsC/{dataset_path}"
            else:
                data_url = server["base_url"]

            source_id = f"thredds_{name[:40].replace('/', '_').replace(' ', '_')}"

            candidate = CandidateSource(
                source_id=source_id,
                name=f"{server['name']} — {name[:80]}",
                url=data_url,
                dataset_type=self._infer_type(name),
                variables_available=self._match_variables(name, request),
                spatial_coverage="See THREDDS catalog",
                temporal_coverage="See THREDDS catalog",
                available_formats=[DownloadFormat.netcdf],
                access_type=AccessType.free,
                requires_login=False,
                requires_payment=False,
                api_type=APIType.thredds,
                api_docs=server["catalog_url"],
                metadata_url=server["catalog_url"],
                description=f"THREDDS dataset: {name}",
                discovery_origin="thredds_discovery",
                catalog_authority_score=0.78,
                catalog_scientific_acceptance=0.75,
                catalog_historical_reliability=0.72,
            )
            candidates.append(candidate)
            count += 1

        return candidates

    def _infer_type(self, name: str) -> str:
        n = name.lower()
        if any(k in n for k in ["ocean", "sea", "sst", "salinity", "wave"]):
            return "ocean"
        if any(k in n for k in ["gfs", "weather", "atmosphere", "wind", "precip"]):
            return "weather"
        return "weather"

    def _match_variables(
        self,
        name: str,
        request: RetrievalRequest,
    ) -> List[str]:
        """CHANGED: matching also checks knowledge-base synonyms. Reports
        the original requested variable name on match."""
        name_lower = name.lower()
        matched = []
        for v in request.variables:
            search_terms = expand_variables([v.variable])
            if any(
                term.lower() in name_lower or term.lower().replace("_", " ") in name_lower
                for term in search_terms
            ):
                matched.append(v.variable)
        return matched if matched else ["unknown"]
