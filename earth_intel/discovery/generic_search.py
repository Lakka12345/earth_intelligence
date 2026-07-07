"""
Generic Search Discoverer — uses a web search API to find datasets
that don't live in any known catalog/ERDDAP/STAC/CKAN/THREDDS server.

Search backend priority:
  1. Tavily Search API   — TAVILY_API_KEY in env (free tier: 1000 req/month)
  2. Google Custom Search — GOOGLE_CSE_KEY + GOOGLE_CSE_CX in env
  3. Brave Search API    — BRAVE_SEARCH_KEY in env (free tier: 2000 req/month)
  (DuckDuckGo Instant Answer API removed — it rarely returns real results.)

After getting URLs the discoverer probes each one:
  - Detects API type from URL patterns AND response headers AND body content.
  - Many APIs don't show their type in the URL, so header/body inspection
    catches ERDDAP, STAC, THREDDS, CKAN instances that are URL-opaque.
  - Detects access requirements (login / payment keywords).
"""

import os
import re
import requests
from typing import List, Optional
from urllib.parse import urlparse

from discovery.base import BaseDiscoverer
from models.discovery_schemas import AccessType, APIType, CandidateSource, DownloadFormat
from models.retrieval_request import RetrievalRequest
from sources.knowledge_base import expand_variables


PROBE_TIMEOUT = 6       # seconds — applied to EVERY requests call

# URL-pattern → API type (fast path — matched before body inspection)
DATA_ENDPOINT_PATTERNS = [
    (re.compile(r"/erddap", re.I),              APIType.erddap),
    (re.compile(r"/thredds|/dodsC", re.I),      APIType.thredds),
    (re.compile(r"/stac|/collections|/items", re.I), APIType.stac),
    (re.compile(r"/api/3/action/package", re.I), APIType.ckan),
    (re.compile(r"/wms|/wfs|/wcs", re.I),       APIType.wms_wfs),
    (re.compile(r"/api/|/v1/|/v2/|/rest/", re.I), APIType.rest),
]

# Body / header signals — checked when URL alone is ambiguous
BODY_SIGNALS = [
    (re.compile(r'"erddapVersion"', re.I),       APIType.erddap),
    (re.compile(r'"stac_version"', re.I),        APIType.stac),
    (re.compile(r'service_type.*thredds|threddsConfig', re.I), APIType.thredds),
    (re.compile(r'"help".*"ckan"', re.I),        APIType.ckan),
    (re.compile(r'<link[^>]+rel="service-desc"', re.I), APIType.rest),  # OGC API
]

REGISTRATION_KEYWORDS = ["login", "register", "account", "sign up", "sign-up",
                          "earthdata", "copernicus", "authenticate"]
PAID_KEYWORDS = ["pricing", "purchase", "subscribe", "commercial",
                 "license fee", "pay per", "checkout"]


class GenericSearchDiscoverer(BaseDiscoverer):

    def search(self, request: RetrievalRequest) -> List[CandidateSource]:
        query = self._build_query(request)
        urls = self._web_search(query)

        candidates = []
        for url in urls[:10]:
            candidate = self._probe_url(url, request)
            if candidate:
                candidates.append(candidate)

        print(f"[GenericSearchDiscoverer] Found {len(candidates)} dataset(s) via web search.")
        return candidates

    def _build_query(self, request: RetrievalRequest) -> str:
        # CHANGED: expand to scientific synonyms before building the web
        # search query. This is the highest-leverage discoverer for
        # expansion -- it's the only general-purpose external search
        # path, used specifically to find providers not already in the
        # registry, so a richer query (e.g. "harmful algal bloom" ->
        # also searching "chlorophyll-a", "ocean colour") materially
        # improves what gets found.
        expanded = expand_variables([v.variable for v in request.variables])
        variables = [t.replace("_", " ") for t in expanded[:3]]

        # spatial_requirements is always a dict in RetrievalRequest
        spatial = request.spatial_requirements
        region = ""
        if isinstance(spatial, dict):
            region = spatial.get("region_name", "") or spatial.get("ocean_basin", "")
        else:
            # Pydantic model — access attributes safely
            region = getattr(spatial, "region_name", "") or getattr(spatial, "ocean_basin", "")

        parts = variables
        if region:
            parts.append(str(region))
        parts.append("dataset api open data")

        return " ".join(parts)

    # ---------------------------------------------------------------- #
    # Search backends                                                   #
    # ---------------------------------------------------------------- #

    def _web_search(self, query: str) -> List[str]:
        """
        Tries backends in priority order.
        Returns at most 10 URLs.
        """
        # 1. Tavily (best quality for science/data queries, free tier)
        tavily_key = os.getenv("TAVILY_API_KEY")
        if tavily_key:
            try:
                urls = self._tavily_search(query, tavily_key)
                if urls:
                    return urls[:10]
            except Exception as exc:
                print(f"[GenericSearchDiscoverer] Tavily failed: {exc}")

        # 2. Google Custom Search
        google_key = os.getenv("GOOGLE_CSE_KEY")
        google_cx = os.getenv("GOOGLE_CSE_CX")
        if google_key and google_cx:
            try:
                urls = self._google_search(query, google_key, google_cx)
                if urls:
                    return urls[:10]
            except Exception as exc:
                print(f"[GenericSearchDiscoverer] Google CSE failed: {exc}")

        # 3. Brave Search API
        brave_key = os.getenv("BRAVE_SEARCH_KEY")
        if brave_key:
            try:
                urls = self._brave_search(query, brave_key)
                if urls:
                    return urls[:10]
            except Exception as exc:
                print(f"[GenericSearchDiscoverer] Brave Search failed: {exc}")

        print("[GenericSearchDiscoverer] No search backend configured. "
              "Set TAVILY_API_KEY, GOOGLE_CSE_KEY+GOOGLE_CSE_CX, or BRAVE_SEARCH_KEY.")
        return []

    def _tavily_search(self, query: str, api_key: str) -> List[str]:
        """
        Tavily Search API — https://docs.tavily.com
        Free tier: 1000 requests/month. No credit card required.
        Sign up at: https://tavily.com
        """
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": 10,
                "include_answer": False,
            },
            timeout=PROBE_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return [r["url"] for r in results if "url" in r]

    def _google_search(self, query: str, api_key: str, cx: str) -> List[str]:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"q": query, "key": api_key, "cx": cx, "num": 10},
            timeout=PROBE_TIMEOUT,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return [item["link"] for item in items if "link" in item]

    def _brave_search(self, query: str, api_key: str) -> List[str]:
        """
        Brave Search API — https://brave.com/search/api
        Free tier: 2000 requests/month.
        """
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": 10},
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            },
            timeout=PROBE_TIMEOUT,
        )
        resp.raise_for_status()
        web = resp.json().get("web", {})
        results = web.get("results", [])
        return [r["url"] for r in results if "url" in r]

    # ---------------------------------------------------------------- #
    # URL probing — detects API type from URL + headers + body         #
    # ---------------------------------------------------------------- #

    def _probe_url(
        self,
        url: str,
        request: RetrievalRequest,
    ) -> Optional[CandidateSource]:
        """
        Visit a URL and classify it as a data source.

        Detection order:
          1. URL pattern match (fast, no body read needed)
          2. Response headers (Content-Type, X-Powered-By, Link rel)
          3. Body snippet — first 2KB only (covers JSON/HTML APIs)

        Returns CandidateSource if it looks like a data endpoint, else None.
        """
        try:
            # Use GET with stream=True so we can read just the first chunk
            resp = requests.get(
                url,
                timeout=PROBE_TIMEOUT,
                allow_redirects=True,
                stream=True,
                headers={
                    "User-Agent": "EarthIntelligenceAgent/1.0",
                    "Accept": "application/json, text/html, */*",
                },
            )
            final_url = resp.url

            # Read only first 2KB for body inspection — avoids large downloads
            body_snippet = resp.raw.read(2048).decode("utf-8", errors="ignore")
            resp.close()

            api_type = self._detect_api_type(final_url, resp.headers, body_snippet)

            # Access classification
            url_lower = final_url.lower()
            body_lower = body_snippet.lower()
            combined = url_lower + " " + body_lower

            requires_login = any(k in combined for k in REGISTRATION_KEYWORDS)
            requires_payment = any(k in combined for k in PAID_KEYWORDS)
            access_type = (
                AccessType.paid if requires_payment
                else AccessType.registration if requires_login
                else AccessType.free
            )

            # Build source_id from hostname
            parsed = urlparse(final_url)
            host_slug = (parsed.hostname or "unknown").replace(".", "_").replace("-", "_")
            path_slug = parsed.path.strip("/").replace("/", "_")[:30]
            source_id = f"web_{host_slug}_{path_slug}"[:60]

            variables = [v.variable for v in request.variables[:5]]

            return CandidateSource(
                source_id=source_id,
                name=f"Web discovery: {parsed.hostname}",
                url=final_url,
                dataset_type=self._infer_type_from_url(final_url, request),
                variables_available=variables,
                spatial_coverage="Unknown — from web search",
                temporal_coverage="Unknown — from web search",
                available_formats=[DownloadFormat.unknown],
                access_type=access_type,
                requires_login=requires_login,
                requires_payment=requires_payment,
                price_estimate="Unknown — check source" if requires_payment else None,
                login_url=final_url if requires_login else None,
                api_type=api_type,
                api_docs=final_url,
                description=f"Discovered via web search for: {', '.join(variables[:3])}",
                discovery_origin="generic_search",
                catalog_authority_score=0.50,
                catalog_scientific_acceptance=0.45,
                catalog_historical_reliability=0.40,
            )

        except Exception:
            return None

    def _detect_api_type(
        self,
        url: str,
        headers,
        body_snippet: str,
    ) -> APIType:
        """
        Three-pass API type detection:
          Pass 1 — URL pattern (cheapest)
          Pass 2 — Response headers
          Pass 3 — Body snippet (first 2KB)
        """
        # Pass 1: URL
        for pattern, detected_type in DATA_ENDPOINT_PATTERNS:
            if pattern.search(url):
                return detected_type

        # Pass 2: Headers
        content_type = headers.get("Content-Type", "").lower()
        powered_by = headers.get("X-Powered-By", "").lower()
        link_header = headers.get("Link", "").lower()

        if "erddap" in powered_by:
            return APIType.erddap
        if "ckan" in powered_by:
            return APIType.ckan
        if "application/geo+json" in content_type or "stac" in link_header:
            return APIType.stac
        if "netcdf" in content_type or "opendap" in content_type:
            return APIType.opendap
        if "wms" in link_header or "wfs" in link_header:
            return APIType.wms_wfs

        # Pass 3: Body
        for pattern, detected_type in BODY_SIGNALS:
            if pattern.search(body_snippet):
                return detected_type

        return APIType.rest

    def _infer_type_from_url(self, url: str, request: RetrievalRequest) -> str:
        url_lower = url.lower()
        if any(k in url_lower for k in ["ocean", "marine", "sea", "erddap", "incois", "cmems"]):
            return "ocean"
        if any(k in url_lower for k in ["weather", "atmosphere", "meteo", "noaa", "imd"]):
            return "weather"
        if any(k in url_lower for k in ["cyclone", "disaster", "flood", "hazard"]):
            return "disaster"
        if request.dataset_requirements:
            return request.dataset_requirements[0].dataset_type
        return "unknown"
