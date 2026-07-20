"""
Agent 3 — Data Discovery Agent

CHANGED from original:
  Phase 1 now does THREE parallel source lookups instead of one:
    1a. Hardcoded catalog (source_catalog.py)          — as before
    1b. DiscoveryEngine (ERDDAP/STAC/CKAN/THREDDS/Web) — NEW
    1c. Qdrant similar-source search                   — NEW (was only reliability lookup)

  After deduplication, phases 2-8 remain exactly the same.

  Phase 7 shows ranked results with access classification.
  Phase 8 (download gate) has been REMOVED — Agent 3 is a discovery-only
  agent. Source selection and retrieval planning are Agent 4's responsibility.
  Agent 3 returns DiscoveryOutput directly to Agent 4 with no user gate.

Flow:
  1a. Catalog query          — pure Python (domain-aware for flood queries)
  1b. Dynamic discovery      — pure Python (network calls, isolated)
  1c. Qdrant similar sources — pure Python
  → Deduplicate
  2.  Qdrant reliability enrichment  — pure Python
  3.  Deep metadata enrichment       — pure Python (HEAD + GET + STAC/CKAN/ERDDAP)
  3b. Provider credibility check     — pure Python (domain-aware scoring boost)
  3c. Coverage analysis              — pure Python (spatial + temporal annotation)
  3d. Variable matching              — pure Python (variable overlap scoring)
  4.  LLM scoring            — BATCHED LLM CALLS (every candidate is actually scored;
                                no candidate is dropped or given placeholder scores
                                just to fit inside one prompt's token budget)
  5.  Four-status ranking    — pure Python (Accepted / Auth-Required / Needs-Eval / Rejected;
                                rejection requires real evidence of unsuitability —
                                wrong domain, broken/empty resource, duplicate, or a
                                confident geographic mismatch — never just "score is low"
                                or "metadata is incomplete")
  6.  Store to Qdrant        — pure Python
  7.  Present results        — pure Python (four buckets displayed; every dataset ranked)
  → Return DiscoveryOutput to Agent 4 (no download gate)
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import requests
from groq import Groq
from pydantic import ValidationError

from discovery.engine import DiscoveryEngine
from discovery.geo_validator import validate_geography
from discovery.temporal_validator import validate_temporal
from discovery.platform_validator import (
    infer_candidate_platform,
    infer_requested_platform,
    validate_platform,
)
from memory import qdrant_store
from models.discovery_schemas import (
    AccessType,
    CandidateSource,
    DiscoveryOutput,
    DiscoveryResult,
    DownloadFormat,
    LLMCandidateScore,
    LLMScoringOutput,
    ParameterScore,
    ScoredSource,
    SourceRecommendation,
    SourceScoreCard,
    SourceStatus,
)
from models.retrieval_request import RetrievalRequest
from prompts.agent3_prompt import build_scoring_prompt, parse_scoring_response
from sources.source_catalog import get_candidates_for_request
from sources.knowledge_base import (
    expand_variables,
    get_preferred_providers,
    get_sensor_instruments,
    is_flood_related,
)
from sources.providers import (
    ALL_PROVIDERS,
    get_provider,
    get_providers_by_dataset_type,
    get_providers_for_variables,
    get_providers_for_flood_domain,
    is_flood_related_query,
    ProviderDefinition,
)


# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
PROBE_TIMEOUT_SECONDS = 5

# CHANGED: This used to be a blunt score cutoff -- any candidate whose
# weighted final_score fell under this number was auto-rejected, even
# if it was a perfectly relevant dataset that simply scored lower than
# others. That violated "no relevant dataset should be rejected simply
# because another dataset has a higher score." It is no longer used as
# a rejection trigger; low scores now just mean a low rank. Kept here
# only as a documented historical constant in case of rollback.
MIN_SCORE_TO_PRESENT = 0.30

# A genuinely unsuitable dataset is rejected based on the LLM's own
# failed_criteria + rejection_confidence fields (per prompts/agent3_prompt.py),
# not on SourceRecommendation, which in this project only has "use"/"consider".
# See REJECTION_CONFIDENCE_FLOOR inside phase5_rank_sources.

# Singleton discovery engine — created once, reused across calls
_discovery_engine: Optional[DiscoveryEngine] = None


def _get_discovery_engine() -> DiscoveryEngine:
    global _discovery_engine
    if _discovery_engine is None:
        _discovery_engine = DiscoveryEngine(enable_generic_search=True)
    return _discovery_engine


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _get_dataset_types_from_request(request: RetrievalRequest) -> List[str]:
    types_set = set()
    for req in request.dataset_requirements:
        types_set.add(req.dataset_type.lower().strip())
    return list(types_set)


def _get_variables_from_request(request: RetrievalRequest) -> List[str]:
    """
    Extract all variables from the request.

    CHANGED: Location and Time are now ALWAYS injected as mandatory variables
    regardless of whether the user explicitly listed them.  They are structural
    metadata required for every retrieval task (Agent 4 needs a spatial anchor
    and a temporal anchor to build any download query), yet they live in
    spatial_requirements / temporal_requirements — NOT in request.variables or
    request.measurements — so the old implementation was silently omitting them.
    This caused Agent 4 to report Location and Time as "Missing / no ranked
    Agent 3 source contributed this variable" for every query, dragging coverage
    down to 33 % even when the primary scientific variable was retrieved cleanly.
    """
    variables = set()

    # User-declared variables and measurements (unchanged)
    for var in request.variables:
        variables.add(var.variable.lower().strip())
    for meas in request.measurements:
        variables.add(meas.variable_measured.lower().strip())

    # ── Mandatory structural variables ──────────────────────────────────
    # Location: inject the concrete place name so source-matching can use
    # it, plus the generic token so downstream variable-coverage checks
    # always find at least one "location" entry.
    location_val = (
        request.spatial_requirements.get("location", "") or ""
    ).strip().lower()
    if location_val and location_val not in ("unknown", "unspecified", ""):
        variables.add(location_val)   # e.g. "bay of bengal", "chennai coast"
    variables.add("location")         # always present as a coverage anchor

    # Time: same pattern — concrete range + generic anchor.
    time_val = (
        request.temporal_requirements.get("date_range", "") or ""
    ).strip().lower()
    if time_val and time_val not in ("unknown", "unspecified", ""):
        variables.add(time_val)
    variables.add("time")             # always present as a coverage anchor

    return list(variables)


def _deduplicate_candidates(candidates: List[CandidateSource]) -> List[CandidateSource]:
    """Remove duplicates by URL. Catalog sources win over dynamic ones."""
    seen_urls = set()
    seen_ids = set()
    unique = []
    for c in candidates:
        url_key = c.url.rstrip("/").lower()
        if url_key in seen_urls or c.source_id in seen_ids:
            continue
        seen_urls.add(url_key)
        seen_ids.add(c.source_id)
        unique.append(c)
    return unique


# URL path segments that indicate a machine-readable data API endpoint.
# Any candidate whose URL does NOT contain at least one of these is likely
# a human-facing landing page and will waste Agent 4's time.
_DATA_API_PATH_MARKERS = ("/api/", "/thredds/", "/erddap/", "/stac/", "/wcs", "/wms",
                          "/wfs", "/ows", "/opendap/", "/dods/", "/catalog/", "/ckan/",
                          "/griddap/", "/tabledap/", "/ftp/")

# TLDs whose bare-domain form (scheme + host only, no meaningful path) are
# routinely returned by the LLM as homepage stand-ins. Extend this list if
# new TLD patterns appear in practice.
_BARE_DOMAIN_TLDS = (
    ".org", ".com", ".net", ".int", ".in", ".gov", ".edu",
    ".io", ".co", ".info", ".eu", ".de", ".uk", ".fr", ".au",
)

# Known source_ids → canonical API base URLs.
# When a bare-domain URL is detected for a source whose real endpoint is
# known here, the URL is corrected rather than the candidate being dropped.
# Add entries whenever a new provider starts appearing as a bare domain.
_CATALOG_ENDPOINT_OVERRIDES: Dict[str, str] = {
    "gdacs":         "https://www.gdacs.org/gdacsapi/api/",
    "reliefweb":     "https://api.reliefweb.int/v1/",
    "openaq":        "https://api.openaq.org/v2/",
    "noaa_ncei":     "https://www.ncei.noaa.gov/access/services/data/v1/",
    "nasa_earthdata":"https://cmr.earthdata.nasa.gov/search/",
    # CHANGED: a single "copernicus" key matched every Copernicus-branded
    # source by substring (Data Space, Land Monitoring, CDS all contain
    # "copernicus" in their name), silently rewriting all of them to the
    # same CDS endpoint whenever any of them tripped the bare-domain
    # check. Each Copernicus service is a distinct product with its own
    # API base -- these must never collapse into one key. Order matters:
    # more specific keys are checked first (see loop below).
    "copernicus climate data store": "https://cds.climate.copernicus.eu/api/",
    "copernicus cds":                "https://cds.climate.copernicus.eu/api/",
    "copernicus data space":         "https://catalogue.dataspace.copernicus.eu/resto/",
    "copernicus land":               "https://land.copernicus.eu/api/",
    "copernicus marine":             "https://data.marine.copernicus.eu/api/",
    "copernicus atmosphere":         "https://ads.atmosphere.copernicus.eu/api/",
    "usgs":          "https://waterservices.usgs.gov/nwis/",
    "ecmwf":         "https://api.ecmwf.int/v1/",
    "imd":           "https://internal.imd.gov.in/section/nhac/dynamic/",
    "incois":        "https://incois.gov.in/erddap/",
    "discomap":      "https://discomap.eea.europa.eu/arcgis/rest/services/",
    "eosdis":        "https://cmr.earthdata.nasa.gov/search/",
    "worldbank":     "https://api.worldbank.org/v2/",
    "ocha":          "https://api.hpc.tools/v2/",
}


# Path segments that mark a page as a human-facing "browse/download" landing
# page rather than a machine-readable endpoint, even when the path has two
# or more segments (e.g. /en/open-data, /data/downloads, /products/catalogue).
# These are checked ONLY when the URL has no API marker and no data-file
# extension — a URL like /erddap/griddap/x.nc still passes fine.
_LANDING_PAGE_SEGMENT_MARKERS = (
    "open-data", "opendata", "open_data", "downloads", "download",
    "datasets", "dataset", "data-portal", "dataportal", "resources",
    "products", "catalogue", "library", "publications", "en", "home",
)


def _is_landing_page_url(url: str) -> bool:
    """
    Returns True when a URL's path is composed entirely of "landing page"
    segments (browse/marketing pages) and it carries no API marker and no
    data-file extension. This catches multi-segment landing pages that
    _is_bare_domain_url misses, e.g. https://www.deltares.nl/en/open-data.
    """
    from urllib.parse import urlparse
    lower = url.lower()

    has_api_marker = any(marker in lower for marker in _DATA_API_PATH_MARKERS)
    has_data_ext = any(lower.rstrip("/").endswith(ext) for ext in
                        (".nc", ".nc4", ".hdf", ".h5", ".csv", ".json", ".tif", ".tiff",
                         ".grib", ".grb", ".grb2", ".zip", ".gz", ".tar"))
    if has_api_marker or has_data_ext:
        return False

    try:
        path = urlparse(url).path.rstrip("/")
    except Exception:
        return False

    segments = [s for s in path.split("/") if s]
    if not segments:
        return False  # handled by _is_bare_domain_url already

    return all(seg.lower() in _LANDING_PAGE_SEGMENT_MARKERS for seg in segments)


def _resolve_landing_page_to_data_endpoint(
    url: str, timeout: int = PROBE_TIMEOUT_SECONDS
) -> Optional[str]:
    """
    Lightweight, best-effort crawl of an HTML landing page to find a real
    data endpoint linked from it — a .nc/.csv/.json/.tif file, or an
    ERDDAP/THREDDS/CKAN/STAC/WCS/WFS entry point. Returns the first strong
    match found, or None if nothing usable is on the page.

    This is intentionally cheap (single GET, one page, no recursion) so it
    never becomes the bottleneck of Phase 1. It exists so that a landing
    page isn't just dropped when it is, in fact, one click away from the
    real endpoint — which is common for provider "open data" pages that
    link out to their ERDDAP/THREDDS server or a downloadable file.
    """
    import re
    from urllib.parse import urljoin

    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "EarthIntelligenceAgent/1.0"})
        if resp.status_code >= 400 or "text/html" not in resp.headers.get("Content-Type", "").lower():
            return None
        html = resp.text
    except Exception:
        return None

    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    candidates = [urljoin(url, h) for h in hrefs]

    # Prefer explicit data-file links first, then API/service entry points.
    data_exts = (".nc", ".nc4", ".hdf", ".h5", ".csv", ".json", ".tif", ".tiff",
                 ".grib", ".grb", ".grb2")
    for link in candidates:
        if link.lower().rstrip("/").endswith(data_exts):
            return link
    for link in candidates:
        if any(marker in link.lower() for marker in _DATA_API_PATH_MARKERS):
            if not _is_portal_url(link):
                return link
    return None


def _is_bare_domain_url(url: str) -> bool:
    """
    Returns True when a URL is nothing more than a bare homepage —
    i.e. the path component is empty, '/', or a single short slug with
    no data-API meaning.

    Logic (applied after stripping scheme and query/fragment):
      1. Parse out the path.  A URL with no path or path == '/' is always bare.
      2. If the path has two or more non-empty segments (e.g. /v1/datasets),
         it is NOT bare — something meaningful is there.
      3. For single-segment paths (e.g. /gdacs, /home, /en):
         - If the host ends with a known landing-page TLD AND the segment
           is five characters or fewer (a typical homepage slug like /en,
           /us, /in), treat it as bare.
         - Otherwise, give benefit of the doubt and call it non-bare.

    This is intentionally conservative: it only rejects the clearest cases
    (pure homepages) so valid single-path API roots like /erddap are kept.
    """
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    path = parsed.path.rstrip("/")

    # No path at all — definitely bare
    if not path:
        return True

    segments = [s for s in path.split("/") if s]

    # Two or more non-empty segments → assume it has real structure
    if len(segments) >= 2:
        return False

    # Single segment: bare only if TLD is in our landing-page list AND the
    # segment is very short (homepage slug, language code, country code, etc.)
    host = parsed.netloc.lower()
    if any(host.endswith(tld) for tld in _BARE_DOMAIN_TLDS):
        return len(segments[0]) <= 5

    return False


# URL path/query patterns that look like an API path marker but actually
# serve HTML map portals or tile viewers rather than raw data.
# Checked AFTER _DATA_API_PATH_MARKERS so we can specifically veto
# WMS/WFS endpoints that are viewer-only (no SERVICE=WMS&REQUEST=GetMap
# or GetFeature parameters attached) and generic map portal paths.
#
# Rule: if a URL contains a marker from _DATA_API_PATH_MARKERS BUT also
# matches a pattern here, it is rejected as a portal.  Extend this list
# as new interactive-portal URL shapes appear in practice.
_HTML_PORTAL_PATH_PATTERNS: tuple = (
    # Bare /wms or /wms/ with no query string → map viewer, not GetMap
    "/wms",
    "/wfs",
    "/wcs",
    # Common interactive map portal path segments
    "/bhuvan/",          # NRSC Bhuvan portal
    "/viewer/",
    "/mapviewer",
    "/portal/",
    "/geoportal/",
    "/webmap/",
    "/explore/",
    "/dashboard/",
    "/visualization/",
    "/interactive/",
)

# Query-string parameters that confirm a WMS/WFS URL is a *real* data
# request rather than a viewer load.  If any of these are present the
# URL is kept even if a portal path pattern matched.
_WMS_DATA_PARAMS: tuple = (
    "request=getmap",
    "request=getfeature",
    "request=getcoverage",
    "request=getdata",
    "request=getcapabilities",   # capabilities XML is machine-readable
    "service=wfs",
    "service=wcs",
    "outputformat=",
    "typenames=",
    "typename=",
)


def _is_portal_url(url: str) -> bool:
    """
    Returns True when a URL contains an API-looking path segment (WMS,
    WFS, /portal/, /viewer/, etc.) but carries no query parameters that
    would make it a real machine-readable data request.

    Examples that return True (portal / viewer):
      https://bhuvan-app1.nrsc.gov.in/bhuvan/wms
      https://example.org/viewer/map
      https://portal.example.com/geoportal/home

    Examples that return False (real data request):
      https://example.org/wms?SERVICE=WMS&REQUEST=GetMap&LAYERS=flood&...
      https://host/ows?service=WFS&request=GetFeature&typeName=...
    """
    lower = url.lower()

    # Only relevant when a portal path pattern is present
    if not any(pat in lower for pat in _HTML_PORTAL_PATH_PATTERNS):
        return False

    # If the URL already has proper WMS/OGC data parameters, it is fine
    if any(param in lower for param in _WMS_DATA_PARAMS):
        return False

    # Portal path present + no data query params → it's a viewer
    return True


def _is_api_url(url: str) -> bool:
    """
    Returns True if the URL looks like a machine-readable data endpoint.

    A URL passes if ALL of:
      1. It contains a known API path marker (_DATA_API_PATH_MARKERS), OR
         ends with a recognised data-file extension.
      2. It does NOT match _is_portal_url() — i.e. it is not a bare WMS
         viewer, map portal, or interactive dashboard.

    NOTE: origin-based trust ("catalog", "provider_registry") has been
    removed from this function.  Those origins are still kept by
    _filter_landing_pages, but ONLY after _is_bare_domain_url() clears
    them — this prevents the LLM from smuggling bare homepages through
    the filter simply by labelling them as catalog sources.
    """
    lower = url.lower()

    has_api_marker = any(marker in lower for marker in _DATA_API_PATH_MARKERS)
    has_data_ext   = any(lower.rstrip("/").endswith(ext) for ext in
                         (".nc", ".nc4", ".hdf", ".h5", ".csv", ".json", ".tif", ".tiff",
                          ".grib", ".grb", ".grb2", ".zip", ".gz", ".tar"))

    if not (has_api_marker or has_data_ext):
        return False

    # Veto portal/viewer URLs even if they have an API-looking path
    if _is_portal_url(url):
        return False

    return True


def _resolve_catalog_endpoint(candidate: CandidateSource) -> Optional[str]:
    """
    Looks up a known API endpoint for a candidate whose URL was detected as
    a bare domain, using _CATALOG_ENDPOINT_OVERRIDES.

    Matching is done against the candidate's source_id and name (both
    lowercased, with hyphens/underscores normalised to spaces) so that
    "gdacs", "gdacs_api", "GDACS", "GDACS Disaster Alerting" all match.

    Returns the replacement URL string, or None if no override is known.
    """
    def _normalise(s: str) -> str:
        return s.lower().replace("-", " ").replace("_", " ")

    sid   = _normalise(candidate.source_id)
    name  = _normalise(candidate.name)

    # Sort by key length descending so a more specific key (e.g.
    # "copernicus land") is always checked before a shorter, broader one
    # that might also appear as a substring -- prevents any future
    # re-introduction of the same collision bug.
    for key, endpoint in sorted(_CATALOG_ENDPOINT_OVERRIDES.items(), key=lambda kv: -len(kv[0])):
        norm_key = _normalise(key)
        if norm_key in sid or norm_key in name:
            return endpoint
    return None


def _filter_landing_pages(candidates: List[CandidateSource]) -> List[CandidateSource]:
    """
    Strict two-stage URL filter applied to ALL candidates regardless of origin.

    Stage 1 — Bare-domain check (_is_bare_domain_url):
        Detects URLs that are nothing more than a homepage
        (e.g. https://www.gdacs.org, https://reliefweb.int).
        Applied universally — even "catalog" / "provider_registry" / "qdrant_cache"
        sources are subject to this check, because the LLM sometimes returns a
        bare homepage labelled as a trusted-origin source, which previously
        bypassed all filtering.

        When a bare URL is found:
          a) If _resolve_catalog_endpoint() knows the real API base, the URL
             is CORRECTED in-place and the candidate is kept (rescued).
          b) Otherwise the candidate is DROPPED with a log entry.

    Stage 2 — API path + portal check (_is_api_url / _is_portal_url):
        For non-bare URLs, confirms the URL contains a recognisable API path
        segment or data-file extension AND is not a bare WMS viewer or
        interactive map portal.  Curated-origin URLs that passed Stage 1 are
        also subject to the portal veto — a Bhuvan /wms landing page is an
        HTML portal regardless of whether the LLM labelled it "catalog".

    Stage 3 — Live content-type sniff (dynamic + curated portal suspects):
        For any URL that passed Stages 1–2 but whose path contains a WMS/OGC
        marker, a lightweight HEAD request checks the Content-Type header.
        If the server responds with text/html (map viewer) rather than
        application/xml, application/json, or a binary data type, the
        candidate is dropped.  Non-fatal: if the HEAD request times out or
        errors, the candidate is kept (benefit of the doubt).

    Logs a summary so the operator can see what was filtered or rescued.
    """
    import urllib.request

    def _head_returns_html(url: str, timeout: int = 4) -> bool:
        """HEAD the URL; return True only when Content-Type is unambiguously HTML."""
        try:
            req = urllib.request.Request(url, method="HEAD",
                                         headers={"User-Agent": "EarthIntelligenceAgent/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                ct = resp.headers.get("Content-Type", "").lower()
            return "text/html" in ct and "xml" not in ct and "json" not in ct
        except Exception:
            return False  # timeout / connection error → keep the candidate

    # WMS/OGC markers whose bare form often serves a map viewer HTML page
    _WMS_MARKERS = ("/wms", "/wfs", "/wcs", "/ows")

    kept:    List[CandidateSource] = []
    rescued: List[str] = []   # (name, old_url → new_url)
    dropped: List[CandidateSource] = []

    for c in candidates:
        origin = getattr(c, "discovery_origin", "") or ""
        was_rescued_by_override = False

        # ── Stage 1: bare-domain check (ALL origins) ─────────────────
        if _is_bare_domain_url(c.url):
            replacement = _resolve_catalog_endpoint(c)
            if replacement:
                rescued.append(f"{c.name}  ({c.url} → {replacement})")
                c.url = replacement
                was_rescued_by_override = True
                # fall through to Stage 2 with the corrected URL
            else:
                dropped.append(c)
                continue   # skip Stage 2 — nothing to salvage

        # ── Stage 2: API path + portal veto (ALL origins) ────────────
        # Portal check applies even to curated sources — a Bhuvan /wms
        # viewer is an HTML portal regardless of discovery_origin label.
        if _is_portal_url(c.url):
            dropped.append(c)
            continue

        # ── Stage 2b: multi-segment landing pages (ALL origins) ──────
        # Catches pages like /en/open-data that _is_bare_domain_url
        # misses because they have 2+ path segments. Try a cheap crawl
        # to rescue the real endpoint before giving up on the source —
        # this is the main fix for HTML pages ending up in Qdrant instead
        # of the actual download/API endpoint.
        if not was_rescued_by_override and _is_landing_page_url(c.url):
            resolved = _resolve_landing_page_to_data_endpoint(c.url)
            if resolved:
                rescued.append(f"{c.name}  ({c.url} → {resolved})")
                c.url = resolved
            else:
                dropped.append(c)
                continue

        # CHANGED: curated origins ("catalog", "provider_registry",
        # "qdrant_cache") no longer get a blanket bypass here. That
        # bypass is exactly what let landing pages like
        # https://www.deltares.nl/en/open-data reach Qdrant when the
        # LLM/catalog labelled them as a trusted origin. Curated origins
        # are still trusted for *authority scoring* elsewhere, but the
        # URL itself must still look like a real data endpoint.
        #
        # CHANGED AGAIN: a URL that was just rescued via
        # _CATALOG_ENDPOINT_OVERRIDES is a hand-curated, manually verified
        # real API endpoint (that's the entire purpose of maintaining that
        # dict) -- re-vetoing it with the generic _is_api_url() marker
        # list (which only recognises a fixed set of path segments like
        # "/api/", "/erddap/", "/dods/") was dropping genuinely correct,
        # rescued endpoints like IMD's "/section/nhac/dynamic/",
        # Copernicus Data Space's "/resto/", ReliefWeb's "/v1/", NASA
        # Earthdata's "/search/", and USGS's "/nwis/" immediately after
        # rescuing them -- silently undoing the rescue for every provider
        # whose real path shape doesn't happen to match that narrow list.
        # Trust the override; skip the generic heuristic for these.
        if not was_rescued_by_override and not _is_api_url(c.url):
            dropped.append(c)
            continue

        # ── Stage 3: live HEAD sniff for WMS/OGC borderline cases ────
        url_lower = c.url.lower()
        if any(m in url_lower for m in _WMS_MARKERS):
            if _head_returns_html(c.url):
                dropped.append(c)
                print(f"  [Phase 1 filter] HEAD probe: {c.name} returned text/html — dropped.")
                continue

        kept.append(c)

    if rescued:
        print(
            f"[Phase 1 filter] Rescued {len(rescued)} bare-domain URL(s) "
            f"using catalog endpoint overrides:"
        )
        for note in rescued:
            print(f"  ↳ {note}")

    if dropped:
        print(
            f"[Phase 1 filter] Dropped {len(dropped)} bare-domain / landing-page URL(s) "
            f"(no API path, no catalog override): "
            + ", ".join(c.name for c in dropped[:5])
            + (" …" if len(dropped) > 5 else "")
        )

    return kept


# ------------------------------------------------------------------ #
# Phase 1a — Catalog query (unchanged)                                #
# ------------------------------------------------------------------ #

def _provider_to_candidate(p: ProviderDefinition) -> CandidateSource:
    """Convert a ProviderDefinition into a CandidateSource for the scoring pipeline."""
    from models.discovery_schemas import APIType as AT, DownloadFormat
    # Map discoverer_type string to APIType enum
    api_map = {
        "erddap": AT.erddap, "stac": AT.stac, "ckan": AT.ckan,
        "thredds": AT.thredds, "cmr": AT.rest, "wcs_wms": AT.wms_wfs,
        "rest": AT.rest, "ftp": AT.ftp, "web": AT.rest,
    }
    return CandidateSource(
        source_id=p.source_id,
        name=p.name,
        url=p.base_url,
        dataset_type=p.dataset_types[0] if p.dataset_types else "unknown",
        variables_available=p.variables_hint,
        spatial_coverage="Global" if not p.regions else ", ".join(p.regions),
        temporal_coverage="Varies by dataset",
        available_formats=[DownloadFormat.unknown],
        access_type=p.access_type,
        requires_login=p.requires_login,
        requires_payment=p.requires_payment,
        price_estimate=p.price_estimate,
        login_url=p.login_url,
        api_docs=p.api_docs_url,
        api_type=api_map.get(p.discoverer_type, AT.rest),
        discovery_origin="provider_registry",
        description=p.description,
        catalog_authority_score=p.authority_score,
        catalog_scientific_acceptance=p.scientific_acceptance,
        catalog_historical_reliability=p.historical_reliability,
    )


def phase1a_catalog_query(request: RetrievalRequest) -> List[CandidateSource]:
    """
    Phase 1a — Queries both the hardcoded source catalog and the provider registry.

    Provider registry (providers.py) is the new source of truth — it contains
    every known data portal with access type, login URL, and API type.
    The old source_catalog.py is kept for backward-compat and its more detailed
    dataset-level entries (specific ERDDAP endpoints, etc.).

    CHANGED: Now consults the Scientific Knowledge Base
    (sources/knowledge_base.py) before falling back to plain type/variable
    matching:
      1. Requested variables are expanded into their scientific synonyms
         (e.g. "SST" -> "sea surface temperature", "ghrsst", "skin
         temperature") via expand_variables(), so the catalog and
         provider matching below see the richer term set, not just the
         user's exact wording.
      2. Variables that have a knowledge-base entry contribute their
         preferred_source_ids FIRST, in scientific priority order --
         this generalizes the flood-domain-only special case that
         existed before (is_flood_related_query /
         get_providers_for_flood_domain) into a reusable mechanism that
         now also applies to rainfall, SST, chlorophyll-a, NDVI, etc.
         The flood-specific routing is kept as an additional signal
         (is_flood_related, the KB's own domain-based check) layered on
         top, not replaced.
      3. Anything not covered by the knowledge base still falls back to
         the original type/variable matching, so behavior for
         unmapped variables is unchanged.
    """
    dataset_types = _get_dataset_types_from_request(request)
    variables = _get_variables_from_request(request)
    expanded_variables = expand_variables(variables)

    # 1. Old catalog — specific dataset-level entries. Uses the expanded
    #    variable set so catalog matching benefits from query expansion too.
    catalog_candidates = get_candidates_for_request(
        dataset_types=dataset_types,
        variables_needed=expanded_variables,
    )

    # 2. Provider registry — provider-level entries (all tiers)
    matched_providers: List[ProviderDefinition] = []
    seen_ids = {c.source_id for c in catalog_candidates}

    # 2a. Knowledge-base-preferred providers first, in scientific
    #     priority order, for every variable that has a KB entry.
    kb_preferred_ids = get_preferred_providers(variables)
    for source_id in kb_preferred_ids:
        provider = get_provider(source_id)
        if provider is not None and source_id not in seen_ids:
            matched_providers.append(provider)
            seen_ids.add(source_id)

    # 2b. Flood-domain routing (existing keyword check OR the
    #     knowledge base's own domain-based signal) adds any
    #     flood-priority providers not already covered above.
    if is_flood_related_query(dataset_types, variables) or is_flood_related(variables, dataset_types):
        for p in get_providers_for_flood_domain():
            if p.source_id not in seen_ids:
                matched_providers.append(p)
                seen_ids.add(p.source_id)

    # 2c. Standard type/variable matching fills in everything else --
    #     unmapped variables, and any provider the KB/flood routing
    #     didn't already surface. Uses the expanded variable set.
    type_matched = []
    for dt in dataset_types:
        type_matched.extend(get_providers_by_dataset_type(dt))

    var_matched = get_providers_for_variables(expanded_variables)

    combined = {p.source_id: p for p in type_matched + var_matched}
    for source_id, provider in combined.items():
        if source_id not in seen_ids:
            matched_providers.append(provider)
            seen_ids.add(source_id)

    provider_candidates = [_provider_to_candidate(p) for p in matched_providers]

    all_candidates = catalog_candidates + provider_candidates
    print(
        f"[Phase 1a] Catalog: {len(catalog_candidates)} entry/entries. "
        f"Provider registry: {len(provider_candidates)} provider(s) "
        f"({len(kb_preferred_ids)} from knowledge-base priority). "
        f"Total: {len(all_candidates)}."
    )
    return all_candidates


# ------------------------------------------------------------------ #
# Phase 1b — Dynamic discovery engine  ← NEW                         #
# ------------------------------------------------------------------ #
def phase1b_dynamic_discovery(request: RetrievalRequest) -> List[CandidateSource]:
    """
    Phase 1b — Runs all discovery plugins concurrently (ERDDAP, STAC,
    CKAN, THREDDS, generic web search) and returns new candidates.

    Non-fatal: if the entire engine fails, returns [].
    """
    try:
        engine = _get_discovery_engine()
        result = engine.search(request)           # DiscoveryResult now
        if result.failures:
            for msg in result.failures:
                print(f"[Phase 1b] Discoverer failure (non-fatal): {msg}")
        print(
            f"[Phase 1b] Dynamic discovery: {len(result.sources)} candidate(s) "
            f"in {result.discovery_time_seconds}s "
            f"via {', '.join(result.discoverers_used) or 'none'}."
        )
        return result.sources
    except Exception as exc:
        print(f"[Phase 1b] Dynamic discovery failed (non-fatal): {exc}")
        return []





# ------------------------------------------------------------------ #
# Phase 1c — Qdrant similar-source search  ← NEW                     #
# ------------------------------------------------------------------ #

def phase1c_qdrant_source_search(request: RetrievalRequest) -> List[CandidateSource]:
    """
    Phase 1c — Searches Qdrant for sources used in past similar queries
    and returns them as CandidateSource objects with from_qdrant_cache=True.

    This turns Qdrant from a "reliability lookup" into a full source
    discovery path for previously seen data providers.

    Non-fatal: if Qdrant is down, returns [].
    """
    if not qdrant_store.is_qdrant_available():
        print("[Phase 1c] Qdrant unavailable — skipping similar-source search.")
        return []

    try:
        dataset_types = _get_dataset_types_from_request(request)
        similar_past = qdrant_store.search_similar_discoveries(
            query_goal=request.goal,
            dataset_types=dataset_types,
        )

        candidates = []
        for record in similar_past:
            source_id = record.get("source_id")
            url = record.get("source_url")
            name = record.get("source_name")

            if not source_id or not url or not name:
                continue

            # Reconstruct a CandidateSource from stored payload
            candidate = CandidateSource(
                source_id=f"qdrant_{source_id}",
                name=name,
                url=url,
                dataset_type=record.get("dataset_type", "unknown"),
                variables_available=record.get("variables_available", []),
                spatial_coverage=record.get("spatial_coverage", "Unknown"),
                temporal_coverage=record.get("temporal_coverage", "Unknown"),
                available_formats=[],
                # Access fields — preserve what was stored, default free if missing
                access_type=AccessType(record.get("access_type", "free")),
                requires_login=record.get("requires_login", False),
                requires_payment=record.get("requires_payment", False),
                description=f"Previously used source from Qdrant memory.",
                discovery_origin="qdrant_cache",
                catalog_authority_score=record.get("authority_score", 0.5),
                catalog_scientific_acceptance=0.5,
                catalog_historical_reliability=record.get("final_score", 0.5),
                from_qdrant_cache=True,
                qdrant_historical_reliability=record.get("final_score"),
                # Restore health_score — degrades over time as Agent 4 reports failures
                health_score=record.get("health_score", 1.0),
            )
            candidates.append(candidate)

        print(f"[Phase 1c] Qdrant similar-source search returned {len(candidates)} candidate(s).")
        return candidates

    except Exception as exc:
        print(f"[Phase 1c] Qdrant source search failed (non-fatal): {exc}")
        return []


# ------------------------------------------------------------------ #
# Phase 2 — RAG — Qdrant reliability enrichment (unchanged logic)    #
# ------------------------------------------------------------------ #

def phase2_qdrant_enrichment(
    candidates: List[CandidateSource],
    request: RetrievalRequest,
) -> List[CandidateSource]:
    """
    Phase 2 — Enriches each candidate with reliability history from Qdrant.
    Logic unchanged. Now receives candidates from all three Phase 1 sources.
    """
    if not qdrant_store.is_qdrant_available():
        print("[Phase 2] Qdrant unavailable — skipping reliability enrichment.")
        return candidates

    for candidate in candidates:
        if candidate.from_qdrant_cache:
            continue   # already has Qdrant data from Phase 1c

        history = qdrant_store.get_source_reliability_history(candidate.source_id)
        if history:
            candidate.qdrant_historical_reliability = history.get("reliability_score", 0.5)
            candidate.qdrant_times_used = history.get("times_used", 0)
            candidate.qdrant_times_succeeded = history.get("times_succeeded", 0)
            candidate.from_qdrant_cache = True

    cache_count = sum(1 for c in candidates if c.from_qdrant_cache)
    print(f"[Phase 2] Reliability enriched {cache_count}/{len(candidates)} candidate(s).")
    return candidates


# ------------------------------------------------------------------ #
# Phase 3 — Deep metadata enrichment                                  #
#                                                                     #
# CHANGED: Was a HEAD-only latency probe. Now performs multi-layer    #
# metadata enrichment before any scoring occurs (Req 2 & 6):          #
#   Layer 1 — HEAD probe: latency + Last-Modified (unchanged)         #
#   Layer 2 — Landing page GET: description, variables, coverage      #
#             extracted from HTML <meta> tags and visible text         #
#   Layer 3 — Protocol-specific endpoints:                            #
#             STAC  → /collections or /collections/{id}               #
#             CKAN  → /api/3/action/package_show?id={id}              #
#             ERDDAP→ /info/{dataset_id}/index.json                   #
#             Generic metadata_url → JSON or HTML fetch               #
#   All layers are non-fatal: failure → fields stay "Unknown".        #
#   Missing metadata is NEVER a rejection reason (Req 3).             #
# ------------------------------------------------------------------ #

_METADATA_TIMEOUT = 8          # slightly longer than probe — full GET
_HTML_DESCRIPTION_CHARS = 400  # max chars to extract from page <meta>

_AGENT_HEADERS = {"User-Agent": "EarthIntelligenceAgent/1.0"}


def _probe_source(candidate: CandidateSource) -> Tuple[Optional[float], Optional[str]]:
    """Layer 1 — HEAD probe for latency and Last-Modified."""
    try:
        start = time.time()
        response = requests.head(
            candidate.url,
            timeout=PROBE_TIMEOUT_SECONDS,
            allow_redirects=True,
            headers=_AGENT_HEADERS,
        )
        latency_ms = (time.time() - start) * 1000
        last_modified = response.headers.get("Last-Modified")
        return round(latency_ms, 1), last_modified
    except Exception:
        return None, None


def _enrich_from_html(candidate: CandidateSource) -> None:
    """
    Layer 2 — GET the landing page and extract whatever metadata is
    publicly available from HTML <meta> tags and page text.

    Populates (only if currently Unknown/empty):
      description, variables_available, spatial_coverage,
      temporal_coverage, metadata_url
    """
    try:
        resp = requests.get(
            candidate.url,
            timeout=_METADATA_TIMEOUT,
            allow_redirects=True,
            headers=_AGENT_HEADERS,
        )
        if resp.status_code != 200:
            return
        html = resp.text

        import re
        if not candidate.description or len(candidate.description) < 20:
            for pattern in [
                r'<meta\s+name=["\']description["\']\s+content=["\'](.*?)["\']',
                r'<meta\s+property=["\']og:description["\']\s+content=["\'](.*?)["\']',
                r'<meta\s+content=["\'](.*?)["\']\s+name=["\']description["\']',
            ]:
                m = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
                if m:
                    candidate.description = m.group(1).strip()[:_HTML_DESCRIPTION_CHARS]
                    break

        if not candidate.metadata_url:
            m = re.search(
                r'href=["\'](https?://[^"\']*(?:metadata|catalog|dataset|info|documentation)[^"\']*)["\']',
                html, re.IGNORECASE,
            )
            if m:
                candidate.metadata_url = m.group(1)

        if candidate.temporal_coverage in ("Unknown", ""):
            year_range = re.search(
                r'\b((?:19|20)\d{2})\s*[-–to]+\s*((?:19|20)\d{2}|[Pp]resent)\b',
                html,
            )
            if year_range:
                candidate.temporal_coverage = (
                    f"{year_range.group(1)} – {year_range.group(2)}"
                )

    except Exception:
        pass   # non-fatal


def _enrich_from_stac(candidate: CandidateSource) -> None:
    """
    Layer 3a — STAC: try /collections and /collections/{source_id}.
    Populates description, spatial_coverage, temporal_coverage,
    variables_available, available_formats, metadata_url.
    """
    base = candidate.url.rstrip("/")
    urls_to_try = [
        f"{base}/collections",
        f"{base}/collections/{candidate.source_id}",
    ]
    for url in urls_to_try:
        try:
            resp = requests.get(url, timeout=_METADATA_TIMEOUT, headers=_AGENT_HEADERS)
            if resp.status_code != 200:
                continue
            data = resp.json()

            collections = data.get("collections", [data])
            if not isinstance(collections, list):
                collections = [collections]

            for col in collections[:3]:
                if not candidate.description or len(candidate.description) < 20:
                    candidate.description = (col.get("description") or col.get("title") or "")[:_HTML_DESCRIPTION_CHARS]

                if candidate.spatial_coverage in ("Unknown", ""):
                    bbox = (
                        col.get("extent", {})
                           .get("spatial", {})
                           .get("bbox", [[]])[0]
                    )
                    if bbox and len(bbox) >= 4:
                        candidate.spatial_coverage = (
                            f"bbox [{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}]"
                        )

                if candidate.temporal_coverage in ("Unknown", ""):
                    interval = (
                        col.get("extent", {})
                           .get("temporal", {})
                           .get("interval", [[None, None]])[0]
                    )
                    if interval and interval[0]:
                        end = interval[1] or "present"
                        candidate.temporal_coverage = f"{interval[0][:10]} – {end[:10] if end != 'present' else 'present'}"

                if not candidate.variables_available:
                    summaries = col.get("summaries", {})
                    eo_bands = summaries.get("eo:bands", [])
                    vars_found = [b.get("name") or b.get("common_name") for b in eo_bands if isinstance(b, dict)]
                    vars_found = [v for v in vars_found if v]
                    if vars_found:
                        candidate.variables_available = vars_found[:20]

                if not candidate.metadata_url:
                    for link in col.get("links", []):
                        if link.get("rel") in ("self", "alternate", "describedby"):
                            candidate.metadata_url = link.get("href")
                            break

            break   # stop after first successful STAC response

        except Exception:
            continue


def _enrich_from_ckan(candidate: CandidateSource) -> None:
    """
    Layer 3b — CKAN: try /api/3/action/package_show?id={source_id}.
    Populates description, variables_available, temporal_coverage,
    spatial_coverage, available_formats, metadata_url.
    """
    base = candidate.url.rstrip("/")
    if "/dataset/" in base:
        base = base.split("/dataset/")[0]
    url = f"{base}/api/3/action/package_show?id={candidate.source_id}"
    try:
        resp = requests.get(url, timeout=_METADATA_TIMEOUT, headers=_AGENT_HEADERS)
        if resp.status_code != 200:
            return
        result = resp.json().get("result", {})
        if not result:
            return

        if not candidate.description or len(candidate.description) < 20:
            candidate.description = (result.get("notes") or result.get("title") or "")[:_HTML_DESCRIPTION_CHARS]

        if candidate.temporal_coverage in ("Unknown", ""):
            start = result.get("temporal_start") or result.get("data_collection_start_date") or ""
            end   = result.get("temporal_end")   or result.get("data_collection_end_date")   or "present"
            if start:
                candidate.temporal_coverage = f"{start} – {end}"

        if candidate.spatial_coverage in ("Unknown", ""):
            bbox = result.get("spatial") or result.get("bbox") or ""
            if bbox:
                candidate.spatial_coverage = str(bbox)[:120]

        if not candidate.variables_available:
            tags = [t.get("name", "") for t in result.get("tags", [])]
            if tags:
                candidate.variables_available = tags[:20]

        if not candidate.metadata_url:
            candidate.metadata_url = result.get("url") or result.get("metadata_url") or ""

        if not candidate.available_formats:
            fmt_map = {
                "netcdf": DownloadFormat.netcdf, "nc": DownloadFormat.netcdf,
                "geotiff": DownloadFormat.geotiff, "tiff": DownloadFormat.geotiff,
                "csv": DownloadFormat.csv, "json": DownloadFormat.json,
                "hdf5": DownloadFormat.hdf5, "hdf": DownloadFormat.hdf5,
                "grib": DownloadFormat.grib, "shapefile": DownloadFormat.shapefile,
                "shp": DownloadFormat.shapefile,
            }
            fmts = set()
            for res in result.get("resources", []):
                fmt = (res.get("format") or "").lower().strip()
                if fmt in fmt_map:
                    fmts.add(fmt_map[fmt])
            if fmts:
                candidate.available_formats = list(fmts)

    except Exception:
        pass


def _enrich_from_erddap(candidate: CandidateSource) -> None:
    """
    Layer 3c — ERDDAP: try /info/{dataset_id}/index.json.
    Populates description, variables_available, temporal_coverage,
    spatial_coverage, spatial_resolution, temporal_resolution.
    """
    base = candidate.url.rstrip("/")
    for suffix in ["/griddap", "/tabledap", "/files", "/info"]:
        if suffix in base:
            base = base.split(suffix)[0]
            break
    dataset_id = candidate.source_id.replace("qdrant_", "")
    url = f"{base}/info/{dataset_id}/index.json"
    try:
        resp = requests.get(url, timeout=_METADATA_TIMEOUT, headers=_AGENT_HEADERS)
        if resp.status_code != 200:
            return
        data = resp.json()
        rows = data.get("table", {}).get("rows", [])
        col_names = data.get("table", {}).get("columnNames", [])

        try:
            rt_idx  = col_names.index("Row Type")
            var_idx = col_names.index("Variable Name")
            att_idx = col_names.index("Attribute Name")
            val_idx = col_names.index("Value")
        except ValueError:
            return

        attrs: Dict[str, str] = {}
        variables: List[str] = []
        for row in rows:
            row_type = row[rt_idx]
            var_name = row[var_idx]
            att_name = row[att_idx]
            value    = str(row[val_idx]) if row[val_idx] is not None else ""

            if row_type == "attribute" and var_name == "NC_GLOBAL":
                attrs[att_name] = value
            elif row_type == "variable" and att_name == "" and var_name not in ("NC_GLOBAL",):
                if var_name not in variables:
                    variables.append(var_name)

        if not candidate.description or len(candidate.description) < 20:
            candidate.description = attrs.get("summary", attrs.get("title", ""))[:_HTML_DESCRIPTION_CHARS]

        if candidate.temporal_coverage in ("Unknown", ""):
            t_start = attrs.get("time_coverage_start", "")
            t_end   = attrs.get("time_coverage_end", attrs.get("time_coverage_duration", "present"))
            if t_start:
                candidate.temporal_coverage = f"{t_start[:10]} – {t_end[:10] if t_end else 'present'}"

        if candidate.temporal_resolution in ("Unknown", ""):
            res = attrs.get("time_coverage_resolution", "")
            if res:
                candidate.temporal_resolution = res

        if candidate.spatial_coverage in ("Unknown", ""):
            lat_min = attrs.get("geospatial_lat_min", "")
            lat_max = attrs.get("geospatial_lat_max", "")
            lon_min = attrs.get("geospatial_lon_min", "")
            lon_max = attrs.get("geospatial_lon_max", "")
            if lat_min and lon_min:
                candidate.spatial_coverage = (
                    f"lat [{lat_min} to {lat_max}], lon [{lon_min} to {lon_max}]"
                )

        if candidate.spatial_resolution in ("Unknown", ""):
            res = attrs.get("geospatial_lat_resolution", "")
            if res:
                candidate.spatial_resolution = res

        if not candidate.variables_available and variables:
            candidate.variables_available = variables[:30]

    except Exception:
        pass


def _enrich_from_metadata_url(candidate: CandidateSource) -> None:
    """
    Layer 3d — If metadata_url was set (by catalog, HTML, or STAC),
    try fetching it as JSON and extract any remaining Unknown fields.
    """
    if not candidate.metadata_url:
        return
    if candidate.metadata_url == candidate.url:
        return   # already covered
    try:
        resp = requests.get(
            candidate.metadata_url,
            timeout=_METADATA_TIMEOUT,
            headers=_AGENT_HEADERS,
        )
        if resp.status_code != 200:
            return
        ct = resp.headers.get("Content-Type", "")
        if "json" not in ct:
            return
        data = resp.json()

        if not candidate.description or len(candidate.description) < 20:
            for key in ("description", "abstract", "summary", "title"):
                val = data.get(key, "")
                if val and len(str(val)) > 20:
                    candidate.description = str(val)[:_HTML_DESCRIPTION_CHARS]
                    break

        if candidate.temporal_coverage in ("Unknown", ""):
            for key in ("temporal_coverage", "time_coverage", "date_range"):
                val = data.get(key, "")
                if val:
                    candidate.temporal_coverage = str(val)[:80]
                    break

        if candidate.spatial_coverage in ("Unknown", ""):
            for key in ("spatial_coverage", "spatial_extent", "bbox", "extent"):
                val = data.get(key, "")
                if val:
                    candidate.spatial_coverage = str(val)[:120]
                    break

        if not candidate.variables_available:
            for key in ("variables", "parameters", "fields"):
                val = data.get(key, [])
                if isinstance(val, list) and val:
                    candidate.variables_available = [str(v) for v in val[:20]]
                    break

    except Exception:
        pass


def phase3_metadata_probe(candidates: List[CandidateSource]) -> List[CandidateSource]:
    """
    Phase 3 — Deep metadata enrichment.

    For every candidate, runs four enrichment layers in order:
      1. HEAD probe (latency + Last-Modified)
      2. Landing page GET (HTML meta tags, temporal patterns)
      3. Protocol-specific endpoint (STAC / CKAN / ERDDAP)
      4. metadata_url JSON fetch (if available)

    All layers are non-fatal. Missing metadata → field stays "Unknown".
    No source is rejected here — that is Phase 5's responsibility.
    """
    from models.discovery_schemas import APIType

    for candidate in candidates:
        # Layer 1 — HEAD probe
        latency, last_modified = _probe_source(candidate)
        candidate.response_latency_ms = latency
        if last_modified:
            candidate.last_updated = last_modified

        # Layers 2–4 only attempted when reachable OR metadata_url already set
        if latency is not None or candidate.metadata_url:
            # Layer 2 — landing page HTML
            _enrich_from_html(candidate)

            # Layer 3 — protocol-specific endpoint
            api = candidate.api_type
            if api == APIType.stac:
                _enrich_from_stac(candidate)
            elif api == APIType.ckan:
                _enrich_from_ckan(candidate)
            elif api == APIType.erddap:
                _enrich_from_erddap(candidate)

            # Layer 4 — metadata_url JSON (any protocol)
            _enrich_from_metadata_url(candidate)

    reachable = sum(1 for c in candidates if c.response_latency_ms is not None)
    enriched  = sum(1 for c in candidates if c.description and len(c.description) > 20)
    print(
        f"[Phase 3] Metadata enrichment: {reachable}/{len(candidates)} reachable, "
        f"{enriched}/{len(candidates)} with description populated."
    )
    return candidates


# ------------------------------------------------------------------ #
# Phase 3b — Provider Credibility Check  (Req 4 & 7)                 #
# ------------------------------------------------------------------ #

# Domain → preferred provider source_ids, in scientific priority order.
_DOMAIN_PREFERRED_PROVIDERS: Dict[str, List[str]] = {
    "flood":       ["nasa_earthdata", "copernicus", "esa", "sentinel", "imd", "incois", "ecmwf", "noaa", "usgs"],
    "ocean":       ["incois", "cmems", "noaa", "copernicus_marine", "nasa_oceancolor"],
    "vegetation":  ["modis", "landsat", "sentinel2", "copernicus_land"],
    "rainfall":    ["imd", "nasa_gpm", "chirps", "ecmwf", "noaa"],
    "cyclone":     ["imd", "noaa", "ecmwf", "nasa_earthdata"],
    "drought":     ["noaa", "nasa_earthdata", "chirps", "modis"],
    "coastal":     ["incois", "cmems", "noaa", "copernicus_marine"],
    "climate":     ["ecmwf", "noaa", "nasa_earthdata", "era5"],
}

# Keywords that map a query goal to a domain key above
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "flood":      ["flood", "inundation", "waterlogging", "river level", "discharge"],
    "ocean":      ["ocean", "sea surface", "sst", "salinity", "current", "wave"],
    "vegetation": ["vegetation", "ndvi", "chlorophyll", "land cover", "crop"],
    "rainfall":   ["rainfall", "precipitation", "rain", "imerg", "monsoon"],
    "cyclone":    ["cyclone", "hurricane", "typhoon", "storm"],
    "drought":    ["drought", "soil moisture", "dry"],
    "coastal":    ["coastal", "estuary", "mangrove", "shoreline"],
    "climate":    ["climate", "era5", "reanalysis", "long-term"],
}


def _infer_domain(goal: str, dataset_types: List[str], variables: List[str]) -> List[str]:
    """Return a list of domain keys that match the query, in order of confidence."""
    text = " ".join([goal] + dataset_types + variables).lower()
    return [domain for domain, keywords in _DOMAIN_KEYWORDS.items() if any(kw in text for kw in keywords)]


def phase3b_provider_credibility(
    candidates: List[CandidateSource],
    request: RetrievalRequest,
) -> List[CandidateSource]:
    """
    Phase 3b — Provider Credibility Check  (Req 4 & 7).

    Boosts catalog_authority_score for providers that appear in the
    domain-preferred provider list for this query's inferred domain(s).
    Penalises sources whose provider is completely unknown.

    No source is rejected here. Only scores are adjusted.
    """
    dataset_types = _get_dataset_types_from_request(request)
    variables     = _get_variables_from_request(request)
    domains       = _infer_domain(request.goal, dataset_types, variables)

    # Build priority set: {source_id_fragment: priority_rank} (lower = higher priority)
    priority_rank: Dict[str, int] = {}
    for domain in domains:
        for rank, sid in enumerate(_DOMAIN_PREFERRED_PROVIDERS.get(domain, [])):
            key = sid.lower()
            if key not in priority_rank:
                priority_rank[key] = rank

    boosted = penalised = 0
    for c in candidates:
        sid_lower  = c.source_id.lower()
        name_lower = c.name.lower()

        matched_rank = None
        for key, rank in priority_rank.items():
            if key in sid_lower or key in name_lower:
                matched_rank = rank
                break

        if matched_rank is not None:
            boost = max(0.0, 0.15 - matched_rank * 0.02)
            c.catalog_authority_score = min(1.0, c.catalog_authority_score + boost)
            boosted += 1
        elif c.catalog_authority_score < 0.3 and not c.description:
            c.catalog_authority_score = max(0.0, c.catalog_authority_score - 0.05)
            penalised += 1

    print(
        f"[Phase 3b] Provider credibility: domains={domains or ['general']}, "
        f"{boosted} boosted, {penalised} penalised."
    )
    return candidates


# ------------------------------------------------------------------ #
# Phase 3c — Coverage Analysis  (Req 4)                              #
# ------------------------------------------------------------------ #

def phase3c_coverage_analysis(
    candidates: List[CandidateSource],
    request: RetrievalRequest,
) -> List[CandidateSource]:
    """
    Phase 3c — Coverage Analysis  (Req 4).

    Annotates each candidate's description with a structured coverage note
    summarising spatial, temporal, and resolution fit vs. what was requested.
    The LLM and deterministic scorers in Phase 5 both read description,
    so this gives them richer context without adding new schema fields.
    """
    import re

    requested_location   = (request.spatial_requirements.get("location", "") or "").lower()
    requested_date_range = (request.temporal_requirements.get("date_range", "") or "").lower()

    for c in candidates:
        notes = []

        # Spatial coverage note
        if requested_location and requested_location not in ("unknown", "unspecified"):
            if requested_location in c.spatial_coverage.lower():
                notes.append(f"Covers requested location ({requested_location}).")
            elif c.spatial_coverage.lower() in ("global", "worldwide"):
                notes.append("Global coverage — includes requested location.")
            else:
                notes.append(f"Spatial coverage: {c.spatial_coverage}.")

        # Temporal coverage note
        if requested_date_range and requested_date_range not in ("unknown", "unspecified"):
            years_requested = re.findall(r'\b((?:19|20)\d{2})\b', requested_date_range)
            years_candidate = re.findall(r'\b((?:19|20)\d{2})\b', c.temporal_coverage)
            if years_requested and years_candidate:
                req_yr    = int(years_requested[0])
                can_years = sorted(int(y) for y in years_candidate)
                if can_years[0] <= req_yr <= can_years[-1] or "present" in c.temporal_coverage.lower():
                    notes.append("Temporal range includes requested period.")
                else:
                    notes.append(
                        f"Temporal range ({c.temporal_coverage}) may not cover "
                        f"requested period ({requested_date_range})."
                    )

        # Resolution presence note
        if c.spatial_resolution not in ("Unknown", ""):
            notes.append(f"Spatial resolution: {c.spatial_resolution}.")
        if c.temporal_resolution not in ("Unknown", ""):
            notes.append(f"Temporal resolution: {c.temporal_resolution}.")

        if notes:
            suffix = "  [Coverage: " + " ".join(notes) + "]"
            c.description = (c.description[:250] + suffix)[:500] if c.description else suffix[:500]

    print(f"[Phase 3c] Coverage analysis complete for {len(candidates)} candidate(s).")
    return candidates


# ------------------------------------------------------------------ #
# Phase 3d — Variable Matching  (Req 4)                              #
# ------------------------------------------------------------------ #

def phase3d_variable_matching(
    candidates: List[CandidateSource],
    request: RetrievalRequest,
) -> List[CandidateSource]:
    """
    Phase 3d — Variable Matching  (Req 4).

    Computes the fraction of requested variables (expanded via knowledge base)
    found in each candidate's variables_available list.  The overlap is blended
    into catalog_scientific_acceptance so it flows into Phase 5 scoring without
    a schema change.
    """
    variables = _get_variables_from_request(request)
    expanded  = set(v.lower() for v in expand_variables(variables))

    # CHANGED: "location" and "time" are mandatory structural variables that
    # virtually every geospatial source implicitly provides (a source with no
    # location or time dimension would be scientifically useless).  They are
    # injected into expanded by _get_variables_from_request, but they almost
    # never appear literally in a source's variables_available list — so
    # counting them as "missing" unfairly penalises every good source.
    # Solution: treat them as implicitly satisfied (matched = True) for any
    # source that has spatial_coverage or temporal_coverage metadata,
    # and exclude them from the denominator when computing overlap ratio
    # so they don't dilute the score of sources that cover the real variables.
    _IMPLICIT_MANDATORY = {"location", "time"}

    # Separate the mandatory structural tokens from the scientific variables
    # so overlap ratio is computed only over the latter.
    scientific_vars = expanded - _IMPLICIT_MANDATORY

    matched = 0
    for c in candidates:
        if not expanded:
            continue

        cand_vars = {v.lower() for v in c.variables_available}

        # Determine how many implicit mandatory variables this source satisfies.
        # A source satisfies "location" if it has any non-empty spatial_coverage,
        # and "time" if it has any non-empty temporal_coverage.
        implicit_satisfied = set()
        if "location" in expanded:
            if c.spatial_coverage and c.spatial_coverage.lower() not in ("unknown", ""):
                implicit_satisfied.add("location")
        if "time" in expanded:
            if c.temporal_coverage and c.temporal_coverage.lower() not in ("unknown", ""):
                implicit_satisfied.add("time")

        # Overlap over scientific variables only (mandatory ones handled above)
        if scientific_vars:
            if not cand_vars:
                # No variable metadata — don't penalise, leave score unchanged.
                # Mandatory variables may still be satisfied implicitly.
                sci_overlap = 0.0
                sci_denominator = len(scientific_vars)
            else:
                sci_overlap = len(scientific_vars & cand_vars)
                sci_denominator = len(scientific_vars)
        else:
            sci_overlap = 0.0
            sci_denominator = 0

        # Build a combined overlap that includes implicitly-satisfied mandatory vars
        total_numerator   = sci_overlap + len(implicit_satisfied)
        total_denominator = sci_denominator + len(_IMPLICIT_MANDATORY & expanded)

        if total_denominator == 0:
            continue

        overlap = total_numerator / total_denominator

        # Blend: don't overwrite a well-established catalog score completely
        c.catalog_scientific_acceptance = round(
            c.catalog_scientific_acceptance * 0.6 + overlap * 0.4, 4
        )
        if overlap > 0:
            matched += 1

    print(
        f"[Phase 3d] Variable matching: {matched}/{len(candidates)} candidate(s) "
        f"have at least one matching variable (location/time satisfied implicitly "
        f"via spatial/temporal coverage metadata)."
    )
    return candidates


# ------------------------------------------------------------------ #
# Phase 3.5 — Batch candidates for LLM scoring (token budget guard)   #
#                                                                     #
# CHANGED: This used to be a *filter* — only the top                  #
# _LLM_CANDIDATE_LIMIT candidates (by a crude pre-score) were sent to #
# the LLM, and everything else was given fake placeholder scores      #
# (relevance=completeness=consistency=0.5) without ever being         #
# evaluated. That violated the core requirement that every            #
# potentially-relevant dataset gets ranked using all ten criteria,    #
# and that nothing is dropped just because another candidate scored   #
# higher in a pre-filter heuristic.                                   #
#                                                                     #
# Now this is a pure *batcher*: every candidate is still sent to the  #
# LLM, just split across multiple calls of <= _LLM_BATCH_SIZE each so #
# no single prompt exceeds the token budget. Results are merged back  #
# into one LLMScoringOutput. No candidate is skipped or faked.        #
# ------------------------------------------------------------------ #

# Candidates per LLM call. Keeps each prompt well under the 12 000 TPM
# limit even with a verbose candidate set. Unlike the old
# _LLM_CANDIDATE_LIMIT, this does NOT cap how many candidates get
# scored overall — it only controls batch size; all batches are sent.
_LLM_BATCH_SIZE = 10


def _batch_candidates(candidates: List[CandidateSource]) -> List[List[CandidateSource]]:
    """
    Splits candidates into batches of at most _LLM_BATCH_SIZE so every
    candidate reaches the LLM regardless of total count.
    """
    if not candidates:
        return []
    return [
        candidates[i:i + _LLM_BATCH_SIZE]
        for i in range(0, len(candidates), _LLM_BATCH_SIZE)
    ]


# ------------------------------------------------------------------ #
# Phase 4 — LLM scoring                                              #
# CHANGED: now also passes relevant_sensors (from the knowledge base) #
# into the prompt, so the LLM's relevance scoring can account for      #
# sensor/instrument suitability, not just variable-name matching.     #
# ------------------------------------------------------------------ #

def _score_one_batch(
    candidates: List[CandidateSource],
    request: RetrievalRequest,
    dataset_types: List[str],
    variables: List[str],
    relevant_sensors: Optional[List[str]],
) -> LLMScoringOutput:
    """Scores a single batch (<= _LLM_BATCH_SIZE candidates) via one LLM call."""
    prompt = build_scoring_prompt(
        goal=request.goal,
        user_intent_type=request.user_intent_type,
        variables_needed=variables,
        dataset_types_needed=dataset_types,
        spatial_requirements=request.spatial_requirements,
        temporal_requirements=request.temporal_requirements,
        candidates=candidates,
        relevant_sensors=relevant_sensors,
    )

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    last_error = None

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw_text = response.choices[0].message.content
            return parse_scoring_response(raw_text)

        except ValidationError:
            raise

        except Exception as exc:
            last_error = exc
            if "503" in str(exc) or "429" in str(exc) or "rate" in str(exc).lower():
                wait = 5 * (attempt + 1)
                print(f"[Phase 4] Groq unavailable. Retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Agent 3 LLM scoring failed: {exc}") from exc

    raise RuntimeError(f"Agent 3 LLM scoring failed after retries: {last_error}")


def phase4_llm_scoring(
    candidates: List[CandidateSource],
    request: RetrievalRequest,
) -> LLMScoringOutput:
    """
    Phase 4 — LLM scoring.

    CHANGED: Every candidate is now scored — none are dropped to fit a
    single prompt's token budget. Candidates are split into batches of
    <= _LLM_BATCH_SIZE and scored with one LLM call per batch; the
    per-batch results are merged into a single LLMScoringOutput so the
    rest of the pipeline (Phase 5) is unaffected by batching.

    If a batch call fails after retries, that failure propagates (it is
    not silently swallowed into placeholder scores) so the caller knows
    discovery did not complete for those candidates.
    """
    dataset_types = _get_dataset_types_from_request(request)
    variables = _get_variables_from_request(request)
    relevant_sensors = get_sensor_instruments(variables)

    batches = _batch_candidates(candidates)
    all_scored: List[LLMCandidateScore] = []

    for batch_num, batch in enumerate(batches, start=1):
        result = _score_one_batch(
            candidates=batch,
            request=request,
            dataset_types=dataset_types,
            variables=variables,
            relevant_sensors=relevant_sensors,
        )
        all_scored.extend(result.scored_candidates)
        print(
            f"[Phase 4] Batch {batch_num}/{len(batches)}: "
            f"scored {len(result.scored_candidates)}/{len(batch)} candidate(s)."
        )

    print(f"[Phase 4] LLM scored {len(all_scored)}/{len(candidates)} candidate(s) total "
          f"across {len(batches)} batch(es). Every candidate evaluated.")

    return LLMScoringOutput(scored_candidates=all_scored)


# ------------------------------------------------------------------ #
# Phase 5 — Ranking                                                   #
# CHANGED: now also computes geographic_match, temporal_match, and    #
# platform_match via the new validators, and compares requested vs.   #
# actual resolution. See phase5_rank_sources docstring below.         #
# ------------------------------------------------------------------ #

def _compute_freshness_score(candidate: CandidateSource) -> float:
    coverage = candidate.temporal_coverage.lower()
    if "present" in coverage or "real-time" in coverage or "real time" in coverage:
        return 0.95
    if "2024" in coverage or "2025" in coverage or "2026" in coverage:
        return 0.85
    if "2020" in coverage or "2021" in coverage or "2022" in coverage or "2023" in coverage:
        return 0.70
    if "2015" in coverage or "2016" in coverage or "2017" in coverage:
        return 0.55
    if "2000" in coverage:
        return 0.45
    return 0.40


def _resolution_tier(text: str) -> Optional[int]:
    """Maps a free-text resolution string onto a coarse 1-4 tier
    (4 = finest/highest resolution, 1 = coarsest/point data).
    Returns None if no recognizable resolution pattern is present."""
    res = (text or "").lower()
    if any(x in res for x in ["1km", "0.01", "30m", "10m", "3km"]):
        return 4
    if any(x in res for x in ["0.1 degree", "0.1degree", "~11km", "11km"]):
        return 3
    if any(x in res for x in ["0.25 degree", "0.25degree", "~28km", "28km"]):
        return 2
    if any(x in res for x in ["station", "point"]):
        return 1
    return None


def _get_requested_resolution_text(request: RetrievalRequest) -> str:
    """
    Collects requested-resolution free text from the two places Agent 1
    actually populates it: SpatialContext.spatial_resolution_requirements
    (via request.spatial_requirements) and each Measurement's
    required_resolution. Returns the combined text, or "" if nothing
    was specified (caller treats that as "no resolution preference").
    """
    parts = []
    spatial_res = request.spatial_requirements.get("spatial_resolution_requirements", "")
    if spatial_res and spatial_res.lower() != "unknown":
        parts.append(spatial_res)
    for m in request.measurements:
        if m.required_resolution and m.required_resolution.lower() != "unknown":
            parts.append(m.required_resolution)
    return " ".join(parts)


# CHANGED: _compute_resolution_score now also takes the request's
# resolution ask and blends two signals into the SAME 'resolution'
# ParameterScore (no schema change -- stays backward compatible):
#
#   1. Absolute quality  -- is this resolution inherently fine-grained?
#      (the original, unchanged logic)
#   2. Match-to-request  -- does it actually meet what was asked for?
#      (NEW -- previously measurement.required_resolution and
#      spatial_resolution_requirements were read from Agent 1 but never
#      passed into scoring at all, so a candidate offering coarse global
#      data could score just as well as one offering exactly the
#      requested fine-grained resolution.)
#
# When the request specifies a resolution preference, the match-to-
# request signal dominates (a candidate that doesn't meet the ask should
# not be rewarded just because its native resolution happens to be
# generically fine elsewhere). When no preference is specified, behavior
# is identical to the original function.
def _compute_resolution_score(
    candidate: CandidateSource,
    requested_resolution_text: str = "",
) -> float:
    res = candidate.spatial_resolution.lower()
    if any(x in res for x in ["1km", "0.01", "30m", "10m", "3km"]):
        absolute_quality = 0.90
    elif any(x in res for x in ["0.1 degree", "0.1degree", "~11km", "11km"]):
        absolute_quality = 0.75
    elif any(x in res for x in ["0.25 degree", "0.25degree", "~28km", "28km"]):
        absolute_quality = 0.60
    elif any(x in res for x in ["station", "point"]):
        absolute_quality = 0.50
    elif "unknown" in res:
        absolute_quality = 0.40
    else:
        absolute_quality = 0.55

    if not requested_resolution_text or requested_resolution_text.lower() == "unknown":
        # No resolution preference -- unchanged original behavior.
        return absolute_quality

    requested_tier = _resolution_tier(requested_resolution_text)
    candidate_tier = _resolution_tier(candidate.spatial_resolution)

    if requested_tier is None or candidate_tier is None:
        # Can't compare -- fall back to absolute quality alone, since we
        # have no reliable signal either way.
        return absolute_quality

    diff = candidate_tier - requested_tier
    if diff >= 0:
        match_score = 0.90   # meets or exceeds what was asked for
    elif diff == -1:
        match_score = 0.55   # one tier coarser than requested
    else:
        match_score = 0.20   # much coarser than requested

    # Blend: match-to-request weighted higher, since matching the
    # actual ask matters more than abstract resolution quality.
    return round((0.7 * match_score) + (0.3 * absolute_quality), 4)


def _compute_real_time_availability_score(candidate: CandidateSource) -> float:
    if candidate.response_latency_ms is None:
        return 0.10
    if candidate.response_latency_ms < 500:
        return 0.95
    if candidate.response_latency_ms < 1500:
        return 0.75
    if candidate.response_latency_ms < 3000:
        return 0.55
    return 0.35


def _compute_metadata_quality_score(candidate: CandidateSource) -> float:
    score = 0.0
    if candidate.description and len(candidate.description) > 20:
        score += 0.3
    if candidate.metadata_url:
        score += 0.2
    if candidate.variables_available:
        score += 0.2
    if candidate.temporal_coverage and candidate.temporal_coverage != "Unknown":
        score += 0.15
    if candidate.spatial_resolution and candidate.spatial_resolution != "Unknown":
        score += 0.15
    return min(score, 1.0)


def _compute_historical_reliability_score(candidate: CandidateSource) -> float:
    """
    Returns the best available historical reliability score.

    If Qdrant has real usage history (times_used ≥ 2), the observed
    success rate takes precedence over the catalog default — it reflects
    what actually happened during previous retrievals, not what the
    catalog assumed. For sources seen only once, we blend catalog default
    and observed rate equally to avoid over-penalising a single failure.
    """
    if candidate.qdrant_historical_reliability is None:
        return candidate.catalog_historical_reliability

    observed = candidate.qdrant_historical_reliability
    times_used = getattr(candidate, "qdrant_times_used", 0) or 0

    if times_used >= 2:
        # Enough evidence — trust the observed rate fully
        return observed
    elif times_used == 1:
        # Blend: one data point is noisy, soften the impact
        return round(0.5 * observed + 0.5 * candidate.catalog_historical_reliability, 4)
    else:
        return candidate.catalog_historical_reliability


def _compute_qdrant_adjusted_scientific_acceptance(candidate: CandidateSource) -> float:
    """
    Adjusts the catalog's scientific_acceptance score downward when Qdrant
    reliability history shows a high failure rate, and upward slightly when
    the source has a consistently good track record.

    Rationale: a source that regularly returns 404s, HTML landing pages,
    or connection errors is less scientifically usable regardless of how
    well-regarded it is in the literature. Conversely, a source that has
    worked reliably many times deserves a modest credibility boost.

    Adjustment rules (applied only when times_used ≥ 2):
      • success_rate < 0.40  →  reduce scientific_acceptance by up to 0.20
      • success_rate > 0.85  →  boost  scientific_acceptance by up to 0.05
      • 0.40 – 0.85          →  no change (neutral zone)
    """
    base = candidate.catalog_scientific_acceptance
    if candidate.qdrant_historical_reliability is None:
        return base

    times_used = getattr(candidate, "qdrant_times_used", 0) or 0
    if times_used < 2:
        return base  # too little evidence to adjust

    success_rate = candidate.qdrant_historical_reliability
    if success_rate < 0.40:
        # Scale penalty: 0% success → -0.20, 40% success → 0
        penalty = 0.20 * (1.0 - success_rate / 0.40)
        return round(max(0.0, base - penalty), 4)
    elif success_rate > 0.85:
        # Modest boost: 85% → +0, 100% → +0.05
        boost = 0.05 * ((success_rate - 0.85) / 0.15)
        return round(min(1.0, base + boost), 4)
    return base


# ------------------------------------------------------------------ #
# Confidence score helper  (Req 5 & 9)                                #
# ------------------------------------------------------------------ #

def _compute_confidence_score(candidate: CandidateSource) -> float:
    """
    Returns a 0.0–1.0 confidence score reflecting how much metadata was
    successfully retrieved for this candidate.  Used by Phase 5 to decide
    whether a source goes to 'Accepted' vs 'Needs Further Evaluation'.
    """
    checks = [
        bool(candidate.description and len(candidate.description) > 20),
        bool(candidate.variables_available),
        candidate.spatial_coverage not in ("Unknown", ""),
        candidate.temporal_coverage not in ("Unknown", ""),
        candidate.spatial_resolution not in ("Unknown", ""),
        candidate.temporal_resolution not in ("Unknown", ""),
        bool(candidate.available_formats),
        bool(candidate.metadata_url),
        bool(candidate.last_updated),
    ]
    return round(sum(checks) / len(checks), 4)


def _build_justification(
    status: SourceStatus,
    final_score: float,
    confidence: float,
    llm_justification: str,
    rejection_reason: Optional[str],
    candidate: CandidateSource,
) -> str:
    """
    Req 9 — Build a structured, human-readable selection_justification
    for every scored source regardless of status.

    Format:
      Status: <status>  |  Score: <final_score> / 1.00  |  Confidence: <confidence>
      <status-specific explanation>  |  LLM: <llm_justification>
    """
    lines = [
        f"Status: {status.value}",
        f"Score: {final_score:.2f} / 1.00  |  Confidence: {confidence:.0%}",
    ]

    if status == SourceStatus.accepted:
        lines.append(
            f"Source meets discovery requirements. "
            f"Provider: {candidate.name}. "
            f"Variables known: {', '.join(candidate.variables_available[:4]) or 'not yet retrieved'}. "
            f"Coverage: {candidate.spatial_coverage}."
        )
    elif status == SourceStatus.authentication_required:
        lines.append(
            "Relevant dataset discovered. Public metadata successfully retrieved. "
            "Download requires authentication. Delegated to Agent 4."
        )
        if candidate.login_url:
            lines.append(f"Login URL: {candidate.login_url}")
    elif status == SourceStatus.needs_further_evaluation:
        lines.append(
            f"Dataset appears promising but metadata is incomplete "
            f"(confidence {confidence:.0%}). "
            "Retained for future consideration. "
            f"Reason: {rejection_reason or 'insufficient metadata to fully evaluate.'}"
        )
    else:  # rejected
        lines.append(
            f"Rejected. Reason: {rejection_reason or 'did not meet discovery requirements.'}"
        )

    lines.append(f"LLM: {llm_justification}")
    return "  |  ".join(lines)


def phase5_rank_sources(
    candidates: List[CandidateSource],
    llm_scores: LLMScoringOutput,
    request: RetrievalRequest,
) -> Tuple[List[ScoredSource], List[ScoredSource], List[ScoredSource], List[ScoredSource]]:
    """
    Phase 5 — Four-status ranking.

    Returns (accepted, auth_required, needs_evaluation, rejected).

    Every candidate passed in has already been scored by the LLM in
    Phase 4 (batched so none are skipped) — there is no longer a
    separate "below token budget" bucket to handle here.

    Priority order for status assignment (first match wins):
      1. geo hard-reject (confident wrong-region match)        → Rejected
      2. requires_login / payment                              → Authentication Required
                                                                   (auth is never a rejection reason)
      3. LLM populated failed_criteria AND rejection_confidence
         is high (>= REJECTION_CONFIDENCE_FLOOR) — i.e. wrong
         domain, duplicate, broken resource, invalid/empty
         dataset, or spam (the only categories the prompt
         allows it to flag)                                    → Rejected
      4. LLM populated failed_criteria but without high
         confidence (ambiguous, not clearly unsuitable)        → Needs Further Evaluation
      5. everything else                                       → Accepted

    A mediocre-but-relevant final_score is NOT a rejection trigger —
    it only affects rank. Incomplete metadata (low confidence_score) is
    NOT a rejection or demotion trigger either — confidence_score is
    reported alongside every source so the consumer can see how much
    metadata was available, but it never removes a relevant dataset
    from the ranked/accepted list.

    Every source gets a structured selection_justification, and every
    rejected source carries dataset name, provider, confidence score,
    exact rejection reason, and failed evaluation criteria.
    """
    llm_lookup: Dict[str, LLMCandidateScore] = {
        s.source_id: s for s in llm_scores.scored_candidates
    }
    # CHANGED (bug fix): also index by a normalized key (lowercased,
    # whitespace/punctuation-stripped). LLMs occasionally return a
    # source_id with different casing or minor formatting drift versus
    # the one we sent (e.g. "NOAA_NCEI_01" vs "noaa-ncei-01"). Under the
    # old exact-match-only lookup, any such drift caused the candidate
    # to be silently dropped from ALL FOUR buckets — it never appeared
    # as accepted, auth-required, needs-evaluation, OR rejected, which
    # is why total_candidates_found could be materially higher than
    # what a person actually saw in the printed results.
    def _normalize_id(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    llm_lookup_normalized: Dict[str, LLMCandidateScore] = {
        _normalize_id(s.source_id): s for s in llm_scores.scored_candidates
    }

    requested_resolution_text = _get_requested_resolution_text(request)
    requested_platform = infer_requested_platform(
        dataset_types_needed=_get_dataset_types_from_request(request),
        measurement_texts=[m.measurement_name for m in request.measurements]
        + [m.variable_measured for m in request.measurements],
    )
    requested_location   = request.spatial_requirements.get("location", "")
    requested_extent     = request.spatial_requirements.get("geographic_extent", "")
    requested_date_range = request.temporal_requirements.get("date_range", "")
    requested_baseline   = request.temporal_requirements.get("historical_baseline", "")

    scored_sources: List[ScoredSource] = []
    unmatched_count = 0

    # ---------------------------------------------------------------- #
    # LLM-scored candidates                                             #
    # ---------------------------------------------------------------- #
    for candidate in candidates:
        llm = llm_lookup.get(candidate.source_id)
        if llm is None:
            llm = llm_lookup_normalized.get(_normalize_id(candidate.source_id))

        if llm is None:
            # CHANGED (bug fix): previously `continue`d here, which
            # silently removed the candidate from every bucket. Every
            # candidate that made it through discovery MUST end up in
            # exactly one of accepted/auth-required/needs-evaluation/
            # rejected so that len(accepted)+len(auth_required)+
            # len(needs_evaluation)+len(rejected) always equals
            # len(candidates). A genuine LLM-scoring gap is now placed
            # in Needs Further Evaluation with an honest, visible note
            # -- never dropped.
            unmatched_count += 1
            llm = LLMCandidateScore(
                source_id=candidate.source_id,
                relevance_score=0.5,
                relevance_explanation="Not scored by the LLM in this run (no matching response); placed in Needs Further Evaluation pending re-scoring rather than dropped.",
                completeness_score=0.5,
                completeness_explanation="Not scored by the LLM in this run.",
                consistency_score=0.5,
                consistency_explanation="Not scored by the LLM in this run.",
                recommendation=SourceRecommendation.consider,
                selection_justification=(
                    f"The LLM scoring response did not include a matching entry for "
                    f"'{candidate.source_id}'. Rather than being silently discarded, this "
                    f"source is retained for further evaluation. This does not reflect on "
                    f"the source's actual quality or relevance."
                ),
                rejection_reason=None,
                failed_criteria=["LLM Scoring Gap"],
                rejection_confidence=None,
            )

        geo_result = validate_geography(
            requested_location=requested_location,
            requested_extent=requested_extent,
            candidate_spatial_coverage=candidate.spatial_coverage,
        )
        temporal_result = validate_temporal(
            requested_date_range=requested_date_range,
            requested_historical_baseline=requested_baseline,
            candidate_temporal_coverage=candidate.temporal_coverage,
        )
        candidate_platform = infer_candidate_platform(
            dataset_type=candidate.dataset_type,
            name=candidate.name,
            description=candidate.description,
            api_type=candidate.api_type.value if candidate.api_type else None,
            discovery_origin=candidate.discovery_origin,
        )
        platform_result = validate_platform(
            requested_platform=requested_platform,
            candidate_platform=candidate_platform,
        )

        score_card = SourceScoreCard(
            authority=ParameterScore(score=candidate.catalog_authority_score, explanation=f"Pre-assigned catalog authority score for {candidate.name}."),
            freshness=ParameterScore(score=_compute_freshness_score(candidate), explanation=f"Temporal coverage: {candidate.temporal_coverage}."),
            relevance=ParameterScore(score=llm.relevance_score, explanation=llm.relevance_explanation),
            resolution=ParameterScore(score=_compute_resolution_score(candidate, requested_resolution_text), explanation=f"Spatial resolution: {candidate.spatial_resolution}."),
            completeness=ParameterScore(score=llm.completeness_score, explanation=llm.completeness_explanation),
            consistency=ParameterScore(score=llm.consistency_score, explanation=llm.consistency_explanation),
            metadata_quality=ParameterScore(score=_compute_metadata_quality_score(candidate), explanation="Scored from description, metadata URL, and variable documentation."),
            historical_reliability=ParameterScore(
                score=_compute_historical_reliability_score(candidate),
                explanation=(
                    f"Qdrant history: {candidate.qdrant_times_used} uses, {candidate.qdrant_times_succeeded} successes."
                    if candidate.from_qdrant_cache else "No prior usage history. Using catalog default."
                ),
            ),
            scientific_acceptance=ParameterScore(
                score=_compute_qdrant_adjusted_scientific_acceptance(candidate),
                explanation=(
                    f"Catalog base {candidate.catalog_scientific_acceptance:.2f}; "
                    f"adjusted down due to Qdrant failure history "
                    f"({getattr(candidate, 'qdrant_times_used', 0)} uses, "
                    f"{getattr(candidate, 'qdrant_times_succeeded', 0)} successes)."
                    if (candidate.qdrant_historical_reliability is not None
                        and (getattr(candidate, "qdrant_times_used", 0) or 0) >= 2
                        and candidate.qdrant_historical_reliability < 0.40)
                    else f"Pre-assigned scientific acceptance for {candidate.name}."
                    + (f" Boosted slightly — {getattr(candidate, 'qdrant_times_used', 0)} "
                       f"reliable uses in Qdrant history."
                       if (candidate.qdrant_historical_reliability is not None
                           and (getattr(candidate, "qdrant_times_used", 0) or 0) >= 2
                           and candidate.qdrant_historical_reliability > 0.85)
                       else "")
                ),
            ),
            real_time_availability=ParameterScore(
                score=_compute_real_time_availability_score(candidate),
                explanation=(f"Response latency: {candidate.response_latency_ms}ms." if candidate.response_latency_ms is not None else "Source unreachable during probe."),
            ),
            geographic_match=ParameterScore(score=geo_result.score, explanation=geo_result.explanation),
            temporal_match=ParameterScore(score=temporal_result.score, explanation=temporal_result.explanation),
            platform_match=ParameterScore(score=platform_result.score, explanation=platform_result.explanation),
        )

        final_score      = score_card.final_score
        confidence       = _compute_confidence_score(candidate)
        recommendation   = llm.recommendation
        rejection_reason = llm.rejection_reason
        failed_criteria: List[str] = list(llm.failed_criteria)
        rejection_confidence: Optional[float] = llm.rejection_confidence

        # Four-status assignment — priority order
        #
        # CHANGED: This project's SourceRecommendation enum only has
        # "use" and "consider" -- the LLM is explicitly instructed
        # (see prompts/agent3_prompt.py) to NEVER emit "reject" and to
        # leave rejection entirely to this deterministic layer. So
        # rejection decisions here must NOT check for
        # SourceRecommendation.reject (that value doesn't exist and
        # raises AttributeError). Instead, the LLM signals genuine
        # unsuitability via failed_criteria (non-empty list) plus
        # rejection_confidence, exactly as the prompt instructs it to.
        #
        # Also removed (kept from an earlier draft, now corrected):
        #   - "final_score < MIN_SCORE_TO_PRESENT" auto-reject. A
        #     relevant but lower-scoring dataset should rank low, not
        #     be discarded.
        #   - "confidence < 0.35" demotion. Incomplete metadata must
        #     never reject or demote a dataset, per spec.
        #
        # Rejection now requires real evidence of unsuitability:
        #   - a confident geographic mismatch (deterministic, unchanged), or
        #   - the LLM populated failed_criteria (wrong domain, duplicate,
        #     broken resource, invalid/empty dataset, spam) AND backed
        #     it with a high rejection_confidence (>= 0.7) -- mirroring
        #     exactly what the prompt asks the LLM to provide.
        # Everything else relevant is accepted (or auth-required) and
        # ranked by final_score, regardless of how that score compares
        # to other candidates.
        REJECTION_CONFIDENCE_FLOOR = 0.7

        if geo_result.hard_reject:
            status           = SourceStatus.rejected
            rejection_reason = geo_result.explanation
            # Deterministic rejection -- the LLM never made this call,
            # so failed_criteria/rejection_confidence are set here
            # directly rather than trusting whatever the LLM happened
            # to return (which may be empty/None, since it might have
            # scored this candidate as "use").
            failed_criteria      = ["Geographic Coverage"]
            rejection_confidence = 0.95

        elif candidate.requires_login or candidate.requires_payment:
            # Auth is NEVER a rejection reason — delegate to Agent 4
            status               = SourceStatus.authentication_required
            recommendation       = SourceRecommendation.consider
            rejection_reason     = None
            failed_criteria      = []
            rejection_confidence = None

        elif failed_criteria and (rejection_confidence or 0.0) >= REJECTION_CONFIDENCE_FLOOR:
            # Genuinely unsuitable: the LLM flagged specific failed
            # criteria (wrong domain, duplicate, broken resource,
            # invalid dataset, spam) with high confidence -- exactly
            # what the prompt asks it to populate for unusable sources.
            status           = SourceStatus.rejected
            rejection_reason = rejection_reason or (
                f"Failed criteria: {', '.join(failed_criteria)}. "
                f"{llm.selection_justification}"
            )

        elif failed_criteria:
            # LLM flagged some failed criteria but without strong
            # confidence -- ambiguous, not clearly unsuitable. Retain
            # rather than discard.
            status           = SourceStatus.needs_further_evaluation
            rejection_reason = (
                rejection_reason
                or f"LLM flagged possible issues ({', '.join(failed_criteria)}) "
                   "without high confidence; retained for further evaluation "
                   "rather than rejected."
            )

        else:
            status               = SourceStatus.accepted
            failed_criteria      = []
            rejection_confidence = None

        structured_justification = _build_justification(
            status=status,
            final_score=final_score,
            confidence=confidence,
            llm_justification=llm.selection_justification,
            rejection_reason=rejection_reason,
            candidate=candidate,
        )

        scored_sources.append(ScoredSource(
            candidate=candidate,
            score_card=score_card,
            final_score=final_score,
            recommendation=recommendation,
            status=status,
            confidence_score=confidence,
            selection_justification=structured_justification,
            rejection_reason=rejection_reason,
            failed_criteria=failed_criteria,
            rejection_confidence=rejection_confidence,
        ))

    scored_sources.sort(key=lambda s: s.final_score, reverse=True)

    accepted        = []
    auth_required   = []
    needs_eval      = []
    rejected        = []
    for rank_pos, source in enumerate(scored_sources, start=1):
        source.rank = rank_pos
        if source.status == SourceStatus.accepted:
            accepted.append(source)
        elif source.status == SourceStatus.authentication_required:
            auth_required.append(source)
        elif source.status == SourceStatus.needs_further_evaluation:
            needs_eval.append(source)
        else:
            rejected.append(source)

    print(
        f"[Phase 5] Ranked {len(accepted)} accepted, "
        f"{len(auth_required)} auth-required, "
        f"{len(needs_eval)} needs-evaluation, "
        f"{len(rejected)} rejected."
        + (f" ({unmatched_count} had no matching LLM score and were placed in "
           f"needs-evaluation rather than dropped.)" if unmatched_count else "")
    )
    return accepted, auth_required, needs_eval, rejected


# ------------------------------------------------------------------ #
# Phase 6 — Store to Qdrant                                           #
# CHANGED: now also stores access_type, requires_login, requires_payment
# so Phase 1c can reconstruct full CandidateSource objects next time. #
# ------------------------------------------------------------------ #

def phase6_store_to_qdrant(
    request: RetrievalRequest,
    scored_sources: List[ScoredSource],
) -> None:
    if not qdrant_store.is_qdrant_available():
        print("[Phase 6] Qdrant unavailable — skipping store.")
        return

    dataset_types = _get_dataset_types_from_request(request)

    source_dicts = [
        {
            "source_id": s.candidate.source_id,
            "name": s.candidate.name,
            "url": s.candidate.url,
            "dataset_type": s.candidate.dataset_type,
            "variables_available": s.candidate.variables_available,
            "spatial_coverage": s.candidate.spatial_coverage,
            "temporal_coverage": s.candidate.temporal_coverage,
            "final_score": s.final_score,
            "recommendation": s.recommendation.value,
            "catalog_authority_score": s.candidate.catalog_authority_score,
            # NEW — store access fields so Phase 1c can reconstruct them
            "access_type": s.candidate.access_type.value,
            "requires_login": s.candidate.requires_login,
            "requires_payment": s.candidate.requires_payment,
            "price_estimate": s.candidate.price_estimate,
            "login_url": s.candidate.login_url,
            "api_type": s.candidate.api_type.value,
            "discovery_origin": s.candidate.discovery_origin,
            "health_score": s.candidate.health_score,
        }
        for s in scored_sources
    ]

    success = qdrant_store.store_discovery_result(
        query_goal=request.goal,
        dataset_types=dataset_types,
        scored_sources=source_dicts,
    )
    if success:
        print(f"[Phase 6] Stored {len(source_dicts)} source(s) to Qdrant.")


# ------------------------------------------------------------------ #
# Phase 7 — Present to user                                           #
# CHANGED: now shows access classification for each source.           #
# ------------------------------------------------------------------ #

def phase7_present_sources(
    accepted: List[ScoredSource],
    auth_required: List[ScoredSource],
    needs_evaluation: List[ScoredSource],
    rejected: List[ScoredSource],
    request: RetrievalRequest,
) -> None:
    total = len(accepted) + len(auth_required) + len(needs_evaluation) + len(rejected)
    print("\n" + "=" * 70)
    print("DATA DISCOVERY RESULTS")
    print("=" * 70)
    print(f"\nScientific Goal:")
    print(f"  {request.goal}")
    print(f"\nTotal sources evaluated : {total}")
    print(f"  Accepted              : {len(accepted)}")
    print(f"  Authentication Required: {len(auth_required)}")
    print(f"  Needs Further Eval.   : {len(needs_evaluation)}")
    print(f"  Rejected              : {len(rejected)}")

    # ---------------------------------------------------------------- #
    # Bucket 1 — Accepted                                              #
    # ---------------------------------------------------------------- #
    if accepted:
        print("\n" + "-" * 70)
        print("ACCEPTED SOURCES (ranked best to worst)")
        print("-" * 70)

        for source in accepted:
            c = source.candidate
            rec_label = "✓ USE" if source.recommendation == SourceRecommendation.use else "~ CONSIDER"

            access_label = c.access_type.value.upper()
            if c.requires_payment:
                access_label += f"  ⚠ PAID — {c.price_estimate or 'price unknown'}"
            elif c.requires_login:
                access_label += "  🔑 LOGIN REQUIRED"

            print(f"\n[{source.rank}] {c.name}  [{rec_label}]")
            print(f"    Status       : {source.status.value}")
            print(f"    Score        : {source.final_score:.2f} / 1.00")
            print(f"    Confidence   : {source.confidence_score:.0%}")
            print(f"    Access       : {access_label}")
            print(f"    Origin       : {c.discovery_origin}")
            print(f"    API type     : {c.api_type.value}")
            print(f"    URL          : {c.url}")
            print(f"    Type         : {c.dataset_type}")
            print(f"    Variables    : {', '.join(c.variables_available[:5])}")
            print(f"    Coverage     : {c.spatial_coverage}")
            print(f"    Time range   : {c.temporal_coverage}")
            print(f"    Resolution   : {c.spatial_resolution}")
            print(f"    Formats      : {', '.join(f.value for f in c.available_formats)}")
            print(f"    Justification: {source.selection_justification}")
            _print_score_breakdown(source)

    # ---------------------------------------------------------------- #
    # Bucket 2 — Authentication Required                               #
    # ---------------------------------------------------------------- #
    if auth_required:
        print("\n" + "-" * 70)
        print("AUTHENTICATION REQUIRED — passed to Agent 4 for retrieval")
        print("-" * 70)

        for source in auth_required:
            c = source.candidate
            print(f"\n  🔑 {c.name}")
            print(f"    Status       : {source.status.value}")
            print(f"    Score        : {source.final_score:.2f} / 1.00")
            print(f"    Confidence   : {source.confidence_score:.0%}")
            print(f"    URL          : {c.url}")
            print(f"    Login URL    : {c.login_url or 'see provider documentation'}")
            print(f"    Variables    : {', '.join(c.variables_available[:5])}")
            print(f"    Coverage     : {c.spatial_coverage}")
            print(f"    Time range   : {c.temporal_coverage}")
            print(f"    Justification: {source.selection_justification}")
            _print_score_breakdown(source)

    # ---------------------------------------------------------------- #
    # Bucket 3 — Needs Further Evaluation                              #
    # ---------------------------------------------------------------- #
    if needs_evaluation:
        print("\n" + "-" * 70)
        print("NEEDS FURTHER EVALUATION — retained for future consideration")
        print("-" * 70)

        for source in needs_evaluation:
            c = source.candidate
            print(f"\n  ~ {c.name}")
            print(f"    Status       : {source.status.value}")
            print(f"    Score        : {source.final_score:.2f} / 1.00")
            print(f"    Confidence   : {source.confidence_score:.0%}")
            print(f"    URL          : {c.url}")
            print(f"    Justification: {source.selection_justification}")

    # ---------------------------------------------------------------- #
    # Bucket 4 — Rejected                                              #
    # ---------------------------------------------------------------- #
    if rejected:
        print("\n" + "-" * 70)
        print("REJECTED SOURCES")
        print("-" * 70)
        for source in rejected:
            c = source.candidate
            print(f"\n  ✗ {c.name}")
            print(f"    Provider     : {c.discovery_origin}")
            print(f"    Score        : {source.final_score:.2f}")
            print(f"    Reason       : {source.rejection_reason or source.selection_justification}")
            print(f"    Failed Criteria: {', '.join(source.failed_criteria) or 'Not specified'}")
            if source.rejection_confidence is not None:
                print(f"    Confidence   : {source.rejection_confidence:.2f}")
            print(f"    Justification: {source.selection_justification}")

    print("\n" + "=" * 70)


def _print_score_breakdown(source: ScoredSource) -> None:
    """Shared score breakdown block used by accepted and auth-required display."""
    sc = source.score_card
    print(f"    Score breakdown:")
    print(f"      Authority          : {sc.authority.score:.2f}")
    print(f"      Freshness          : {sc.freshness.score:.2f}")
    print(f"      Relevance          : {sc.relevance.score:.2f}  ← {sc.relevance.explanation}")
    print(f"      Resolution         : {sc.resolution.score:.2f}")
    print(f"      Completeness       : {sc.completeness.score:.2f}  ← {sc.completeness.explanation}")
    print(f"      Consistency        : {sc.consistency.score:.2f}  ← {sc.consistency.explanation}")
    print(f"      Metadata Quality   : {sc.metadata_quality.score:.2f}")
    print(f"      Historical Reliab. : {sc.historical_reliability.score:.2f}")
    print(f"      Scientific Accept. : {sc.scientific_acceptance.score:.2f}")
    print(f"      Real-time Avail.   : {sc.real_time_availability.score:.2f}")
    print(f"      Geographic Match   : {sc.geographic_match.score:.2f}  ← {sc.geographic_match.explanation}")
    print(f"      Temporal Match     : {sc.temporal_match.score:.2f}  ← {sc.temporal_match.explanation}")
    print(f"      Platform Match     : {sc.platform_match.score:.2f}  ← {sc.platform_match.explanation}")
    c = source.candidate
    if c.from_qdrant_cache:
        print(f"    Memory: Used {c.qdrant_times_used}x before, {c.qdrant_times_succeeded} successes.")





# ------------------------------------------------------------------ #
# Main runner                                                         #
#                                                                     #
# Phase 1 is 3 sub-phases (1a + 1b + 1c) with dedup.                #
# Phase 8 (download gate) removed. Agent 3 is discovery-only.        #
# Agent 4 is responsible for source selection and retrieval planning. #
# ------------------------------------------------------------------ #

def run_agent3(
    request: RetrievalRequest,
    extra_context: dict | None = None,
) -> DiscoveryOutput:
    print("\n" + "=" * 70)
    print("AGENT 3 — DATA DISCOVERY")
    print("=" * 70)

    # ---------------------------------------------------------------- #
    # Extra context from a previous Agent 4 failure round              #
    # ---------------------------------------------------------------- #
    # When main.py re-runs Agent 3 after Agent 4 flagged incomplete
    # coverage, it passes extra_context containing:
    #   previously_failed_source_ids  – source_ids Agent 4 could not use
    #   discovery_feedback            – free-form notes from Agent 4
    # We surface both in the console so the operator can see what Agent 3
    # is working around, then filter the failed IDs out of Phase 1
    # results so those sources are not re-ranked and re-passed to Agent 4.
    previously_failed_ids: set[str] = set()
    if extra_context:
        failed_ids = extra_context.get("previously_failed_source_ids") or []
        if failed_ids:
            previously_failed_ids = {str(sid).strip() for sid in failed_ids}
            print(
                f"[Agent 3] Re-discovery round: excluding {len(previously_failed_ids)} "
                f"previously-failed source(s): {sorted(previously_failed_ids)}"
            )

        feedback = extra_context.get("discovery_feedback") or {}
        if feedback:
            print("[Agent 3] Agent 4 feedback for this round:")
            for key, value in feedback.items():
                if key == "previously_failed_source_ids":
                    continue  # already printed above
                print(f"  {key}: {value}")

    if qdrant_store.is_qdrant_available():
        qdrant_store.ensure_collections_exist()

    # ---------------------------------------------------------------- #
    # Phase 1 — THREE-SOURCE DISCOVERY FAN-OUT  ← CHANGED             #
    # ---------------------------------------------------------------- #
    catalog_candidates = phase1a_catalog_query(request)
    dynamic_candidates = phase1b_dynamic_discovery(request)
    qdrant_candidates = phase1c_qdrant_source_search(request)

    # Merge: catalog first (highest authority), then dynamic, then Qdrant cache
    all_candidates = catalog_candidates + dynamic_candidates + qdrant_candidates
    candidates = _deduplicate_candidates(all_candidates)
    candidates = _filter_landing_pages(candidates)

    # Drop any source that Agent 4 already tried and failed on so this
    # round finds genuinely new alternatives rather than re-proposing the
    # same broken sources.
    if previously_failed_ids:
        before = len(candidates)
        candidates = [
            c for c in candidates
            if str(c.source_id).strip() not in previously_failed_ids
        ]
        dropped = before - len(candidates)
        if dropped:
            print(
                f"[Phase 1] Excluded {dropped} previously-failed source(s) "
                f"from this discovery round."
            )

    print(f"[Phase 1] Combined: {len(catalog_candidates)} catalog + "
          f"{len(dynamic_candidates)} dynamic + "
          f"{len(qdrant_candidates)} Qdrant = "
          f"{len(candidates)} unique candidate(s) after dedup + bare-domain/landing-page filter.")

    if not candidates:
        print("[Agent 3] No candidates found. Discovery complete with no results.")
        return DiscoveryOutput(
            retrieval_request_goal=request.goal,
            total_candidates_found=0,
            total_candidates_scored=0,
            total_from_qdrant_cache=0,
            ranked_sources=[],
            rejected_sources=[],
            discovery_notes=[
                "No matching sources found across catalog, dynamic discovery, or Qdrant.",
                "Discovery complete with no results. Agent 4 will receive empty ranked_sources.",
            ],
        )

    # ---------------------------------------------------------------- #
    # Phases 2–8 — updated flow                                        #
    # ---------------------------------------------------------------- #
    candidates = phase2_qdrant_enrichment(candidates, request)
    total_from_cache = sum(1 for c in candidates if c.from_qdrant_cache)

    # Phase 3 — Deep metadata enrichment (Req 2 & 6)
    candidates = phase3_metadata_probe(candidates)

    # Phase 3b — Provider credibility check (Req 4 & 7)
    candidates = phase3b_provider_credibility(candidates, request)

    # Phase 3c — Coverage analysis (Req 4)
    candidates = phase3c_coverage_analysis(candidates, request)

    # Phase 3d — Variable matching (Req 4)
    candidates = phase3d_variable_matching(candidates, request)

    # Phase 4 — LLM scoring. Every candidate is scored (batched
    # internally so the token budget per call stays safe) — nothing is
    # dropped or faked here just because the candidate count is large.
    llm_scores = phase4_llm_scoring(candidates, request)

    accepted, auth_required, needs_evaluation, rejected = phase5_rank_sources(
        candidates, llm_scores, request
    )
    all_scored = accepted + auth_required + needs_evaluation + rejected

    # CHANGED (bug fix): hard integrity check. Every candidate that
    # survives Phase 1 dedup MUST end up in exactly one of the four
    # buckets. If this ever fails, it means something in Phase 5 is
    # dropping candidates again -- surface that loudly rather than
    # silently under-reporting totals to the person running Agent 3.
    if len(all_scored) != len(candidates):
        missing = len(candidates) - len(all_scored)
        print(
            f"[Agent 3] INTEGRITY WARNING: {missing} candidate(s) discovered but "
            f"not present in any ranked bucket. This should never happen -- "
            f"please report this as a bug."
        )

    phase6_store_to_qdrant(request, all_scored)

    # Renumber accepted sources sequentially (1, 2, 3, ...) before output.
    for position, source in enumerate(accepted, start=1):
        source.rank = position

    phase7_present_sources(accepted, auth_required, needs_evaluation, rejected, request)

    # Agent 3 finishes here. No download gate, no user interaction.
    # Agent 4 receives ranked_sources, auth_required_sources,
    # needs_evaluation_sources, and rejected_sources.
    return DiscoveryOutput(
        retrieval_request_goal=request.goal,
        total_candidates_found=len(candidates),
        total_candidates_scored=len(all_scored),
        total_from_qdrant_cache=total_from_cache,
        ranked_sources=accepted,
        auth_required_sources=auth_required,
        needs_evaluation_sources=needs_evaluation,
        rejected_sources=rejected,
        discovery_notes=[
            f"Catalog: {len(catalog_candidates)} | "
            f"Dynamic: {len(dynamic_candidates)} | "
            f"Qdrant cache: {len(qdrant_candidates)} | "
            f"Unique after dedup: {len(candidates)}.",
            f"{total_from_cache} enriched from Qdrant reliability history.",
            f"{len(accepted)} accepted + {len(auth_required)} auth-required + "
            f"{len(needs_evaluation)} needs-evaluation + {len(rejected)} rejected "
            f"= {len(all_scored)} of {len(candidates)} discovered candidate(s) accounted for.",
            "Discovery complete. Source selection delegated to Agent 4.",
        ],
    )
