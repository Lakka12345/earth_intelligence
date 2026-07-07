"""
Geographic Validator — deterministic geographic-compatibility check for
Agent 3 candidate sources.

Problem this solves:
  Candidates discovered by ERDDAP/THREDDS/CKAN/STAC/generic-search (and many
  catalog providers) carry no structured geographic metadata — region info
  exists only as a free-text "spatial_coverage" string, or is missing
  entirely ("Unknown — check ERDDAP metadata"). Nothing in the pipeline
  ever compared that string against the request's spatial_requirements
  (location / geographic_extent), so a clearly wrong-region candidate
  (e.g. an Irish ERDDAP buoy for a Bay-of-Bengal query) could still win on
  variable-keyword match and authority/historical-reliability scores.

Design (small, pure-Python, no geocoding/API calls — fast, offline,
debuggable, and easy to extend):
  - Region clusters group mutually-compatible aliases. Two strings that
    fall in the same cluster (or share a cluster) are treated as a match.
  - Three outcomes per candidate:
      1. Clear match            -> high score, never rejected.
      2. Clear, confident,      -> low score, hard_reject=True.
         disjoint mismatch
      3. Global or unknown      -> treated as valid (per spec refinement
         coverage                 #3 — NASA/Copernicus/GHRSST/MODIS-style
                                   global products genuinely cover any
                                   requested region, and ERDDAP's
                                   "Unknown — check ERDDAP metadata"
                                   placeholder must NOT be punished).

This module is intentionally conservative: when in doubt, it returns a
reduced-confidence "unknown overlap" result rather than a hard reject, so
we never silently drop a legitimate source just because our alias table
doesn't cover its region.
"""

from dataclasses import dataclass
from typing import List, Optional


# ------------------------------------------------------------------ #
# Region clusters                                                     #
# ------------------------------------------------------------------ #
# Each cluster is a set of mutually-compatible region aliases (any string
# in a cluster is considered geographically compatible with any other
# string in the SAME cluster). Clusters are intentionally coarse —
# this is a structural sanity check, not precise geofencing.
#
# IMPORTANT: ocean-basin names (atlantic, pacific, indian ocean) and
# generic compass directions (east coast, west coast) are deliberately
# NOT shared across multiple landmass clusters here. An earlier version
# of this table put "atlantic" in the Europe, North America, AND Africa
# clusters (since all three border the Atlantic), and "east coast" /
# "west coast" in the North America cluster alone -- but because cluster
# membership is transitive for matching purposes (any shared cluster
# index counts as overlap), this created false bridges: a request
# mentioning "...India east coast..." matched the North America cluster
# purely through the generic substring "east coast", which then
# overlapped with an Irish ERDDAP candidate that also (correctly)
# matched the North America cluster via "atlantic" -- producing a false
# geographic match between Ireland and the Bay of Bengal. Ocean-basin
# names are large enough to border multiple, non-adjacent landmasses, so
# they are kept in only ONE cluster each (their most distinctive/primary
# association) rather than used as bridges. This is intentionally
# conservative: a real edge case (e.g. a literal mid-Atlantic dataset)
# will fall through to the "unknown overlap" neutral path rather than
# being wrongly matched OR wrongly rejected.

REGION_CLUSTERS: List[List[str]] = [
    # South / Southeast Asia & surrounding seas
    [
        "india", "india coastal", "south asia", "indian ocean",
        "bay of bengal", "arabian sea", "andaman sea", "lakshadweep sea",
        "sri lanka", "bangladesh", "myanmar", "indonesia", "maldives",
    ],
    # Europe & adjacent waters
    [
        "europe", "european waters", "ireland", "uk", "united kingdom",
        "north atlantic", "north sea", "baltic", "mediterranean",
        "arctic",
    ],
    # North America & adjacent waters
    [
        "united states", "usa", "north america", "gulf of mexico",
        "caribbean", "pacific northwest", "california", "florida",
    ],
    # Oceania / Southern hemisphere
    [
        "australia", "southern ocean", "new zealand",
    ],
    # East Asia & Northwest Pacific
    [
        "japan", "northwest pacific", "east asia", "china", "korea",
    ],
    # Africa
    [
        "africa", "east africa", "west africa", "south africa",
    ],
    # South America
    [
        "south america", "brazil",
    ],
]

# Tokens that mean "this candidate covers the whole planet" — never
# rejected, always treated as compatible with any requested region.
GLOBAL_TOKENS = [
    "global", "world", "worldwide", "all regions", "planetary",
]

# Tokens that mean "we genuinely don't know" — must NOT be punished.
# (e.g. ERDDAP discoverer's placeholder, STAC/THREDDS "see catalog" text)
UNKNOWN_TOKENS = [
    "unknown", "see erddap metadata", "see thredds catalog",
    "see stac item geometry", "see portal metadata", "from web search",
    "varies", "n/a", "",
]


@dataclass
class GeoValidationResult:
    score: float                 # 0.0 - 1.0
    hard_reject: bool
    explanation: str


def _normalize(text: Optional[str]) -> str:
    if not text:
        return ""
    return text.lower().strip()


def _find_clusters(text: str) -> List[int]:
    """Return indices of all REGION_CLUSTERS that this text matches."""
    matches = []
    for idx, cluster in enumerate(REGION_CLUSTERS):
        for alias in cluster:
            if alias in text:
                matches.append(idx)
                break
    return matches


def _is_global(text: str) -> bool:
    return any(tok in text for tok in GLOBAL_TOKENS)


def _is_unknown(text: str) -> bool:
    if text.strip() in UNKNOWN_TOKENS:
        return True
    return any(tok in text for tok in UNKNOWN_TOKENS if tok)


def validate_geography(
    requested_location: Optional[str],
    requested_extent: Optional[str],
    candidate_spatial_coverage: Optional[str],
) -> GeoValidationResult:
    """
    Compares the request's geographic intent against a candidate's
    spatial_coverage string.

    Args:
        requested_location: SpatialContext.location (e.g. "Bay of Bengal").
        requested_extent: SpatialContext.geographic_extent (free text).
        candidate_spatial_coverage: CandidateSource.spatial_coverage.

    Returns:
        GeoValidationResult with a 0-1 score and a hard_reject flag.
        hard_reject is only ever True for a confident, known-disjoint
        mismatch between two well-recognized clusters.
    """
    requested_text = " ".join(
        filter(None, [_normalize(requested_location), _normalize(requested_extent)])
    )
    candidate_text = _normalize(candidate_spatial_coverage)

    # No geographic ask at all -> nothing to validate against.
    if not requested_text or requested_text in ("unknown", ""):
        return GeoValidationResult(
            score=0.75,
            hard_reject=False,
            explanation="Request has no specific geographic constraint; geography not evaluated.",
        )

    # Candidate explicitly declares global coverage -> always valid.
    if _is_global(candidate_text):
        return GeoValidationResult(
            score=0.85,
            hard_reject=False,
            explanation="Candidate declares global coverage, which satisfies any requested region.",
        )

    # Candidate's coverage is unknown/unparsed -> don't punish, don't reward.
    # (This is the ERDDAP/THREDDS/STAC placeholder case — see module docstring.)
    if _is_unknown(candidate_text):
        return GeoValidationResult(
            score=0.55,
            hard_reject=False,
            explanation=(
                "Candidate's spatial coverage is not yet known structurally "
                "(placeholder text); geography treated as unverified, not rejected."
            ),
        )

    requested_clusters = set(_find_clusters(requested_text))
    candidate_clusters = set(_find_clusters(candidate_text))

    # Requested region isn't in our alias table -> unknown overlap.
    if not requested_clusters:
        return GeoValidationResult(
            score=0.55,
            hard_reject=False,
            explanation=(
                f"Requested region '{requested_text}' is not in the known region table; "
                "geographic match unverified, treated as neutral rather than rejected."
            ),
        )

    # Candidate's stated coverage isn't in our alias table either -> unknown overlap.
    if not candidate_clusters:
        return GeoValidationResult(
            score=0.55,
            hard_reject=False,
            explanation=(
                f"Candidate's stated coverage '{candidate_text}' is not in the known region "
                "table; geographic match unverified, treated as neutral rather than rejected."
            ),
        )

    # Both sides resolved to known clusters -> check overlap.
    if requested_clusters & candidate_clusters:
        return GeoValidationResult(
            score=0.95,
            hard_reject=False,
            explanation=(
                f"Candidate coverage '{candidate_text}' overlaps the requested region "
                f"'{requested_text}'."
            ),
        )

    # Confident, known, disjoint mismatch -> hard reject.
    return GeoValidationResult(
        score=0.05,
        hard_reject=True,
        explanation=(
            f"Candidate coverage '{candidate_text}' is a known region disjoint from the "
            f"requested region '{requested_text}'. Geographic mismatch."
        ),
    )
