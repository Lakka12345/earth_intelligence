from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


# ------------------------------------------------------------------ #
# Enums                                                                #
# ------------------------------------------------------------------ #

class SourceRecommendation(str, Enum):
    use = "use"
    consider = "consider"


# Four-status classification replacing the binary accept/reject model.
# Agent 3 sets this; Agent 4 reads it to decide retrieval strategy.
class SourceStatus(str, Enum):
    accepted = "Accepted"
    authentication_required = "Authentication Required"
    needs_further_evaluation = "Needs Further Evaluation"
    rejected = "Rejected"


class DownloadFormat(str, Enum):
    netcdf = "NetCDF"
    geotiff = "GeoTIFF"
    csv = "CSV"
    json = "JSON"
    hdf5 = "HDF5"
    grib = "GRIB"
    shapefile = "Shapefile"
    geojson = "GeoJSON"
    unknown = "Unknown"


# NEW — classifies how a source can be accessed
class AccessType(str, Enum):
    free = "free"                       # no auth, no registration
    registration = "registration"       # free but needs account/login
    api_key = "api_key"                 # free key, no payment
    paid = "paid"                       # costs money
    unknown = "unknown"


# NEW — classifies the API protocol/type
class APIType(str, Enum):
    rest = "REST"
    opendap = "OPeNDAP"
    wms_wfs = "WMS/WFS"
    ftp = "FTP"
    graphql = "GraphQL"
    web_scrape = "WebScrape"
    database = "Database"
    stac = "STAC"
    erddap = "ERDDAP"
    ckan = "CKAN"
    thredds = "THREDDS"
    unknown = "Unknown"


# ------------------------------------------------------------------ #
# Source scoring — one object per scoring parameter                    #
# ------------------------------------------------------------------ #

class ParameterScore(BaseModel):
    """Score for a single parameter, with explanation."""
    score: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=3)


class SourceScoreCard(BaseModel):
    """
    13-parameter score card for one candidate source.
    7 original parameters are computed by Python (deterministic).
    3 parameters (relevance, completeness, consistency)
    are scored by the LLM in a single call.

    CHANGED: Added 3 new deterministic parameters -- geographic_match,
    temporal_match, platform_match -- computed in Agent 3 Phase 5 via
    discovery/geo_validator.py, discovery/temporal_validator.py, and
    discovery/platform_validator.py. agent3_discovery.py already
    constructs SourceScoreCard with these three keyword arguments and
    reads them back in Phase 7's printout; without these fields
    declared here, pydantic silently drops the unrecognized kwargs at
    construction (BaseModel's default extra="ignore"), and the later
    `.geographic_match` attribute access then raises AttributeError --
    so this was a real, if deferred, crash.

    final_score weights rebalanced so the new-3 sum to 0.38 (geographic
    weighted highest at 0.20) and the original-7 are scaled down
    proportionally, preserving their original relative ordering. Sums
    to exactly 1.00.
    """
    authority: ParameterScore
    freshness: ParameterScore
    relevance: ParameterScore           # LLM scored
    resolution: ParameterScore
    completeness: ParameterScore        # LLM scored
    consistency: ParameterScore         # LLM scored
    metadata_quality: ParameterScore
    historical_reliability: ParameterScore
    scientific_acceptance: ParameterScore
    real_time_availability: ParameterScore

    # NEW -- deterministic validators (Phase 5).
    geographic_match: ParameterScore
    temporal_match: ParameterScore
    platform_match: ParameterScore

    @property
    def final_score(self) -> float:
        weights = {
            "authority": 0.10,
            "freshness": 0.06,
            "relevance": 0.13,
            "resolution": 0.06,
            "completeness": 0.06,
            "consistency": 0.06,
            "metadata_quality": 0.03,
            "historical_reliability": 0.06,
            "scientific_acceptance": 0.03,
            "real_time_availability": 0.03,
            "geographic_match": 0.20,
            "temporal_match": 0.09,
            "platform_match": 0.09,
        }
        scores = {
            "authority": self.authority.score,
            "freshness": self.freshness.score,
            "relevance": self.relevance.score,
            "resolution": self.resolution.score,
            "completeness": self.completeness.score,
            "consistency": self.consistency.score,
            "metadata_quality": self.metadata_quality.score,
            "historical_reliability": self.historical_reliability.score,
            "scientific_acceptance": self.scientific_acceptance.score,
            "real_time_availability": self.real_time_availability.score,
            "geographic_match": self.geographic_match.score,
            "temporal_match": self.temporal_match.score,
            "platform_match": self.platform_match.score,
        }
        return round(
            sum(scores[k] * weights[k] for k in weights), 4
        )


# ------------------------------------------------------------------ #
# Candidate source — before scoring                                    #
# ------------------------------------------------------------------ #

class CandidateSource(BaseModel):
    """
    A data source identified from the catalog, discovery engine, or Qdrant.
    Not yet scored. Passed to the scoring phase.

    CHANGED: Added requires_login, requires_payment, api_type, access_type,
    price_estimate, login_url, api_docs fields.
    These are used by Agent 4 to classify and plan access before retrieval.
    """
    source_id: str = Field(min_length=2)
    name: str = Field(min_length=2)
    url: str = Field(min_length=5)
    dataset_type: str = Field(min_length=2)

    # Populated from catalog or live metadata probe
    variables_available: List[str] = Field(default_factory=list)
    spatial_coverage: str = Field(default="Unknown")
    temporal_coverage: str = Field(default="Unknown")
    temporal_resolution: str = Field(default="Unknown")
    spatial_resolution: str = Field(default="Unknown")
    available_formats: List[DownloadFormat] = Field(default_factory=list)

    # CHANGED: access_type is now the AccessType enum, not a plain string
    # OLD: access_type: str = Field(default="API")
    access_type: AccessType = Field(default=AccessType.free)

    # CHANGED: renamed requires_auth → requires_login for clarity
    # OLD: requires_auth: bool = Field(default=False)
    requires_login: bool = Field(default=False)

    # NEW: explicit payment flag — Agent 4 MUST show human approval gate if True
    requires_payment: bool = Field(default=False)

    # NEW: estimated price string shown to user before any purchase decision
    price_estimate: Optional[str] = None       # e.g. "$120 per scene", "Free"

    # NEW: URL for the login/registration page (used by auth_manager in Agent 4)
    login_url: Optional[str] = None

    # NEW: URL for the API documentation (used by downloader in Agent 4)
    api_docs: Optional[str] = None

    # NEW: API protocol type — tells Agent 4 which downloader to use
    api_type: APIType = Field(default=APIType.rest)

    # NEW: source origin — where was this candidate found?
    # "catalog" | "erddap_discovery" | "stac_discovery" | "ckan_discovery"
    # | "thredds_discovery" | "generic_search" | "qdrant_cache"
    discovery_origin: str = Field(default="catalog")

    response_latency_ms: Optional[float] = None
    last_updated: Optional[str] = None
    metadata_url: Optional[str] = None
    description: str = Field(default="")

    # Pre-filled from catalog (deterministic scores)
    catalog_authority_score: float = Field(ge=0.0, le=1.0, default=0.5)
    catalog_scientific_acceptance: float = Field(ge=0.0, le=1.0, default=0.5)
    catalog_historical_reliability: float = Field(ge=0.0, le=1.0, default=0.5)

    # From Qdrant memory
    qdrant_historical_reliability: Optional[float] = None
    qdrant_times_used: int = Field(default=0)
    qdrant_times_succeeded: int = Field(default=0)
    from_qdrant_cache: bool = Field(default=False)

    # Health score — degrades each time Agent 4 fails on this source.
    # Stored in Qdrant and restored on cache hit.
    # Agent 4 calls: source.health_score *= 0.9 on failure, then persists.
    health_score: float = Field(ge=0.0, le=1.0, default=1.0)


# ------------------------------------------------------------------ #
# Scored source — after scoring                                        #
# ------------------------------------------------------------------ #

class ScoredSource(BaseModel):
    """
    A candidate source after full scoring.
    Contains the score card, final score, and recommendation.

    CHANGED: Added status (SourceStatus) and confidence_score.
    - status replaces the binary accept/reject model with four categories:
      Accepted, Authentication Required, Needs Further Evaluation, Rejected.
    - confidence_score (0.0–1.0) reflects how much metadata was available
      during evaluation (populated by Agent 3 Phase 5).
    - rejection_reason is now populated for every non-accepted status.
    """
    candidate: CandidateSource
    score_card: SourceScoreCard
    final_score: float = Field(ge=0.0, le=1.0)
    recommendation: SourceRecommendation

    # Four-status classification
    status: SourceStatus = Field(default=SourceStatus.accepted)

    # Fraction of expected metadata fields that were populated (0.0–1.0)
    confidence_score: float = Field(ge=0.0, le=1.0, default=1.0)

    selection_justification: str = Field(min_length=5)
    rejection_reason: Optional[str] = None

    # NEW -- mirrors LLMCandidateScore's new fields (see that class for
    # rationale). Populated for EVERY rejection, whether the rejection
    # came from the LLM's recommendation or from a deterministic Phase 5
    # rule (geo hard-reject, below-threshold score) -- those
    # deterministic paths set failed_criteria/rejection_confidence
    # themselves in Phase 5, since the LLM never saw those candidates
    # rejected for those reasons in the first place.
    failed_criteria: List[str] = Field(default_factory=list)
    rejection_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    rank: Optional[int] = None


# ------------------------------------------------------------------ #
# LLM scoring input / output                                           #
# ------------------------------------------------------------------ #

class LLMScoringInput(BaseModel):
    goal: str
    user_intent_type: str
    variables_needed: List[str]
    dataset_types_needed: List[str]
    spatial_requirements: Dict
    temporal_requirements: Dict
    candidates: List[CandidateSource]


class LLMCandidateScore(BaseModel):
    source_id: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    relevance_explanation: str = Field(min_length=3)
    completeness_score: float = Field(ge=0.0, le=1.0)
    completeness_explanation: str = Field(min_length=3)
    consistency_score: float = Field(ge=0.0, le=1.0)
    consistency_explanation: str = Field(min_length=3)
    recommendation: SourceRecommendation
    selection_justification: str = Field(min_length=5)
    rejection_reason: Optional[str] = None

    # NEW -- structured rejection detail, per spec requirement #3
    # ("Every rejection must include... Failed Evaluation Criteria").
    # Only meaningful when recommendation == reject; empty/default
    # otherwise. Free-text rejection_reason is kept for the human-
    # readable explanation; failed_criteria is the machine-readable
    # list (e.g. ["Relevance", "Required Variables"]) Agent 4 or a UI
    # can render as tags without parsing prose.
    failed_criteria: List[str] = Field(default_factory=list)

    # NEW -- confidence specifically in the rejection decision itself,
    # distinct from ScoredSource.confidence_score (which measures
    # metadata completeness, not decision confidence). Per spec
    # requirement #3's worked examples (e.g. "Confidence: 0.99").
    # Defaults to the same value as relevance for non-rejected
    # candidates is NOT assumed here -- defaults to None, and Phase 5
    # only treats it as meaningful when recommendation == reject.
    rejection_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class LLMScoringOutput(BaseModel):
    scored_candidates: List[LLMCandidateScore]

    @model_validator(mode="after")
    def validate_all_candidates_scored(self):
        if not self.scored_candidates:
            raise ValueError("LLM must score at least one candidate.")
        return self


# ------------------------------------------------------------------ #
# Discovery output — final result of Agent 3                           #
# ------------------------------------------------------------------ #

class DiscoveryOutput(BaseModel):
    """
    Final output of the data discovery agent.

    CHANGED: ranked_sources now carry full CandidateSource objects
    including access_type, requires_login, requires_payment, api_type.
    Agent 4 reads these directly to plan retrieval without re-discovery.

    CHANGED: Added auth_required_sources and needs_evaluation_sources.
    These replace the old binary ranked/rejected model with four buckets:
      - ranked_sources            → SourceStatus.accepted
      - auth_required_sources     → SourceStatus.authentication_required
      - needs_evaluation_sources  → SourceStatus.needs_further_evaluation
      - rejected_sources          → SourceStatus.rejected (only truly unusable)

    Agent 4 uses auth_required_sources directly for authenticated retrieval.
    needs_evaluation_sources are kept for future consideration.
    """
    retrieval_request_goal: str
    total_candidates_found: int
    total_candidates_scored: int
    total_from_qdrant_cache: int

    ranked_sources: List[ScoredSource]
    auth_required_sources: List[ScoredSource] = Field(default_factory=list)
    needs_evaluation_sources: List[ScoredSource] = Field(default_factory=list)
    rejected_sources: List[ScoredSource]

    sources_selected_for_download: List[str] = Field(default_factory=list)
    download_requested: bool = Field(default=False)

    # Credentials collected by the access gate — passed to Agent 4.
    # Keys are source_ids. Values are dicts with username/password/api_key/token.
    # NEVER written to disk or Qdrant — in-memory only for this session.
    retrieval_credentials: Dict[str, Dict] = Field(default_factory=dict)

    discovery_notes: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ranks(self):
        for i, source in enumerate(self.ranked_sources, start=1):
            if source.rank != i:
                raise ValueError(
                    f"Rank mismatch at position {i}: "
                    f"source '{source.candidate.source_id}' has rank {source.rank}."
                )
        return self


# ------------------------------------------------------------------ #
# DiscoveryResult — per-run diagnostics wrapper                        #
# Returned by DiscoveryEngine.search() so callers can log failures.   #
# ------------------------------------------------------------------ #

class DiscoveryResult(BaseModel):
    """
    Wraps the raw output of one DiscoveryEngine.search() call.
    Separate from DiscoveryOutput — this is the engine-level result
    before scoring. Agent 3 unpacks .sources and discards the rest.
    """
    sources: List[CandidateSource]
    discovery_time_seconds: float
    discoverers_used: List[str]
    failures: List[str]          # "DiscovererName: error message"
