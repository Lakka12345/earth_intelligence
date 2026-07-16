"""
Website Analysis Schemas — additive, self-contained.

Nothing here modifies models/discovery_schemas.py. discovery/website_analyzer.py
populates these purely from fields the existing Agent 3 pipeline already
extracted (or, for the accessibility keyword-scan fields, from
candidate.description text) -- never from a new network call.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ===================================================================== #
# ACCESS CLASSIFICATION                                                  #
# ===================================================================== #

class AccessClassification(str, Enum):
    free = "free"
    simple_login_required = "simple_login_required"          # agent can self-register
    credential_verification_required = "credential_verification_required"  # user must supply real identity
    paid_access = "paid_access"


class CredentialEase(str, Enum):
    """The specific question you asked for: can Agent 4 create the
    account itself with throwaway/random details, or must the person
    supply their own real information?"""
    no_credentials_needed = "no_credentials_needed"
    agent_can_self_register = "agent_can_self_register"        # any email/random signup works
    user_must_provide_real_credentials = "user_must_provide_real_credentials"  # institutional email, ID, approval, etc.
    unknown = "unknown"


class AvailabilityTimeline(str, Enum):
    immediate = "immediate"
    manual_approval = "manual_approval"
    few_days_approval = "few_days_approval"
    waiting_period = "waiting_period"
    unknown = "unknown"


class DataAvailabilityStatus(str, Enum):
    full = "full"
    partial = "partial"
    not_available = "not_available"
    unknown = "unknown"  # variable metadata wasn't extracted for this source -- not the same as "confirmed missing"


class TriState(str, Enum):
    """For factors we can only sometimes determine from metadata text
    (e.g. rate limits, server uptime) -- yes/no/unknown, never guessed."""
    yes = "yes"
    no = "no"
    unknown = "unknown"


# ===================================================================== #
# ACCESSIBILITY breakdown -- exactly the factors requested:              #
# Authentication required, Anonymous access available, API availability, #
# Download restrictions, Rate limits, Registration required,             #
# Real-Time Availability, Retrieval speed, Server uptime,                #
# Supported download formats -- plus the credential-ease question and    #
# explicit payment flag.                                                 #
# ===================================================================== #

class AccessibilityProfile(BaseModel):
    authentication_required: bool
    anonymous_access_available: bool
    api_available: bool
    api_type: str = "unknown"

    download_restrictions: TriState = TriState.unknown
    download_restrictions_notes: str = ""

    rate_limits: TriState = TriState.unknown
    rate_limits_notes: str = ""

    registration_required: bool = False
    credential_ease: CredentialEase = CredentialEase.unknown
    credential_ease_notes: str = ""

    real_time_availability_score: float = Field(ge=0.0, le=1.0, default=0.5)
    retrieval_speed_ms: Optional[float] = None
    retrieval_speed_label: str = "unknown"

    server_uptime: TriState = TriState.unknown
    server_uptime_notes: str = ""

    supported_download_formats: List[str] = Field(default_factory=list)

    payment_required: bool = False
    payment_notes: str = ""

    access_classification: AccessClassification = AccessClassification.free
    availability_timeline: AvailabilityTimeline = AvailabilityTimeline.unknown
    timeline_notes: str = ""

    accessibility_composite_score: float = Field(ge=0.0, le=1.0, default=0.5)


# ===================================================================== #
# AVAILABILITY breakdown -- exactly the factors requested:               #
# Relevance, Completeness, Variable availability, Spatial coverage,      #
# Temporal coverage, Resolution, Historical coverage, Data continuity,   #
# Missing values.                                                        #
# ===================================================================== #

class AvailabilityProfile(BaseModel):
    relevance_score: float = Field(ge=0.0, le=1.0)
    completeness_score: float = Field(ge=0.0, le=1.0)

    requested_variables: List[str] = Field(default_factory=list)
    covered_variables: List[str] = Field(default_factory=list)
    missing_variables: List[str] = Field(default_factory=list)
    variable_availability_score: float = Field(ge=0.0, le=1.0, default=0.0)

    spatial_coverage_score: float = Field(ge=0.0, le=1.0)
    spatial_coverage_notes: str = ""

    temporal_coverage_score: float = Field(ge=0.0, le=1.0)
    temporal_coverage_notes: str = ""

    resolution_score: float = Field(ge=0.0, le=1.0)
    resolution_notes: str = ""

    # Best-effort only: Agent 3 works from metadata, not the raw data
    # files, so these two are text-derived proxies, not verified facts.
    historical_coverage_score: float = Field(ge=0.0, le=1.0, default=0.5)
    historical_coverage_notes: str = "Estimated from stated temporal coverage span; not verified against actual archive depth."

    data_continuity: TriState = TriState.unknown
    data_continuity_notes: str = "Cannot be verified without inspecting the actual data files; based on provider-stated update cadence only."

    missing_values: TriState = TriState.unknown
    missing_values_notes: str = "Cannot be verified without inspecting the actual data files."

    availability_status: DataAvailabilityStatus = DataAvailabilityStatus.partial
    availability_composite_score: float = Field(ge=0.0, le=1.0, default=0.5)


# ===================================================================== #
# ACCURACY breakdown -- exactly the factors requested:                   #
# Authority, Credibility, Scientific Acceptance, Consistency,            #
# Historical Reliability, Metadata Quality.                              #
# ===================================================================== #

class AccuracyProfile(BaseModel):
    authority_score: float = Field(ge=0.0, le=1.0)
    # "Credibility" has no separate existing factor in the pipeline's
    # SourceScoreCard -- it is derived here as a blend of authority and
    # historical reliability (a source is credible if it's both an
    # authoritative provider AND has a track record). Documented rather
    # than silently invented.
    credibility_score: float = Field(ge=0.0, le=1.0)
    credibility_notes: str = "Derived from (authority + historical reliability) / 2 -- no standalone 'credibility' signal exists in the underlying pipeline."

    scientific_acceptance_score: float = Field(ge=0.0, le=1.0)
    consistency_score: float = Field(ge=0.0, le=1.0)
    historical_reliability_score: float = Field(ge=0.0, le=1.0)
    metadata_quality_score: float = Field(ge=0.0, le=1.0)

    accuracy_composite_score: float = Field(ge=0.0, le=1.0, default=0.5)


class WebsiteAnalysisResult(BaseModel):
    source_id: str
    website_name: str

    accessibility: AccessibilityProfile
    availability: AvailabilityProfile
    accuracy: AccuracyProfile

    data_policy_summary: str = ""


# ===================================================================== #
# Ranking preference (user-selected)                                     #
# ===================================================================== #

class RankingCriterion(str, Enum):
    accuracy = "accuracy"
    accessibility = "accessibility"
    availability = "availability"


class RankingPreference(BaseModel):
    selected_criteria: List[RankingCriterion] = Field(min_length=1)


# ===================================================================== #
# Final Agent 3 -> Agent 4 handoff payload                               #
# ===================================================================== #

class Agent3ToAgent4Mode(str, Enum):
    ranked_selection = "ranked_selection"
    user_override = "user_override"


class SourceSnapshot(BaseModel):
    """
    A minimal, decoupled copy of the fields Agent 4 actually needs from
    a CandidateSource. Deliberately NOT importing CandidateSource here
    -- Agent 4 should depend on this stable, narrow contract rather than
    reaching into Agent 3's internal schema.
    """
    source_id: str
    name: str
    url: str
    api_type: str = "unknown"
    dataset_type: str = "unknown"
    variables_available: List[str] = Field(default_factory=list)
    login_url: Optional[str] = None
    registration_url: Optional[str] = None
    price_estimate: Optional[str] = None


class Agent3ToAgent4Payload(BaseModel):
    mode: Agent3ToAgent4Mode
    final_ranked_source_ids: List[str] = Field(default_factory=list)
    # Explicit per-source flag Agent 4 can act on directly, without
    # re-deriving it: which sources Agent 4 is free to self-register
    # for using throwaway credentials, vs. which require the person's
    # own real information.
    self_registerable_source_ids: List[str] = Field(default_factory=list)
    real_credentials_required_source_ids: List[str] = Field(default_factory=list)
    paid_source_ids: List[str] = Field(default_factory=list)
    unconfirmed_credential_source_ids: List[str] = Field(default_factory=list)

    # NEW: full context so Agent 4 never has to re-derive or re-import
    # Agent 3 internals -- keyed by source_id, same ids as the lists above.
    website_analyses: Dict[str, WebsiteAnalysisResult] = Field(default_factory=dict)
    source_snapshots: Dict[str, SourceSnapshot] = Field(default_factory=dict)
    requested_variables: List[str] = Field(default_factory=list)

    # NEW: mirrors DiscoveryOutput.retrieval_credentials -- credentials
    # already collected during Agent 3's own flow (by whatever access
    # gate populates that field), keyed by source_id, each a dict with
    # username/password/api_key/token. Agent 4 should use these BEFORE
    # asking the user again or attempting self-registration.
    pre_collected_credentials: Dict[str, Dict] = Field(default_factory=dict)

    override_website: Optional[str] = None
    override_notes: str = "User explicitly asked to bypass the ranked recommendations."

    ranking_preference: Optional[RankingPreference] = None
    notes: List[str] = Field(default_factory=list)
