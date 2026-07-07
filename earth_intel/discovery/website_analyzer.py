"""
Website Analyzer — Discovery Agent extension.

Builds three separate factor breakdowns per surviving candidate
(everything Phase 5 did NOT reject), using ONLY fields the existing
Agent 3 pipeline already extracted. No new network calls.

  - AccessibilityProfile : authentication/registration/API/rate-limits/
                            formats/credential-ease/payment
  - AvailabilityProfile  : relevance/completeness/variable coverage/
                            spatial/temporal/resolution/historical/
                            continuity/missing-values
  - AccuracyProfile      : authority/credibility/scientific-acceptance/
                            consistency/historical-reliability/
                            metadata-quality
"""

from typing import Dict, List

from models.retrieval_request import RetrievalRequest
from models.website_analysis_schemas import (
    AccessClassification,
    AccessibilityProfile,
    AccuracyProfile,
    AvailabilityProfile,
    AvailabilityTimeline,
    CredentialEase,
    DataAvailabilityStatus,
    TriState,
    WebsiteAnalysisResult,
)

# --------------------------------------------------------------------- #
# Keyword heuristics                                                     #
# --------------------------------------------------------------------- #

_SELF_REGISTER_KEYWORDS = (
    "sign up with any email", "create a free account", "self-service registration",
    "instant access", "register online", "no verification required",
    "quick registration", "email verification only",
)
_REAL_CREDENTIALS_KEYWORDS = (
    "institutional email", "credential", "proof of affiliation",
    "approved researcher", "vetting", "organization verification",
    "id verification", "apply for access", "access request form",
    "justify your request", "government id", "orcid required",
    "your real name", "employer verification",
)
_MANUAL_APPROVAL_KEYWORDS = ("manual approval", "reviewed by", "subject to approval", "pending review")
_FEW_DAYS_KEYWORDS = ("business days", "working days", "2-3 days", "3-5 days", "within a week", "processing time")
_WAITING_PERIOD_KEYWORDS = ("embargo", "release delay", "waiting period", "quarantine period")
_RATE_LIMIT_KEYWORDS = ("rate limit", "requests per", "throttle", "quota")
_NO_RATE_LIMIT_KEYWORDS = ("no rate limit", "unlimited requests")
_DOWNLOAD_RESTRICTION_KEYWORDS = ("bulk download not permitted", "download limit", "restricted download", "license required for download")
_UPTIME_GOOD_KEYWORDS = ("99.9% uptime", "high availability", "sla")
_UPTIME_BAD_KEYWORDS = ("frequent outages", "maintenance window", "known downtime")


def _text_blob(candidate) -> str:
    return " ".join(filter(None, [
        getattr(candidate, "description", "") or "",
        getattr(candidate, "login_url", "") or "",
        getattr(candidate, "metadata_url", "") or "",
    ])).lower()


def _tristate_from_keywords(blob: str, yes_kw, no_kw) -> TriState:
    if any(k in blob for k in yes_kw):
        return TriState.yes
    if any(k in blob for k in no_kw):
        return TriState.no
    return TriState.unknown


# --------------------------------------------------------------------- #
# ACCESSIBILITY                                                          #
# --------------------------------------------------------------------- #

def _classify_access(candidate) -> AccessClassification:
    if getattr(candidate, "requires_payment", False):
        return AccessClassification.paid_access
    access_type_value = (getattr(getattr(candidate, "access_type", None), "value", "") or "").lower()
    if access_type_value == "paid":
        return AccessClassification.paid_access
    # CHANGED (bug fix): the real AccessType enum (models/discovery_schemas.py)
    # has values free/registration/api_key/paid/unknown -- "login_required"
    # and "restricted" never existed in that enum, so this check only ever
    # worked by accident via the requires_login boolean fallback below. Fixed
    # to check the actual enum values, so a candidate with access_type=api_key
    # and requires_login=False (a legitimate combination -- an API key isn't
    # really a "login") is no longer silently misclassified as Free.
    if getattr(candidate, "requires_login", False) or access_type_value in ("registration", "api_key"):
        blob = _text_blob(candidate)
        if any(kw in blob for kw in _REAL_CREDENTIALS_KEYWORDS):
            return AccessClassification.credential_verification_required
        return AccessClassification.simple_login_required
    return AccessClassification.free


# Curated, high-confidence knowledge about specific well-documented
# login systems -- NOT a guess. These are publicly documented facts
# about how these specific providers' registration actually works.
# Matched against the candidate's own url/login_url domain.
_KNOWN_SELF_SERVICE_DOMAINS = (
    "urs.earthdata.nasa.gov", "earthdata.nasa.gov",   # NASA Earthdata Login: any email, instant, no verification
    "dataspace.copernicus.eu", "copernicus.eu",        # Copernicus Data Space: self-service email signup
    "planetarycomputer.microsoft.com",                  # Sign in with any Microsoft/GitHub account
    "esgf.llnl.gov", "esgf-node",                       # ESGF: self-service OpenID registration
)
_KNOWN_VERIFICATION_REQUIRED_DOMAINS = (
    # Intentionally left sparse -- only add a domain here when its
    # verification requirement is publicly documented, not assumed.
)


def _known_domain_match(candidate, domains) -> bool:
    haystack = " ".join(filter(None, [
        getattr(candidate, "url", "") or "",
        getattr(candidate, "login_url", "") or "",
    ])).lower()
    return any(d in haystack for d in domains)


def _classify_credential_ease(candidate, access: AccessClassification, blob: str) -> tuple[CredentialEase, str]:
    if access == AccessClassification.free:
        return CredentialEase.no_credentials_needed, "No account needed."
    if access == AccessClassification.paid_access:
        return (
            CredentialEase.user_must_provide_real_credentials,
            "Paid access requires real payment/billing information; Agent 4 cannot self-register around this.",
        )

    # Tier 1: curated, documented knowledge about this specific provider.
    if _known_domain_match(candidate, _KNOWN_SELF_SERVICE_DOMAINS):
        return (
            CredentialEase.agent_can_self_register,
            "This provider's registration system is documented to accept self-service signup with any working email address, no identity verification -- Agent 4 can register directly.",
        )
    if _known_domain_match(candidate, _KNOWN_VERIFICATION_REQUIRED_DOMAINS):
        return (
            CredentialEase.user_must_provide_real_credentials,
            "This provider's registration process is documented to require identity/affiliation verification -- the user must supply their own real details.",
        )

    # Tier 2: explicit keyword signal in whatever metadata text exists.
    if any(kw in blob for kw in _REAL_CREDENTIALS_KEYWORDS):
        return (
            CredentialEase.user_must_provide_real_credentials,
            "The provider's stated policy requires institutional affiliation, identity verification, or a reviewed application -- the person must supply their own real details.",
        )
    if any(kw in blob for kw in _SELF_REGISTER_KEYWORDS):
        return (
            CredentialEase.agent_can_self_register,
            "The provider explicitly describes lightweight self-service signup -- Agent 4 can register with any working email address on the user's behalf.",
        )

    # Tier 3: genuinely no signal. CHANGED (bug fix): this used to
    # default to "agent_can_self_register" -- a guess dressed up as a
    # finding, and since most sources hit exactly this fallback, it
    # produced the same boilerplate line for nearly every login-required
    # source, which is indistinguishable from not having this feature
    # at all. Now it honestly reports that this specific question is
    # unanswered rather than asserting an unverified answer.
    return (
        CredentialEase.unknown,
        "Login is required, but nothing in this provider's known profile or extracted metadata indicates whether registration is self-service or requires real identity/affiliation. This has NOT been determined -- confirm by visiting the provider's registration page, or have Agent 4 attempt self-registration as a probe.",
    )


def _classify_timeline(access: AccessClassification, blob: str) -> AvailabilityTimeline:
    if access == AccessClassification.free:
        return AvailabilityTimeline.immediate
    if any(kw in blob for kw in _WAITING_PERIOD_KEYWORDS):
        return AvailabilityTimeline.waiting_period
    if any(kw in blob for kw in _FEW_DAYS_KEYWORDS):
        return AvailabilityTimeline.few_days_approval
    if any(kw in blob for kw in _MANUAL_APPROVAL_KEYWORDS):
        return AvailabilityTimeline.manual_approval
    if access == AccessClassification.simple_login_required:
        return AvailabilityTimeline.immediate
    return AvailabilityTimeline.manual_approval


def _retrieval_speed_label(latency_ms) -> str:
    if latency_ms is None:
        return "unknown (source unreachable during probe)"
    if latency_ms < 500:
        return "fast"
    if latency_ms < 2000:
        return "moderate"
    return "slow"


def _build_accessibility(candidate, score_card) -> AccessibilityProfile:
    blob = _text_blob(candidate)
    access = _classify_access(candidate)
    credential_ease, credential_notes = _classify_credential_ease(candidate, access, blob)
    timeline = _classify_timeline(access, blob)

    authentication_required = access != AccessClassification.free
    real_time_score = score_card.real_time_availability.score if score_card else 0.5
    latency = getattr(candidate, "response_latency_ms", None)

    rate_limits = _tristate_from_keywords(blob, _RATE_LIMIT_KEYWORDS, _NO_RATE_LIMIT_KEYWORDS)
    download_restrictions = TriState.yes if any(k in blob for k in _DOWNLOAD_RESTRICTION_KEYWORDS) else TriState.unknown
    uptime = _tristate_from_keywords(blob, _UPTIME_GOOD_KEYWORDS, _UPTIME_BAD_KEYWORDS)

    formats = [f.value if hasattr(f, "value") else str(f) for f in (getattr(candidate, "available_formats", None) or [])]

    payment_required = access == AccessClassification.paid_access
    price = getattr(candidate, "price_estimate", None)
    payment_notes = (
        f"Paid access. {'Estimated cost: ' + price if price else 'Provider does not state a price in the available metadata -- confirm on the provider site before proceeding.'}"
        if payment_required else "No payment required."
    )

    # Composite accessibility score -- equal-weighted blend of the
    # requested factors that are numerically expressible. Ease of
    # access (credential_ease) is weighted most heavily since that was
    # the specific concern raised: an agent-self-registerable free-tier
    # source should clearly outrank one needing the person's real
    # identity, which should in turn outrank paid access.
    ease_score = {
        CredentialEase.no_credentials_needed: 1.0,
        CredentialEase.agent_can_self_register: 0.75,
        CredentialEase.user_must_provide_real_credentials: 0.35,
        CredentialEase.unknown: 0.5,
    }[credential_ease]
    timeline_score = {
        AvailabilityTimeline.immediate: 1.0,
        AvailabilityTimeline.manual_approval: 0.55,
        AvailabilityTimeline.few_days_approval: 0.35,
        AvailabilityTimeline.waiting_period: 0.2,
        AvailabilityTimeline.unknown: 0.5,
    }[timeline]
    payment_score = 0.2 if payment_required else 1.0
    api_score = 1.0 if getattr(candidate, "api_type", None) else 0.6
    format_score = min(1.0, 0.4 + 0.15 * len(formats))

    composite = (
        0.30 * ease_score
        + 0.20 * payment_score
        + 0.15 * timeline_score
        + 0.15 * real_time_score
        + 0.10 * api_score
        + 0.10 * format_score
    )

    return AccessibilityProfile(
        authentication_required=authentication_required,
        anonymous_access_available=not authentication_required,
        api_available=bool(getattr(candidate, "api_type", None)),
        api_type=(candidate.api_type.value if getattr(candidate, "api_type", None) else "unknown"),
        download_restrictions=download_restrictions,
        download_restrictions_notes="Restriction language found in provider metadata." if download_restrictions == TriState.yes else "No restriction language found; not confirmed unrestricted.",
        rate_limits=rate_limits,
        rate_limits_notes="Rate-limit language found in provider metadata." if rate_limits == TriState.yes else "",
        registration_required=authentication_required,
        credential_ease=credential_ease,
        credential_ease_notes=credential_notes,
        real_time_availability_score=real_time_score,
        retrieval_speed_ms=latency,
        retrieval_speed_label=_retrieval_speed_label(latency),
        server_uptime=uptime,
        server_uptime_notes="" if uptime == TriState.unknown else "Derived from provider-stated language.",
        supported_download_formats=formats,
        payment_required=payment_required,
        payment_notes=payment_notes,
        access_classification=access,
        availability_timeline=timeline,
        timeline_notes={
            AvailabilityTimeline.immediate: "Usable as soon as access is obtained.",
            AvailabilityTimeline.manual_approval: "A human reviewer must approve the request.",
            AvailabilityTimeline.few_days_approval: "Provider states an approval turnaround of roughly a few days.",
            AvailabilityTimeline.waiting_period: "An embargo or release-delay period applies.",
            AvailabilityTimeline.unknown: "Timeline not stated in available metadata.",
        }[timeline],
        accessibility_composite_score=round(composite, 3),
    )


# --------------------------------------------------------------------- #
# AVAILABILITY                                                           #
# --------------------------------------------------------------------- #

def _requested_variables(request: RetrievalRequest) -> List[str]:
    variables = set()
    for var in getattr(request, "variables", []) or []:
        v = getattr(var, "variable", None)
        if v:
            variables.add(v.lower().strip())
    for meas in getattr(request, "measurements", []) or []:
        v = getattr(meas, "variable_measured", None)
        if v:
            variables.add(v.lower().strip())
    return sorted(variables)


# Public alias -- callers outside this module (e.g. the interactive
# runner) should use this name rather than reaching into the
# underscore-prefixed internal helper.
get_requested_variables = _requested_variables


# --------------------------------------------------------------------- #
# Variable matching                                                      #
# --------------------------------------------------------------------- #
# CHANGED (bug fix): the previous version compared requested variable
# names to a source's variables_available list with exact string
# equality ("sea surface temperature" == "sst" -> False). Real-world
# catalogs almost never use full, identically-phrased variable names,
# so this made nearly every source -- including obvious SST providers
# like NASA PO.DAAC, HYCOM, and Copernicus -- come back as "0/1
# covered, Not Available". Replaced with a small synonym table plus
# substring/acronym matching. Still heuristic, not perfect, but far
# closer to reality; documented as such rather than presented as exact.

_VARIABLE_SYNONYMS = {
    "sea surface temperature": {"sst", "skin sst", "temperature", "sea temperature", "water temperature"},
    "chlorophyll": {"chl", "chl-a", "chlorophyll-a", "chlorophyll a", "ocean color", "ocean colour"},
    "salinity": {"sss", "sea surface salinity", "psu", "psal"},
    "sea level": {"ssh", "sea surface height", "altimetry", "sea level anomaly", "sla"},
    "wind speed": {"wind", "u10", "v10", "wind vector"},
    "precipitation": {"rain", "rainfall", "precip"},
    "wave height": {"swh", "significant wave height", "wave"},
    "ocean currents": {"current", "currents", "ocean current velocity", "uv velocity"},
}


def _acronym(phrase: str) -> str:
    return "".join(word[0] for word in phrase.split() if word)


def _variable_matches(requested: str, candidate_var: str) -> bool:
    """True if `candidate_var` (from the source's own metadata) plausibly
    satisfies `requested` (from the user's query). Order: exact match,
    substring either direction, synonym table, acronym match."""
    r, c = requested.strip(), candidate_var.strip()
    if not r or not c:
        return False
    if r == c or r in c or c in r:
        return True
    synonyms = _VARIABLE_SYNONYMS.get(r, set())
    if c in synonyms or any(s in c for s in synonyms):
        return True
    if _acronym(r) == c or _acronym(r) in c:
        return True
    return False


def _build_availability(candidate, score_card, request: RetrievalRequest) -> AvailabilityProfile:
    requested_vars = _requested_variables(request)
    raw_candidate_vars = [v.lower().strip() for v in (getattr(candidate, "variables_available", None) or []) if v]

    # CHANGED (bug fix): distinguish "we checked and it's genuinely
    # missing" from "this source's variable list was never extracted".
    # Asserting the latter as "Not Available" was actively misleading --
    # a source can be an excellent SST provider even if Agent 3's Phase
    # 3 enrichment simply didn't populate variables_available for it.
    variables_extracted = bool(raw_candidate_vars)

    if variables_extracted:
        covered = [rv for rv in requested_vars if any(_variable_matches(rv, cv) for cv in raw_candidate_vars)]
        missing = [rv for rv in requested_vars if rv not in covered]
        var_score = (len(covered) / len(requested_vars)) if requested_vars else 1.0
    else:
        covered, missing = [], []
        # Neutral, not zero: unextracted metadata is not evidence of absence.
        var_score = 0.5 if requested_vars else 1.0

    spatial_score = score_card.geographic_match.score if score_card else 0.5
    temporal_score = score_card.temporal_match.score if score_card else 0.5
    resolution_score = score_card.resolution.score if score_card else 0.5
    relevance_score = score_card.relevance.score if score_card else 0.5
    completeness_score = score_card.completeness.score if score_card else 0.5

    coverage_text = (getattr(candidate, "temporal_coverage", "") or "").lower()
    # Best-effort historical depth proxy: longer explicit year ranges
    # score higher. This is NOT a verified fact -- see the notes field.
    historical_score = 0.5
    if "present" in coverage_text or "real-time" in coverage_text:
        historical_score = 0.7
    for decade_start in (1980, 1990, 2000):
        if str(decade_start) in coverage_text:
            historical_score = 0.9
            break

    data_continuity = TriState.unknown
    missing_values = TriState.unknown

    if not requested_vars and var_score == 1.0:
        status = DataAvailabilityStatus.full
    elif not variables_extracted:
        status = DataAvailabilityStatus.unknown
    elif covered and not missing:
        status = DataAvailabilityStatus.full
    elif covered:
        status = DataAvailabilityStatus.partial
    else:
        status = DataAvailabilityStatus.not_available if requested_vars else DataAvailabilityStatus.full

    # Composite -- equal-weighted average across the nine requested
    # factors (the two unverifiable ones contribute their neutral 0.5
    # default rather than being excluded, so they don't silently
    # distort the composite in either direction).
    composite = (
        relevance_score + completeness_score + var_score + spatial_score
        + temporal_score + resolution_score + historical_score + 0.5 + 0.5
    ) / 9.0

    return AvailabilityProfile(
        relevance_score=relevance_score,
        completeness_score=completeness_score,
        requested_variables=requested_vars,
        covered_variables=covered,
        missing_variables=missing,
        variable_availability_score=round(var_score, 3),
        spatial_coverage_score=spatial_score,
        spatial_coverage_notes=score_card.geographic_match.explanation if score_card else "",
        temporal_coverage_score=temporal_score,
        temporal_coverage_notes=score_card.temporal_match.explanation if score_card else "",
        resolution_score=resolution_score,
        resolution_notes=f"Spatial resolution: {getattr(candidate, 'spatial_resolution', 'Unknown')}.",
        historical_coverage_score=historical_score,
        data_continuity=data_continuity,
        missing_values=missing_values,
        availability_status=status,
        availability_composite_score=round(composite, 3),
    )


# --------------------------------------------------------------------- #
# ACCURACY                                                                #
# --------------------------------------------------------------------- #

def _build_accuracy(score_card) -> AccuracyProfile:
    authority = score_card.authority.score if score_card else 0.5
    historical_reliability = score_card.historical_reliability.score if score_card else 0.5
    credibility = round((authority + historical_reliability) / 2.0, 3)
    scientific_acceptance = score_card.scientific_acceptance.score if score_card else 0.5
    consistency = score_card.consistency.score if score_card else 0.5
    metadata_quality = score_card.metadata_quality.score if score_card else 0.5

    composite = (
        authority + credibility + scientific_acceptance
        + consistency + historical_reliability + metadata_quality
    ) / 6.0

    return AccuracyProfile(
        authority_score=authority,
        credibility_score=credibility,
        scientific_acceptance_score=scientific_acceptance,
        consistency_score=consistency,
        historical_reliability_score=historical_reliability,
        metadata_quality_score=metadata_quality,
        accuracy_composite_score=round(composite, 3),
    )


# --------------------------------------------------------------------- #
# Entry point                                                            #
# --------------------------------------------------------------------- #

def analyze_websites(scored_sources: List, request: RetrievalRequest) -> Dict[str, WebsiteAnalysisResult]:
    """Analyzes every candidate that survived Phase 5's rejection process
    (accepted + auth_required + needs_evaluation -- never call this with
    rejected sources). Never raises; a failure on one candidate is
    logged and skipped rather than aborting the batch."""
    results: Dict[str, WebsiteAnalysisResult] = {}
    for scored in scored_sources:
        candidate = getattr(scored, "candidate", scored)
        score_card = getattr(scored, "score_card", None)
        try:
            results[candidate.source_id] = WebsiteAnalysisResult(
                source_id=candidate.source_id,
                website_name=candidate.name,
                accessibility=_build_accessibility(candidate, score_card),
                availability=_build_availability(candidate, score_card, request),
                accuracy=_build_accuracy(score_card),
                data_policy_summary=(getattr(candidate, "description", "") or "No data policy text extracted for this source.")[:400],
            )
        except Exception as exc:
            print(f"[Website Analyzer] Skipped {getattr(candidate, 'source_id', '?')}: {exc}")
            continue
    return results


def suggest_complementary_combination(
    analyses: Dict[str, WebsiteAnalysisResult],
    requested_variables: List[str],
) -> List[str]:
    """
    Implements the "Website A + B > C" example: greedily picks the
    smallest set of sources whose COMBINED covered_variables satisfies
    the most of requested_variables, breaking ties by higher individual
    availability_composite_score. Returns an ordered list of source_ids.
    Purely a presentation aid -- does not change any individual
    source's rank or status.
    """
    if not requested_variables:
        return []

    remaining = set(v.lower().strip() for v in requested_variables)
    candidates = list(analyses.values())
    chosen: List[str] = []

    while remaining and candidates:
        best = max(
            candidates,
            key=lambda a: (
                len(remaining & set(a.availability.covered_variables)),
                a.availability.availability_composite_score,
            ),
        )
        gained = remaining & set(best.availability.covered_variables)
        if not gained:
            break
        chosen.append(best.source_id)
        remaining -= gained
        candidates.remove(best)

    return chosen
