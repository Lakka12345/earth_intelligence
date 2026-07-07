"""
security/provider_trust.py

Provider Trust Evaluation Module
=================================

Independent, standalone security module for the Earth Intelligence
multi-agent platform.

Purpose
-------
Evaluate every candidate scientific data provider against ten trust
criteria and produce a transparent, explainable ``overall_trust_score``
in the range ``0.0`` - ``1.0``.

This module NEVER rejects a provider. It only evaluates, scores, ranks
and explains its reasoning. The decision of which dataset to actually
download is left entirely to Agent 3, which will consume the reports
produced here.

Design principle: no score is ever hardcoded per-provider. Every
``evaluate_*`` function derives its score exclusively from the metadata
dictionary supplied for that provider (plus, where relevant, the
requested scientific task and the historical retrieval ledger stored in
``provider_trust_db.json``). Two providers that supply identical
metadata will always receive identical scores. Each score is returned
together with a short human-readable explanation describing exactly
which metadata fields produced it.

This module is fully standalone:

    python security/provider_trust.py

It does not import, call, or depend on Agent 3, Agent 4, or any other
security module.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(MODULE_DIR, "provider_trust_db.json")

# Relative weight of each criterion in the final weighted trust score.
# Weights are fixed policy, not per-provider scores, and sum to 1.0.
CRITERIA_WEIGHTS: Dict[str, float] = {
    "authority_score": 0.15,
    "freshness_score": 0.08,
    "relevance_score": 0.15,
    "resolution_score": 0.10,
    "completeness_score": 0.10,
    "consistency_score": 0.08,
    "metadata_quality_score": 0.08,
    "historical_reliability_score": 0.10,
    "scientific_acceptance_score": 0.10,
    "real_time_availability_score": 0.06,
}

TRUST_LEVEL_THRESHOLDS: Tuple[Tuple[float, str], ...] = (
    (0.90, "Very High Trust"),
    (0.75, "High Trust"),
    (0.60, "Moderate Trust"),
    (0.40, "Low Trust"),
    (0.00, "Very Low Trust"),
)

# Domain suffixes conventionally associated with government / international
# / academic bodies. Used only as one transparent signal among several for
# the authority criterion -- never as a direct score assignment.
GOVERNMENT_OR_TREATY_DOMAIN_SUFFIXES = (".gov", ".int", ".europa.eu", ".mil")
ACADEMIC_OR_NONPROFIT_DOMAIN_SUFFIXES = (".edu", ".org", ".ac.uk")

# Keyword sets describing which data domains satisfy which requested
# scientific tasks. This is generic policy configuration (not tied to any
# specific provider) used by evaluate_relevance().
TASK_KEYWORD_MAP: Dict[str, Sequence[str]] = {
    "flood_analysis": ("rainfall", "precipitation", "hydrology", "river_discharge",
                        "soil_moisture", "weather", "flood"),
    "drought_monitoring": ("rainfall", "precipitation", "soil_moisture",
                            "vegetation_index", "temperature", "drought"),
    "ocean_forecasting": ("ocean", "sea_surface_temperature", "currents",
                           "salinity", "wave_height", "marine"),
    "cyclone_tracking": ("wind", "pressure", "satellite_imagery", "ocean",
                          "weather", "cyclone", "storm"),
    "earth_observation": ("satellite_imagery", "land_cover", "vegetation_index",
                           "climate", "atmosphere", "ocean", "weather"),
}

# Reasonable default resolution requirements per task, used only when the
# caller does not supply explicit requirements. Generic policy, not
# provider-specific.
DEFAULT_RESOLUTION_REQUIREMENTS: Dict[str, Dict[str, float]] = {
    "flood_analysis": {"spatial_resolution_km": 5.0, "temporal_resolution_hours": 6.0},
    "drought_monitoring": {"spatial_resolution_km": 10.0, "temporal_resolution_hours": 24.0},
    "ocean_forecasting": {"spatial_resolution_km": 10.0, "temporal_resolution_hours": 12.0},
    "cyclone_tracking": {"spatial_resolution_km": 5.0, "temporal_resolution_hours": 3.0},
    "earth_observation": {"spatial_resolution_km": 10.0, "temporal_resolution_hours": 24.0},
}

# Bayesian smoothing prior for historical reliability, applied so that a
# provider with little or no retrieval history is not unfairly scored at
# the extremes. This is a neutral statistical prior, not a per-provider
# score.
HISTORICAL_PRIOR_SUCCESS_RATE = 0.5
HISTORICAL_PRIOR_WEIGHT = 4.0


# --------------------------------------------------------------------------
# EXCEPTIONS
# --------------------------------------------------------------------------

class ProviderTrustError(Exception):
    """Base exception for provider trust evaluation failures."""


class ProviderMetadataError(ProviderTrustError):
    """Raised when required provider metadata is missing or malformed."""


class ProviderTrustDatabaseError(ProviderTrustError):
    """Raised when the provider trust database cannot be read or written."""


class ProviderNotFoundError(ProviderTrustError):
    """Raised when a requested provider has no stored report."""


# --------------------------------------------------------------------------
# SMALL HELPERS
# --------------------------------------------------------------------------

def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp ``value`` into the inclusive range [low, high]."""
    return max(low, min(high, value))


def _round_score(value: float) -> float:
    """Round a score to 4 decimal places for stable, readable output."""
    return round(_clamp(value), 4)


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _extract_domain_suffix(url: str) -> str:
    """Best-effort extraction of a domain suffix (e.g. '.gov') from a URL."""
    if not url:
        return ""
    lowered = url.lower()
    for suffix in GOVERNMENT_OR_TREATY_DOMAIN_SUFFIXES + ACADEMIC_OR_NONPROFIT_DOMAIN_SUFFIXES + (".com", ".net", ".io"):
        if suffix in lowered:
            return suffix
    return ""


def _days_since(date_str: Optional[str]) -> Optional[float]:
    """Return the number of days between ``date_str`` (ISO date) and now."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            parsed = datetime.strptime(date_str, fmt)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            delta = datetime.now(timezone.utc) - parsed
            return max(delta.total_seconds() / 86400.0, 0.0)
        except ValueError:
            continue
    return None


@dataclass
class CriterionResult:
    """Result of evaluating a single trust criterion."""
    score: float
    explanation: str
    signals: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"score": _round_score(self.score), "explanation": self.explanation}


# --------------------------------------------------------------------------
# CRITERION 1: AUTHORITY
# --------------------------------------------------------------------------

def evaluate_authority(provider: Dict[str, Any]) -> CriterionResult:
    """
    Evaluate the credibility/authority of the organization behind a
    provider, derived entirely from the metadata it supplies.

    Signals used (all optional, default to the conservative/low end when
    absent):
      - is_government_agency (bool)
      - is_international_organization (bool)
      - has_regulatory_mandate (bool): official mandate over the domain
        (e.g. national meteorological/space authority)
      - provider_url domain suffix (.gov/.int/.europa.eu -> strong signal,
        .edu/.org -> moderate signal, .com -> neutral)
      - founding_year (int): longevity/establishment bonus
      - partner_agencies (list[str]): collaborations with other
        recognized scientific/regulatory bodies
    """
    reasons: List[str] = []
    score = 0.25  # conservative baseline for an unknown organization

    if provider.get("is_government_agency"):
        score += 0.22
        reasons.append("is a government agency (+0.22)")
    if provider.get("is_international_organization"):
        score += 0.18
        reasons.append("is an international/treaty-based organization (+0.18)")
    if provider.get("has_regulatory_mandate"):
        score += 0.15
        reasons.append("holds an official regulatory mandate for this data domain (+0.15)")

    suffix = _extract_domain_suffix(provider.get("provider_url", ""))
    if suffix in GOVERNMENT_OR_TREATY_DOMAIN_SUFFIXES:
        score += 0.10
        reasons.append(f"domain suffix '{suffix}' associated with government/treaty bodies (+0.10)")
    elif suffix in ACADEMIC_OR_NONPROFIT_DOMAIN_SUFFIXES:
        score += 0.05
        reasons.append(f"domain suffix '{suffix}' associated with academic/nonprofit bodies (+0.05)")

    founding_year = provider.get("founding_year")
    if isinstance(founding_year, (int, float)):
        age_years = max(datetime.now(timezone.utc).year - int(founding_year), 0)
        longevity_bonus = _clamp(age_years / 30.0, 0.0, 0.10)
        if longevity_bonus > 0:
            score += longevity_bonus
            reasons.append(f"established {age_years} years ago (+{longevity_bonus:.2f})")

    partner_agencies = provider.get("partner_agencies") or []
    if isinstance(partner_agencies, list) and partner_agencies:
        partner_bonus = _clamp(len(partner_agencies) * 0.02, 0.0, 0.10)
        score += partner_bonus
        reasons.append(f"collaborates with {len(partner_agencies)} recognized partner agencies (+{partner_bonus:.2f})")

    if not reasons:
        reasons.append("no authority-related metadata supplied; scored at conservative baseline")

    explanation = "Authority derived from: " + "; ".join(reasons) + "."
    return CriterionResult(score=score, explanation=explanation)


# --------------------------------------------------------------------------
# CRITERION 2: FRESHNESS
# --------------------------------------------------------------------------

def evaluate_freshness(provider: Dict[str, Any]) -> CriterionResult:
    """
    Evaluate data freshness from last_updated timestamp, declared update
    frequency, and whether a recent dataset is flagged as available.
    """
    reasons: List[str] = []
    last_updated = provider.get("last_updated")
    update_frequency_days = provider.get("update_frequency_days")
    has_recent_dataset = provider.get("has_recent_dataset")

    age_days = _days_since(last_updated)

    if age_days is None:
        score = 0.30
        reasons.append("no 'last_updated' timestamp supplied; scored at low baseline")
    else:
        if isinstance(update_frequency_days, (int, float)) and update_frequency_days > 0:
            # How many update cycles behind is the data?
            cycles_behind = age_days / float(update_frequency_days)
            score = _clamp(1.0 - (cycles_behind * 0.35))
            reasons.append(
                f"last updated {age_days:.1f} days ago against a declared "
                f"{update_frequency_days}-day update cycle ({cycles_behind:.2f} cycles behind)"
            )
        else:
            # No declared cadence: fall back to absolute recency bands.
            if age_days <= 1:
                score = 0.95
            elif age_days <= 7:
                score = 0.85
            elif age_days <= 30:
                score = 0.65
            elif age_days <= 90:
                score = 0.45
            else:
                score = 0.20
            reasons.append(f"last updated {age_days:.1f} days ago; no declared update cadence supplied")

    if has_recent_dataset is True:
        score = _clamp(score + 0.05)
        reasons.append("provider flags a recent dataset as available (+0.05)")
    elif has_recent_dataset is False:
        score = _clamp(score - 0.10)
        reasons.append("provider explicitly flags no recent dataset available (-0.10)")

    explanation = "Freshness derived from: " + "; ".join(reasons) + "."
    return CriterionResult(score=score, explanation=explanation)


# --------------------------------------------------------------------------
# CRITERION 3: RELEVANCE
# --------------------------------------------------------------------------

def evaluate_relevance(provider: Dict[str, Any], requested_task: Optional[str]) -> CriterionResult:
    """
    Measure how well a provider's declared data domains match the
    scientific task being requested (e.g. 'flood_analysis').
    """
    provider_topics = {t.lower().strip() for t in provider.get("data_domains", []) if isinstance(t, str)}

    if not requested_task:
        if provider_topics:
            explanation = (
                "No requested_task supplied for comparison; relevance cannot be "
                "computed against a target task, scored at neutral baseline."
            )
            return CriterionResult(score=0.50, explanation=explanation)
        explanation = "No requested_task and no provider data_domains supplied; scored at low baseline."
        return CriterionResult(score=0.20, explanation=explanation)

    task_keywords = {k.lower() for k in TASK_KEYWORD_MAP.get(requested_task.lower(), (requested_task.lower(),))}

    if not provider_topics:
        explanation = (
            f"Provider supplied no 'data_domains' metadata to compare against the "
            f"requested task '{requested_task}'; scored at low baseline."
        )
        return CriterionResult(score=0.15, explanation=explanation)

    overlap = task_keywords & provider_topics
    ratio = len(overlap) / len(task_keywords) if task_keywords else 0.0
    score = _clamp(0.05 + ratio * 0.95)  # small floor so total mismatch isn't literally zero

    if overlap:
        explanation = (
            f"{len(overlap)}/{len(task_keywords)} keywords for task '{requested_task}' matched "
            f"provider data domains ({sorted(overlap)})."
        )
    else:
        explanation = (
            f"None of the keywords for task '{requested_task}' "
            f"({sorted(task_keywords)}) matched provider data domains "
            f"({sorted(provider_topics)})."
        )

    return CriterionResult(score=score, explanation=explanation)


# --------------------------------------------------------------------------
# CRITERION 4: RESOLUTION
# --------------------------------------------------------------------------

def evaluate_resolution(provider: Dict[str, Any], requested_task: Optional[str]) -> CriterionResult:
    """
    Evaluate spatial and temporal resolution suitability against either
    explicit requirements or task-based defaults.
    """
    requirements = DEFAULT_RESOLUTION_REQUIREMENTS.get(
        (requested_task or "").lower(),
        {"spatial_resolution_km": 10.0, "temporal_resolution_hours": 24.0},
    )
    required_spatial = provider.get("required_spatial_resolution_km", requirements["spatial_resolution_km"])
    required_temporal = provider.get("required_temporal_resolution_hours", requirements["temporal_resolution_hours"])

    spatial = provider.get("spatial_resolution_km")
    temporal = provider.get("temporal_resolution_hours")

    reasons: List[str] = []
    components: List[float] = []

    if isinstance(spatial, (int, float)) and spatial > 0:
        spatial_ratio = _clamp(required_spatial / spatial)  # finer than required -> up to 1.0, capped
        components.append(spatial_ratio)
        reasons.append(
            f"spatial resolution {spatial}km vs required {required_spatial}km "
            f"(ratio {spatial_ratio:.2f})"
        )
    else:
        components.append(0.30)
        reasons.append("no spatial_resolution_km supplied; scored at low baseline for this component")

    if isinstance(temporal, (int, float)) and temporal > 0:
        temporal_ratio = _clamp(required_temporal / temporal)
        components.append(temporal_ratio)
        reasons.append(
            f"temporal resolution {temporal}h vs required {required_temporal}h "
            f"(ratio {temporal_ratio:.2f})"
        )
    else:
        components.append(0.30)
        reasons.append("no temporal_resolution_hours supplied; scored at low baseline for this component")

    score = sum(components) / len(components)
    explanation = "Resolution derived from: " + "; ".join(reasons) + "."
    return CriterionResult(score=score, explanation=explanation)


# --------------------------------------------------------------------------
# CRITERION 5: COMPLETENESS
# --------------------------------------------------------------------------

def evaluate_completeness(provider: Dict[str, Any]) -> CriterionResult:
    """
    Evaluate variable coverage, documentation presence, and declared
    spatial/temporal coverage percentage.
    """
    reasons: List[str] = []
    components: List[float] = []

    variables_provided = {v.lower() for v in provider.get("variables_provided", []) if isinstance(v, str)}
    expected_variables = {v.lower() for v in provider.get("expected_variables", []) if isinstance(v, str)}

    if expected_variables:
        overlap = variables_provided & expected_variables
        var_ratio = len(overlap) / len(expected_variables)
        components.append(var_ratio)
        reasons.append(
            f"{len(overlap)}/{len(expected_variables)} expected variables present ({sorted(overlap)})"
        )
    elif variables_provided:
        # No expectation supplied; reward breadth modestly.
        var_ratio = _clamp(len(variables_provided) / 10.0)
        components.append(var_ratio)
        reasons.append(f"{len(variables_provided)} variables declared (no expected-variable list supplied)")
    else:
        components.append(0.10)
        reasons.append("no variables_provided or expected_variables supplied")

    coverage_percent = provider.get("coverage_percent")
    if isinstance(coverage_percent, (int, float)):
        coverage_ratio = _clamp(coverage_percent / 100.0)
        components.append(coverage_ratio)
        reasons.append(f"declared spatial/temporal coverage {coverage_percent}%")
    else:
        components.append(0.40)
        reasons.append("no coverage_percent supplied; scored at low-moderate baseline")

    if provider.get("has_documentation"):
        components.append(0.90)
        reasons.append("documentation is available")
    else:
        components.append(0.20)
        reasons.append("no documentation flagged as available")

    score = sum(components) / len(components)
    explanation = "Completeness derived from: " + "; ".join(reasons) + "."
    return CriterionResult(score=score, explanation=explanation)


# --------------------------------------------------------------------------
# CRITERION 6: CONSISTENCY
# --------------------------------------------------------------------------

def evaluate_consistency(provider: Dict[str, Any]) -> CriterionResult:
    """
    Evaluate agreement with other candidate providers for the same
    variables/region, using a declared agreement ratio if supplied.
    """
    agreement = provider.get("agreement_with_other_providers")

    if isinstance(agreement, (int, float)):
        score = _clamp(float(agreement))
        explanation = (
            f"Consistency derived from a declared agreement ratio of {agreement:.2f} "
            f"with other candidate providers covering the same variables/region."
        )
        return CriterionResult(score=score, explanation=explanation)

    deviation_flags = provider.get("deviation_flags")
    if isinstance(deviation_flags, list):
        if not deviation_flags:
            explanation = "No deviation flags recorded against other providers; scored high."
            return CriterionResult(score=0.85, explanation=explanation)
        score = _clamp(1.0 - (len(deviation_flags) * 0.15))
        explanation = (
            f"{len(deviation_flags)} deviation flag(s) recorded against other providers "
            f"({deviation_flags}); score reduced accordingly."
        )
        return CriterionResult(score=score, explanation=explanation)

    explanation = "No cross-provider comparison data supplied; scored at neutral baseline."
    return CriterionResult(score=0.50, explanation=explanation)


# --------------------------------------------------------------------------
# CRITERION 7: METADATA QUALITY
# --------------------------------------------------------------------------

def evaluate_metadata_quality(provider: Dict[str, Any]) -> CriterionResult:
    """
    Evaluate documentation, units, CRS, variable descriptions and
    licensing clarity.
    """
    reasons: List[str] = []
    checks = {
        "has_documentation": provider.get("has_documentation"),
        "units_documented": provider.get("units_documented"),
        "crs_documented": provider.get("crs_documented"),
        "has_metadata_standard": provider.get("has_metadata_standard"),
    }
    passed = sum(1 for v in checks.values() if v is True)
    total = len(checks)
    base = passed / total
    for name, value in checks.items():
        reasons.append(f"{name}={bool(value)}")

    license_type = str(provider.get("license_type", "unknown")).lower()
    license_bonus = {"open": 0.10, "restricted": 0.0, "proprietary": -0.05, "unknown": -0.05}.get(license_type, -0.05)
    reasons.append(f"license_type='{license_type}' ({license_bonus:+.2f})")

    score = _clamp(base * 0.9 + 0.10 + license_bonus)
    explanation = "Metadata quality derived from: " + "; ".join(reasons) + "."
    return CriterionResult(score=score, explanation=explanation)


# --------------------------------------------------------------------------
# CRITERION 8: HISTORICAL RELIABILITY
# --------------------------------------------------------------------------

def get_mock_historical_data(provider_name: str, db: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the historical retrieval ledger for ``provider_name`` from the
    trust database.

    If no history exists yet, an empty ledger (0 successes, 0 failures)
    is returned. This is intentionally NOT a hardcoded score -- with zero
    samples, evaluate_historical_reliability() will fall back to the
    Bayesian prior, i.e. a neutral score with an explicit low-confidence
    explanation.

    Agent 4 is expected to update this ledger automatically in
    production as real retrievals complete; for now, the demo below
    seeds a few illustrative mock ledgers directly into the database.
    """
    history = db.get("historical_data", {})
    return history.get(provider_name, {"successful_retrievals": 0, "failed_retrievals": 0, "last_retrieval": None})


def evaluate_historical_reliability(provider: Dict[str, Any], db: Dict[str, Any]) -> CriterionResult:
    """
    Evaluate historical reliability using a Bayesian-smoothed success
    rate computed from successful/failed retrieval counts stored in the
    trust database (mocked for now; Agent 4 updates these later).
    """
    provider_name = provider.get("provider_name", "")
    ledger = get_mock_historical_data(provider_name, db)
    successes = int(ledger.get("successful_retrievals", 0) or 0)
    failures = int(ledger.get("failed_retrievals", 0) or 0)
    total = successes + failures

    smoothed = (successes + HISTORICAL_PRIOR_SUCCESS_RATE * HISTORICAL_PRIOR_WEIGHT) / \
               (total + HISTORICAL_PRIOR_WEIGHT)
    score = _clamp(smoothed)

    if total == 0:
        explanation = (
            "No historical retrieval records found for this provider; score defaults "
            "to the neutral statistical prior pending data from Agent 4."
        )
    else:
        raw_rate = successes / total
        explanation = (
            f"{successes} successful / {failures} failed retrievals on record "
            f"(raw success rate {raw_rate:.2%}), Bayesian-smoothed toward the "
            f"neutral prior for sample size (n={total}) to avoid overconfidence "
            f"on small histories."
        )

    return CriterionResult(score=score, explanation=explanation, signals={"successes": successes, "failures": failures})


# --------------------------------------------------------------------------
# CRITERION 9: SCIENTIFIC ACCEPTANCE
# --------------------------------------------------------------------------

def evaluate_scientific_acceptance(provider: Dict[str, Any]) -> CriterionResult:
    """
    Evaluate acceptance by the scientific community using citation
    counts, peer-reviewed publication counts, and the number of other
    known scientific/operational agencies that use this provider.
    """
    reasons: List[str] = []
    components: List[float] = []

    citation_count = provider.get("citation_count")
    if isinstance(citation_count, (int, float)):
        citation_score = _clamp(citation_count / 2000.0)
        components.append(citation_score)
        reasons.append(f"{int(citation_count)} scientific citations (scaled score {citation_score:.2f})")
    else:
        components.append(0.20)
        reasons.append("no citation_count supplied; scored at low baseline for this component")

    peer_reviewed_publications = provider.get("peer_reviewed_publications")
    if isinstance(peer_reviewed_publications, (int, float)):
        pub_score = _clamp(peer_reviewed_publications / 500.0)
        components.append(pub_score)
        reasons.append(f"{int(peer_reviewed_publications)} peer-reviewed publications referencing this provider")
    else:
        components.append(0.20)
        reasons.append("no peer_reviewed_publications supplied; scored at low baseline for this component")

    used_by_agencies = provider.get("used_by_agencies") or []
    if isinstance(used_by_agencies, list) and used_by_agencies:
        usage_score = _clamp(len(used_by_agencies) / 10.0)
        components.append(usage_score)
        reasons.append(f"used operationally by {len(used_by_agencies)} other known agencies")
    else:
        components.append(0.15)
        reasons.append("no used_by_agencies supplied; scored at low baseline for this component")

    score = sum(components) / len(components)
    explanation = "Scientific acceptance derived from: " + "; ".join(reasons) + "."
    return CriterionResult(score=score, explanation=explanation)


# --------------------------------------------------------------------------
# CRITERION 10: REAL-TIME AVAILABILITY
# --------------------------------------------------------------------------

def evaluate_real_time_availability(provider: Dict[str, Any]) -> CriterionResult:
    """
    Evaluate API uptime, accessibility, and response latency.
    """
    reasons: List[str] = []
    components: List[float] = []

    uptime_percent = provider.get("api_uptime_percent")
    if isinstance(uptime_percent, (int, float)):
        uptime_score = _clamp(uptime_percent / 100.0)
        components.append(uptime_score)
        reasons.append(f"API uptime {uptime_percent}%")
    else:
        components.append(0.30)
        reasons.append("no api_uptime_percent supplied; scored at low baseline for this component")

    latency_ms = provider.get("avg_response_latency_ms")
    if isinstance(latency_ms, (int, float)) and latency_ms >= 0:
        # 100ms -> ~1.0, 2000ms -> ~0.05, decays smoothly.
        latency_score = _clamp(1.0 - (latency_ms / 2000.0))
        components.append(latency_score)
        reasons.append(f"average response latency {latency_ms}ms")
    else:
        components.append(0.30)
        reasons.append("no avg_response_latency_ms supplied; scored at low baseline for this component")

    accessible = provider.get("accessible")
    if accessible is True:
        components.append(0.95)
        reasons.append("endpoint reported as currently accessible")
    elif accessible is False:
        components.append(0.05)
        reasons.append("endpoint reported as currently INACCESSIBLE")
    else:
        components.append(0.40)
        reasons.append("no accessible flag supplied; scored at low-moderate baseline")

    score = sum(components) / len(components)
    explanation = "Real-time availability derived from: " + "; ".join(reasons) + "."
    return CriterionResult(score=score, explanation=explanation)


# --------------------------------------------------------------------------
# OVERALL SCORE
# --------------------------------------------------------------------------

def _trust_level_for(score: float) -> str:
    """Map a numeric overall trust score to its human-readable trust level."""
    for threshold, label in TRUST_LEVEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "Very Low Trust"


def calculate_overall_trust_score(criterion_scores: Dict[str, float]) -> float:
    """
    Compute the weighted overall trust score from the ten individual
    criterion scores using the fixed, transparent weights in
    ``CRITERIA_WEIGHTS``.
    """
    try:
        total = sum(criterion_scores[key] * weight for key, weight in CRITERIA_WEIGHTS.items())
    except KeyError as exc:
        raise ProviderMetadataError(f"Missing criterion score required for overall calculation: {exc}") from exc
    return _round_score(total)


# --------------------------------------------------------------------------
# REPORT GENERATION
# --------------------------------------------------------------------------

def generate_provider_trust_report(
    provider: Dict[str, Any],
    requested_task: Optional[str] = None,
    db: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run all ten evaluation criteria for a single provider and assemble a
    complete, explainable trust report.

    Parameters
    ----------
    provider:
        Dictionary of provider metadata. Must contain at least
        'provider_name'. All other fields are optional; missing fields
        degrade individual criterion scores transparently rather than
        raising errors.
    requested_task:
        Optional scientific task identifier (e.g. 'flood_analysis') used
        by the relevance and resolution criteria.
    db:
        Optional pre-loaded trust database (as returned by ``load_db``).
        If not supplied, the database is loaded from disk automatically
        so historical reliability can be evaluated.
    """
    if not isinstance(provider, dict) or not provider.get("provider_name"):
        raise ProviderMetadataError("provider metadata must be a dict containing at least 'provider_name'")

    if db is None:
        db = load_db()

    results: Dict[str, CriterionResult] = {
        "authority_score": evaluate_authority(provider),
        "freshness_score": evaluate_freshness(provider),
        "relevance_score": evaluate_relevance(provider, requested_task),
        "resolution_score": evaluate_resolution(provider, requested_task),
        "completeness_score": evaluate_completeness(provider),
        "consistency_score": evaluate_consistency(provider),
        "metadata_quality_score": evaluate_metadata_quality(provider),
        "historical_reliability_score": evaluate_historical_reliability(provider, db),
        "scientific_acceptance_score": evaluate_scientific_acceptance(provider),
        "real_time_availability_score": evaluate_real_time_availability(provider),
    }

    numeric_scores = {key: result.score for key, result in results.items()}
    overall = calculate_overall_trust_score(numeric_scores)
    trust_level = _trust_level_for(overall)

    ranked_criteria = sorted(results.items(), key=lambda kv: kv[1].score, reverse=True)
    strengths = [
        f"{name.replace('_score', '')}: {result.explanation}"
        for name, result in ranked_criteria[:3]
    ]
    weaknesses = [
        f"{name.replace('_score', '')}: {result.explanation}"
        for name, result in ranked_criteria[-3:]
    ]

    top_criterion = ranked_criteria[0][0].replace("_score", "")
    bottom_criterion = ranked_criteria[-1][0].replace("_score", "")
    ranking_reason = (
        f"Overall trust score {overall:.2f} ({trust_level}) is driven most positively by "
        f"'{top_criterion}' (score {ranked_criteria[0][1].score:.2f}) and most negatively by "
        f"'{bottom_criterion}' (score {ranked_criteria[-1][1].score:.2f}), combined via fixed "
        f"criterion weights {CRITERIA_WEIGHTS}."
    )

    recommendations: List[str] = []
    for name, result in results.items():
        if result.score < 0.40:
            readable = name.replace("_score", "").replace("_", " ")
            recommendations.append(
                f"Improve or verify '{readable}' before relying on this provider "
                f"(current score {result.score:.2f})."
            )
    if not recommendations:
        recommendations.append("No individual criterion fell below the Low Trust threshold (0.40).")

    report: Dict[str, Any] = {
        "provider_name": provider.get("provider_name"),
        "provider_url": provider.get("provider_url", ""),
        "requested_task": requested_task,
        "evaluation_timestamp": _now_iso(),
        "authority_score": results["authority_score"].as_dict(),
        "freshness_score": results["freshness_score"].as_dict(),
        "relevance_score": results["relevance_score"].as_dict(),
        "resolution_score": results["resolution_score"].as_dict(),
        "completeness_score": results["completeness_score"].as_dict(),
        "consistency_score": results["consistency_score"].as_dict(),
        "metadata_quality_score": results["metadata_quality_score"].as_dict(),
        "historical_reliability_score": results["historical_reliability_score"].as_dict(),
        "scientific_acceptance_score": results["scientific_acceptance_score"].as_dict(),
        "real_time_availability_score": results["real_time_availability_score"].as_dict(),
        "overall_trust_score": overall,
        "trust_level": trust_level,
        "ranking_reason": ranking_reason,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "recommendations": recommendations,
    }
    return report


# --------------------------------------------------------------------------
# DATABASE (ATOMIC READ / WRITE)
# --------------------------------------------------------------------------

def _empty_db() -> Dict[str, Any]:
    return {"reports": [], "historical_data": {}}


def ensure_db_exists(db_path: str = DB_PATH) -> None:
    """Create the provider trust database file with an empty skeleton if missing."""
    if not os.path.exists(db_path):
        _atomic_write_json(db_path, _empty_db())


def load_db(db_path: str = DB_PATH) -> Dict[str, Any]:
    """Load the provider trust database, creating it if necessary."""
    ensure_db_exists(db_path)
    try:
        with open(db_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        raise ProviderTrustDatabaseError(f"Failed to read provider trust database at {db_path}: {exc}") from exc

    if "reports" not in data or "historical_data" not in data:
        raise ProviderTrustDatabaseError(f"Provider trust database at {db_path} is missing required keys")
    return data


def _atomic_write_json(db_path: str, data: Dict[str, Any]) -> None:
    """Write ``data`` to ``db_path`` atomically (write to temp file, then replace)."""
    directory = os.path.dirname(db_path) or "."
    os.makedirs(directory, exist_ok=True)
    try:
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".provider_trust_db_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, db_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
    except OSError as exc:
        raise ProviderTrustDatabaseError(f"Failed to atomically write provider trust database at {db_path}: {exc}") from exc


def save_provider_trust_report(report: Dict[str, Any], db_path: str = DB_PATH) -> None:
    """
    Append a report to the trust database. Existing reports are never
    overwritten or removed.
    """
    if not isinstance(report, dict) or not report.get("provider_name"):
        raise ProviderMetadataError("report must be a dict containing at least 'provider_name'")

    db = load_db(db_path)
    db["reports"].append(report)
    _atomic_write_json(db_path, db)


def get_provider_trust_report(provider_name: str, db_path: str = DB_PATH) -> Dict[str, Any]:
    """
    Retrieve the most recent stored trust report for ``provider_name``.

    Raises ``ProviderNotFoundError`` if no report exists for that provider.
    """
    db = load_db(db_path)
    matches = [r for r in db["reports"] if r.get("provider_name") == provider_name]
    if not matches:
        raise ProviderNotFoundError(f"No trust report found for provider '{provider_name}'")
    matches.sort(key=lambda r: r.get("evaluation_timestamp", ""))
    return matches[-1]


def list_provider_trust_reports(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Return every stored trust report, in the order they were saved."""
    db = load_db(db_path)
    return list(db["reports"])


def seed_mock_historical_data(historical_data: Dict[str, Dict[str, Any]], db_path: str = DB_PATH) -> None:
    """
    Merge mock historical retrieval ledgers into the database. Intended
    for demo/bootstrap use only; in production Agent 4 will update this
    ledger automatically after each real retrieval.
    """
    db = load_db(db_path)
    db["historical_data"].update(historical_data)
    _atomic_write_json(db_path, db)


# --------------------------------------------------------------------------
# CONSOLE FORMATTING HELPERS
# --------------------------------------------------------------------------

def _print_header(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def _print_report_summary(report: Dict[str, Any]) -> None:
    print(f"\nProvider: {report['provider_name']}  ({report.get('provider_url', 'n/a')})")
    print(f"Evaluated: {report['evaluation_timestamp']}  |  Task: {report.get('requested_task')}")
    print("-" * 78)
    for key in CRITERIA_WEIGHTS:
        entry = report[key]
        label = key.replace("_score", "").replace("_", " ").title()
        print(f"  {label:<26} {entry['score']:.4f}  weight={CRITERIA_WEIGHTS[key]:.2f}")
        print(f"      -> {entry['explanation']}")
    print("-" * 78)
    print(f"  OVERALL TRUST SCORE: {report['overall_trust_score']:.4f}  ({report['trust_level']})")
    print(f"  Ranking reason: {report['ranking_reason']}")
    print("  Strengths:")
    for s in report["strengths"]:
        print(f"    + {s}")
    print("  Weaknesses:")
    for w in report["weaknesses"]:
        print(f"    - {w}")
    print("  Recommendations:")
    for r in report["recommendations"]:
        print(f"    * {r}")


# --------------------------------------------------------------------------
# DEMO
# --------------------------------------------------------------------------

def _demo_providers() -> List[Dict[str, Any]]:
    """
    Build metadata for five illustrative candidate providers for an
    earth-observation / flood-analysis task.

    NOTE: These are metadata inputs (organizational facts, technical
    capabilities), not trust scores. Every score below is computed by
    the evaluate_* functions from this metadata at run time.
    """
    return [
        {
            "provider_name": "NASA",
            "provider_url": "https://earthdata.nasa.gov",
            "is_government_agency": True,
            "is_international_organization": False,
            "has_regulatory_mandate": True,
            "founding_year": 1958,
            "partner_agencies": ["ESA", "NOAA", "JAXA", "CNES"],
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "update_frequency_days": 1,
            "has_recent_dataset": True,
            "data_domains": ["rainfall", "precipitation", "soil_moisture", "satellite_imagery", "atmosphere"],
            "spatial_resolution_km": 1.0,
            "temporal_resolution_hours": 1.0,
            "variables_provided": ["precipitation", "soil_moisture", "surface_temperature", "vegetation_index"],
            "expected_variables": ["precipitation", "soil_moisture", "surface_temperature"],
            "coverage_percent": 98.0,
            "has_documentation": True,
            "agreement_with_other_providers": 0.93,
            "units_documented": True,
            "crs_documented": True,
            "has_metadata_standard": True,
            "license_type": "open",
            "citation_count": 1800,
            "peer_reviewed_publications": 620,
            "used_by_agencies": ["NOAA", "ESA", "INCOIS", "CMEMS", "WMO", "USGS"],
            "api_uptime_percent": 99.5,
            "avg_response_latency_ms": 180,
            "accessible": True,
        },
        {
            "provider_name": "NOAA",
            "provider_url": "https://www.noaa.gov",
            "is_government_agency": True,
            "is_international_organization": False,
            "has_regulatory_mandate": True,
            "founding_year": 1970,
            "partner_agencies": ["NASA", "ESA", "WMO"],
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "update_frequency_days": 1,
            "has_recent_dataset": True,
            "data_domains": ["weather", "rainfall", "ocean", "sea_surface_temperature", "atmosphere"],
            "spatial_resolution_km": 2.0,
            "temporal_resolution_hours": 1.0,
            "variables_provided": ["precipitation", "wind", "surface_temperature", "sea_surface_temperature"],
            "expected_variables": ["precipitation", "soil_moisture", "surface_temperature"],
            "coverage_percent": 96.0,
            "has_documentation": True,
            "agreement_with_other_providers": 0.90,
            "units_documented": True,
            "crs_documented": True,
            "has_metadata_standard": True,
            "license_type": "open",
            "citation_count": 1500,
            "peer_reviewed_publications": 540,
            "used_by_agencies": ["NASA", "INCOIS", "CMEMS", "WMO"],
            "api_uptime_percent": 99.2,
            "avg_response_latency_ms": 210,
            "accessible": True,
        },
        {
            "provider_name": "INCOIS",
            "provider_url": "https://incois.gov.in",
            "is_government_agency": True,
            "is_international_organization": False,
            "has_regulatory_mandate": True,
            "founding_year": 1999,
            "partner_agencies": ["NOAA", "CMEMS"],
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "update_frequency_days": 1,
            "has_recent_dataset": True,
            "data_domains": ["ocean", "rainfall", "cyclone", "wave_height", "weather"],
            "spatial_resolution_km": 4.0,
            "temporal_resolution_hours": 3.0,
            "variables_provided": ["precipitation", "wave_height", "wind", "sea_surface_temperature"],
            "expected_variables": ["precipitation", "soil_moisture", "surface_temperature"],
            "coverage_percent": 80.0,
            "has_documentation": True,
            "agreement_with_other_providers": 0.82,
            "units_documented": True,
            "crs_documented": True,
            "has_metadata_standard": False,
            "license_type": "open",
            "citation_count": 320,
            "peer_reviewed_publications": 110,
            "used_by_agencies": ["NOAA", "CMEMS"],
            "api_uptime_percent": 97.0,
            "avg_response_latency_ms": 340,
            "accessible": True,
        },
        {
            "provider_name": "CMEMS",
            "provider_url": "https://marine.copernicus.eu",
            "is_government_agency": False,
            "is_international_organization": True,
            "has_regulatory_mandate": True,
            "founding_year": 2015,
            "partner_agencies": ["ESA", "NOAA", "INCOIS"],
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "update_frequency_days": 1,
            "has_recent_dataset": True,
            "data_domains": ["ocean", "currents", "salinity", "sea_surface_temperature", "marine"],
            "spatial_resolution_km": 3.0,
            "temporal_resolution_hours": 6.0,
            "variables_provided": ["sea_surface_temperature", "salinity", "currents"],
            "expected_variables": ["precipitation", "soil_moisture", "surface_temperature"],
            "coverage_percent": 90.0,
            "has_documentation": True,
            "agreement_with_other_providers": 0.88,
            "units_documented": True,
            "crs_documented": True,
            "has_metadata_standard": True,
            "license_type": "open",
            "citation_count": 900,
            "peer_reviewed_publications": 300,
            "used_by_agencies": ["ESA", "NOAA", "INCOIS", "WMO"],
            "api_uptime_percent": 98.6,
            "avg_response_latency_ms": 260,
            "accessible": True,
        },
        {
            "provider_name": "Bloomberg Finance",
            "provider_url": "https://www.bloomberg.com",
            "is_government_agency": False,
            "is_international_organization": False,
            "has_regulatory_mandate": False,
            "founding_year": 1981,
            "partner_agencies": [],
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "update_frequency_days": 1,
            "has_recent_dataset": True,
            "data_domains": ["equities", "bonds", "commodities", "financial_markets", "currency_exchange"],
            "spatial_resolution_km": None,
            "temporal_resolution_hours": None,
            "variables_provided": ["stock_price", "exchange_rate", "bond_yield"],
            "expected_variables": ["precipitation", "soil_moisture", "surface_temperature"],
            "coverage_percent": 100.0,
            "has_documentation": True,
            "agreement_with_other_providers": None,
            "units_documented": True,
            "crs_documented": False,
            "has_metadata_standard": False,
            "license_type": "restricted",
            "citation_count": 40,
            "peer_reviewed_publications": 5,
            "used_by_agencies": [],
            "api_uptime_percent": 99.9,
            "avg_response_latency_ms": 90,
            "accessible": True,
        },
    ]


def run_demo() -> None:
    """Standalone demonstration of the provider trust evaluation module."""
    requested_task = "flood_analysis"

    _print_header("EARTH INTELLIGENCE PLATFORM - PROVIDER TRUST EVALUATION DEMO")
    print(f"Requested scientific task: {requested_task}")
    print(f"Database location: {DB_PATH}")

    ensure_db_exists()

    # Seed a small illustrative historical ledger (Agent 4 will maintain this
    # automatically in production; here it is bootstrapped for the demo).
    seed_mock_historical_data({
        "NASA": {"successful_retrievals": 48, "failed_retrievals": 2, "last_retrieval": _now_iso()},
        "NOAA": {"successful_retrievals": 44, "failed_retrievals": 1, "last_retrieval": _now_iso()},
        "INCOIS": {"successful_retrievals": 30, "failed_retrievals": 5, "last_retrieval": _now_iso()},
        "CMEMS": {"successful_retrievals": 40, "failed_retrievals": 3, "last_retrieval": _now_iso()},
        # Bloomberg Finance intentionally left with no history: it has never
        # been retrieved for an earth-observation task.
    })

    db = load_db()
    providers = _demo_providers()

    _print_header("STEP 1-3: EVALUATE EACH PROVIDER (CRITERION-BY-CRITERION)")
    reports: List[Dict[str, Any]] = []
    for provider in providers:
        report = generate_provider_trust_report(provider, requested_task=requested_task, db=db)
        reports.append(report)
        _print_report_summary(report)

    _print_header("STEP 4: RANK ALL PROVIDERS (HIGHEST TO LOWEST TRUST)")
    ranked = sorted(reports, key=lambda r: r["overall_trust_score"], reverse=True)
    for position, report in enumerate(ranked, start=1):
        print(f"  #{position}  {report['provider_name']:<20} "
              f"overall={report['overall_trust_score']:.4f}  ({report['trust_level']})")

    _print_header("STEP 5: SAVE EVERY REPORT (APPEND-ONLY)")
    for report in reports:
        save_provider_trust_report(report)
        print(f"  Saved report for {report['provider_name']} at {report['evaluation_timestamp']}")

    _print_header("STEP 6: RETRIEVE ONE REPORT")
    retrieved = get_provider_trust_report("NASA")
    print(f"  Retrieved latest report for NASA: overall_trust_score="
          f"{retrieved['overall_trust_score']:.4f} ({retrieved['trust_level']})")

    _print_header("STEP 7: LIST ALL STORED REPORTS")
    all_reports = list_provider_trust_reports()
    print(f"  Total reports stored in database: {len(all_reports)}")
    for report in all_reports:
        print(f"    - {report['provider_name']:<20} {report['evaluation_timestamp']}  "
              f"overall={report['overall_trust_score']:.4f}")

    _print_header("STEP 8: WHY DOES BLOOMBERG FINANCE SCORE LOW FOR AN EARTH OBSERVATION REQUEST?")
    bloomberg = next(r for r in reports if r["provider_name"] == "Bloomberg Finance")
    nasa = next(r for r in reports if r["provider_name"] == "NASA")
    print(f"  Bloomberg Finance overall_trust_score = {bloomberg['overall_trust_score']:.4f} "
          f"({bloomberg['trust_level']})")
    print(f"  NASA overall_trust_score              = {nasa['overall_trust_score']:.4f} "
          f"({nasa['trust_level']})")
    print("\n  Criterion-by-criterion comparison:")
    for key in CRITERIA_WEIGHTS:
        label = key.replace("_score", "").replace("_", " ").title()
        b = bloomberg[key]["score"]
        n = nasa[key]["score"]
        print(f"    {label:<26} Bloomberg={b:.2f}   NASA={n:.2f}")
    print(f"\n  {bloomberg['ranking_reason']}")
    print("  Bloomberg's low relevance/resolution/scientific-acceptance scores are a direct,")
    print("  transparent consequence of its own declared metadata (financial data domains,")
    print("  no spatial/temporal resolution, no earth-science variables, no peer-reviewed")
    print("  earth-science citations) -- not a hardcoded penalty against the name 'Bloomberg'.")

    _print_header("DEMO COMPLETE")
    print("This module made no accept/reject decisions. All scores, rankings, and")
    print("explanations above are advisory input for Agent 3's downstream selection logic.")


if __name__ == "__main__":
    run_demo()
