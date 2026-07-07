"""
security/cross_agent_verification.py

Cross-Agent Verification Module
================================

Independent, standalone security module for the Earth Intelligence
multi-agent platform.

Purpose
-------
Act as an independent security auditor that checks whether Agent 3's
selection of a scientific data provider is justified, given:

  - the user's scientific query,
  - Agent 3's full ranked provider list,
  - the provider Agent 3 actually selected,
  - and the Provider Trust reports produced by ``provider_trust.py``.

This module NEVER changes Agent 3's decision, NEVER downloads datasets,
and NEVER rejects a provider. It only audits the decision that has
already been made and produces a transparent, explainable verification
report. Anything flagged here is advisory input for a human or for a
future policy-enforcement stage -- not an automatic override.

This module is fully standalone:

    python security/cross_agent_verification.py

Its core verification functions operate on plain dictionaries (a
ranked-provider list, a selected-provider name, and Provider Trust
report dictionaries) and have no import-time dependency on Agent 3 or
Agent 4. The bundled demo optionally reuses ``provider_trust.py``
(security module #4, already completed) purely to generate realistic
example input -- exactly the kind of data this module is specified to
consume -- but the verification logic itself never calls into Agent 3
or Agent 4 code.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(MODULE_DIR, "cross_agent_verification_db.json")

# Relative weight of each verification stage in the final weighted
# verification_score. Fixed policy, sums to 1.0.
STAGE_WEIGHTS: Dict[str, float] = {
    "ranking_verification": 0.15,
    "trust_verification": 0.20,
    "relevance_verification": 0.15,
    "metadata_verification": 0.10,
    "availability_verification": 0.10,
    "scientific_consistency_verification": 0.10,
    "policy_verification": 0.15,
    "explanation_verification": 0.05,
}

# Stages whose FAIL status is severe enough, on its own, to prevent an
# overall "VERIFIED" outcome regardless of the numeric score.
CRITICAL_STAGES = {"relevance_verification", "policy_verification"}

STATUS_PASS = "PASS"
STATUS_WARNING = "WARNING"
STATUS_FLAGGED = "FLAGGED"
STATUS_FAIL = "FAIL"
STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"

OVERALL_VERIFIED = "VERIFIED"
OVERALL_FLAGGED = "FLAGGED FOR REVIEW"
OVERALL_NOT_VERIFIED = "NOT VERIFIED"

# Default policy applied when the caller does not supply one. This is
# generic security policy configuration, not tied to any provider.
DEFAULT_POLICY: Dict[str, Any] = {
    "minimum_trust_threshold": 0.55,
    "required_metadata_fields": ["variables_provided", "has_documentation"],
    "accepted_provider_categories": [
        "government_agency",
        "international_agency",
        "research_institution",
    ],
}

TRUST_GAP_THRESHOLD = 0.15          # trust-score gap considered "significant"
RELEVANCE_PASS_THRESHOLD = 0.60
RELEVANCE_WARNING_THRESHOLD = 0.35
CONSISTENCY_OUTLIER_Z_SCORE = 1.5   # |z| beyond this is flagged as an outlier


# --------------------------------------------------------------------------
# EXCEPTIONS
# --------------------------------------------------------------------------

class CrossAgentVerificationError(Exception):
    """Base exception for cross-agent verification failures."""


class VerificationInputError(CrossAgentVerificationError):
    """Raised when required verification input is missing or malformed."""


class VerificationDatabaseError(CrossAgentVerificationError):
    """Raised when the verification database cannot be read or written."""


class VerificationNotFoundError(CrossAgentVerificationError):
    """Raised when a requested verification report does not exist."""


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


def _get_trust_score(entry: Dict[str, Any]) -> float:
    """Safely extract overall_trust_score from a ranked-provider entry."""
    trust_report = entry.get("trust_report") or {}
    score = trust_report.get("overall_trust_score")
    if isinstance(score, (int, float)):
        return float(score)
    return 0.0


def _get_criterion_score(entry: Dict[str, Any], criterion_key: str) -> Optional[float]:
    """Safely extract a nested criterion score (e.g. 'relevance_score') from a trust report."""
    trust_report = entry.get("trust_report") or {}
    criterion = trust_report.get(criterion_key)
    if isinstance(criterion, dict):
        value = criterion.get("score")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _find_entry(ranked_providers: List[Dict[str, Any]], provider_name: str) -> Optional[Dict[str, Any]]:
    """Find a provider's entry within the ranked-provider list by name."""
    for entry in ranked_providers:
        if entry.get("provider_name") == provider_name:
            return entry
    return None


@dataclass
class VerificationResult:
    """Result of running a single verification stage."""
    status: str
    score: float
    confidence: float
    explanation: str
    details: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "score": _round_score(self.score),
            "confidence": _round_score(self.confidence),
            "explanation": self.explanation,
        }


# --------------------------------------------------------------------------
# STAGE 1: RANKING VERIFICATION
# --------------------------------------------------------------------------

def verify_ranking(
    ranked_providers: List[Dict[str, Any]],
    selected_provider_name: str,
    agent3_justification: Optional[str],
) -> VerificationResult:
    """
    Verify whether the selected provider was Agent 3's own top-ranked
    candidate. If not, check whether a justification was supplied for
    deviating from that ranking.
    """
    if not ranked_providers:
        return VerificationResult(
            status=STATUS_FAIL, score=0.0, confidence=0.0,
            explanation="No ranked_providers list was supplied; ranking cannot be verified.",
        )

    selected_entry = _find_entry(ranked_providers, selected_provider_name)
    if selected_entry is None:
        return VerificationResult(
            status=STATUS_FAIL, score=0.0, confidence=1.0,
            explanation=f"Selected provider '{selected_provider_name}' does not appear in Agent 3's ranked list at all.",
        )

    ordered = sorted(ranked_providers, key=lambda e: e.get("rank", 10 ** 6))
    top_provider = ordered[0].get("provider_name")
    selected_rank = selected_entry.get("rank")

    if selected_rank == 1 or selected_provider_name == top_provider:
        return VerificationResult(
            status=STATUS_PASS, score=1.0, confidence=1.0,
            explanation=f"Selected provider '{selected_provider_name}' was Agent 3's own rank #1 candidate.",
            details={"selected_rank": selected_rank, "top_provider": top_provider},
        )

    has_justification = bool(agent3_justification and len(agent3_justification.strip()) >= 20)
    if has_justification:
        return VerificationResult(
            status=STATUS_FLAGGED, score=0.5, confidence=0.8,
            explanation=(
                f"Selected provider '{selected_provider_name}' was ranked #{selected_rank} "
                f"(top candidate was '{top_provider}'). A justification was supplied for this "
                f"deviation and requires downstream review to confirm it is sound."
            ),
            details={"selected_rank": selected_rank, "top_provider": top_provider},
        )

    return VerificationResult(
        status=STATUS_FAIL, score=0.0, confidence=1.0,
        explanation=(
            f"Selected provider '{selected_provider_name}' was ranked #{selected_rank}, not #1 "
            f"(top candidate was '{top_provider}'), and no justification was supplied for "
            f"overriding Agent 3's own ranking."
        ),
        details={"selected_rank": selected_rank, "top_provider": top_provider},
    )


# --------------------------------------------------------------------------
# STAGE 2: TRUST VERIFICATION
# --------------------------------------------------------------------------

def verify_trust(ranked_providers: List[Dict[str, Any]], selected_provider_name: str) -> VerificationResult:
    """
    Compare the selected provider's overall trust score against every
    other candidate to determine whether a significantly higher-trust
    provider was available and ignored.
    """
    selected_entry = _find_entry(ranked_providers, selected_provider_name)
    if selected_entry is None:
        return VerificationResult(
            status=STATUS_FAIL, score=0.0, confidence=1.0,
            explanation=f"Selected provider '{selected_provider_name}' has no trust report available for comparison.",
        )

    selected_score = _get_trust_score(selected_entry)
    others = [e for e in ranked_providers if e.get("provider_name") != selected_provider_name]
    if not others:
        return VerificationResult(
            status=STATUS_PASS, score=1.0, confidence=0.5,
            explanation="No other candidate providers were supplied for comparison; trust verification is trivially satisfied.",
        )

    best_other_entry = max(others, key=_get_trust_score)
    best_other_score = _get_trust_score(best_other_entry)
    best_other_name = best_other_entry.get("provider_name")
    gap = best_other_score - selected_score

    if gap <= 0:
        return VerificationResult(
            status=STATUS_PASS, score=1.0, confidence=1.0,
            explanation=(
                f"Selected provider trust score ({selected_score:.2f}) is at or above every "
                f"other candidate's (highest other: '{best_other_name}' at {best_other_score:.2f})."
            ),
            details={"selected_score": selected_score, "best_other_provider": best_other_name, "best_other_score": best_other_score},
        )

    if gap <= TRUST_GAP_THRESHOLD:
        return VerificationResult(
            status=STATUS_PASS, score=_round_score(1.0 - gap), confidence=1.0,
            explanation=(
                f"Selected provider trust score ({selected_score:.2f}) is only marginally below the "
                f"highest-trust alternative '{best_other_name}' ({best_other_score:.2f}); gap of "
                f"{gap:.2f} is within the acceptable threshold of {TRUST_GAP_THRESHOLD:.2f}."
            ),
            details={"selected_score": selected_score, "best_other_provider": best_other_name, "best_other_score": best_other_score},
        )

    return VerificationResult(
        status=STATUS_FAIL, score=_round_score(_clamp(1.0 - gap)), confidence=1.0,
        explanation=(
            f"Selected provider trust score ({selected_score:.2f}) is significantly lower than "
            f"'{best_other_name}' ({best_other_score:.2f}) -- a gap of {gap:.2f}, exceeding the "
            f"{TRUST_GAP_THRESHOLD:.2f} threshold for a significant difference. A materially "
            f"higher-trust provider appears to have been available and was not selected."
        ),
        details={"selected_score": selected_score, "best_other_provider": best_other_name, "best_other_score": best_other_score},
    )


# --------------------------------------------------------------------------
# STAGE 3: RELEVANCE VERIFICATION
# --------------------------------------------------------------------------

def verify_relevance(ranked_providers: List[Dict[str, Any]], selected_provider_name: str, user_query: Dict[str, Any]) -> VerificationResult:
    """
    Verify that the selected provider satisfies the requested scientific
    task, using the relevance_score already computed by provider_trust.py
    (falling back to a direct comparison if unavailable).
    """
    selected_entry = _find_entry(ranked_providers, selected_provider_name)
    if selected_entry is None:
        return VerificationResult(
            status=STATUS_FAIL, score=0.0, confidence=1.0,
            explanation=f"No trust report available for '{selected_provider_name}'; relevance cannot be verified.",
        )

    relevance_score = _get_criterion_score(selected_entry, "relevance_score")
    requested_task = user_query.get("requested_task", "unspecified")

    if relevance_score is None:
        return VerificationResult(
            status=STATUS_WARNING, score=0.4, confidence=0.2,
            explanation=(
                f"Selected provider's trust report has no relevance_score for task "
                f"'{requested_task}'; relevance cannot be confidently verified."
            ),
        )

    if relevance_score >= RELEVANCE_PASS_THRESHOLD:
        status = STATUS_PASS
    elif relevance_score >= RELEVANCE_WARNING_THRESHOLD:
        status = STATUS_WARNING
    else:
        status = STATUS_FAIL

    trust_report = selected_entry.get("trust_report") or {}
    underlying_explanation = (trust_report.get("relevance_score") or {}).get("explanation", "")

    explanation = (
        f"Relevance score {relevance_score:.2f} for task '{requested_task}' "
        f"({'meets' if status == STATUS_PASS else 'falls short of'} the "
        f"{RELEVANCE_PASS_THRESHOLD:.2f} PASS threshold). Underlying provider_trust.py finding: "
        f"{underlying_explanation}"
    )
    return VerificationResult(status=status, score=relevance_score, confidence=1.0, explanation=explanation)


# --------------------------------------------------------------------------
# STAGE 4: METADATA VERIFICATION
# --------------------------------------------------------------------------

def verify_metadata(ranked_providers: List[Dict[str, Any]], selected_provider_name: str, user_query: Dict[str, Any]) -> VerificationResult:
    """
    Verify that the selected provider's raw metadata contains the
    variables required by the user query, plus documentation, as
    reflected in both raw_metadata and the provider_trust.py
    metadata_quality_score / completeness_score.
    """
    selected_entry = _find_entry(ranked_providers, selected_provider_name)
    if selected_entry is None:
        return VerificationResult(
            status=STATUS_FAIL, score=0.0, confidence=1.0,
            explanation=f"No metadata available for '{selected_provider_name}'.",
        )

    raw_metadata = selected_entry.get("raw_metadata") or {}
    required_variables = [v.lower() for v in user_query.get("required_variables", [])]
    provided_variables = {v.lower() for v in raw_metadata.get("variables_provided", [])}

    reasons: List[str] = []
    components: List[float] = []

    if required_variables:
        missing = [v for v in required_variables if v not in provided_variables]
        present_ratio = 1.0 - (len(missing) / len(required_variables))
        components.append(present_ratio)
        if missing:
            reasons.append(f"missing required variable(s): {missing}")
        else:
            reasons.append("all required variables are present")
    else:
        components.append(0.7)
        reasons.append("no required_variables specified in user_query; scored at moderate baseline")

    has_documentation = bool(raw_metadata.get("has_documentation"))
    components.append(1.0 if has_documentation else 0.0)
    reasons.append(f"has_documentation={has_documentation}")

    metadata_quality = _get_criterion_score(selected_entry, "metadata_quality_score")
    if metadata_quality is not None:
        components.append(metadata_quality)
        reasons.append(f"provider_trust.py metadata_quality_score={metadata_quality:.2f}")

    score = sum(components) / len(components)
    if score >= 0.75:
        status = STATUS_PASS
    elif score >= 0.45:
        status = STATUS_WARNING
    else:
        status = STATUS_FAIL

    explanation = "Metadata verification: " + "; ".join(reasons) + "."
    return VerificationResult(status=status, score=score, confidence=1.0, explanation=explanation)


# --------------------------------------------------------------------------
# STAGE 5: DATASET AVAILABILITY VERIFICATION
# --------------------------------------------------------------------------

def verify_dataset_availability(ranked_providers: List[Dict[str, Any]], selected_provider_name: str) -> VerificationResult:
    """
    Verify that the selected provider's dataset exists, is reported as
    available, and has a usable download endpoint.
    """
    selected_entry = _find_entry(ranked_providers, selected_provider_name)
    if selected_entry is None:
        return VerificationResult(
            status=STATUS_FAIL, score=0.0, confidence=1.0,
            explanation=f"No metadata available for '{selected_provider_name}'; availability cannot be verified.",
        )

    raw_metadata = selected_entry.get("raw_metadata") or {}
    dataset_exists = raw_metadata.get("dataset_exists")
    availability_reported = raw_metadata.get("accessible")
    download_endpoint = raw_metadata.get("download_endpoint")

    checks = {
        "dataset_exists": dataset_exists is True,
        "availability_reported": availability_reported is True,
        "download_endpoint_present": bool(download_endpoint),
    }
    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    score = passed / total

    if score == 1.0:
        status = STATUS_PASS
    elif score >= 0.5:
        status = STATUS_WARNING
    else:
        status = STATUS_FAIL

    explanation = (
        f"Dataset availability checks: dataset_exists={dataset_exists}, "
        f"availability_reported={availability_reported}, "
        f"download_endpoint={'present' if download_endpoint else 'missing'}."
    )
    return VerificationResult(status=status, score=score, confidence=1.0, explanation=explanation, details=checks)


# --------------------------------------------------------------------------
# STAGE 6: SCIENTIFIC CONSISTENCY VERIFICATION
# --------------------------------------------------------------------------

def verify_scientific_consistency(ranked_providers: List[Dict[str, Any]], selected_provider_name: str) -> VerificationResult:
    """
    Compare the selected provider's overall trust score against the
    distribution of all candidates' trust scores. A strong statistical
    outlier is flagged for review -- it is NEVER rejected here.
    """
    if len(ranked_providers) < 3:
        return VerificationResult(
            status=STATUS_PASS, score=0.7, confidence=0.3,
            explanation="Fewer than 3 candidate providers supplied; statistical outlier detection is not meaningful.",
        )

    scores = [_get_trust_score(e) for e in ranked_providers]
    mean_score = statistics.mean(scores)
    stdev_score = statistics.pstdev(scores)
    selected_score = _get_trust_score(_find_entry(ranked_providers, selected_provider_name) or {})

    if stdev_score == 0:
        return VerificationResult(
            status=STATUS_PASS, score=1.0, confidence=1.0,
            explanation="All candidate providers share an identical trust score; no outlier possible.",
        )

    z_score = (selected_score - mean_score) / stdev_score

    if abs(z_score) <= CONSISTENCY_OUTLIER_Z_SCORE:
        return VerificationResult(
            status=STATUS_PASS, score=_round_score(1.0 - min(abs(z_score) / (2 * CONSISTENCY_OUTLIER_Z_SCORE), 1.0)),
            confidence=1.0,
            explanation=(
                f"Selected provider trust score ({selected_score:.2f}) is consistent with the "
                f"candidate pool (mean={mean_score:.2f}, stdev={stdev_score:.2f}, z={z_score:.2f})."
            ),
        )

    direction = "below" if z_score < 0 else "above"
    return VerificationResult(
        status=STATUS_FLAGGED, score=_round_score(_clamp(1.0 - abs(z_score) / (2 * CONSISTENCY_OUTLIER_Z_SCORE))),
        confidence=1.0,
        explanation=(
            f"Selected provider trust score ({selected_score:.2f}) is a statistical outlier "
            f"{direction} the candidate pool (mean={mean_score:.2f}, stdev={stdev_score:.2f}, "
            f"z={z_score:.2f}); flagged for review, not rejected."
        ),
    )


# --------------------------------------------------------------------------
# STAGE 7: POLICY VERIFICATION
# --------------------------------------------------------------------------

def verify_policy(ranked_providers: List[Dict[str, Any]], selected_provider_name: str, policy: Optional[Dict[str, Any]]) -> VerificationResult:
    """
    Verify whether the selected provider complies with security policy:
    a minimum trust threshold, required metadata fields, and accepted
    provider categories.
    """
    active_policy = policy or DEFAULT_POLICY
    selected_entry = _find_entry(ranked_providers, selected_provider_name)
    if selected_entry is None:
        return VerificationResult(
            status=STATUS_FAIL, score=0.0, confidence=1.0,
            explanation=f"No data available for '{selected_provider_name}'; policy cannot be verified.",
        )

    raw_metadata = selected_entry.get("raw_metadata") or {}
    violations: List[str] = []

    min_trust = active_policy.get("minimum_trust_threshold", 0.0)
    selected_score = _get_trust_score(selected_entry)
    if selected_score < min_trust:
        violations.append(f"trust score {selected_score:.2f} is below the minimum policy threshold of {min_trust:.2f}")

    required_fields = active_policy.get("required_metadata_fields", [])
    missing_fields = [f for f in required_fields if not raw_metadata.get(f)]
    if missing_fields:
        violations.append(f"missing required metadata field(s): {missing_fields}")

    accepted_categories = active_policy.get("accepted_provider_categories")
    provider_category = raw_metadata.get("provider_category")
    if accepted_categories and provider_category not in accepted_categories:
        violations.append(
            f"provider category '{provider_category}' is not in the accepted list {accepted_categories}"
        )

    if not violations:
        return VerificationResult(
            status=STATUS_PASS, score=1.0, confidence=1.0,
            explanation="Selected provider complies with all active security policy rules.",
            details={"policy": active_policy},
        )

    score = _clamp(1.0 - (len(violations) / max(len(active_policy), 1)))
    return VerificationResult(
        status=STATUS_FAIL, score=score, confidence=1.0,
        explanation="Policy violation(s): " + "; ".join(violations) + ".",
        details={"policy": active_policy, "violations": violations},
    )


# --------------------------------------------------------------------------
# STAGE 8: EXPLANATION VERIFICATION
# --------------------------------------------------------------------------

def verify_explanation(
    agent3_justification: Optional[str],
    ranking_result: VerificationResult,
    trust_result: VerificationResult,
    relevance_result: VerificationResult,
) -> VerificationResult:
    """
    Verify that Agent 3 provided a clear explanation for its selection,
    and that the explanation is consistent with the ranking, trust, and
    relevance findings above (e.g. a deviation from top rank or a trust
    gap should be acknowledged in the explanation).
    """
    text = (agent3_justification or "").strip()

    if not text:
        return VerificationResult(
            status=STATUS_FAIL, score=0.0, confidence=1.0,
            explanation="Agent 3 supplied no justification for its provider selection.",
        )

    if len(text) < 20:
        return VerificationResult(
            status=STATUS_WARNING, score=0.4, confidence=0.8,
            explanation=f"Agent 3's justification is very short ('{text}') and may not adequately explain the decision.",
        )

    lowered = text.lower()
    gaps_needing_acknowledgement: List[str] = []
    if ranking_result.status != STATUS_PASS and not any(w in lowered for w in ("rank", "top", "first", "priorit")):
        gaps_needing_acknowledgement.append("deviation from the top-ranked provider")
    if trust_result.status != STATUS_PASS and not any(w in lowered for w in ("trust", "score", "confidence", "reliab")):
        gaps_needing_acknowledgement.append("the trust-score gap versus a higher-trust alternative")
    if relevance_result.status != STATUS_PASS and not any(w in lowered for w in ("relevan", "task", "suitab", "match")):
        gaps_needing_acknowledgement.append("the relevance shortfall for the requested task")

    if not gaps_needing_acknowledgement:
        return VerificationResult(
            status=STATUS_PASS, score=1.0, confidence=0.7,
            explanation="Agent 3's justification is present and, where deviations occurred, appears to address them.",
        )

    return VerificationResult(
        status=STATUS_FAIL, score=0.2, confidence=0.7,
        explanation=(
            "Agent 3's justification does not address: " + "; ".join(gaps_needing_acknowledgement) +
            ". The stated explanation is inconsistent with the other verification findings."
        ),
    )


# --------------------------------------------------------------------------
# OVERALL SCORE / OUTCOME
# --------------------------------------------------------------------------

def calculate_verification_score(stage_results: Dict[str, VerificationResult]) -> float:
    """Compute the weighted overall verification_score from all eight stages."""
    try:
        total = sum(stage_results[key].score * weight for key, weight in STAGE_WEIGHTS.items())
    except KeyError as exc:
        raise VerificationInputError(f"Missing stage result required for score calculation: {exc}") from exc
    return _round_score(total)


def calculate_verification_confidence(stage_results: Dict[str, VerificationResult]) -> float:
    """
    Compute verification_confidence: how much reliable input data backed
    this audit, independent of whether the outcome itself was good or
    bad. Low-confidence stages (missing data, small sample sizes) pull
    this down even when the resulting scores looked favorable.
    """
    try:
        total = sum(stage_results[key].confidence * weight for key, weight in STAGE_WEIGHTS.items())
    except KeyError as exc:
        raise VerificationInputError(f"Missing stage result required for confidence calculation: {exc}") from exc
    return _round_score(total)


def _determine_overall_verification(stage_results: Dict[str, VerificationResult], verification_score: float) -> str:
    """Determine the overall_verification label from stage outcomes and the composite score."""
    critical_failed = any(stage_results[key].status == STATUS_FAIL for key in CRITICAL_STAGES)
    if critical_failed:
        return OVERALL_NOT_VERIFIED

    any_fail_or_flag = any(
        result.status in (STATUS_FAIL, STATUS_FLAGGED) for result in stage_results.values()
    )
    if any_fail_or_flag:
        return OVERALL_FLAGGED

    if verification_score >= 0.85:
        return OVERALL_VERIFIED

    return OVERALL_FLAGGED


# --------------------------------------------------------------------------
# REPORT GENERATION
# --------------------------------------------------------------------------

def generate_verification_report(
    user_query: Dict[str, Any],
    ranked_providers: List[Dict[str, Any]],
    selected_provider_name: str,
    agent3_justification: Optional[str] = None,
    policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run all eight verification stages against Agent 3's decision and
    assemble a complete Cross-Agent Verification Report.

    Parameters
    ----------
    user_query:
        Dict describing the user's scientific request, e.g.
        {'query_text': ..., 'requested_task': 'flood_analysis',
         'required_variables': [...]}.
    ranked_providers:
        Agent 3's full ranked candidate list. Each entry:
        {'provider_name': str, 'rank': int,
         'trust_report': <report from provider_trust.py>,
         'raw_metadata': <the metadata dict originally scored>}.
    selected_provider_name:
        The provider Agent 3 actually chose.
    agent3_justification:
        Agent 3's stated reason for the selection, if any.
    policy:
        Optional security policy dict; DEFAULT_POLICY is used otherwise.
    """
    if not isinstance(user_query, dict):
        raise VerificationInputError("user_query must be a dict")
    if not isinstance(ranked_providers, list) or not ranked_providers:
        raise VerificationInputError("ranked_providers must be a non-empty list")
    if not selected_provider_name:
        raise VerificationInputError("selected_provider_name is required")

    ranking_result = verify_ranking(ranked_providers, selected_provider_name, agent3_justification)
    trust_result = verify_trust(ranked_providers, selected_provider_name)
    relevance_result = verify_relevance(ranked_providers, selected_provider_name, user_query)
    metadata_result = verify_metadata(ranked_providers, selected_provider_name, user_query)
    availability_result = verify_dataset_availability(ranked_providers, selected_provider_name)
    consistency_result = verify_scientific_consistency(ranked_providers, selected_provider_name)
    policy_result = verify_policy(ranked_providers, selected_provider_name, policy)
    explanation_result = verify_explanation(agent3_justification, ranking_result, trust_result, relevance_result)

    stage_results: Dict[str, VerificationResult] = {
        "ranking_verification": ranking_result,
        "trust_verification": trust_result,
        "relevance_verification": relevance_result,
        "metadata_verification": metadata_result,
        "availability_verification": availability_result,
        "scientific_consistency_verification": consistency_result,
        "policy_verification": policy_result,
        "explanation_verification": explanation_result,
    }

    verification_score = calculate_verification_score(stage_results)
    verification_confidence = calculate_verification_confidence(stage_results)
    overall_verification = _determine_overall_verification(stage_results, verification_score)

    ordered = sorted(ranked_providers, key=lambda e: e.get("rank", 10 ** 6))
    highest_ranked_entry = ordered[0]
    selected_entry = _find_entry(ranked_providers, selected_provider_name) or {}

    warnings: List[str] = [
        f"[{name.replace('_verification', '')}] {result.explanation}"
        for name, result in stage_results.items()
        if result.status in (STATUS_FAIL, STATUS_FLAGGED, STATUS_WARNING)
    ]

    recommendations: List[str] = []
    if ranking_result.status != STATUS_PASS:
        recommendations.append(
            f"Confirm whether '{highest_ranked_entry.get('provider_name')}' (Agent 3's own top-ranked "
            f"provider) should have been selected instead, or strengthen the justification for overriding it."
        )
    if trust_result.status == STATUS_FAIL:
        recommendations.append("Re-evaluate the trust gap against the higher-trust alternative before proceeding.")
    if relevance_result.status != STATUS_PASS:
        recommendations.append("Select a provider whose data domains better match the requested scientific task.")
    if metadata_result.status != STATUS_PASS:
        recommendations.append("Obtain the missing required variables, documentation, or metadata before proceeding.")
    if availability_result.status != STATUS_PASS:
        recommendations.append("Confirm dataset existence, availability, and a working download endpoint before proceeding.")
    if consistency_result.status == STATUS_FLAGGED:
        recommendations.append("Manually review the selection given its statistical divergence from other candidates.")
    if policy_result.status == STATUS_FAIL:
        recommendations.append("Resolve the listed security policy violation(s) before this selection can be trusted.")
    if explanation_result.status != STATUS_PASS:
        recommendations.append("Request a clearer, more complete justification from Agent 3 for this selection.")
    if not recommendations:
        recommendations.append("No corrective action required; selection is well-supported by all verification stages.")

    summary = (
        f"Cross-agent verification of Agent 3's selection of '{selected_provider_name}' for task "
        f"'{user_query.get('requested_task', 'unspecified')}' resulted in '{overall_verification}' "
        f"(verification_score={verification_score:.2f}, verification_confidence={verification_confidence:.2f}). "
        f"{len(warnings)} stage(s) raised a warning, flag, or failure."
    )

    report: Dict[str, Any] = {
        "verification_id": str(uuid.uuid4()),
        "verification_timestamp": _now_iso(),
        "user_query": user_query,
        "selected_provider": selected_provider_name,
        "selected_provider_score": _round_score(_get_trust_score(selected_entry)),
        "highest_ranked_provider": highest_ranked_entry.get("provider_name"),
        "highest_ranked_score": _round_score(_get_trust_score(highest_ranked_entry)),
        "ranking_verification": ranking_result.as_dict(),
        "trust_verification": trust_result.as_dict(),
        "relevance_verification": relevance_result.as_dict(),
        "metadata_verification": metadata_result.as_dict(),
        "availability_verification": availability_result.as_dict(),
        "scientific_consistency_verification": consistency_result.as_dict(),
        "policy_verification": policy_result.as_dict(),
        "explanation_verification": explanation_result.as_dict(),
        "overall_verification": overall_verification,
        "verification_score": verification_score,
        "verification_confidence": verification_confidence,
        "verification_summary": summary,
        "warnings": warnings,
        "recommendations": recommendations,
    }
    return report


# --------------------------------------------------------------------------
# DATABASE (ATOMIC READ / WRITE)
# --------------------------------------------------------------------------

def _empty_db() -> Dict[str, Any]:
    return {"reports": []}


def ensure_db_exists(db_path: str = DB_PATH) -> None:
    """Create the verification database file with an empty skeleton if missing."""
    if not os.path.exists(db_path):
        _atomic_write_json(db_path, _empty_db())


def load_db(db_path: str = DB_PATH) -> Dict[str, Any]:
    """Load the verification database, creating it if necessary."""
    ensure_db_exists(db_path)
    try:
        with open(db_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        raise VerificationDatabaseError(f"Failed to read verification database at {db_path}: {exc}") from exc

    if "reports" not in data:
        raise VerificationDatabaseError(f"Verification database at {db_path} is missing required keys")
    return data


def _atomic_write_json(db_path: str, data: Dict[str, Any]) -> None:
    """Write ``data`` to ``db_path`` atomically (write to temp file, then replace)."""
    directory = os.path.dirname(db_path) or "."
    os.makedirs(directory, exist_ok=True)
    try:
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".cross_agent_verification_db_", suffix=".tmp")
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
        raise VerificationDatabaseError(f"Failed to atomically write verification database at {db_path}: {exc}") from exc


def save_verification_report(report: Dict[str, Any], db_path: str = DB_PATH) -> None:
    """Append a verification report to the database. Existing reports are never overwritten or removed."""
    if not isinstance(report, dict) or not report.get("verification_id"):
        raise VerificationInputError("report must be a dict containing at least 'verification_id'")

    db = load_db(db_path)
    db["reports"].append(report)
    _atomic_write_json(db_path, db)


def get_verification_report(verification_id: str, db_path: str = DB_PATH) -> Dict[str, Any]:
    """Retrieve a stored verification report by its verification_id."""
    db = load_db(db_path)
    for report in db["reports"]:
        if report.get("verification_id") == verification_id:
            return report
    raise VerificationNotFoundError(f"No verification report found with verification_id '{verification_id}'")


def list_verification_reports(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Return every stored verification report, in the order they were saved."""
    db = load_db(db_path)
    return list(db["reports"])


# --------------------------------------------------------------------------
# CONSOLE FORMATTING HELPERS
# --------------------------------------------------------------------------

def _print_header(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def _print_report(report: Dict[str, Any]) -> None:
    print(f"\nVerification ID: {report['verification_id']}")
    print(f"Timestamp: {report['verification_timestamp']}")
    print(f"User query: {report['user_query'].get('query_text')}  (task={report['user_query'].get('requested_task')})")
    print(f"Selected provider: {report['selected_provider']}  (score={report['selected_provider_score']:.4f})")
    print(f"Highest-ranked provider (Agent 3's own ranking): {report['highest_ranked_provider']}  "
          f"(score={report['highest_ranked_score']:.4f})")
    print("-" * 78)
    for key in STAGE_WEIGHTS:
        stage = report[key]
        label = key.replace("_verification", "").replace("_", " ").title()
        print(f"  {label:<24} {stage['status']:<9} score={stage['score']:.2f}  "
              f"confidence={stage['confidence']:.2f}  weight={STAGE_WEIGHTS[key]:.2f}")
        print(f"      -> {stage['explanation']}")
    print("-" * 78)
    print(f"  OVERALL VERIFICATION: {report['overall_verification']}")
    print(f"  verification_score={report['verification_score']:.4f}   "
          f"verification_confidence={report['verification_confidence']:.4f}")
    print(f"  Summary: {report['verification_summary']}")
    if report["warnings"]:
        print("  Warnings:")
        for w in report["warnings"]:
            print(f"    ! {w}")
    print("  Recommendations:")
    for r in report["recommendations"]:
        print(f"    * {r}")


# --------------------------------------------------------------------------
# DEMO
# --------------------------------------------------------------------------

def _build_demo_ranked_providers() -> List[Dict[str, Any]]:
    """
    Build a demo ranked-provider list for a flood-analysis query, reusing
    provider_trust.py (security module #4) to generate real Provider
    Trust reports from provider metadata -- exactly the kind of input
    this module is specified to consume. If provider_trust.py cannot be
    imported for any reason, a minimal built-in fallback is used instead
    so this module still runs fully standalone.
    """
    provider_metadata = [
        {
            "provider_name": "NASA",
            "provider_url": "https://earthdata.nasa.gov",
            "is_government_agency": True,
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
            "dataset_exists": True,
            "download_endpoint": "https://earthdata.nasa.gov/download/precip",
            "provider_category": "government_agency",
        },
        {
            "provider_name": "NOAA",
            "provider_url": "https://www.noaa.gov",
            "is_government_agency": True,
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
            "dataset_exists": True,
            "download_endpoint": "https://www.noaa.gov/download/precip",
            "provider_category": "government_agency",
        },
        {
            "provider_name": "INCOIS",
            "provider_url": "https://incois.gov.in",
            "is_government_agency": True,
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
            "dataset_exists": True,
            "download_endpoint": "https://incois.gov.in/download/precip",
            "provider_category": "government_agency",
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
            "dataset_exists": True,
            "download_endpoint": "https://marine.copernicus.eu/download/currents",
            "provider_category": "international_agency",
        },
        {
            "provider_name": "Bloomberg Finance",
            "provider_url": "https://www.bloomberg.com",
            "is_government_agency": False,
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
            "dataset_exists": False,
            "download_endpoint": "",
            "provider_category": "commercial_financial_data_vendor",
        },
    ]

    requested_task = "flood_analysis"
    trust_reports: Dict[str, Dict[str, Any]] = {}
    try:
        sys.path.insert(0, MODULE_DIR)
        import provider_trust as pt  # security module #4, reused only to generate demo input

        pt_db = pt.load_db()
        for metadata in provider_metadata:
            trust_reports[metadata["provider_name"]] = pt.generate_provider_trust_report(
                metadata, requested_task=requested_task, db=pt_db
            )
    except Exception as exc:  # pragma: no cover - defensive fallback, keeps this module standalone
        print(f"[warning] Could not use provider_trust.py to build demo trust reports ({exc}); "
              f"falling back to a minimal built-in placeholder.")
        for metadata in provider_metadata:
            trust_reports[metadata["provider_name"]] = {
                "provider_name": metadata["provider_name"],
                "overall_trust_score": 0.5,
                "relevance_score": {"score": 0.5, "explanation": "fallback placeholder"},
                "metadata_quality_score": {"score": 0.5, "explanation": "fallback placeholder"},
            }

    ranked = sorted(
        provider_metadata,
        key=lambda m: trust_reports[m["provider_name"]].get("overall_trust_score", 0.0),
        reverse=True,
    )

    ranked_providers: List[Dict[str, Any]] = []
    for rank, metadata in enumerate(ranked, start=1):
        ranked_providers.append({
            "provider_name": metadata["provider_name"],
            "rank": rank,
            "trust_report": trust_reports[metadata["provider_name"]],
            "raw_metadata": metadata,
        })
    return ranked_providers


def run_demo() -> None:
    """Standalone demonstration of the cross-agent verification module."""
    _print_header("EARTH INTELLIGENCE PLATFORM - CROSS-AGENT VERIFICATION DEMO")
    print(f"Database location: {DB_PATH}")
    ensure_db_exists()

    ranked_providers = _build_demo_ranked_providers()
    user_query = {
        "query_text": "Assess flood risk for the upcoming monsoon season using recent rainfall and soil moisture data.",
        "requested_task": "flood_analysis",
        "required_variables": ["precipitation", "soil_moisture"],
    }

    print("\nAgent 3's ranked provider list (by trust score, for this demo):")
    for entry in ranked_providers:
        print(f"  #{entry['rank']}  {entry['provider_name']:<20} "
              f"trust={entry['trust_report'].get('overall_trust_score', 0.0):.4f}")

    # ----------------------------------------------------------------
    # SCENARIO 1: Agent 3 selects NASA -- its own top-ranked, highest
    # trust provider. Verification should PASS / VERIFIED.
    # ----------------------------------------------------------------
    _print_header("SCENARIO 1: Agent 3 selects NASA (top-ranked, highest trust)")
    scenario_1_justification = (
        "NASA was selected because it has the highest overall trust score among all "
        "candidates, strong relevance to flood analysis through its rainfall and soil "
        "moisture variables, complete documentation and metadata, and a proven "
        "historical retrieval record."
    )
    report_1 = generate_verification_report(
        user_query=user_query,
        ranked_providers=ranked_providers,
        selected_provider_name="NASA",
        agent3_justification=scenario_1_justification,
    )
    _print_report(report_1)
    save_verification_report(report_1)

    # ----------------------------------------------------------------
    # SCENARIO 2: Agent 3 selects Bloomberg Finance instead, while NASA
    # has a much higher trust score for flood analysis. Verification
    # should FAIL / NOT VERIFIED.
    # ----------------------------------------------------------------
    _print_header("SCENARIO 2: Agent 3 selects Bloomberg Finance (much lower trust, irrelevant task match)")
    scenario_2_justification = (
        "Bloomberg Finance was selected because it has the lowest response latency "
        "and highest API uptime among all candidates."
    )
    report_2 = generate_verification_report(
        user_query=user_query,
        ranked_providers=ranked_providers,
        selected_provider_name="Bloomberg Finance",
        agent3_justification=scenario_2_justification,
    )
    _print_report(report_2)
    save_verification_report(report_2)

    _print_header("WHY SCENARIO 2 FAILS VERIFICATION")
    print(f"  Bloomberg Finance was selected (trust={report_2['selected_provider_score']:.2f}) while "
          f"'{report_2['highest_ranked_provider']}' (trust={report_2['highest_ranked_score']:.2f}) "
          f"was Agent 3's own top-ranked, highest-trust candidate.")
    print("  Failing / flagged stages:")
    for key in STAGE_WEIGHTS:
        stage = report_2[key]
        if stage["status"] != STATUS_PASS:
            label = key.replace("_verification", "").replace("_", " ").title()
            print(f"    - {label}: {stage['status']} -> {stage['explanation']}")
    print("\n  Agent 3's stated justification (latency/uptime) never addresses the trust gap,")
    print("  the relevance shortfall, or the deviation from its own top ranking -- which is")
    print("  exactly what explanation_verification is designed to catch.")
    print("\n  Recommendations for Agent 3:")
    for r in report_2["recommendations"]:
        print(f"    * {r}")

    _print_header("STEP: RETRIEVE ONE REPORT BY ID")
    retrieved = get_verification_report(report_1["verification_id"])
    print(f"  Retrieved report {retrieved['verification_id']}: "
          f"overall_verification={retrieved['overall_verification']}")

    _print_header("STEP: LIST ALL STORED VERIFICATION REPORTS")
    all_reports = list_verification_reports()
    print(f"  Total verification reports stored: {len(all_reports)}")
    for report in all_reports:
        print(f"    - {report['verification_id']}  selected={report['selected_provider']:<18} "
              f"overall={report['overall_verification']}")

    _print_header("DEMO COMPLETE")
    print("This module changed nothing about Agent 3's decisions, downloaded no datasets,")
    print("and rejected no providers. It only produced independent, transparent audit reports.")


if __name__ == "__main__":
    run_demo()
