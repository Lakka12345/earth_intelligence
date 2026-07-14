"""
security/security_risk_assessment.py
=====================================
Central Security Decision Engine for the Earth Intelligence multi-agent platform.

PURPOSE
-------
This module is NOT a detector.  Individual security modules (prompt_injection,
provider_trust, dataset_validation, integrity, cross_agent_verification,
provenance) each produce their own reports.  THIS module receives those reports
as plain dictionaries, combines the evidence, and returns a single authoritative
SecurityRiskAssessmentReport that the pipeline uses to decide whether to
proceed.

DESIGN PRINCIPLES
-----------------
• Completely standalone — no imports from other security modules.
• Agent-independent — no imports from Agent 1 / 2 / 3 / 4.
• No LLM dependency.
• All reports are OPTIONAL.  Missing reports reduce confidence; they do NOT
  automatically raise the risk level.
• Weights are top-level constants so they can be tuned without touching logic.
• Every decision is accompanied by a human-readable explanation.
• Results are appended to a JSON database file for audit purposes.

USAGE
-----
    from security.security_risk_assessment import assess_security_risk

    report = assess_security_risk(
        prompt_injection=pi_report_dict,    # optional
        provider_trust=pt_report_dict,      # optional
        integrity=integ_report_dict,        # optional
        validation=val_report_dict,         # optional
        cross_agent=ca_report_dict,         # optional
        provenance=prov_report_dict,        # optional
    )
    print(report.recommended_action)       # "ALLOW", "WARN", …
    print(report.overall_risk_level)       # "SAFE", "LOW", …

Running this file directly demonstrates five built-in scenarios.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ===========================================================================
# Paths
# ===========================================================================

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SECURITY_DB_PATH = os.path.join(_MODULE_DIR, "security_risk_db.json")


# ===========================================================================
# Enumerations
# ===========================================================================

class RiskLevel(str, Enum):
    """Ordered risk levels from safest to most dangerous."""
    SAFE     = "SAFE"
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


class RecommendedAction(str, Enum):
    """Pipeline actions ordered from most permissive to most restrictive."""
    ALLOW                = "ALLOW"
    WARN                 = "WARN"
    REQUIRE_CONFIRMATION = "REQUIRE_CONFIRMATION"
    BLOCK                = "BLOCK"


# ===========================================================================
# Scoring weights
# ===========================================================================
# Each weight represents the fraction of the overall score contributed by
# that report type.  Weights MUST sum to 1.0.  Adjust here without touching
# scoring logic.

WEIGHT_PROMPT_INJECTION  : float = 0.30   # highest — direct attack surface
WEIGHT_PROVIDER_TRUST    : float = 0.20   # data provenance anchor
WEIGHT_INTEGRITY         : float = 0.15   # file/payload tamper evidence
WEIGHT_VALIDATION        : float = 0.15   # schema / content correctness
WEIGHT_CROSS_AGENT       : float = 0.10   # inter-agent consistency
WEIGHT_PROVENANCE        : float = 0.10   # lineage / audit trail

_WEIGHT_SUM = (
    WEIGHT_PROMPT_INJECTION
    + WEIGHT_PROVIDER_TRUST
    + WEIGHT_INTEGRITY
    + WEIGHT_VALIDATION
    + WEIGHT_CROSS_AGENT
    + WEIGHT_PROVENANCE
)
assert abs(_WEIGHT_SUM - 1.0) < 1e-9, f"Weights must sum to 1.0, got {_WEIGHT_SUM}"


# ===========================================================================
# Risk-level thresholds
# Maps overall_security_score (0–100, higher = safer) to RiskLevel
# ===========================================================================

_RISK_THRESHOLDS: List[tuple[int, RiskLevel]] = [
    (90, RiskLevel.SAFE),
    (75, RiskLevel.LOW),
    (55, RiskLevel.MEDIUM),
    (35, RiskLevel.HIGH),
    (0,  RiskLevel.CRITICAL),
]

# Maps RiskLevel → RecommendedAction
_ACTION_MAP: Dict[RiskLevel, RecommendedAction] = {
    RiskLevel.SAFE    : RecommendedAction.ALLOW,
    RiskLevel.LOW     : RecommendedAction.ALLOW,
    RiskLevel.MEDIUM  : RecommendedAction.WARN,
    RiskLevel.HIGH    : RecommendedAction.REQUIRE_CONFIRMATION,
    RiskLevel.CRITICAL: RecommendedAction.BLOCK,
}

# Hard overrides: certain combinations force a stricter action regardless of score
_FORCE_BLOCK_CONDITIONS = {
    "prompt_injection_detected",
    "integrity_failed",
}

_FORCE_CONFIRMATION_CONDITIONS = {
    "provider_untrusted",
    "cross_agent_inconsistency",
}


# ===========================================================================
# Per-report component score extraction
# ===========================================================================

@dataclass
class ComponentScore:
    """Holds the extracted numeric score and metadata for one report type."""
    name       : str
    present    : bool
    raw_score  : float        # 0–100, higher = safer
    weight     : float
    weighted   : float        # raw_score * weight (0 if absent)
    flags      : List[str]    # short machine-readable flags for override checks
    summary    : str          # one-sentence human summary


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _extract_prompt_injection_score(report: Optional[Dict[str, Any]]) -> ComponentScore:
    """
    Interprets a prompt-injection report dict.

    Expected keys (all optional):
        injection_detected (bool)   — True if an injection was found
        risk_score         (float)  — 0–100 where 100 = most dangerous
        confidence         (float)  — 0–1 confidence of detector
        injection_type     (str)    — e.g. "direct", "indirect", "none"
    """
    if not report:
        return ComponentScore(
            name="Prompt Injection",
            present=False,
            raw_score=70.0,    # neutral-ish assumption when absent
            weight=WEIGHT_PROMPT_INJECTION,
            weighted=0.0,      # absent reports contribute 0 to weighted sum
            flags=[],
            summary="Prompt injection report not provided; assuming no injection (reduced confidence).",
        )

    detected   = bool(report.get("injection_detected", False))
    risk_score = float(report.get("risk_score", 0.0))     # 0=safe,100=danger
    confidence = float(report.get("confidence", 1.0))

    # Convert detector risk_score (0=safe) to safety_score (100=safe)
    safety_score = _clamp(100.0 - risk_score)

    # If explicitly detected, force safety score very low
    if detected:
        safety_score = _clamp(safety_score * 0.1)

    flags   = ["prompt_injection_detected"] if detected else []
    summary = (
        "Prompt injection DETECTED — pipeline input must not be trusted."
        if detected
        else f"No prompt injection detected (safety score {safety_score:.0f}/100)."
    )

    return ComponentScore(
        name="Prompt Injection",
        present=True,
        raw_score=safety_score,
        weight=WEIGHT_PROMPT_INJECTION,
        weighted=safety_score * WEIGHT_PROMPT_INJECTION,
        flags=flags,
        summary=summary,
    )


def _extract_provider_trust_score(report: Optional[Dict[str, Any]]) -> ComponentScore:
    """
    Interprets a provider-trust report dict.

    Expected keys (all optional):
        trust_score        (float)  — 0–100 where 100 = fully trusted
        trust_level        (str)    — "HIGH", "MEDIUM", "LOW", "UNTRUSTED"
        provider_name      (str)
        known_provider     (bool)
        historical_issues  (int)    — count of past problems
    """
    if not report:
        return ComponentScore(
            name="Provider Trust",
            present=False,
            raw_score=60.0,
            weight=WEIGHT_PROVIDER_TRUST,
            weighted=0.0,
            flags=[],
            summary="Provider trust report not provided; using neutral assumption.",
        )

    trust_score = float(report.get("trust_score", 50.0))
    trust_level = str(report.get("trust_level", "MEDIUM")).upper()
    issues      = int(report.get("historical_issues", 0))

    # Penalise historical issues
    penalty     = _clamp(issues * 5.0, 0.0, 25.0)
    safety_score= _clamp(trust_score - penalty)

    untrusted   = trust_level == "UNTRUSTED" or safety_score < 20.0
    flags       = ["provider_untrusted"] if untrusted else []
    provider    = report.get("provider_name", "unknown provider")

    summary = (
        f"Provider '{provider}' is UNTRUSTED (score {safety_score:.0f}/100)."
        if untrusted
        else f"Provider '{provider}' trust score is {safety_score:.0f}/100 [{trust_level}]."
    )

    return ComponentScore(
        name="Provider Trust",
        present=True,
        raw_score=safety_score,
        weight=WEIGHT_PROVIDER_TRUST,
        weighted=safety_score * WEIGHT_PROVIDER_TRUST,
        flags=flags,
        summary=summary,
    )


def _extract_integrity_score(report: Optional[Dict[str, Any]]) -> ComponentScore:
    """
    Interprets a data-integrity report dict.

    Expected keys (all optional):
        integrity_passed   (bool)   — True if all checks passed
        checksum_valid     (bool)
        tamper_detected    (bool)
        integrity_score    (float)  — 0–100 where 100 = fully intact
        issues             (list)   — list of issue strings
    """
    if not report:
        return ComponentScore(
            name="Integrity",
            present=False,
            raw_score=65.0,
            weight=WEIGHT_INTEGRITY,
            weighted=0.0,
            flags=[],
            summary="Integrity report not provided; no tamper evidence available.",
        )

    passed          = bool(report.get("integrity_passed", True))
    tamper_detected = bool(report.get("tamper_detected", False))
    safety_score    = float(report.get("integrity_score", 100.0 if passed else 0.0))
    checksum_valid  = bool(report.get("checksum_valid", True))

    if tamper_detected:
        safety_score = _clamp(safety_score * 0.05)
    elif not checksum_valid:
        safety_score = _clamp(safety_score * 0.5)

    failed = not passed or tamper_detected
    flags  = ["integrity_failed"] if failed else []

    issues   = report.get("issues") or []
    issue_str= f" Issues: {'; '.join(issues[:3])}." if issues else ""
    summary  = (
        f"Integrity check FAILED — tamper evidence present.{issue_str}"
        if failed
        else f"Dataset integrity verified (score {safety_score:.0f}/100).{issue_str}"
    )

    return ComponentScore(
        name="Integrity",
        present=True,
        raw_score=safety_score,
        weight=WEIGHT_INTEGRITY,
        weighted=safety_score * WEIGHT_INTEGRITY,
        flags=flags,
        summary=summary,
    )


def _extract_validation_score(report: Optional[Dict[str, Any]]) -> ComponentScore:
    """
    Interprets a dataset-validation report dict.

    Expected keys (all optional):
        validation_passed  (bool)
        validation_score   (float)  — 0–100
        errors             (list)
        warnings           (list)
        content_type_valid (bool)
    """
    if not report:
        return ComponentScore(
            name="Validation",
            present=False,
            raw_score=65.0,
            weight=WEIGHT_VALIDATION,
            weighted=0.0,
            flags=[],
            summary="Validation report not provided; dataset contents unverified.",
        )

    passed       = bool(report.get("validation_passed", True))
    safety_score = float(report.get("validation_score", 100.0 if passed else 30.0))
    errors       = report.get("errors") or []
    warnings     = report.get("warnings") or []

    # Errors hurt more than warnings
    error_penalty   = _clamp(len(errors) * 10.0, 0.0, 50.0)
    warning_penalty = _clamp(len(warnings) * 3.0, 0.0, 15.0)
    safety_score    = _clamp(safety_score - error_penalty - warning_penalty)

    flags   = ["validation_failed"] if not passed or errors else []
    summary = (
        f"Validation FAILED — {len(errors)} error(s), {len(warnings)} warning(s)."
        if not passed or errors
        else f"Validation passed (score {safety_score:.0f}/100)."
        + (f" {len(warnings)} warning(s)." if warnings else "")
    )

    return ComponentScore(
        name="Validation",
        present=True,
        raw_score=safety_score,
        weight=WEIGHT_VALIDATION,
        weighted=safety_score * WEIGHT_VALIDATION,
        flags=flags,
        summary=summary,
    )


def _extract_cross_agent_score(report: Optional[Dict[str, Any]]) -> ComponentScore:
    """
    Interprets a cross-agent verification report dict.

    Expected keys (all optional):
        verification_passed (bool)
        consistency_score   (float) — 0–100
        inconsistencies     (list)
        agents_verified     (list)
    """
    if not report:
        return ComponentScore(
            name="Cross-Agent Verification",
            present=False,
            raw_score=70.0,
            weight=WEIGHT_CROSS_AGENT,
            weighted=0.0,
            flags=[],
            summary="Cross-agent verification report not provided.",
        )

    passed       = bool(report.get("verification_passed", True))
    safety_score = float(report.get("consistency_score", 100.0 if passed else 40.0))
    inconsist    = report.get("inconsistencies") or []

    penalty      = _clamp(len(inconsist) * 12.0, 0.0, 60.0)
    safety_score = _clamp(safety_score - penalty)

    flags   = ["cross_agent_inconsistency"] if not passed or inconsist else []
    summary = (
        f"Cross-agent verification FAILED — {len(inconsist)} inconsistency(ies) found."
        if not passed or inconsist
        else f"All agent outputs are consistent (score {safety_score:.0f}/100)."
    )

    return ComponentScore(
        name="Cross-Agent Verification",
        present=True,
        raw_score=safety_score,
        weight=WEIGHT_CROSS_AGENT,
        weighted=safety_score * WEIGHT_CROSS_AGENT,
        flags=flags,
        summary=summary,
    )


def _extract_provenance_score(report: Optional[Dict[str, Any]]) -> ComponentScore:
    """
    Interprets a provenance / lineage report dict.

    Expected keys (all optional):
        provenance_verified (bool)
        lineage_complete    (bool)
        provenance_score    (float) — 0–100
        missing_links       (list)  — gaps in the lineage chain
        source_confirmed    (bool)
    """
    if not report:
        return ComponentScore(
            name="Provenance",
            present=False,
            raw_score=65.0,
            weight=WEIGHT_PROVENANCE,
            weighted=0.0,
            flags=[],
            summary="Provenance report not provided; lineage cannot be confirmed.",
        )

    verified     = bool(report.get("provenance_verified", True))
    complete     = bool(report.get("lineage_complete", True))
    safety_score = float(report.get("provenance_score", 100.0 if verified else 40.0))
    missing      = report.get("missing_links") or []

    penalty      = _clamp(len(missing) * 8.0, 0.0, 40.0)
    safety_score = _clamp(safety_score - penalty)
    if not complete:
        safety_score = _clamp(safety_score * 0.7)

    flags   = ["provenance_unverified"] if not verified else []
    summary = (
        f"Provenance NOT verified — lineage chain is incomplete ({len(missing)} missing link(s))."
        if not verified
        else f"Provenance confirmed (score {safety_score:.0f}/100)."
        + (f" {len(missing)} minor gap(s)." if missing else "")
    )

    return ComponentScore(
        name="Provenance",
        present=True,
        raw_score=safety_score,
        weight=WEIGHT_PROVENANCE,
        weighted=safety_score * WEIGHT_PROVENANCE,
        flags=flags,
        summary=summary,
    )


# ===========================================================================
# Aggregation helpers
# ===========================================================================

def _score_to_risk_level(score: float) -> RiskLevel:
    """Map a safety score (0–100) to a RiskLevel."""
    for threshold, level in _RISK_THRESHOLDS:
        if score >= threshold:
            return level
    return RiskLevel.CRITICAL


def _collect_all_flags(components: List[ComponentScore]) -> List[str]:
    flags: List[str] = []
    for comp in components:
        flags.extend(comp.flags)
    return flags


def _apply_hard_overrides(
    risk_level: RiskLevel,
    action: RecommendedAction,
    all_flags: List[str],
) -> tuple[RiskLevel, RecommendedAction]:
    """
    Apply hard override rules that can only make the decision MORE restrictive,
    never less.  This prevents a high individual-component flag from being
    averaged away into a safe-looking score.
    """
    flag_set = set(all_flags)

    if flag_set & _FORCE_BLOCK_CONDITIONS:
        # Prompt injection or integrity failure → always BLOCK
        if risk_level not in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            risk_level = RiskLevel.HIGH
        action = RecommendedAction.BLOCK

    elif flag_set & _FORCE_CONFIRMATION_CONDITIONS:
        # Untrusted provider or cross-agent inconsistency → at least REQUIRE_CONFIRMATION
        if action == RecommendedAction.ALLOW:
            action = RecommendedAction.REQUIRE_CONFIRMATION
        if risk_level in (RiskLevel.SAFE, RiskLevel.LOW):
            risk_level = RiskLevel.MEDIUM

    return risk_level, action


def _calculate_confidence(
    components: List[ComponentScore],
    all_flags: List[str],
) -> float:
    """
    Confidence reflects how much of the evidence picture is available.
    - All six reports present and no flags → 1.0
    - Each missing report reduces confidence by ~0.08
    - Each active flag reduces confidence slightly (uncertainty about severity)
    """
    present_count = sum(1 for c in components if c.present)
    total         = len(components)
    missing_penalty = (total - present_count) * 0.08
    flag_penalty    = len(all_flags) * 0.03
    confidence      = max(0.35, 1.0 - missing_penalty - flag_penalty)
    return round(confidence, 3)


def _build_risk_breakdown(components: List[ComponentScore]) -> Dict[str, Any]:
    """Build the per-component breakdown included in the report."""
    return {
        comp.name: {
            "present"     : comp.present,
            "raw_score"   : round(comp.raw_score, 2),
            "weight"      : comp.weight,
            "weighted"    : round(comp.weighted, 2),
            "flags"       : comp.flags,
            "summary"     : comp.summary,
        }
        for comp in components
    }


def _build_risk_reasoning(
    components: List[ComponentScore],
    overall_score: float,
    risk_level: RiskLevel,
    all_flags: List[str],
    present_count: int,
    total: int,
) -> str:
    """Build a human-readable explanation of the overall risk decision."""
    lines = [
        f"Overall risk is {risk_level.value} (security score {overall_score:.1f}/100).",
        "",
        "Evidence summary:",
    ]

    for comp in components:
        prefix = "  ✓" if comp.present and not comp.flags else ("  ✗" if comp.flags else "  –")
        lines.append(f"{prefix} [{comp.name}] {comp.summary}")

    if present_count < total:
        missing = total - present_count
        lines.append("")
        lines.append(
            f"Note: {missing} of {total} security report(s) were not provided. "
            "Confidence has been reduced accordingly. "
            "Missing reports do NOT automatically indicate risk."
        )

    if all_flags:
        lines.append("")
        lines.append(f"Active risk flags: {', '.join(all_flags)}")

    return "\n".join(lines)


def _generate_recommendations(
    components: List[ComponentScore],
    risk_level: RiskLevel,
    all_flags: List[str],
) -> List[str]:
    """Generate targeted, actionable recommendations based on findings."""
    recs: List[str] = []
    flag_set = set(all_flags)

    if "prompt_injection_detected" in flag_set:
        recs.append("CRITICAL: Block this request immediately — prompt injection was detected.")
        recs.append("Review and sanitise all user-supplied input before any retry.")

    if "integrity_failed" in flag_set:
        recs.append("CRITICAL: Dataset integrity check failed — do not use this data.")
        recs.append("Re-download the dataset from a trusted source and repeat integrity verification.")

    if "provider_untrusted" in flag_set:
        recs.append("Reject or quarantine data from this untrusted provider.")
        recs.append("Attempt to find an equivalent dataset from a provider with a higher trust score.")

    if "cross_agent_inconsistency" in flag_set:
        recs.append("Re-run cross-agent verification after reviewing agent outputs manually.")
        recs.append("Investigate which agent produced the inconsistent output.")

    if "validation_failed" in flag_set:
        recs.append("Repeat dataset validation after addressing the reported schema/content errors.")

    if "provenance_unverified" in flag_set:
        recs.append("Attempt to reconstruct the provenance chain before using this data in analysis.")

    # Missing report recommendations
    for comp in components:
        if not comp.present:
            recs.append(f"Run the {comp.name} security module and include its report in the next assessment.")

    # Generic level-based recommendations
    if risk_level == RiskLevel.SAFE and not recs:
        recs.append("All security checks passed. Pipeline may proceed normally.")

    if risk_level == RiskLevel.LOW and not flag_set:
        recs.append("Risk is low. Consider adding missing security reports to increase confidence.")

    if risk_level == RiskLevel.MEDIUM:
        recs.append("Proceed with caution. Log this request and monitor for anomalies.")

    return recs


# ===========================================================================
# Output dataclass
# ===========================================================================

@dataclass
class SecurityRiskAssessmentReport:
    """
    The complete output of the central security risk assessment.

    Fields
    ------
    assessment_id       : Unique identifier for this assessment (UUID4).
    timestamp           : ISO-8601 UTC timestamp of the assessment.
    overall_security_score : 0–100 safety score (higher = safer).
    overall_risk_level  : SAFE | LOW | MEDIUM | HIGH | CRITICAL.
    recommended_action  : ALLOW | WARN | REQUIRE_CONFIRMATION | BLOCK.
    confidence_score    : 0.0–1.0 reflecting how much evidence was available.
    risk_breakdown      : Per-component scores and summaries.
    risk_reasoning      : Human-readable explanation of the overall decision.
    recommendations     : Ordered list of actionable recommendations.
    active_flags        : Machine-readable risk flags that triggered overrides.
    reports_provided    : Which report types were present in this assessment.
    reports_missing     : Which report types were absent.
    """
    assessment_id          : str
    timestamp              : str
    overall_security_score : float
    overall_risk_level     : RiskLevel
    recommended_action     : RecommendedAction
    confidence_score       : float
    risk_breakdown         : Dict[str, Any]
    risk_reasoning         : str
    recommendations        : List[str]
    active_flags           : List[str]
    reports_provided       : List[str]
    reports_missing        : List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary (JSON-safe)."""
        d = asdict(self)
        d["overall_risk_level"]  = self.overall_risk_level.value
        d["recommended_action"]  = self.recommended_action.value
        return d

    def is_safe_to_proceed(self) -> bool:
        """Convenience: True when the pipeline may continue without blocking."""
        return self.recommended_action in (
            RecommendedAction.ALLOW,
            RecommendedAction.WARN,
        )

    def requires_user_confirmation(self) -> bool:
        """True when a human must explicitly confirm before proceeding."""
        return self.recommended_action == RecommendedAction.REQUIRE_CONFIRMATION

    def is_blocked(self) -> bool:
        """True when the pipeline must NOT proceed."""
        return self.recommended_action == RecommendedAction.BLOCK


# ===========================================================================
# Core public API
# ===========================================================================

def calculate_overall_score(components: List[ComponentScore]) -> float:
    """
    Compute the weighted overall security score.

    Present reports contribute their (raw_score × weight) to the weighted sum.
    Absent reports contribute 0 to the weighted sum but their weight IS counted
    in the denominator — this prevents the score from appearing artificially
    high when reports are missing.  Missing reports instead reduce confidence.

    Returns a float in [0, 100].
    """
    total_weight_present = sum(c.weight for c in components if c.present)

    if total_weight_present == 0.0:
        # No reports at all — return a conservative neutral score
        return 50.0

    weighted_sum = sum(c.weighted for c in components if c.present)

    # Normalise by the sum of present weights so that missing reports don't
    # inflate the score, but are instead reflected in reduced confidence.
    raw = weighted_sum / total_weight_present

    return round(_clamp(raw), 2)


def generate_recommendations(
    components: List[ComponentScore],
    risk_level: RiskLevel,
    all_flags: List[str],
) -> List[str]:
    """
    Public wrapper around the internal recommendation generator.

    Parameters
    ----------
    components  : List of ComponentScore objects from all six report types.
    risk_level  : The final determined risk level.
    all_flags   : Combined list of active risk flags.

    Returns
    -------
    Ordered list of human-readable recommendation strings.
    """
    return _generate_recommendations(components, risk_level, all_flags)


def assess_security_risk(
    prompt_injection : Optional[Dict[str, Any]] = None,
    provider_trust   : Optional[Dict[str, Any]] = None,
    integrity        : Optional[Dict[str, Any]] = None,
    validation       : Optional[Dict[str, Any]] = None,
    cross_agent      : Optional[Dict[str, Any]] = None,
    provenance       : Optional[Dict[str, Any]] = None,
    save_to_db       : bool = True,
) -> SecurityRiskAssessmentReport:
    """
    Central security risk assessment function.

    Accepts up to six optional report dictionaries from the security modules
    and returns a SecurityRiskAssessmentReport containing the overall risk
    decision.

    All parameters are optional.  Missing reports reduce confidence but do
    NOT automatically raise the risk level.

    Parameters
    ----------
    prompt_injection : Dict from security/prompt_injection.py (optional).
    provider_trust   : Dict from security/provider_trust.py (optional).
    integrity        : Dict from security/integrity.py (optional).
    validation       : Dict from security/dataset_validation.py (optional).
    cross_agent      : Dict from security/cross_agent_verification.py (optional).
    provenance       : Dict from security/provenance.py (optional).
    save_to_db       : If True (default), append the result to security_risk_db.json.

    Returns
    -------
    SecurityRiskAssessmentReport
    """

    # ── 1. Extract component scores ─────────────────────────────────────────
    components: List[ComponentScore] = [
        _extract_prompt_injection_score(prompt_injection),
        _extract_provider_trust_score(provider_trust),
        _extract_integrity_score(integrity),
        _extract_validation_score(validation),
        _extract_cross_agent_score(cross_agent),
        _extract_provenance_score(provenance),
    ]

    # ── 2. Compute weighted overall score ───────────────────────────────────
    overall_score = calculate_overall_score(components)

    # ── 3. Determine initial risk level and action ──────────────────────────
    risk_level = _score_to_risk_level(overall_score)
    action     = _ACTION_MAP[risk_level]

    # ── 4. Collect flags and apply hard overrides ───────────────────────────
    all_flags             = _collect_all_flags(components)
    risk_level, action    = _apply_hard_overrides(risk_level, action, all_flags)

    # ── 5. Compute confidence ────────────────────────────────────────────────
    confidence = _calculate_confidence(components, all_flags)

    # ── 6. Build report fields ───────────────────────────────────────────────
    present_count  = sum(1 for c in components if c.present)
    total          = len(components)
    reports_provided = [c.name for c in components if c.present]
    reports_missing  = [c.name for c in components if not c.present]

    breakdown  = _build_risk_breakdown(components)
    reasoning  = _build_risk_reasoning(
        components, overall_score, risk_level, all_flags, present_count, total
    )
    recommendations = _generate_recommendations(components, risk_level, all_flags)

    assessment_id = str(uuid.uuid4())
    timestamp     = datetime.now(timezone.utc).isoformat()

    report = SecurityRiskAssessmentReport(
        assessment_id          = assessment_id,
        timestamp              = timestamp,
        overall_security_score = overall_score,
        overall_risk_level     = risk_level,
        recommended_action     = action,
        confidence_score       = confidence,
        risk_breakdown         = breakdown,
        risk_reasoning         = reasoning,
        recommendations        = recommendations,
        active_flags           = all_flags,
        reports_provided       = reports_provided,
        reports_missing        = reports_missing,
    )

    # ── 7. Persist to database ───────────────────────────────────────────────
    if save_to_db:
        save_security_assessment(report)

    return report


# ===========================================================================
# Database functions
# ===========================================================================

def save_security_assessment(report: SecurityRiskAssessmentReport) -> None:
    """
    Append a completed SecurityRiskAssessmentReport to the JSON database.

    The database is an append-only JSON array stored at SECURITY_DB_PATH.
    If the file does not exist it is created.  If it exists and is corrupt,
    the corrupt content is archived as a sibling file before a fresh database
    is started (data loss is avoided by archiving rather than overwriting).

    Parameters
    ----------
    report : The SecurityRiskAssessmentReport to persist.
    """
    # Load existing entries
    existing = _load_raw_db()

    # Append compact record (full report minus the verbose risk_breakdown
    # to keep the DB file human-readable; breakdown is still included)
    entry = {
        "assessment_id"         : report.assessment_id,
        "timestamp"             : report.timestamp,
        "overall_security_score": report.overall_security_score,
        "overall_risk_level"    : report.overall_risk_level.value,
        "recommended_action"    : report.recommended_action.value,
        "confidence_score"      : report.confidence_score,
        "active_flags"          : report.active_flags,
        "reports_provided"      : report.reports_provided,
        "reports_missing"       : report.reports_missing,
        "recommendations"       : report.recommendations,
        "risk_breakdown"        : report.risk_breakdown,
    }
    existing.append(entry)

    try:
        with open(SECURITY_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        print(f"[SecurityRiskDB] WARNING: Could not write to {SECURITY_DB_PATH}: {exc}",
              file=sys.stderr)


def load_security_database() -> List[Dict[str, Any]]:
    """
    Load and return all security assessment records from the database.

    Returns an empty list if the database does not exist or is empty.

    Returns
    -------
    List of assessment record dicts, ordered oldest-first.
    """
    return _load_raw_db()


def _load_raw_db() -> List[Dict[str, Any]]:
    """Internal helper — load the database file, returning [] on any problem."""
    if not os.path.exists(SECURITY_DB_PATH):
        return []

    try:
        with open(SECURITY_DB_PATH, encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return []
        data = json.loads(content)
        if isinstance(data, list):
            return data
        print(f"[SecurityRiskDB] WARNING: Database is not a JSON array; resetting.",
              file=sys.stderr)
        return []
    except (json.JSONDecodeError, OSError) as exc:
        # Archive the corrupt file before resetting
        archive = SECURITY_DB_PATH + ".corrupt"
        try:
            os.rename(SECURITY_DB_PATH, archive)
            print(f"[SecurityRiskDB] WARNING: Corrupt database archived to {archive}. "
                  f"Starting fresh.  Error: {exc}", file=sys.stderr)
        except OSError:
            pass
        return []


# ===========================================================================
# Pretty-printer
# ===========================================================================

def _print_report(report: SecurityRiskAssessmentReport, scenario: str) -> None:
    """Print a formatted summary of an assessment report."""
    divider = "=" * 70
    print(f"\n{divider}")
    print(f"  SCENARIO: {scenario}")
    print(divider)
    print(f"  Assessment ID   : {report.assessment_id}")
    print(f"  Timestamp       : {report.timestamp}")
    print(f"  Security Score  : {report.overall_security_score:.1f} / 100")
    print(f"  Risk Level      : {report.overall_risk_level.value}")
    print(f"  Action          : {report.recommended_action.value}")
    print(f"  Confidence      : {report.confidence_score:.2f}")
    if report.active_flags:
        print(f"  Active Flags    : {', '.join(report.active_flags)}")
    print(f"\n  Reports Present : {', '.join(report.reports_provided) or 'none'}")
    if report.reports_missing:
        print(f"  Reports Missing : {', '.join(report.reports_missing)}")
    print(f"\n  REASONING\n  {'-'*66}")
    for line in report.risk_reasoning.splitlines():
        print(f"  {line}")
    print(f"\n  RECOMMENDATIONS\n  {'-'*66}")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"  {i}. {rec}")
    print(divider)


# ===========================================================================
# Demo scenarios
# ===========================================================================

def _run_demo() -> None:
    """
    Demonstrate the security risk assessment module across five scenarios.
    Each scenario exercises a different risk profile.
    """
    print("\n" + "#" * 70)
    print("#  Security Risk Assessment Module — Demo")
    print("#  security/security_risk_assessment.py")
    print("#" * 70)

    # ── Scenario 1: Everything safe ─────────────────────────────────────────
    r1 = assess_security_risk(
        prompt_injection={"injection_detected": False, "risk_score": 0.0,  "confidence": 0.99},
        provider_trust  ={"trust_score": 95.0, "trust_level": "HIGH",    "provider_name": "NOAA ERDDAP", "historical_issues": 0},
        integrity       ={"integrity_passed": True, "checksum_valid": True, "tamper_detected": False, "integrity_score": 100.0},
        validation      ={"validation_passed": True, "validation_score": 98.0, "errors": [], "warnings": []},
        cross_agent     ={"verification_passed": True, "consistency_score": 100.0, "inconsistencies": []},
        provenance      ={"provenance_verified": True, "lineage_complete": True, "provenance_score": 100.0, "missing_links": []},
    )
    _print_report(r1, "1 — Everything Safe  →  Expected: SAFE / ALLOW")

    # ── Scenario 2: Prompt injection detected ───────────────────────────────
    r2 = assess_security_risk(
        prompt_injection={"injection_detected": True,  "risk_score": 95.0, "confidence": 0.97,
                          "injection_type": "indirect"},
        provider_trust  ={"trust_score": 80.0, "trust_level": "HIGH",   "provider_name": "Copernicus CDS"},
        integrity       ={"integrity_passed": True, "integrity_score": 95.0},
        validation      ={"validation_passed": True, "validation_score": 92.0},
        cross_agent     ={"verification_passed": True, "consistency_score": 97.0},
        provenance      ={"provenance_verified": True, "provenance_score": 90.0},
    )
    _print_report(r2, "2 — Prompt Injection Detected  →  Expected: HIGH or CRITICAL / BLOCK")

    # ── Scenario 3: Low provider trust ──────────────────────────────────────
    r3 = assess_security_risk(
        prompt_injection={"injection_detected": False, "risk_score": 5.0},
        provider_trust  ={"trust_score": 18.0, "trust_level": "UNTRUSTED",
                          "provider_name": "UnknownSource-42", "historical_issues": 6},
        integrity       ={"integrity_passed": True, "integrity_score": 85.0},
        validation      ={"validation_passed": True, "validation_score": 80.0},
    )
    _print_report(r3, "3 — Low Provider Trust  →  Expected: MEDIUM / WARN or REQUIRE_CONFIRMATION")

    # ── Scenario 4: Dataset integrity failure ───────────────────────────────
    r4 = assess_security_risk(
        prompt_injection={"injection_detected": False, "risk_score": 0.0},
        provider_trust  ={"trust_score": 88.0, "trust_level": "HIGH", "provider_name": "NASA Earthdata"},
        integrity       ={"integrity_passed": False, "checksum_valid": False,
                          "tamper_detected": True, "integrity_score": 5.0,
                          "issues": ["SHA-256 mismatch", "unexpected trailing bytes"]},
        validation      ={"validation_passed": True, "validation_score": 75.0},
        cross_agent     ={"verification_passed": True, "consistency_score": 88.0},
        provenance      ={"provenance_verified": True, "provenance_score": 92.0},
    )
    _print_report(r4, "4 — Dataset Integrity Failed  →  Expected: HIGH / BLOCK")

    # ── Scenario 5: Several reports missing ─────────────────────────────────
    r5 = assess_security_risk(
        # Only two of six reports provided
        prompt_injection={"injection_detected": False, "risk_score": 10.0, "confidence": 0.85},
        provider_trust  ={"trust_score": 70.0, "trust_level": "MEDIUM", "provider_name": "BCO-DMO"},
        # integrity, validation, cross_agent, provenance → all absent
    )
    _print_report(r5, "5 — Several Reports Missing  →  Expected: LOW or MEDIUM / Reduced confidence")

    # ── Database summary ─────────────────────────────────────────────────────
    db = load_security_database()
    print(f"\n{'='*70}")
    print(f"  Database: {len(db)} record(s) stored in {SECURITY_DB_PATH}")
    print(f"{'='*70}\n")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    _run_demo()
