"""
Observation Platform Validator -- deterministic platform-type check for
Agent 3 candidate sources.

Problem this solves (per approved refinement #2):
  If the user asks for satellite observations, in-situ platforms (buoys,
  weather stations, river gauges) should receive a heavy penalty or be
  rejected unless explicitly requested -- and symmetrically, if the user
  specifically asks for in-situ measurements, satellite-only datasets
  should not be automatically preferred. Nothing in the existing pipeline
  ever checked this: dataset_type matching is keyword-based on "ocean" /
  "weather" / "satellite" / etc, but a "satellite" dataset_type tag does
  not guarantee the discovered candidate itself is satellite-derived
  (e.g. an ERDDAP buoy network dataset can still be tagged "ocean" while
  being a strictly in-situ product).

Design (small, pure-Python, no schema change required):
  - Infers the CANDIDATE's platform from fields already on every
    CandidateSource: dataset_type, name, description, api_type,
    discovery_origin. No new CandidateSource fields are needed.
  - Infers what the user REQUESTED from the dataset_type strings and
    measurement/variable text supplied by the caller (Agent 3 already
    has these from RetrievalRequest).
  - Stays NEUTRAL (does not penalize) when the user's request doesn't
    specify a platform preference at all -- the validator only acts
    when the user was specific.
  - Penalizes symmetrically: satellite-only when in-situ was requested
    is treated the same as in-situ-only when satellite was requested.
"""

from dataclasses import dataclass
from typing import List, Optional


SATELLITE_TOKENS = [
    "satellite", "modis", "viirs", "landsat", "sentinel", "goes",
    "avhrr", "ndvi", "remote sensing", "remote-sensing", "imagery",
    "spaceborne", "space-based", "ocean color", "sar ", "synthetic aperture",
    "planetary computer", "earth observation", "geostationary",
]

INSITU_TOKENS = [
    "buoy", "weather station", "river gauge", "stream gauge", "tide gauge",
    "in-situ", "in situ", "moored", "mooring", "ground station",
    "rain gauge", "observation station", "argo float", "ctd cast",
    "fixed station", "monitoring station",
]

MODEL_TOKENS = [
    "model output", "reanalysis", "forecast model", "hindcast",
    "numerical model", "simulation", "era5", "hycom", "gfs ", "ecmwf",
]


@dataclass
class PlatformValidationResult:
    score: float                  # 0.0 - 1.0
    hard_reject: bool
    explanation: str
    inferred_candidate_platform: str
    inferred_requested_platform: Optional[str]


def _normalize(text: Optional[str]) -> str:
    if not text:
        return ""
    return text.lower().strip()


def _infer_platform(text: str) -> str:
    """
    Classifies a blob of text as 'satellite', 'in_situ', 'model', or
    'unknown'. If multiple signals are present, satellite/in-situ tokens
    take priority over generic model tokens (a model that ingests
    satellite products is still usefully labeled 'satellite' for this
    coarse check), and an explicit in-situ token alongside a satellite
    token signals a hybrid/unclear case, which we leave as 'unknown'
    rather than guessing wrong.
    """
    has_satellite = any(tok in text for tok in SATELLITE_TOKENS)
    has_insitu = any(tok in text for tok in INSITU_TOKENS)
    has_model = any(tok in text for tok in MODEL_TOKENS)

    if has_satellite and has_insitu:
        return "unknown"   # hybrid / ambiguous signal -- don't guess
    if has_satellite:
        return "satellite"
    if has_insitu:
        return "in_situ"
    if has_model:
        return "model"
    return "unknown"


def infer_candidate_platform(
    dataset_type: str,
    name: str,
    description: str,
    api_type: Optional[str] = None,
    discovery_origin: Optional[str] = None,
) -> str:
    """
    Infers the platform type of a discovered CandidateSource from fields
    that already exist on the model -- no schema change required.
    """
    text = " ".join(filter(None, [
        _normalize(dataset_type),
        _normalize(name),
        _normalize(description),
        _normalize(api_type),
        _normalize(discovery_origin),
    ]))
    return _infer_platform(text)


def infer_requested_platform(
    dataset_types_needed: List[str],
    measurement_texts: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Infers what platform the USER asked for, from the dataset_type
    strings and any measurement/variable free text supplied by Agent 3.

    Returns None when the request doesn't specify a platform preference
    -- the validator must stay neutral in that case rather than
    penalizing candidates for a preference the user never expressed.
    """
    combined = " ".join(filter(None, [
        " ".join(_normalize(t) for t in (dataset_types_needed or [])),
        " ".join(_normalize(t) for t in (measurement_texts or [])),
    ]))
    if not combined.strip():
        return None

    platform = _infer_platform(combined)
    return platform if platform != "unknown" else None


def validate_platform(
    requested_platform: Optional[str],
    candidate_platform: str,
) -> PlatformValidationResult:
    """
    Compares the inferred requested platform against the inferred
    candidate platform.

    Symmetric penalty: satellite-only when in-situ was requested is
    scored the same as in-situ-only when satellite was requested.
    Stays neutral whenever requested_platform is None (user didn't
    specify) or either side is 'unknown'/'model' (model-output products
    are not penalized by this validator -- they are a separate axis from
    the satellite vs in-situ distinction the spec asks about).
    """
    if requested_platform is None:
        return PlatformValidationResult(
            score=0.75,
            hard_reject=False,
            explanation="Request does not specify a platform preference; platform not evaluated.",
            inferred_candidate_platform=candidate_platform,
            inferred_requested_platform=None,
        )

    if candidate_platform == "unknown":
        return PlatformValidationResult(
            score=0.55,
            hard_reject=False,
            explanation=(
                f"Candidate's platform type could not be determined; user requested "
                f"'{requested_platform}'. Treated as neutral, not penalized."
            ),
            inferred_candidate_platform=candidate_platform,
            inferred_requested_platform=requested_platform,
        )

    if candidate_platform == "model":
        return PlatformValidationResult(
            score=0.65,
            hard_reject=False,
            explanation=(
                f"Candidate is model/reanalysis output; user requested "
                f"'{requested_platform}' observational data. Mild reduction, not a "
                "satellite-vs-in-situ mismatch."
            ),
            inferred_candidate_platform=candidate_platform,
            inferred_requested_platform=requested_platform,
        )

    if candidate_platform == requested_platform:
        return PlatformValidationResult(
            score=0.95,
            hard_reject=False,
            explanation=(
                f"Candidate platform '{candidate_platform}' matches the requested "
                f"platform '{requested_platform}'."
            ),
            inferred_candidate_platform=candidate_platform,
            inferred_requested_platform=requested_platform,
        )

    # Symmetric mismatch: satellite-only vs in-situ-requested, or vice versa.
    return PlatformValidationResult(
        score=0.15,
        hard_reject=False,
        explanation=(
            f"Candidate platform '{candidate_platform}' does not match the requested "
            f"platform '{requested_platform}'. Heavy penalty (not a hard reject, since "
            "platform inference is coarse and a worthwhile source could still be "
            "miscategorized)."
        ),
        inferred_candidate_platform=candidate_platform,
        inferred_requested_platform=requested_platform,
    )
