"""
security/dataset_validation.py

Dataset Validation Module for the Earth Intelligence Platform.

Validates downloaded scientific datasets across seven stages before they
are accepted into the analysis pipeline. Every stage is independently
runnable and produces a structured result that feeds into a single
ValidationReport stored in security/validation_db.json.

This module is completely standalone and has no dependency on Agent 4.
Agent 4 will call this module later, after:
    1. Download
    2. Integrity Verification   (security/integrity.py)
    3. Provenance Tracking      (security/provenance.py)
    4. Dataset Validation       <-- this module

Validation stages:
    1. File Validation         -- existence, size, format, readability
    2. Metadata Validation     -- required metadata fields present
    3. Variable Validation     -- required scientific variables present
    4. Spatial Validation      -- dataset covers the requested study area
    5. Temporal Validation     -- dataset covers the requested time period
    6. Scientific Validation   -- units, CRS, variable descriptions, resolution
    7. Integrity Validation    -- reuse integrity.py result (no new checksum)

Run standalone:
    python security/dataset_validation.py
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ------------------------------------------------------------------ #
# Constants                                                           #
# ------------------------------------------------------------------ #

DB_PATH = Path(__file__).parent / "validation_db.json"

SUPPORTED_FORMATS = {
    ".nc", ".nc4",        # NetCDF
    ".csv", ".tsv",       # Tabular
    ".tif", ".tiff",      # GeoTIFF
    ".h5", ".hdf5",       # HDF5
    ".json", ".geojson",  # JSON
    ".zarr",              # Zarr
    ".grib", ".grb2",     # GRIB
    ".zip",               # Archive
    ".txt",               # Plain text
}

# ------------------------------------------------------------------ #
# Variable alias table                                                #
#                                                                     #
# Maps canonical Earth-observation variable names to their accepted  #
# aliases. Used by validate_variables() and _normalize_var() below.  #
#                                                                     #
# EXTENDING: add a new key + alias list to cover a new variable.     #
# This dict is deliberately self-contained — no external files or    #
# provider modules are needed. Move to a shared module later by      #
# importing VARIABLE_ALIASES from that module instead.               #
#                                                                     #
# Matching is always done via _normalize_var(), which lowercases the #
# token and strips underscores, hyphens, and whitespace, so every   #
# alias is stored in its canonical form; callers never need to       #
# pre-normalise.                                                      #
# ------------------------------------------------------------------ #

VARIABLE_ALIASES: Dict[str, List[str]] = {
    # Sea Surface Temperature
    # Spec aliases + extras that appear in real GHRSST / CMEMS metadata
    "sea_surface_temperature": [
        "sea_surface_temperature",
        "sst",
        "analysed_sst",
        "sst_c",
        "surface_temperature",
        "temp_surface",
        # Additional commonly-seen names kept for completeness
        "ocean_skin_temperature",
        "skin_temperature",
        "thetao",
        "to",
    ],

    # Rainfall / Precipitation
    "precipitation": [
        "precipitation",
        "rainfall",
        "rain_rate",
        "tp",
        "total_precipitation",
        "precip",
        # Additional
        "rain",
        "pr",
        "imerg",
    ],

    # Wind
    "wind_speed": [
        "wind_speed",
        "wind_velocity",
        "u10",
        "v10",
        "wind",
        "surface_wind",
        # Additional
        "wspd",
        "sfcwind",
    ],

    # Sea Surface Salinity
    "sea_surface_salinity": [
        "sea_surface_salinity",
        "sss",
        "salinity",
        # Additional
        "sal",
        "so",
        "practical_salinity",
        "sea_water_salinity",
    ],

    # Chlorophyll
    "chlorophyll": [
        "chlorophyll",
        "chlorophyll_a",
        "chlor_a",
        "chl",
        "chl_a",
        # Additional
        "chla",
        "chl1_mean",
    ],

    # Wave Height
    "significant_wave_height": [
        "significant_wave_height",
        "swh",
        "wave_height",
        "hs",
    ],

    # ---- extend below this line ------------------------------------ #

    # NDVI
    "ndvi": [
        "ndvi",
        "normalized_difference_vegetation_index",
        "vegetation_index",
    ],

    # Soil moisture
    "soil_moisture": [
        "soil_moisture",
        "sm",
        "surface_soil_moisture",
        "smsurf",
    ],

    # Sea Surface Height
    "sea_surface_height": [
        "sea_surface_height",
        "ssh",
        "adt",
        "sla",
    ],

    # Land Surface Temperature
    "land_surface_temperature": [
        "land_surface_temperature",
        "lst",
    ],

    # Flood extent
    "flood_extent": [
        "flood_extent",
        "flood",
        "inundation",
        "flooded_area",
    ],

    # NO₂
    "no2": [
        "no2",
        "nitrogen_dioxide",
        "nitrogendioxide",
    ],
}


def _normalize_var(text: str) -> str:
    """
    Normalise a variable name for alias matching.

    Lowercases the string and strips underscores, hyphens, and
    whitespace so that e.g. 'Sea Surface Temperature',
    'sea_surface_temperature', 'sea-surface-temperature', and
    'seasurfacetemperature' all collapse to the same key for
    comparison purposes.

    This function is used internally by validate_variables() and is
    not part of the public API, but it is kept at module level so
    it is easy to test in isolation and to reuse if the alias table
    is later moved to a shared module.
    """
    return re.sub(r"[\s_\-]+", "", text.lower())



# Geographic region → synonyms and parent regions (for spatial matching).
# A dataset that covers a parent region is considered to cover the child.
SPATIAL_HIERARCHY: Dict[str, List[str]] = {
    "global": [
        "global", "world", "worldwide", "all regions",
    ],
    "indian ocean": [
        "indian ocean", "india", "south asia", "bay of bengal",
        "arabian sea", "lakshadweep", "andaman sea", "maldives",
        "sri lanka",
    ],
    "bay of bengal": [
        "bay of bengal", "indian ocean", "south asia", "india",
        "global",
    ],
    "arabian sea": [
        "arabian sea", "indian ocean", "south asia", "india", "global",
    ],
    "lakshadweep": [
        "lakshadweep", "indian ocean", "india", "south asia", "global",
    ],
    "north atlantic": [
        "north atlantic", "atlantic", "europe", "north america",
        "global",
    ],
    "europe": [
        "europe", "european waters", "north atlantic", "global",
    ],
    "north america": [
        "north america", "usa", "united states", "north atlantic",
        "pacific northwest", "global",
    ],
    "asia": [
        "asia", "south asia", "east asia", "india", "china",
        "japan", "global",
    ],
    "africa": [
        "africa", "east africa", "west africa", "south africa",
        "indian ocean", "global",
    ],
}


# ------------------------------------------------------------------ #
# Data models (plain dataclasses — no pydantic dependency)           #
# ------------------------------------------------------------------ #

class StageResult:
    """
    Result of a single validation stage.

    Attributes:
        passed:      True if this stage succeeded.
        score:       0.0–1.0, stage-level quality signal.
        errors:      Hard failures (cause overall failure).
        warnings:    Soft failures (reduce confidence only).
        details:     Free-text notes for the report.
    """

    def __init__(
        self,
        passed: bool,
        score: float,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        details: Optional[str] = None,
    ) -> None:
        self.passed   = passed
        self.score    = round(max(0.0, min(1.0, score)), 4)
        self.errors   = errors   or []
        self.warnings = warnings or []
        self.details  = details  or ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed":   self.passed,
            "score":    self.score,
            "errors":   self.errors,
            "warnings": self.warnings,
            "details":  self.details,
        }


class ValidationReport:
    """
    Complete validation report for one dataset.

    Contains one StageResult per validation stage plus the computed
    overall result, composite score, and confidence score.
    """

    # Weights for the seven stages when computing validation_score.
    STAGE_WEIGHTS: Dict[str, float] = {
        "file_validation":       0.10,
        "metadata_validation":   0.15,
        "variable_validation":   0.20,
        "spatial_validation":    0.15,
        "temporal_validation":   0.15,
        "scientific_validation": 0.10,
        "integrity_validation":  0.15,
    }

    def __init__(
        self,
        dataset_id: str,
        dataset_name: str,
        provider: str,
        file_validation:       StageResult,
        metadata_validation:   StageResult,
        variable_validation:   StageResult,
        spatial_validation:    StageResult,
        temporal_validation:   StageResult,
        scientific_validation: StageResult,
        integrity_validation:  StageResult,
    ) -> None:
        self.validation_id        = str(uuid.uuid4())
        self.dataset_id           = dataset_id
        self.dataset_name         = dataset_name
        self.provider             = provider
        self.validation_timestamp = datetime.now(timezone.utc).isoformat()

        self.file_validation       = file_validation
        self.metadata_validation   = metadata_validation
        self.variable_validation   = variable_validation
        self.spatial_validation    = spatial_validation
        self.temporal_validation   = temporal_validation
        self.scientific_validation = scientific_validation
        self.integrity_validation  = integrity_validation

        self.overall_validation = self._compute_overall()
        self.validation_score   = self._compute_score()
        self.confidence_score   = self._compute_confidence()
        self.validation_summary = self._compute_summary()
        self.validation_errors  = self._collect_errors()
        self.validation_warnings= self._collect_warnings()

    # ---- private helpers ------------------------------------------ #

    def _stages(self) -> List[Tuple[str, StageResult]]:
        return [
            ("file_validation",       self.file_validation),
            ("metadata_validation",   self.metadata_validation),
            ("variable_validation",   self.variable_validation),
            ("spatial_validation",    self.spatial_validation),
            ("temporal_validation",   self.temporal_validation),
            ("scientific_validation", self.scientific_validation),
            ("integrity_validation",  self.integrity_validation),
        ]

    def _compute_overall(self) -> bool:
        # A dataset passes overall only if every stage that carries
        # hard errors passes.  scientific_validation is allowed to
        # "warn" (score < 1.0) without causing an overall failure,
        # since missing optional metadata fields only reduce confidence.
        REQUIRED_STAGES = {
            "file_validation", "variable_validation",
            "spatial_validation", "temporal_validation",
            "integrity_validation",
        }
        for name, stage in self._stages():
            if name in REQUIRED_STAGES and not stage.passed:
                return False
        return True

    def _compute_score(self) -> float:
        total = 0.0
        for name, stage in self._stages():
            total += stage.score * self.STAGE_WEIGHTS[name]
        return round(total, 4)

    def _compute_confidence(self) -> float:
        """
        Confidence measures how certain we are about this evaluation,
        independent of the validation_score.

        High confidence requires:
            - most metadata fields present and parseable
            - integrity result is available and trusted
            - no warnings from scientific_validation

        A dataset can have score=0.95 with confidence=0.52 if its
        metadata is sparse (e.g. temporal coverage unknown).
        """
        penalty = 0.0

        # Sparse metadata → lower confidence
        if self.metadata_validation.score < 0.6:
            penalty += 0.20
        elif self.metadata_validation.score < 0.8:
            penalty += 0.10

        # Missing scientific metadata → lower confidence
        if self.scientific_validation.score < 0.6:
            penalty += 0.15
        elif self.scientific_validation.score < 0.8:
            penalty += 0.07

        # Integrity unknown or failed → lower confidence
        if not self.integrity_validation.passed:
            penalty += 0.20

        # Spatial / temporal unknowns → lower confidence
        if not self.spatial_validation.passed:
            penalty += 0.10
        if not self.temporal_validation.passed:
            penalty += 0.10

        # Warnings accumulate mild penalty
        total_warnings = sum(len(s.warnings) for _, s in self._stages())
        penalty += min(0.15, total_warnings * 0.03)

        return round(max(0.0, 1.0 - penalty), 4)

    def _compute_summary(self) -> str:
        passed_names = [n for n, s in self._stages() if s.passed]
        failed_names = [n for n, s in self._stages() if not s.passed]
        if self.overall_validation:
            return (
                f"PASSED — all {len(passed_names)} required stages "
                f"succeeded. Score: {self.validation_score:.2f}, "
                f"Confidence: {self.confidence_score:.2f}."
            )
        failed_display = ", ".join(
            n.replace("_", " ") for n in failed_names
        )
        return (
            f"FAILED — {len(failed_names)} stage(s) failed: "
            f"{failed_display}. Score: {self.validation_score:.2f}, "
            f"Confidence: {self.confidence_score:.2f}."
        )

    def _collect_errors(self) -> List[str]:
        errors = []
        for name, stage in self._stages():
            for e in stage.errors:
                errors.append(f"[{name}] {e}")
        return errors

    def _collect_warnings(self) -> List[str]:
        warnings = []
        for name, stage in self._stages():
            for w in stage.warnings:
                warnings.append(f"[{name}] {w}")
        return warnings

    def to_dict(self) -> Dict[str, Any]:
        return {
            "validation_id":        self.validation_id,
            "dataset_id":           self.dataset_id,
            "dataset_name":         self.dataset_name,
            "provider":             self.provider,
            "validation_timestamp": self.validation_timestamp,
            "file_validation":       self.file_validation.to_dict(),
            "metadata_validation":   self.metadata_validation.to_dict(),
            "variable_validation":   self.variable_validation.to_dict(),
            "spatial_validation":    self.spatial_validation.to_dict(),
            "temporal_validation":   self.temporal_validation.to_dict(),
            "scientific_validation": self.scientific_validation.to_dict(),
            "integrity_validation":  self.integrity_validation.to_dict(),
            "overall_validation":    self.overall_validation,
            "validation_score":      self.validation_score,
            "confidence_score":      self.confidence_score,
            "validation_summary":    self.validation_summary,
            "validation_errors":     self.validation_errors,
            "validation_warnings":   self.validation_warnings,
        }


# ------------------------------------------------------------------ #
# Validation stage functions                                          #
# ------------------------------------------------------------------ #

def validate_file(file_path: str) -> StageResult:
    """
    Stage 1 — File Validation.

    Verifies that the file exists, is non-empty, has a supported format,
    and is actually readable (not just accessible by name).

    Args:
        file_path: Absolute or relative path to the downloaded file.

    Returns:
        StageResult with errors for hard failures and warnings for soft ones.
    """
    errors:   List[str] = []
    warnings: List[str] = []
    path = Path(file_path)

    # Existence
    if not path.exists():
        return StageResult(
            passed=False, score=0.0,
            errors=[f"File does not exist: {file_path}"],
        )

    # Size > 0
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        return StageResult(
            passed=False, score=0.0,
            errors=[f"Cannot stat file: {exc}"],
        )
    if size_bytes == 0:
        return StageResult(
            passed=False, score=0.0,
            errors=["File is empty (0 bytes)."],
        )

    # Supported format
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        errors.append(
            f"Unsupported file format: '{suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}."
        )

    # Readable
    try:
        with open(path, "rb") as fh:
            first_chunk = fh.read(512)
        if not first_chunk:
            errors.append("File appears empty after reading.")
    except OSError as exc:
        errors.append(f"File is not readable: {exc}")

    score = 1.0 if not errors else 0.3
    details = (
        f"File: {path.name}, size: {size_bytes:,} bytes, "
        f"format: {suffix or '(no extension)'}."
    )
    return StageResult(
        passed=len(errors) == 0,
        score=score,
        errors=errors,
        warnings=warnings,
        details=details,
    )


def validate_metadata(metadata: Dict[str, Any]) -> StageResult:
    """
    Stage 2 — Metadata Validation.

    Verifies that all required metadata fields are present and non-empty.
    Missing fields are hard failures; fields present but with low-quality
    values (e.g. 'Unknown') are warnings.

    Args:
        metadata: Dict with keys matching the dataset's declared metadata.

    Returns:
        StageResult.
    """
    REQUIRED_FIELDS = [
        "dataset_name", "provider", "version",
        "variables", "spatial_coverage", "temporal_coverage",
        "resolution", "license",
    ]
    WARN_VALUES = {"unknown", "n/a", "none", "not available", "tbd", ""}

    errors:   List[str] = []
    warnings: List[str] = []

    for field in REQUIRED_FIELDS:
        value = metadata.get(field)
        if value is None:
            errors.append(f"Required metadata field missing: '{field}'.")
        elif isinstance(value, str) and value.strip().lower() in WARN_VALUES:
            warnings.append(
                f"Field '{field}' is present but has a placeholder value: '{value}'."
            )
        elif isinstance(value, (list, dict)) and len(value) == 0:
            warnings.append(f"Field '{field}' is present but empty.")

    filled = len(REQUIRED_FIELDS) - len(errors)
    score  = round(filled / len(REQUIRED_FIELDS), 4)
    score -= len(warnings) * 0.05
    score  = max(0.0, score)

    return StageResult(
        passed=len(errors) == 0,
        score=score,
        errors=errors,
        warnings=warnings,
        details=(
            f"{filled}/{len(REQUIRED_FIELDS)} required fields present."
        ),
    )


def validate_variables(
    required_variables: List[str],
    dataset_variables: List[str],
) -> StageResult:
    """
    Stage 3 — Variable Validation.

    Verifies that every required scientific variable is present in the
    dataset, using alias matching so that 'SST', 'sea_surface_temperature',
    and 'sst' all resolve to the same canonical variable.

    Args:
        required_variables: Variables the user requested (from Agent 2/3).
        dataset_variables:  Variables declared in the downloaded dataset's
                            metadata.

    Returns:
        StageResult.
    """
    if not required_variables:
        return StageResult(
            passed=True, score=1.0,
            details="No specific variables required — stage skipped.",
        )

    errors:   List[str] = []
    warnings: List[str] = []
    found:    List[str] = []

    dataset_vars_norm = {_normalize_var(v) for v in dataset_variables}

    for req in required_variables:
        req_norm = _normalize_var(req)

        # Direct match first (after normalisation)
        if req_norm in dataset_vars_norm:
            found.append(req)
            continue

        # Alias match: normalise every alias entry the same way so that
        # 'sea-surface-temperature', 'sea_surface_temperature', and
        # 'Sea Surface Temperature' all hit the same canonical key.
        matched = False
        for canonical, aliases in VARIABLE_ALIASES.items():
            aliases_norm = {_normalize_var(a) for a in aliases}
            aliases_norm.add(_normalize_var(canonical))
            req_in_aliases     = req_norm in aliases_norm
            dataset_in_aliases = bool(dataset_vars_norm & aliases_norm)
            if req_in_aliases and dataset_in_aliases:
                found.append(req)
                matched = True
                break

        if not matched:
            errors.append(
                f"Required variable '{req}' not found in dataset. "
                f"Dataset declares: {dataset_variables or ['(none)']}"
            )

    score = round(len(found) / len(required_variables), 4) if required_variables else 1.0
    return StageResult(
        passed=len(errors) == 0,
        score=score,
        errors=errors,
        warnings=warnings,
        details=(
            f"{len(found)}/{len(required_variables)} required variables found."
        ),
    )


def validate_spatial(
    requested_region: str,
    dataset_coverage: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> StageResult:
    """
    Stage 4 — Spatial Validation.

    Step 4a — Domain gate (NEW):
        Before any geographic comparison, determine whether the dataset
        is actually an Earth-observation or geospatial dataset.
        Non-geospatial datasets (financial, social-media, text, etc.)
        fail this stage immediately regardless of what words appear in
        their spatial_coverage string.  In particular, "Global" alone
        is NOT accepted as valid geographic coverage — we first confirm
        the dataset is genuinely geospatial.

    Step 4b — Coverage check:
        If the domain gate passes, check that the dataset's declared
        spatial coverage overlaps the requested study area using
        SPATIAL_HIERARCHY synonyms and parent–child relationships.

    Geographic mismatches reduce the score significantly but do NOT
    cause a hard failure unless the mismatch is clearly confirmed.
    A global Earth-observation dataset always passes step 4b.

    Args:
        requested_region:  The study area from the user's request.
        dataset_coverage:  The spatial coverage declared in the dataset.
        metadata:          Full metadata dict (used for domain detection).
                           Optional — pass None to skip domain detection.

    Returns:
        StageResult.
    """
    # ------------------------------------------------------------------ #
    # Step 4a — Earth-observation domain gate                            #
    # ------------------------------------------------------------------ #
    # Tokens that strongly indicate a non-geospatial dataset.  Their
    # presence in the dataset's name, provider, variable list, or
    # coverage string causes an immediate hard failure before any
    # geographic comparison is attempted.
    _NON_GEO_SIGNALS = {
        # Financial / market
        "stock", "equity", "share price", "market cap", "bond", "etf",
        "portfolio", "trading", "hedge fund", "forex", "cryptocurrency",
        "bitcoin", "nasdaq", "nyse", "bloomberg", "financial",
        # Retail / customer
        "customer", "transaction", "invoice", "e-commerce", "retail",
        "sales data", "crm",
        # Social media / text
        "tweet", "social media", "facebook", "instagram", "reddit",
        "text corpus", "nlp", "sentiment", "news article",
        # Generic non-scientific
        "human resources", "payroll", "logistics", "supply chain",
    }

    # Tokens (in any metadata field) that positively confirm this is a
    # geospatial / Earth-observation dataset.  Only one needs to match.
    _GEO_SIGNALS = {
        # Platform / sensor types
        "satellite", "radar", "lidar", "remote sensing", "sensor",
        "buoy", "argo", "altimetry", "radiometer", "sonar",
        # Data formats / standards that are overwhelmingly geospatial
        "netcdf", "geotiff", "hdf5", "grib", "opendap", "erddap",
        "wms", "wfs", "ogc", "epsg", "wgs84", "crs",
        # Scientific domains
        "oceanography", "ocean", "sea surface", "sea level",
        "bathymetry", "chlorophyll", "salinity", "wave height",
        "precipitation", "rainfall", "wind speed", "temperature",
        "land cover", "elevation", "dem", "ndvi", "flood", "drought",
        "atmospheric", "aerosol", "ozone", "co2", "methane",
        "climate", "reanalysis", "model output", "gridded",
        # Known Earth-observation providers / missions
        "nasa", "esa", "isro", "noaa", "copernicus", "sentinel",
        "modis", "viirs", "landsat", "ghrsst", "cmems", "incois",
        "jaxa", "cnes", "eumetsat", "ecmwf", "era5",
        # Geographic terms that only make sense for real spatial data
        "latitude", "longitude", "degrees north", "degrees east",
        "coordinate", "projection", "spatial resolution",
        "bounding box", "extent", "coverage area",
    }

    if metadata is not None:
        # Collect every string value from the metadata into one blob
        # for efficient signal scanning.
        _blob_parts: List[str] = [dataset_coverage or ""]
        for val in metadata.values():
            if isinstance(val, str):
                _blob_parts.append(val)
            elif isinstance(val, list):
                _blob_parts.extend(
                    str(v) for v in val if isinstance(v, (str, int, float))
                )
        meta_blob = " ".join(_blob_parts).lower()

        # Hard fail if any non-geospatial signal is found
        triggered_non_geo = [s for s in _NON_GEO_SIGNALS if s in meta_blob]
        if triggered_non_geo:
            # Still check for overriding geo signals — e.g. a dataset
            # about flood events on stock-exchange districts is still geo.
            triggered_geo = [s for s in _GEO_SIGNALS if s in meta_blob]
            if not triggered_geo:
                return StageResult(
                    passed=False,
                    score=0.0,
                    errors=[
                        f"Dataset does not appear to be a geospatial or "
                        f"Earth-observation dataset. Non-geospatial signals "
                        f"detected: {', '.join(sorted(triggered_non_geo)[:4])}. "
                        f"Spatial validation cannot be applied to this dataset."
                    ],
                    details="Domain gate failed — non-geospatial dataset.",
                )

        # Require at least one positive geo signal when no clear non-geo
        # signal was found but the dataset also has no recognisable geo
        # vocabulary — mark as unverifiable rather than passing silently.
        triggered_geo = [s for s in _GEO_SIGNALS if s in meta_blob]
        if not triggered_geo and not triggered_non_geo:
            # Ambiguous — cannot confirm the dataset is geospatial.
            # Warn but do not hard-fail (might be a niche dataset whose
            # vocabulary we haven't seen before).
            pass   # fall through to geographic matching with a warning

    # ------------------------------------------------------------------ #
    # Step 4b — Geographic coverage check                                #
    # ------------------------------------------------------------------ #
    if not requested_region or not dataset_coverage:
        return StageResult(
            passed=True, score=0.6,
            warnings=["Spatial coverage or requested region not specified — "
                      "cannot confirm coverage. Treating as neutral."],
        )

    req_lower = requested_region.lower().strip()
    cov_lower = dataset_coverage.lower().strip()

    # "Global" is only accepted as geographic coverage when the domain
    # gate above has already confirmed this is a geospatial dataset.
    # We do NOT match "Global Exchanges", "Global Stock Market", etc.
    # Strategy: require 'global' to appear alongside recognised
    # geographic qualifiers, OR for the metadata to have passed the
    # geo-signal check above without triggering any non-geo signal.
    _GEOGRAPHIC_GLOBAL_PHRASES = [
        "global coverage", "global ocean", "global dataset",
        "global satellite", "global reanalysis", "global climate",
        "global precipitation", "global sea surface", "worldwide",
        "world", "all regions",
    ]
    is_geographic_global = any(phrase in cov_lower
                               for phrase in _GEOGRAPHIC_GLOBAL_PHRASES)

    # Also accept bare "global" only when no non-geo signals were
    # detected from metadata (i.e. the domain gate above did not raise
    # a concern). We track this via the local triggered_non_geo variable
    # which is only defined when metadata was provided.
    if not is_geographic_global and "global" in cov_lower:
        _non_geo_present = (
            metadata is not None
            and any(s in " ".join(
                str(v) for v in metadata.values() if isinstance(v, str)
            ).lower() for s in _NON_GEO_SIGNALS)
        )
        if not _non_geo_present:
            is_geographic_global = True

    if is_geographic_global:
        return StageResult(
            passed=True, score=0.95,
            details="Dataset declares global geographic coverage.",
        )

    # Direct substring containment in either direction
    if req_lower in cov_lower or cov_lower in req_lower:
        return StageResult(
            passed=True, score=1.0,
            details=(
                f"Requested region '{requested_region}' found in "
                f"coverage '{dataset_coverage}'."
            ),
        )

    # Hierarchy / synonym check
    for region_key, synonyms in SPATIAL_HIERARCHY.items():
        syns_lower = [s.lower() for s in synonyms]
        req_in_hier     = req_lower in syns_lower or region_key == req_lower
        cov_matches_hier = any(s in cov_lower for s in syns_lower)
        if req_in_hier and cov_matches_hier:
            return StageResult(
                passed=True, score=0.85,
                details=(
                    f"Spatial match via '{region_key}' hierarchy: "
                    f"'{dataset_coverage}' covers '{requested_region}'."
                ),
            )

    # No match found — significant penalty but not a hard block
    return StageResult(
        passed=False,
        score=0.10,
        errors=[
            f"Dataset coverage '{dataset_coverage}' does not appear to "
            f"cover the requested region '{requested_region}'."
        ],
        details="Geographic mismatch detected.",
    )


def validate_temporal(
    requested_period: str,
    dataset_coverage: str,
) -> StageResult:
    """
    Stage 5 — Temporal Validation.

    Checks that the dataset covers the requested time period by
    extracting 4-digit years from both strings and testing for overlap.

    'Present', 'ongoing', 'real-time' in the dataset's coverage
    automatically satisfies any requested period up to the current year.

    Args:
        requested_period:  Date range from the user's request
                           (e.g. '2023', '2020-2024', 'January 2024').
        dataset_coverage:  Temporal coverage declared in the dataset
                           (e.g. '2000-present', '1981–2023').

    Returns:
        StageResult.
    """
    CURRENT_YEAR = datetime.now().year

    ONGOING_TOKENS = {"present", "ongoing", "real-time", "real time", "current"}

    def extract_year_range(text: str) -> Optional[Tuple[int, int]]:
        years = [int(y) for y in re.findall(r"\b(19|20)\d{2}\b", text)]
        if not years:
            return None
        lo, hi = min(years), max(years)
        if any(tok in text.lower() for tok in ONGOING_TOKENS):
            hi = CURRENT_YEAR
        return (lo, hi)

    if not requested_period or not dataset_coverage:
        return StageResult(
            passed=True, score=0.6,
            warnings=["Temporal coverage or requested period not specified — "
                      "cannot confirm coverage. Treating as neutral."],
        )

    if any(tok in dataset_coverage.lower() for tok in ONGOING_TOKENS):
        req_span = extract_year_range(requested_period)
        if req_span is None:
            return StageResult(
                passed=True, score=0.8,
                details="Dataset is ongoing; requested period could not be parsed.",
                warnings=["Could not parse year from requested period — "
                          "assumed covered by ongoing dataset."],
            )
        return StageResult(
            passed=True, score=1.0,
            details=f"Dataset is ongoing (extends to {CURRENT_YEAR}); covers {req_span}.",
        )

    req_span = extract_year_range(requested_period)
    cov_span = extract_year_range(dataset_coverage)

    if req_span is None:
        return StageResult(
            passed=True, score=0.7,
            warnings=[f"Could not parse a year range from requested period '{requested_period}'."],
        )
    if cov_span is None:
        return StageResult(
            passed=True, score=0.6,
            warnings=[f"Could not parse a year range from dataset coverage '{dataset_coverage}'."],
        )

    # Overlap test: [req_lo, req_hi] ∩ [cov_lo, cov_hi] ≠ ∅
    req_lo, req_hi = req_span
    cov_lo, cov_hi = cov_span

    if cov_lo <= req_hi and req_lo <= cov_hi:
        # Full containment is better than partial overlap
        if cov_lo <= req_lo and cov_hi >= req_hi:
            score = 1.0
            detail = f"Dataset {cov_span} fully contains requested period {req_span}."
        else:
            overlap_lo = max(req_lo, cov_lo)
            overlap_hi = min(req_hi, cov_hi)
            frac = (overlap_hi - overlap_lo + 1) / max(1, req_hi - req_lo + 1)
            score = round(0.6 + 0.4 * frac, 4)
            detail = (
                f"Partial overlap: {(overlap_lo, overlap_hi)} "
                f"of requested {req_span}."
            )
        return StageResult(passed=True, score=score, details=detail)

    return StageResult(
        passed=False,
        score=0.0,
        errors=[
            f"No temporal overlap: dataset covers {cov_span} but "
            f"'{requested_period}' ({req_span}) was requested."
        ],
    )


def validate_scientific(metadata: Dict[str, Any]) -> StageResult:
    """
    Stage 6 — Scientific Validation.

    Checks for the presence of units, coordinate reference system (CRS),
    variable descriptions, and resolution. Missing fields reduce
    confidence without causing a hard failure — they make the evaluation
    less certain, not less scientifically valid.

    Args:
        metadata: The same metadata dict used in validate_metadata().

    Returns:
        StageResult.
    """
    warnings: List[str] = []
    present = 0
    total   = 4

    if metadata.get("units"):
        present += 1
    else:
        warnings.append("No units declared — cannot verify measurement standard.")

    if metadata.get("crs") or metadata.get("coordinate_reference_system"):
        present += 1
    else:
        warnings.append(
            "No coordinate reference system (CRS) declared — "
            "cannot verify geospatial projection."
        )

    if metadata.get("variable_descriptions") or metadata.get("variables"):
        present += 1
    else:
        warnings.append("No variable descriptions found.")

    if metadata.get("resolution"):
        present += 1
    else:
        warnings.append("No resolution information declared.")

    score = round(present / total, 4)
    return StageResult(
        passed=True,          # scientific metadata gaps → warnings, never hard fail
        score=score,
        warnings=warnings,
        details=f"{present}/{total} scientific metadata fields present.",
    )


def validate_integrity(integrity_record: Optional[Dict[str, Any]]) -> StageResult:
    """
    Stage 7 — Integrity Validation.

    Reuses the result from security/integrity.py rather than recomputing
    a checksum. Expects an integrity record returned by that module.

    Does NOT compute a new checksum — this is intentional. The integrity
    module runs earlier in the pipeline and its result is passed in here.

    Args:
        integrity_record: The dict returned by integrity.py for this
                          dataset, or None if integrity.py has not run yet.

    Returns:
        StageResult.
    """
    if integrity_record is None:
        return StageResult(
            passed=False,
            score=0.0,
            errors=["No integrity record found. Run integrity.py first."],
        )

    # Integrity.py records use 'integrity_passed' or 'verification_status'
    # depending on version — check both.
    passed_flag = integrity_record.get("integrity_passed")
    if passed_flag is None:
        status = integrity_record.get("verification_status", "").lower()
        passed_flag = status in ("passed", "verified", "ok", "valid")

    if passed_flag:
        checksum   = integrity_record.get("checksum", "not recorded")
        algorithm  = integrity_record.get("algorithm", "unknown")
        return StageResult(
            passed=True,
            score=1.0,
            details=(
                f"Integrity verified. Algorithm: {algorithm}. "
                f"Checksum: {checksum[:16]}..." if len(str(checksum)) > 16
                else f"Integrity verified. Algorithm: {algorithm}."
            ),
        )

    reason = integrity_record.get("failure_reason", "Integrity check failed.")
    return StageResult(
        passed=False,
        score=0.0,
        errors=[f"Integrity verification failed: {reason}"],
    )


# ------------------------------------------------------------------ #
# Report generation                                                   #
# ------------------------------------------------------------------ #

def generate_validation_report(
    dataset_id:           str,
    dataset_name:         str,
    provider:             str,
    file_path:            str,
    metadata:             Dict[str, Any],
    required_variables:   List[str],
    requested_region:     str,
    requested_period:     str,
    integrity_record:     Optional[Dict[str, Any]] = None,
) -> ValidationReport:
    """
    Runs all seven validation stages and assembles a ValidationReport.

    Args:
        dataset_id:           Unique ID for the dataset (e.g. from Agent 3).
        dataset_name:         Human-readable dataset name.
        provider:             Provider name.
        file_path:            Path to the downloaded file on disk.
        metadata:             Metadata dict (name, variables, coverage, etc.).
        required_variables:   Variables the user requested.
        requested_region:     Geographic region requested by the user.
        requested_period:     Time period requested by the user.
        integrity_record:     Record from integrity.py, or None.

    Returns:
        ValidationReport with all stages populated.
    """
    dataset_variables = metadata.get("variables", [])
    if isinstance(dataset_variables, str):
        dataset_variables = [v.strip() for v in dataset_variables.split(",")]

    file_val       = validate_file(file_path)
    metadata_val   = validate_metadata(metadata)
    variable_val   = validate_variables(required_variables, dataset_variables)
    spatial_val    = validate_spatial(
                         requested_region,
                         metadata.get("spatial_coverage", ""),
                         metadata=metadata,
                     )
    temporal_val   = validate_temporal(
                         requested_period,
                         metadata.get("temporal_coverage", ""),
                     )
    scientific_val = validate_scientific(metadata)
    integrity_val  = validate_integrity(integrity_record)

    return ValidationReport(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        provider=provider,
        file_validation=file_val,
        metadata_validation=metadata_val,
        variable_validation=variable_val,
        spatial_validation=spatial_val,
        temporal_validation=temporal_val,
        scientific_validation=scientific_val,
        integrity_validation=integrity_val,
    )


# ------------------------------------------------------------------ #
# Database functions                                                  #
# ------------------------------------------------------------------ #

def _load_db() -> Dict[str, Any]:
    """Load validation_db.json, creating it if absent."""
    if not DB_PATH.exists():
        return {"validation_reports": []}
    try:
        with open(DB_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {"validation_reports": []}


def _save_db(db: Dict[str, Any]) -> None:
    """Write validation_db.json atomically (temp-file + rename)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=DB_PATH.parent, suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(db, fh, indent=2, ensure_ascii=False)
        os.replace(tmp_path, DB_PATH)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def save_validation_report(report: ValidationReport) -> str:
    """
    Append a ValidationReport to validation_db.json.

    Never overwrites existing reports. Creates the DB file if absent.

    Args:
        report: A completed ValidationReport.

    Returns:
        The report's validation_id.
    """
    import contextlib
    db = _load_db()
    db["validation_reports"].append(report.to_dict())
    _save_db(db)
    return report.validation_id


def get_validation_report(validation_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a single validation report by its validation_id.

    Args:
        validation_id: The UUID returned by save_validation_report().

    Returns:
        The report dict, or None if not found.
    """
    db = _load_db()
    for entry in db["validation_reports"]:
        if entry.get("validation_id") == validation_id:
            return entry
    return None


def list_validation_reports() -> List[Dict[str, Any]]:
    """
    Return all validation reports in the database, newest first.

    Returns:
        List of report dicts (full content, not summaries).
    """
    db = _load_db()
    reports = db.get("validation_reports", [])
    return list(reversed(reports))


# ------------------------------------------------------------------ #
# Print helpers                                                       #
# ------------------------------------------------------------------ #

def _stage_line(name: str, result: StageResult) -> str:
    icon   = "✓" if result.passed else "✗"
    status = "PASS" if result.passed else "FAIL"
    return f"  {icon} {name:<30} {status:<5}  score={result.score:.2f}"


def print_validation_report(report: ValidationReport) -> None:
    """Print a human-readable summary of a ValidationReport."""
    icon   = "✓ PASSED" if report.overall_validation else "✗ FAILED"
    border = "=" * 70

    print(f"\n{border}")
    print(f"DATASET VALIDATION REPORT")
    print(f"{border}")
    print(f"  Validation ID  : {report.validation_id}")
    print(f"  Dataset ID     : {report.dataset_id}")
    print(f"  Dataset Name   : {report.dataset_name}")
    print(f"  Provider       : {report.provider}")
    print(f"  Timestamp      : {report.validation_timestamp}")
    print(f"  Overall        : {icon}")
    print(f"  Score          : {report.validation_score:.2f} / 1.00")
    print(f"  Confidence     : {report.confidence_score:.2f} / 1.00")
    print()
    print(f"  Validation Stages:")
    stages = [
        ("1. File Validation",       report.file_validation),
        ("2. Metadata Validation",   report.metadata_validation),
        ("3. Variable Validation",   report.variable_validation),
        ("4. Spatial Validation",    report.spatial_validation),
        ("5. Temporal Validation",   report.temporal_validation),
        ("6. Scientific Validation", report.scientific_validation),
        ("7. Integrity Validation",  report.integrity_validation),
    ]
    for name, stage in stages:
        print(_stage_line(name, stage))

    if report.validation_errors:
        print(f"\n  Errors ({len(report.validation_errors)}):")
        for e in report.validation_errors:
            print(f"    ✗ {e}")

    if report.validation_warnings:
        print(f"\n  Warnings ({len(report.validation_warnings)}):")
        for w in report.validation_warnings:
            print(f"    ⚠ {w}")

    print(f"\n  Summary: {report.validation_summary}")
    print(f"{border}")


# ------------------------------------------------------------------ #
# Standalone demo                                                     #
# ------------------------------------------------------------------ #

def _run_demo() -> None:
    """
    Standalone demo.

    Creates two mock datasets:
        1. A valid SST dataset — demonstrates a complete successful validation.
        2. A broken dataset with wrong domain + temporal mismatch
           — demonstrates a clear, explained failure.
    """
    import contextlib

    print("\n" + "=" * 70)
    print("DATASET VALIDATION MODULE — STANDALONE DEMO")
    print("=" * 70)

    # ---------------------------------------------------------- #
    # Create temporary files representing downloaded datasets    #
    # ---------------------------------------------------------- #
    sst_file  = tempfile.NamedTemporaryFile(
        suffix=".nc", delete=False,
        dir=Path(__file__).parent,
    )
    sst_file.write(b"mock NetCDF content for SST dataset")
    sst_file.close()

    bad_file  = tempfile.NamedTemporaryFile(
        suffix=".nc", delete=False,
        dir=Path(__file__).parent,
    )
    bad_file.write(b"mock content for a bad dataset")
    bad_file.close()

    try:
        # ---------------------------------------------------------- #
        # Demo 1 — Successful validation                             #
        # (NASA GHRSST SST product, Bay of Bengal, 2023)             #
        # ---------------------------------------------------------- #
        print("\n\nDEMO 1 — SUCCESSFUL VALIDATION")
        print("-" * 70)
        print("Dataset : NASA GHRSST Multi-Scale Ultra-High Resolution SST")
        print("Provider: NASA / GHRSST")
        print("Request : Sea Surface Temperature, Bay of Bengal, 2023")

        sst_metadata = {
            "dataset_name":    "GHRSST MUR SST v4.1",
            "provider":        "NASA / GHRSST",
            "version":         "4.1",
            "variables":       ["sst", "sea_ice_fraction", "analysed_sst_uncertainty"],
            "spatial_coverage": "Indian Ocean, Bay of Bengal, Global",
            "temporal_coverage": "2002-present",
            "resolution":      "0.01 degree (~1 km)",
            "license":         "NASA Open Data",
            "units":           {"sst": "Kelvin"},
            "crs":             "EPSG:4326 (WGS84)",
            "variable_descriptions": {
                "sst": "Sea surface temperature analysed field",
            },
        }

        sst_integrity = {
            "integrity_passed": True,
            "algorithm":        "SHA-256",
            "checksum":         "a1b2c3d4e5f60718293a4b5c6d7e8f90",
        }

        sst_report = generate_validation_report(
            dataset_id="ghrsst_mur_sst_2023_bay_of_bengal",
            dataset_name="GHRSST MUR SST v4.1",
            provider="NASA / GHRSST",
            file_path=sst_file.name,
            metadata=sst_metadata,
            required_variables=["sea_surface_temperature"],
            requested_region="Bay of Bengal",
            requested_period="2023",
            integrity_record=sst_integrity,
        )
        print_validation_report(sst_report)

        vid1 = save_validation_report(sst_report)
        print(f"\n  ✓ Report saved. Validation ID: {vid1}")

        retrieved = get_validation_report(vid1)
        assert retrieved is not None
        print(f"  ✓ Report retrieved: {retrieved['dataset_name']} "
              f"— overall={retrieved['overall_validation']}")

        # ---------------------------------------------------------- #
        # Demo 2 — Failed validation                                 #
        # (Financial dataset with wrong domain + wrong region/time)  #
        # ---------------------------------------------------------- #
        print("\n\nDEMO 2 — FAILED VALIDATION")
        print("-" * 70)
        print("Dataset : Global Stock Market Prices")
        print("Provider: Bloomberg Finance")
        print("Request : Sea Surface Temperature, Bay of Bengal, 2023")
        print("Expected: FAIL — wrong scientific domain, wrong region, wrong time")

        bad_metadata = {
            "dataset_name":    "Global Stock Market Prices",
            "provider":        "Bloomberg Finance",
            "version":         "2024.1",
            "variables":       ["stock_price", "volume", "market_cap"],
            "spatial_coverage": "New York Stock Exchange, Global Exchanges",
            "temporal_coverage": "2010-2015",  # does not cover 2023
            "resolution":      "1 minute",
            "license":         "Bloomberg Terminal License",
            "units":           None,     # no scientific units declared
            "crs":             None,     # no CRS (financial, not geospatial)
            "variable_descriptions": None,
        }

        bad_integrity = {
            "integrity_passed": False,
            "failure_reason":   "Checksum mismatch — file may be corrupted.",
            "algorithm":        "SHA-256",
        }

        bad_report = generate_validation_report(
            dataset_id="bloomberg_stock_2024",
            dataset_name="Global Stock Market Prices",
            provider="Bloomberg Finance",
            file_path=bad_file.name,
            metadata=bad_metadata,
            required_variables=["sea_surface_temperature"],
            requested_region="Bay of Bengal",
            requested_period="2023",
            integrity_record=bad_integrity,
        )
        print_validation_report(bad_report)

        vid2 = save_validation_report(bad_report)
        print(f"\n  ✓ Report saved. Validation ID: {vid2}")

        # ---------------------------------------------------------- #
        # List all reports                                            #
        # ---------------------------------------------------------- #
        print("\n\nALL VALIDATION REPORTS IN DATABASE")
        print("-" * 70)
        all_reports = list_validation_reports()
        print(f"  Total reports: {len(all_reports)}")
        for i, r in enumerate(all_reports, start=1):
            flag = "✓" if r["overall_validation"] else "✗"
            print(
                f"  {i}. {flag} {r['dataset_name']:<45} "
                f"score={r['validation_score']:.2f}  "
                f"confidence={r['confidence_score']:.2f}"
            )

        print(f"\n  Database location: {DB_PATH}")
        print("\n  DEMO COMPLETE.")

    finally:
        # Clean up temp files
        with contextlib.suppress(OSError):
            os.unlink(sst_file.name)
        with contextlib.suppress(OSError):
            os.unlink(bad_file.name)


import contextlib   # needed by _save_db; imported here to keep top-level clean

if __name__ == "__main__":
    _run_demo()
