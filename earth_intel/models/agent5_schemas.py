"""
Pydantic schemas for the Agent 4 -> Agent 5 handoff, and for Agent 5's
own internal pipeline and final output.

NOTE ON SCOPE: Agent4Output / RetrievedDataset did not exist as a
separate contract before this file -- Agent 4 hasn't been fully wired
up to Agent 5 yet. These are defined here, reusing the same enums
Agent 3 already established (DownloadFormat, APIType) so the pipeline
speaks one consistent vocabulary end to end rather than Agent 5
inventing its own. If Agent 4's real output ends up shaped differently,
only this file (and agent5.py's ingestion step) should need to change.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from models.discovery_schemas import APIType, DownloadFormat


# ------------------------------------------------------------------ #
# Agent 4 -> Agent 5 handoff contract                                  #
# ------------------------------------------------------------------ #

class RetrievalStatus(str, Enum):
    success = "success"
    partial = "partial"       # e.g. resumed download, some files missing
    failed = "failed"


class RetrievedDataset(BaseModel):
    """
    One physically-downloaded dataset, as handed off by Agent 4.
    Mirrors CandidateSource's identity fields so Agent 5 can trace any
    dataset back to its Agent 3 discovery record without re-deriving
    anything Agent 3/4 already know.
    """
    source_id: str
    name: str
    provider_url: str

    # Where the actual bytes live on disk. A source can produce more
    # than one file (e.g. one NetCDF per day/tile).
    local_file_paths: List[str] = Field(min_length=1)
    file_format: DownloadFormat
    api_type: APIType = Field(default=APIType.unknown)

    variables_requested: List[str] = Field(default_factory=list)
    variables_retrieved: List[str] = Field(default_factory=list)

    spatial_coverage_retrieved: str = Field(default="Unknown")
    temporal_coverage_retrieved: str = Field(default="Unknown")

    retrieval_status: RetrievalStatus
    retrieval_notes: str = Field(default="")

    # Provider-native metadata Agent 4 already extracted while
    # downloading (units, CRS, fill values, etc.), if any -- Agent 5
    # should prefer this over re-deriving it from the file when present.
    provider_metadata: Dict[str, Any] = Field(default_factory=dict)


class Agent4Output(BaseModel):
    """
    Full handoff from Agent 4 to Agent 5. Only ever constructed when
    the user chose "Preprocess" (never "Raw Files") -- see
    agent5.py's own docstring for why Agent 5 never re-asks this.
    """
    retrieval_request_goal: str
    retrieved_datasets: List[RetrievedDataset]
    user_chose_preprocessing: bool = True

    @model_validator(mode="after")
    def validate_handoff(self):
        if not self.user_chose_preprocessing:
            raise ValueError(
                "Agent5 must never be constructed/invoked unless the "
                "user explicitly chose preprocessing over raw files."
            )
        if not self.retrieved_datasets:
            raise ValueError(
                "Agent4Output requires at least one retrieved dataset."
            )
        return self


# ------------------------------------------------------------------ #
# STEP 1 — Data profiling                                             #
# ------------------------------------------------------------------ #

class DatasetProfile(BaseModel):
    source_id: str
    file_format: DownloadFormat
    variables_found: List[str] = Field(default_factory=list)
    units_by_variable: Dict[str, str] = Field(default_factory=dict)
    dimensions: Dict[str, int] = Field(default_factory=dict)
    coordinate_reference_system: Optional[str] = None
    spatial_resolution: Optional[str] = None
    temporal_resolution: Optional[str] = None
    metadata_completeness_score: float = Field(ge=0.0, le=1.0, default=0.5)

    missing_value_fraction: float = Field(ge=0.0, le=1.0, default=0.0)
    duplicate_record_count: int = Field(default=0)
    invalid_value_count: int = Field(default=0)
    outlier_count: int = Field(default=0)
    time_gaps: List[str] = Field(default_factory=list)
    spatial_gaps: List[str] = Field(default_factory=list)

    profiling_notes: str = Field(default="")


# ------------------------------------------------------------------ #
# STEP 2 — Data validation                                            #
# ------------------------------------------------------------------ #

class ValidationSeverity(str, Enum):
    blocking = "blocking"        # objective cannot be met
    significant = "significant"  # handled automatically, but material
    minor = "minor"


class ValidationIssue(BaseModel):
    source_id: str
    issue: str
    severity: ValidationSeverity
    affected_variable: Optional[str] = None


class DatasetValidationResult(BaseModel):
    source_id: str
    required_variables_present: bool
    missing_required_variables: List[str] = Field(default_factory=list)
    temporal_coverage_sufficient: bool
    spatial_coverage_sufficient: bool
    unit_consistency_ok: bool
    coordinate_validity_ok: bool
    issues: List[ValidationIssue] = Field(default_factory=list)

    @property
    def has_blocking_issue(self) -> bool:
        return any(i.severity == ValidationSeverity.blocking for i in self.issues)


# ------------------------------------------------------------------ #
# STEP 3/4 — Preprocessing plan + execution                           #
# ------------------------------------------------------------------ #

class PreprocessingStepName(str, Enum):
    missing_value_handling = "missing_value_handling"
    duplicate_removal = "duplicate_removal"
    invalid_value_filtering = "invalid_value_filtering"
    unit_conversion = "unit_conversion"
    coordinate_normalization = "coordinate_normalization"
    crs_transformation = "crs_transformation"
    variable_standardization = "variable_standardization"
    time_alignment = "time_alignment"
    spatial_alignment = "spatial_alignment"
    resampling = "resampling"
    interpolation = "interpolation"
    dataset_merging = "dataset_merging"
    feature_extraction = "feature_extraction"
    derived_variable_computation = "derived_variable_computation"


class PlannedStep(BaseModel):
    step: PreprocessingStepName
    applies_to_source_ids: List[str]
    rationale: str = Field(min_length=3)
    method: str = Field(min_length=3)  # e.g. "linear interpolation", "CF-standard unit conversion via pint"


class PreprocessingPlan(BaseModel):
    """
    The dynamic pipeline for THIS request. Never a fixed template --
    built from what STEP 1/2 actually found plus the user's objective.
    """
    steps: List[PlannedStep] = Field(default_factory=list)
    plan_rationale: str = Field(default="")


class ExecutedStep(BaseModel):
    step: PreprocessingStepName
    applies_to_source_ids: List[str]
    method_used: str
    result_summary: str
    success: bool = True
    warning: Optional[str] = None


# ------------------------------------------------------------------ #
# STEP 6/7 — Analysis + quality assessment                            #
# ------------------------------------------------------------------ #

class AnalysisResult(BaseModel):
    analysis_type: str
    description: str
    output_variable_names: List[str] = Field(default_factory=list)
    output_file_path: Optional[str] = None
    summary_statistics: Dict[str, float] = Field(default_factory=dict)
    notes: str = Field(default="")


class QualityAssessment(BaseModel):
    overall_confidence: float = Field(ge=0.0, le=1.0)
    completeness_score: float = Field(ge=0.0, le=1.0)
    processing_quality_score: float = Field(ge=0.0, le=1.0)
    remaining_limitations: List[str] = Field(default_factory=list)
    reliability_notes: str = Field(default="")


# ------------------------------------------------------------------ #
# STEP 8 — Final output                                                #
# ------------------------------------------------------------------ #

class ProcessingLogEntry(BaseModel):
    stage: str
    detail: str


class Agent5Status(str, Enum):
    completed = "completed"
    completed_with_limitations = "completed_with_limitations"
    stopped_objective_unreachable = "stopped_objective_unreachable"


class Agent5Output(BaseModel):
    status: Agent5Status
    stop_reason: Optional[str] = None  # only set when status == stopped_objective_unreachable

    dataset_profiles: List[DatasetProfile] = Field(default_factory=list)
    validation_results: List[DatasetValidationResult] = Field(default_factory=list)
    preprocessing_plan: Optional[PreprocessingPlan] = None
    executed_steps: List[ExecutedStep] = Field(default_factory=list)

    clean_dataset_paths: List[str] = Field(default_factory=list)
    analysis_results: List[AnalysisResult] = Field(default_factory=list)
    quality_assessment: Optional[QualityAssessment] = None
    processing_log: List[ProcessingLogEntry] = Field(default_factory=list)
