import json

from enum import Enum
from typing import List

from pydantic import (
    BaseModel,
    Field,
    model_validator,
    field_validator,
)


class UserIntentType(str, Enum):
    monitoring = "Monitoring"
    forecasting_preparation = "Forecasting Preparation"
    risk_assessment = "Risk Assessment"
    impact_assessment = "Impact Assessment"
    trend_analysis = "Trend Analysis"
    change_detection = "Change Detection"
    resource_planning = "Resource Planning"
    environmental_assessment = "Environmental Assessment"
    disaster_assessment = "Disaster Assessment"
    scientific_research = "Scientific Research"
    exploratory_analysis = "Exploratory Analysis"
    unknown = "Unknown"


class AmbiguityLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class RetrievalReadiness(str, Enum):
    ready = "ready"
    proceed_with_assumptions = "proceed_with_assumptions"
    clarification_required = "clarification_required"
    partially_ready = "partially_ready"


class PriorityLevel(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class DomainName(str, Enum):
    oceanography = "Oceanography"
    meteorology = "Meteorology"
    climate_science = "Climate Science"
    hydrology = "Hydrology"
    gis = "GIS"
    remote_sensing = "Remote Sensing"
    fisheries = "Fisheries"
    coastal_processes = "Coastal Processes"
    disaster_management = "Disaster Management"
    environmental_monitoring = "Environmental Monitoring"
    urban_planning = "Urban Planning"
    civil_engineering = "Civil Engineering"
    agricultural_science = "Agricultural Science"


class EventType(str, Enum):
    cyclone = "Cyclone"
    flood = "Flood"
    storm_surge = "Storm Surge"
    heatwave = "Heatwave"
    drought = "Drought"
    landslide = "Landslide"
    tsunami = "Tsunami"
    unknown = "Unknown"


class ResearchQuestion(BaseModel):
    question: str = Field(min_length=5)
    importance: PriorityLevel


class DecisionContext(BaseModel):
    inferred_decision_context: str = Field(min_length=3)
    intended_use_case: str = Field(min_length=3)


class ScientificObjective(BaseModel):
    objective: str = Field(min_length=5)
    rationale: str = Field(min_length=5)
    priority: PriorityLevel


class ScientificVariable(BaseModel):
    variable: str = Field(min_length=2)
    scientific_meaning: str = Field(min_length=5)
    relevance: str = Field(min_length=5)
    priority: PriorityLevel


class VariablePriority(BaseModel):
    variable: str = Field(min_length=2)
    priority: PriorityLevel
    justification: str = Field(min_length=5)


class DomainConfidence(BaseModel):
    domain: DomainName
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=5)


class CrossDomainDependency(BaseModel):
    domains: List[DomainName] = Field(min_length=2)
    dependency_description: str = Field(min_length=5)
    why_it_mattered: str = Field(min_length=5)


class VariableDependency(BaseModel):
    variable: str = Field(min_length=2)
    depends_on: List[str] = Field(default_factory=list)
    reason: str = Field(min_length=5)


class PlanningStep(BaseModel):
    step_number: int = Field(ge=1)
    stage: str = Field(min_length=3)
    output: str = Field(min_length=3)


class UncertaintyItem(BaseModel):
    uncertainty: str = Field(min_length=3)
    impact_on_planning: str = Field(min_length=3)
    can_proceed: bool


class SpatialContext(BaseModel):
    location: str = Field(description="Location mentioned by user, or Unknown.")
    geographic_extent: str = Field(description="Study area extent, or Unknown.")
    study_boundary_type: str = Field(
        description="Coastal zone, district, river basin, offshore, estuary, etc."
    )
    coordinate_requirements: str = Field(
        description="Coordinates, bounding box, polygon, station points, admin boundary, or Unknown."
    )
    spatial_resolution_requirements: str = Field(
        description="Required spatial granularity, or Unknown."
    )
    spatial_context: str = Field(
        description="Coastal Zone, River Basin, Watershed, EEZ, District, State, Offshore, Estuary, Urban Area, or Unknown."
    )


class TemporalContext(BaseModel):
    date_range: str = Field(
        description="Explicit date range, inferred range, or Unknown."
    )
    event_window: str = Field(
        description="Before, during, after, recovery period, or Unknown."
    )
    historical_baseline: str = Field(
        description="Baseline period needed, or Unknown."
    )
    temporal_resolution: str = Field(
        description="Hourly, daily, weekly, monthly, seasonal, event-based, or Unknown."
    )
    temporal_analysis_type: str = Field(
        description="Before Event, During Event, After Event, Recovery Period, Historical Trend, Seasonal Analysis, or Unknown."
    )


class Measurement(BaseModel):
    measurement_name: str = Field(min_length=2)
    variable_measured: str = Field(min_length=2)
    why_measurement_is_needed: str = Field(min_length=5)
    preferred_unit_or_representation: str = Field(default="Unknown")
    required_resolution: str = Field(default="Unknown")


class DatasetRequirement(BaseModel):
    requirement_name: str = Field(min_length=3)
    dataset_type: str = Field(min_length=3)
    variables_or_measurements_needed: List[str] = Field(min_length=1)
    spatial_coverage_needed: str = Field(min_length=3)
    temporal_coverage_needed: str = Field(min_length=3)
    metadata_needed: List[str] = Field(min_length=1)
    quality_constraints: List[str] = Field(min_length=1)
    why_required: str = Field(min_length=5)


class ClarificationQuestion(BaseModel):
    question: str = Field(min_length=5)
    reason: str = Field(min_length=5)
    importance: PriorityLevel


class ReasoningSummary(BaseModel):
    goal_summary: str = Field(min_length=5)
    objective_summary: str = Field(min_length=5)
    variable_summary: str = Field(min_length=5)
    measurement_summary: str = Field(min_length=5)
    dataset_requirement_summary: str = Field(min_length=5)
    readiness_summary: str = Field(min_length=5)


FORBIDDEN_DATASET_NAMES = [
    "MODIS",
    "VIIRS",
    "Sentinel",
    "Landsat",
    "ERA5",
    "GFS",
    "ECMWF",
    "CMEMS",
    "Copernicus Marine",
    "HYCOM",
    "NOAA OISST",
    "CHIRPS",
    "IMERG",
    "TRMM",
    "GRACE",
    "SMAP",
    "SARAL",
    "Jason",
    "INSAT",
]


class ScientificIntentOutput(BaseModel):
    query: str = Field(min_length=3)

    inferred_user_research_goal: str = Field(min_length=5)

    user_intent_type: UserIntentType
    ambiguity_level: AmbiguityLevel
    retrieval_readiness: RetrievalReadiness

    hazard_type: EventType = Field(
        default=EventType.unknown,
        description=(
            "Primary hazard/event type identified from the query "
            "(Cyclone, Flood, Storm Surge, Heatwave, Drought, Landslide, "
            "Tsunami), or Unknown if the query does not state or clearly "
            "imply one."
        ),
    )
    hazard_mechanism_reasoning: str = Field(
        default="Unknown",
        description=(
            "Physical mechanism/subtype reasoning for the identified "
            "hazard_type (e.g. whether a flood is pluvial/urban-drainage, "
            "riverine/fluvial, coastal/storm-surge, or flash), including "
            "any ambiguity about the underlying mechanism that downstream "
            "agents should resolve. 'Unknown' if hazard_type is Unknown or "
            "the query gives no basis to reason about mechanism."
        ),
    )

    research_questions: List[ResearchQuestion] = Field(min_length=1)
    decision_context: DecisionContext

    scientific_objectives: List[ScientificObjective] = Field(min_length=1)
    scientific_variables: List[ScientificVariable] = Field(min_length=1)
    variable_priorities: List[VariablePriority] = Field(min_length=1)

    domain_confidences: List[DomainConfidence] = Field(min_length=1)
    cross_domain_dependencies: List[CrossDomainDependency] = Field(default_factory=list)
    variable_dependencies: List[VariableDependency] = Field(default_factory=list)

    spatial_context: SpatialContext
    temporal_context: TemporalContext

    measurements: List[Measurement] = Field(min_length=1)
    dataset_requirements: List[DatasetRequirement] = Field(min_length=1)

    planning_steps: List[PlanningStep] = Field(
        min_length=1,
        description=(
            "High-level OUTPUT steps summarising the analysis plan "
            "(typically 3-10 items). NOT the 23 internal reasoning stages "
            "the LLM executes silently before producing this JSON."
        ),
    )

    uncertainty_items: List[UncertaintyItem] = Field(default_factory=list)

    clarification_questions: List[str] = Field(default_factory=list)
    clarification_questions_structured: List[ClarificationQuestion] = Field(
        default_factory=list
    )

    hidden_assumptions_detected: List[str] = Field(default_factory=list)
    expected_scientific_outputs: List[str] = Field(min_length=1)
    required_data_fusion: List[str] = Field(default_factory=list)
    critical_constraints: List[str] = Field(default_factory=list)

    reasoning_summary: ReasoningSummary

    @field_validator("dataset_requirements")
    @classmethod
    def reject_specific_dataset_names(cls, value):
        blob = json.dumps(
            [item.model_dump() for item in value],
            ensure_ascii=False,
        ).lower()

        for name in FORBIDDEN_DATASET_NAMES:
            if name.lower() in blob:
                raise ValueError(
                    f"Specific dataset name detected: {name}. "
                    "Agent 1 may only produce dataset requirements, not dataset names."
                )

        return value

    @model_validator(mode="after")
    def validate_contract(self):

        step_numbers = [step.step_number for step in self.planning_steps]
        expected = list(range(1, len(step_numbers) + 1))

        if step_numbers != expected:
            raise ValueError(
                "Planning steps must be sequential starting from 1."
            )

        # NOTE: the stage-name ordering check against the 23 internal
        # reasoning stages has been removed. Those stages are the LLM's
        # silent chain-of-thought, not output fields; enforcing their names
        # and order on the JSON output was the root cause of the min_length=23
        # crash and would break valid plans that use different stage labels.

        # Cross-reference: every measurement must reference a known variable.
        variable_names = {
            v.variable.lower().strip()
            for v in self.scientific_variables
        }

        for measurement in self.measurements:

            if (
                measurement.variable_measured.lower().strip()
                not in variable_names
            ):
                raise ValueError(
                    f"Measurement '{measurement.measurement_name}' "
                    f"references unknown variable "
                    f"'{measurement.variable_measured}'."
                )

        # Cross-reference: every scientific variable must have a priority record.
        priority_variables = {
            v.variable.lower().strip()
            for v in self.variable_priorities
        }

        missing_priorities = variable_names - priority_variables

        if missing_priorities:
            raise ValueError(
                f"Variables missing priority records: "
                f"{sorted(missing_priorities)}"
            )

        return self


class GapSeverity(str, Enum):
    critical = "critical"
    non_critical = "non_critical"


class QuestionPriority(str, Enum):
    critical = "critical"
    important = "important"
    optional = "optional"


class ResolutionSource(str, Enum):
    user = "user"
    inferred = "inferred"
    inherited_from_agent1 = "inherited_from_agent1"


class Agent2RetrievalReadiness(str, Enum):
    ready = "ready"
    proceed_with_assumptions = "proceed_with_assumptions"
    clarification_required = "clarification_required"


class GapItem(BaseModel):
    gap_name: str = Field(min_length=2)
    description: str = Field(min_length=5)
    severity: GapSeverity
    blocks_retrieval: bool


class Agent2ClarificationQuestion(BaseModel):
    question: str = Field(min_length=5)
    reason: str = Field(min_length=5)
    priority: QuestionPriority
    resolves_gaps: List[str] = Field(default_factory=list)

    # Structured options, kept separate from the question text so the
    # UI (main.py) can render them as a numbered list deterministically
    # instead of relying on the LLM to embed "1. ... 2. ..." inside the
    # question string (which was the root cause of inconsistent
    # rendering). Empty list = genuinely open-ended question (rare;
    # the LLM is instructed to prefer options wherever possible).
    # When non-empty, the LAST entry must always be a free-text
    # catch-all (e.g. "None of the above - type your own answer").
    # This is enforced defensively in Python (see agent2.py) even if
    # the LLM forgets it.
    options: List[str] = Field(default_factory=list)

    # True only for the deterministic, Python-injected spatial/location
    # question (asks whether the user wants to give lat/lon directly or
    # name a place for the agent to resolve). Lets main.py branch to
    # specialized handling instead of generic numbered-option handling.
    is_location_question: bool = False

    # True only for the deterministic, Python-injected time-period
    # question. Selecting a category here (e.g. "a range of years")
    # is never treated as a final answer -- main.py always follows up
    # for the concrete date(s), the same way it does for coordinates
    # on the location question.
    is_time_period_question: bool = False

    # True only for the scope-narrowing question, asked when a study
    # area WAS stated but is too broad to retrieve sensibly on its own
    # (e.g. "Bay of Bengal", "the Indian coastline", "Arabian Sea").
    # This is distinct from is_location_question, which only fires when
    # no area was given at all.
    is_scope_question: bool = False

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower()

            mapping = {
                "high": "critical",
                "urgent": "critical",
                "required": "critical",
                "medium": "important",
                "moderate": "important",
                "normal": "important",
                "low": "optional",
                "minor": "optional",
            }

            return mapping.get(normalized, normalized)

        return value


class UserResponse(BaseModel):
    field_name: str = Field(min_length=2)
    user_answer: str = Field(min_length=1)


class Assumption(BaseModel):
    assumption: str = Field(min_length=5)
    reason: str = Field(min_length=5)
    confidence: float = Field(ge=0.0, le=1.0)
    risk_if_wrong: str = Field(min_length=5)


class ResolvedInformation(BaseModel):
    field_name: str = Field(min_length=2)
    resolved_value: str = Field(min_length=1)
    resolution_source: ResolutionSource


class ClarificationRound(BaseModel):
    round_number: int = Field(ge=1)
    questions_asked: List[Agent2ClarificationQuestion] = Field(
        default_factory=list
    )
    responses_received: List[UserResponse] = Field(
        default_factory=list
    )


class RefinedScientificPlan(BaseModel):
    agent1_plan_preserved: ScientificIntentOutput
    updated_missing_information: List[str] = Field(
        default_factory=list
    )
    resolved_information: List[ResolvedInformation] = Field(
        default_factory=list
    )
    remaining_gaps: List[GapItem] = Field(
        default_factory=list
    )
    active_assumptions: List[Assumption] = Field(
        default_factory=list
    )
    retrieval_readiness: Agent2RetrievalReadiness
    completeness_score: float = Field(
        ge=0.0,
        le=1.0
    )


class ClarificationAgentOutput(BaseModel):
    clarification_needed: bool

    critical_gaps: List[GapItem] = Field(default_factory=list)
    non_critical_gaps: List[GapItem] = Field(default_factory=list)

    prioritized_questions: List[
        Agent2ClarificationQuestion
    ] = Field(default_factory=list)

    active_assumptions: List[Assumption] = Field(
        default_factory=list
    )

    resolved_information: List[ResolvedInformation] = Field(
        default_factory=list
    )

    remaining_gaps: List[GapItem] = Field(
        default_factory=list
    )

    clarification_history: List[ClarificationRound] = Field(
        default_factory=list
    )

    refined_scientific_plan: RefinedScientificPlan

    retrieval_readiness: Agent2RetrievalReadiness

    completeness_score: float = Field(
        ge=0.0,
        le=1.0
    )

    confidence_score: float = Field(
        ge=0.0,
        le=1.0
    )

    @model_validator(mode="after")
    def validate_agent2_contract(self):

        if (
            self.retrieval_readiness
            != self.refined_scientific_plan.retrieval_readiness
        ):
            raise ValueError(
                "Top-level retrieval_readiness must match "
                "refined_scientific_plan.retrieval_readiness."
            )

        if (
            self.completeness_score
            != self.refined_scientific_plan.completeness_score
        ):
            raise ValueError(
                "Top-level completeness_score must match "
                "refined_scientific_plan.completeness_score."
            )

        if (
            self.retrieval_readiness
            == Agent2RetrievalReadiness.ready
            and self.critical_gaps
        ):
            raise ValueError(
                "Plan cannot be ready while critical gaps remain."
            )

        if (
            self.retrieval_readiness
            == Agent2RetrievalReadiness.clarification_required
            and not self.prioritized_questions
        ):
            raise ValueError(
                "Clarification-required plans must include prioritized questions."
            )

        if len(self.prioritized_questions) > 6:
            raise ValueError(
                "Agent 2 must ask no more than 6 questions."
            )

        resolved_fields = {
            item.field_name.lower().strip()
            for item in self.resolved_information
        }

        unresolved_critical_gap_names = {
            gap.gap_name.lower().strip()
            for gap in self.remaining_gaps
            if (
                gap.severity == GapSeverity.critical
                and gap.blocks_retrieval
            )
        }

        conflicts = resolved_fields.intersection(
            unresolved_critical_gap_names
        )

        if conflicts:
            raise ValueError(
                "Critical unresolved conflicts cannot appear in "
                "resolved_information. "
                f"Conflicting fields: {sorted(conflicts)}"
            )

        return self
