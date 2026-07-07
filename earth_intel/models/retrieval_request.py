from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class RetrievalVariable(BaseModel):
    variable: str
    scientific_meaning: str
    relevance: str
    priority: str


class RetrievalMeasurement(BaseModel):
    measurement_name: str
    variable_measured: str
    why_measurement_is_needed: str
    preferred_unit_or_representation: str
    required_resolution: str


class RetrievalDatasetRequirement(BaseModel):
    requirement_name: str
    dataset_type: str
    variables_or_measurements_needed: List[str]
    spatial_coverage_needed: str
    temporal_coverage_needed: str
    metadata_needed: List[str]
    quality_constraints: List[str]
    why_required: str


class RetrievalRequest(BaseModel):
    goal: str
    user_intent_type: str

    variables: List[RetrievalVariable]
    measurements: List[RetrievalMeasurement]

    spatial_requirements: Dict[str, Any]
    temporal_requirements: Dict[str, Any]

    dataset_requirements: List[RetrievalDatasetRequirement]

    expected_scientific_outputs: List[str]
    required_data_fusion: List[str]
    critical_constraints: List[str]

    retrieval_readiness: str
    user_approved_retrieval: bool

    source_plan_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_retrieval_request(self):
        if not self.user_approved_retrieval:
            raise ValueError(
                "Cannot create RetrievalRequest unless user approved retrieval."
            )
        if not self.goal:
            raise ValueError("RetrievalRequest requires a scientific goal.")
        if not self.variables:
            raise ValueError("RetrievalRequest requires scientific variables.")
        if not self.measurements:
            raise ValueError("RetrievalRequest requires measurements.")
        if not self.dataset_requirements:
            raise ValueError("RetrievalRequest requires dataset requirements.")
        return self


def build_retrieval_request(
    approved_plan,
    user_approved_retrieval: bool,
) -> RetrievalRequest:
    """
    Converts an approved ScientificIntentOutput into a RetrievalRequest
    for Agent 3. Called after user approval gate passes.
    """
    return RetrievalRequest(
        goal=approved_plan.inferred_user_research_goal,
        user_intent_type=str(approved_plan.user_intent_type),
        variables=[
            RetrievalVariable(
                variable=item.variable,
                scientific_meaning=item.scientific_meaning,
                relevance=item.relevance,
                priority=str(item.priority),
            )
            for item in approved_plan.scientific_variables
        ],
        measurements=[
            RetrievalMeasurement(
                measurement_name=item.measurement_name,
                variable_measured=item.variable_measured,
                why_measurement_is_needed=item.why_measurement_is_needed,
                preferred_unit_or_representation=item.preferred_unit_or_representation,
                required_resolution=item.required_resolution,
            )
            for item in approved_plan.measurements
        ],
        spatial_requirements=approved_plan.spatial_context.model_dump(),
        temporal_requirements=approved_plan.temporal_context.model_dump(),
        dataset_requirements=[
            RetrievalDatasetRequirement(
                requirement_name=item.requirement_name,
                dataset_type=item.dataset_type,
                variables_or_measurements_needed=item.variables_or_measurements_needed,
                spatial_coverage_needed=item.spatial_coverage_needed,
                temporal_coverage_needed=item.temporal_coverage_needed,
                metadata_needed=item.metadata_needed,
                quality_constraints=item.quality_constraints,
                why_required=item.why_required,
            )
            for item in approved_plan.dataset_requirements
        ],
        expected_scientific_outputs=approved_plan.expected_scientific_outputs,
        required_data_fusion=approved_plan.required_data_fusion,
        critical_constraints=approved_plan.critical_constraints,
        retrieval_readiness=str(approved_plan.retrieval_readiness),
        user_approved_retrieval=user_approved_retrieval,
        source_plan_id="approved_scientific_plan_v1",
    )
