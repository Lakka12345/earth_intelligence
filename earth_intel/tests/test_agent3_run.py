from agents.agent3_discovery import run_agent3

from models.retrieval_request import (
    RetrievalRequest,
    RetrievalVariable,
    RetrievalMeasurement,
    RetrievalDatasetRequirement,
)

request = RetrievalRequest(
    goal="Analyze coastal flood risk in Andhra Pradesh during cyclone events",

    user_intent_type="analysis",

    variables=[
        RetrievalVariable(
            variable="storm_surge",
            scientific_meaning="Increase in sea level caused by cyclones",
            relevance="Primary driver of coastal flooding",
            priority="high",
        )
    ],

    measurements=[
        RetrievalMeasurement(
            measurement_name="Storm Surge Height",
            variable_measured="storm_surge",
            why_measurement_is_needed="Needed for flood risk estimation",
            preferred_unit_or_representation="meters",
            required_resolution="hourly",
        )
    ],

    spatial_requirements={},
    temporal_requirements={},

    dataset_requirements=[
        RetrievalDatasetRequirement(
            requirement_name="Storm Surge Dataset",
            dataset_type="disaster",
            variables_or_measurements_needed=[
                "storm_surge"
            ],
            spatial_coverage_needed="Andhra Pradesh Coast",
            temporal_coverage_needed="Cyclone Events",
            metadata_needed=[],
            quality_constraints=[],
            why_required="Required for flood-risk assessment",
        )
    ],

    expected_scientific_outputs=[
        "Flood risk map"
    ],

    required_data_fusion=[],

    critical_constraints=[],

    retrieval_readiness="ready",

    user_approved_retrieval=True,
)

result = run_agent3(request)

print("\nAGENT 3 SUCCESS\n")

print(
    result.model_dump_json(
        indent=2
    )
)