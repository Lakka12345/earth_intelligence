from security.hallucination_guard import (
    validate_agent2_consistency,
    HallucinationException,
)

from models.schemas import Assumption


class FakeOutput:

    retrieval_readiness = None

    critical_gaps = []

    clarification_needed = True

    prioritized_questions = []

    completeness_score = 0.5

    confidence_score = 0.5

    resolved_information = []

    remaining_gaps = []

    active_assumptions = [

        Assumption(
            assumption="Study area is Andhra Pradesh",
            reason="Inferred from context",
            confidence=0.95,
            risk_if_wrong="catastrophic",
        )

    ]


try:

    validate_agent2_consistency(
        FakeOutput()
    )

    print(
        "Rule #8 FAILED."
    )

except HallucinationException as e:

    print(
        "Rule #8 working."
    )

    print(e)