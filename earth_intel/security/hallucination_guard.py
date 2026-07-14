from models.schemas import (
    ClarificationAgentOutput,
    Agent2RetrievalReadiness,
)


class HallucinationException(Exception):
    pass


def validate_agent2_consistency(
    output: ClarificationAgentOutput
):
       #Rule 1
    if (
        output.retrieval_readiness
        == Agent2RetrievalReadiness.ready
        and output.critical_gaps
    ):

        raise HallucinationException(
            "Plan cannot be ready while critical gaps remain."
        )
       #Rule 1b
    if (
        output.retrieval_readiness
        == Agent2RetrievalReadiness.ready
        and output.remaining_gaps
    ):

        raise HallucinationException(
            "Plan cannot be ready while remaining gaps exist."
        )
       #Rule 1c
    if (
        output.retrieval_readiness
        == Agent2RetrievalReadiness.ready
        and output.completeness_score != 1.0
    ):

        raise HallucinationException(
            "Ready plans must have completeness_score=1.0."
        )
       #Rule 2
    if (
        output.clarification_needed is False
        and output.prioritized_questions
    ):

        raise HallucinationException(
            "Clarification questions exist despite clarification_needed=False."
        )
        # Rule 2b
    if (
        output.clarification_needed is False
        and output.retrieval_readiness
        == Agent2RetrievalReadiness.clarification_required
    ):

        raise HallucinationException(
            "clarification_needed=False conflicts with "
            "retrieval_readiness=clarification_required."
        )
        # Rule 3

    if (
        output.completeness_score == 1.0
        and output.remaining_gaps
    ):

        raise HallucinationException(
            "Plan cannot be fully complete while gaps remain."
        )
        # Rule 4

    if (
        output.retrieval_readiness
        == Agent2RetrievalReadiness.clarification_required
        and not output.prioritized_questions
    ):

        raise HallucinationException(
            "Clarification-required plans must include questions."
        )
        # Rule 5

    if (
        output.confidence_score
        > output.completeness_score + 0.3
    ):

        raise HallucinationException(
            "Confidence score is disproportionately high "
            "relative to completeness score."
        )
        # Rule 6

    resolved_fields = {
        item.field_name.lower().strip()
        for item in output.resolved_information
    }

    unresolved_critical_gap_names = {
        gap.gap_name.lower().strip()
        for gap in output.remaining_gaps
        if (
            gap.severity.value == "critical"
            and gap.blocks_retrieval
        )
    }

    conflicts = (
        resolved_fields
        .intersection(
            unresolved_critical_gap_names
        )
    )

    if conflicts:

        raise HallucinationException(
            "Resolved information cannot also be "
            "an unresolved critical gap. "
            f"Conflicting fields: {sorted(conflicts)}"
        )
        # Rule 7

    if (
        len(output.active_assumptions)
        > len(output.resolved_information) + 3
    ):

        raise HallucinationException(
            "Too many assumptions relative "
            "to resolved information."
        )
        # Rule 8

    for assumption in output.active_assumptions:

        if (
            assumption.confidence >= 0.9
            and assumption.risk_if_wrong.lower()
            in [
                "high",
                "severe",
                "critical",
                "catastrophic",
            ]
        ):

            raise HallucinationException(
                "High-confidence assumption has "
                "dangerous consequences if wrong."
            )
