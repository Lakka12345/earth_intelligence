from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict


class EarthGraphState(TypedDict, total=False):

    # ------------------------------------------------------------------ #
    # User input                                                           #
    # ------------------------------------------------------------------ #
    user_query: str

    # ------------------------------------------------------------------ #
    # Agent 1 — Scientific planning                                        #
    # ------------------------------------------------------------------ #
    agent1_output: Any                  # ScientificIntentOutput

    # ------------------------------------------------------------------ #
    # Agent 2 — Clarification                                              #
    # ------------------------------------------------------------------ #
    agent2_output: Any                  # ClarificationAgentOutput
    clarification_answers: List[Dict]
    revision_history: List[Dict]
    user_feedback: Optional[str]

    # ------------------------------------------------------------------ #
    # Approval gate                                                        #
    # ------------------------------------------------------------------ #
    approved_scientific_plan: Any       # ScientificIntentOutput (approved)
    user_approved_retrieval: bool
    revision_cycle: int
    max_revision_cycles: int

    # ------------------------------------------------------------------ #
    # Agent 3 input contract                                               #
    # ------------------------------------------------------------------ #
    retrieval_request: Any              # RetrievalRequest

    # ------------------------------------------------------------------ #
    # Agent 3 — Discovery                                                  #
    # ------------------------------------------------------------------ #
    agent3_output: Any                  # DiscoveryOutput

    # Ranked sources after scoring — list of ScoredSource summary dicts
    ranked_sources: List[Dict]

    # Sources the user confirmed for download
    sources_to_download: List[str]      # list of source_ids

    # Whether user requested a download at all
    download_requested: bool

    # ------------------------------------------------------------------ #
    # NEW — Access classification summary passed to Agent 4               #
    # ------------------------------------------------------------------ #
    # Subset of sources_to_download that are free (no auth needed)
    free_sources_to_retrieve: List[str]

    # Subset requiring registration/login — Agent 4 will ask for credentials
    login_required_sources: List[str]

    # Subset requiring payment — Agent 4 MUST show payment gate to user
    payment_required_sources: List[str]

    # ------------------------------------------------------------------ #
    # Graph control                                                        #
    # ------------------------------------------------------------------ #
    agent3_input_ready: bool
    current_stage: str
