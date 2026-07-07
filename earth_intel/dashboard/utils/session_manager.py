"""
utils/session_manager.py
Initialises and resets all st.session_state keys used across the dashboard.
"""

import streamlit as st


def init_session_state() -> None:
    """Initialise every session key with a safe default if not already set."""
    defaults = {
        # Pipeline stage tracking
        "pipeline_stage": "idle",          # idle | running_a1 | running_a2 | clarifying | running_a3 | done | error
        "current_query": "",

        # Agent outputs (raw pydantic objects)
        "agent1_result": None,
        "agent2_result": None,
        "agent3_result": None,

        # Clarification state
        "clarification_round": 0,
        "clarification_history": [],       # list of ClarificationRound dicts
        "pending_user_responses": [],      # list of UserResponse dicts for current round

        # Timing
        "timings": {},                     # {"agent1": 2.3, ...}

        # Logs
        "logs": [],                        # list of {"ts": "HH:MM:SS", "msg": str}

        # Selected dataset for detail panel
        "selected_dataset_id": None,

        # Settings
        "max_datasets": 10,
        "enable_clarification": True,
        "show_reasoning": True,
        "dark_mode": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session() -> None:
    """Clear all pipeline state so the user can start a fresh query."""
    keys_to_reset = [
        "pipeline_stage",
        "current_query",
        "agent1_result",
        "agent2_result",
        "agent3_result",
        "clarification_round",
        "clarification_history",
        "pending_user_responses",
        "timings",
        "logs",
        "selected_dataset_id",
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    init_session_state()


def append_log(message: str) -> None:
    """Append a timestamped line to the execution log."""
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append({"ts": ts, "msg": message})
