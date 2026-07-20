"""Navigation sidebar plus analysis controls."""

import streamlit as st

from dashboard.components.auth import logout_user
from dashboard.utils.session_manager import reset_session


EXAMPLE_QUERIES = [
    "Find chlorophyll concentration datasets around Lakshadweep between 2018 and 2024.",
    "Retrieve sea surface temperature data for the Bay of Bengal during monsoon 2022.",
    "Discover flood inundation datasets for Chennai after the 2015 event.",
    "Find NDVI satellite imagery for the Sundarbans mangrove region 2019-2023.",
    "Get wave height and ocean current datasets for the Arabian Sea 2020-2024.",
]

NAV_ITEMS = [
    ("Dashboard", "◇"),
    ("New Analysis", "+"),
    ("Voice Interface", "🎙"),
    ("Previous Analyses", "◷"),
    ("Pipeline", "▣"),
    ("Agent Outputs", "▤"),
    ("Security Reports", "◈"),
    ("Downloads", "⇩"),
    ("Settings", "⚙"),
    ("Help", "?"),
]


def _render_nav() -> None:
    st.markdown("<div class='nav-label'>Workspace</div>", unsafe_allow_html=True)
    current = st.session_state.get("active_page", "Dashboard")
    for label, icon in NAV_ITEMS:
        active = label == current
        button_label = f"{icon}  {label}"
        if st.button(button_label, key=f"nav_{label}", use_container_width=True, type="primary" if active else "secondary"):
            st.session_state.active_page = label
            st.rerun()


def render_sidebar() -> tuple[str, bool]:
    """Render navigation and analysis controls. Returns query and run click."""
    with st.sidebar:
        user = st.session_state.get("auth_user") or {}
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <div class="sidebar-logo">EI</div>
                <div>
                    <strong>Earth Intelligence</strong>
                    <span>Scientific AI Platform</span>
                </div>
            </div>
            <div class="sidebar-user">
                <div class="sidebar-avatar">{user.get('avatar', 'EI')}</div>
                <div>
                    <strong>{user.get('name', 'Research User')}</strong>
                    <span>{user.get('email', '')}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _render_nav()
        st.divider()

        st.markdown("**Research Query**")
        query = st.text_area(
            label="Query",
            label_visibility="collapsed",
            placeholder="Describe your scientific data need...",
            height=120,
            key="sidebar_query",
        )

        with st.expander("Example Queries", expanded=False):
            for i, example in enumerate(EXAMPLE_QUERIES):
                if st.button(example[:62] + "...", key=f"example_{i}", use_container_width=True):
                    st.session_state["sidebar_query"] = example
                    st.session_state.active_page = "New Analysis"
                    st.rerun()

        run_clicked = st.button(
            "Run Analysis",
            type="primary",
            use_container_width=True,
            disabled=(st.session_state.pipeline_stage not in ("idle", "done", "error")),
        )

        reset_col, logout_col = st.columns(2)
        with reset_col:
            if st.button("Reset", use_container_width=True):
                reset_session()
                st.rerun()
        with logout_col:
            if st.button("Logout", use_container_width=True):
                logout_user()
                st.rerun()

        st.divider()
        stage = st.session_state.pipeline_stage
        stage_labels = {
            "idle": "Idle",
            "running_a1": "Running Planner",
            "running_a2": "Clarifying",
            "clarifying": "Waiting for Input",
            "confirm_understanding": "Confirm Understanding",
            "running_a3": "Discovering Datasets",
            "agent3_ranking": "Ranking & Reviewing Sources",
            "running_a4": "Downloading Data",
            "running_a5": "Preprocessing Data",
            "done": "Complete",
            "error": "Error",
        }
        st.markdown(f"<div class='status-pill'>Status: {stage_labels.get(stage, stage.title())}</div>", unsafe_allow_html=True)

    return query, run_clicked
