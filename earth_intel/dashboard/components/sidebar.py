"""
components/sidebar.py
Renders the left sidebar: logo, query input, example queries, settings,
Run and Reset buttons.
"""

import streamlit as st
from utils.session_manager import reset_session

EXAMPLE_QUERIES = [
    "Find chlorophyll concentration datasets around Lakshadweep between 2018 and 2024.",
    "Retrieve sea surface temperature data for the Bay of Bengal during monsoon 2022.",
    "Discover flood inundation datasets for Chennai after the 2015 event.",
    "Find NDVI satellite imagery for the Sundarbans mangrove region 2019-2023.",
    "Get wave height and ocean current datasets for the Arabian Sea 2020-2024.",
]


def render_sidebar() -> tuple[str, bool]:
    """
    Render the full sidebar.

    Returns
    -------
    query : str
        The query the user has typed (may be empty).
    run_clicked : bool
        True on the frame the Run Analysis button is pressed.
    """
    with st.sidebar:
        # ── Logo / branding ──────────────────────────────────────────── #
        st.markdown(
            """
            <div style='text-align:center; padding: 16px 0 8px 0;'>
                <div style='font-size:32px;'>🌍</div>
                <div style='font-size:16px; font-weight:800; color:#1e293b;
                            letter-spacing:0.04em; margin-top:4px;'>
                    Earth Intelligence
                </div>
                <div style='font-size:11px; color:#94a3b8; margin-top:2px;'>
                    Scientific Dataset Discovery
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Research query ───────────────────────────────────────────── #
        st.markdown("**🔍 Research Query**")
        query = st.text_area(
            label="Query",
            label_visibility="collapsed",
            placeholder="Describe your scientific data need...",
            height=110,
            key="sidebar_query",
        )

        # ── Example queries ──────────────────────────────────────────── #
        with st.expander("💡 Example Queries", expanded=False):
            for i, example in enumerate(EXAMPLE_QUERIES):
                if st.button(
                    f"{'📍' if i == 0 else '📌'} {example[:55]}...",
                    key=f"example_{i}",
                    use_container_width=True,
                ):
                    st.session_state["sidebar_query"] = example
                    st.rerun()

        st.markdown("")

        # ── Action buttons ───────────────────────────────────────────── #
        run_clicked = st.button(
            "🚀  Run Analysis",
            type="primary",
            use_container_width=True,
            disabled=(st.session_state.pipeline_stage not in ("idle", "done", "error")),
        )

        if st.button("🔄  Reset", use_container_width=True):
            reset_session()
            st.rerun()

        st.divider()

        # ── Settings ─────────────────────────────────────────────────── #
        st.markdown("**⚙️ Settings**")

        st.session_state.max_datasets = st.slider(
            "Maximum Datasets",
            min_value=3,
            max_value=25,
            value=st.session_state.get("max_datasets", 10),
            step=1,
        )

        st.session_state.enable_clarification = st.toggle(
            "Enable Clarification",
            value=st.session_state.get("enable_clarification", True),
        )

        st.session_state.show_reasoning = st.toggle(
            "Show Reasoning",
            value=st.session_state.get("show_reasoning", True),
        )

        st.session_state.dark_mode = st.toggle(
            "Dark Mode (coming soon)",
            value=False,
            disabled=True,
        )

        st.divider()

        # ── Pipeline status summary ───────────────────────────────────── #
        stage = st.session_state.pipeline_stage
        stage_labels = {
            "idle":        ("⬜", "Idle"),
            "running_a1":  ("🔵", "Running Agent 1"),
            "running_a2":  ("🔵", "Running Agent 2"),
            "clarifying":  ("🟡", "Waiting for Input"),
            "running_a3":  ("🔵", "Running Agent 3"),
            "done":        ("🟢", "Complete"),
            "error":       ("🔴", "Error"),
        }
        icon, label = stage_labels.get(stage, ("⬜", stage.title()))
        st.markdown(f"**Status:** {icon} {label}")

    return query, run_clicked
