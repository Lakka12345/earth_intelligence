"""
components/clarification.py
Renders Agent 2 clarification questions and collects user answers.
Supports multiple rounds. Returns True when the user has submitted answers.
"""

import streamlit as st


def render_clarification_panel(agent2_result) -> bool:
    """
    Display all pending clarification questions from agent2_result.

    Returns True when the user clicks Submit (answers are stored in
    st.session_state.pending_user_responses ready for the next
    run_agent2 call). Returns False if nothing has been submitted yet.
    """
    questions = agent2_result.prioritized_questions
    if not questions:
        return False

    round_num = st.session_state.get("clarification_round", 1)

    st.markdown(
        f"""
        <div style='background:#fffbeb;border:1px solid #fde68a;border-radius:12px;
                    padding:16px 20px;margin-bottom:16px;'>
            <div style='font-size:15px;font-weight:700;color:#92400e;margin-bottom:4px;'>
                💬 Clarification Needed  <span style='font-size:12px;font-weight:400;'>(Round {round_num})</span>
            </div>
            <div style='font-size:13px;color:#78350f;'>
                Please answer the following questions to improve dataset discovery.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show previously answered rounds (read-only)
    history = st.session_state.get("clarification_history", [])
    if history:
        with st.expander(f"📜 Previous answers ({len(history)} round(s))", expanded=False):
            for past_round in history:
                rn = past_round.get("round_number", "?")
                st.markdown(f"**Round {rn}**")
                for q in past_round.get("questions_asked", []):
                    st.markdown(f"- *{q.get('question', '')}*")
                for r in past_round.get("responses_received", []):
                    st.markdown(f"  → {r.get('user_answer', '')}")

    # Render current questions
    answers: dict[str, str] = {}

    with st.form(key=f"clarification_form_round_{round_num}"):
        for idx, q in enumerate(questions, start=1):
            question_text = q.question
            options       = getattr(q, "options", None)
            priority      = getattr(q, "priority", "medium")
            rationale     = getattr(q, "rationale", "")

            priority_colours = {
                "critical": "#dc2626",
                "high":     "#ea580c",
                "medium":   "#ca8a04",
                "low":      "#64748b",
            }
            p_colour = priority_colours.get(str(priority).lower(), "#64748b")

            st.markdown(
                f"""
                <div style='margin-bottom:6px;'>
                    <span style='font-size:13px;font-weight:700;color:#1e293b;'>
                        {idx}. {question_text}
                    </span>
                    <span style='font-size:11px;color:{p_colour};margin-left:8px;
                                 text-transform:uppercase;font-weight:600;'>
                        {priority}
                    </span>
                </div>
                {f"<div style='font-size:12px;color:#64748b;margin-bottom:6px;'>{rationale}</div>" if rationale else ""}
                """,
                unsafe_allow_html=True,
            )

            field_key = f"clarification_q_{round_num}_{idx}"

            if options and isinstance(options, list) and len(options) > 0:
                answer = st.radio(
                    label=f"Answer {idx}",
                    options=options,
                    label_visibility="collapsed",
                    key=field_key,
                )
            else:
                answer = st.text_input(
                    label=f"Your answer {idx}",
                    label_visibility="collapsed",
                    placeholder="Type your answer here…",
                    key=field_key,
                )

            answers[f"question_{idx}"] = answer
            st.markdown("")

        submitted = st.form_submit_button(
            "✅  Submit Answers",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        # Store as list of dicts matching UserResponse schema
        responses = [
            {"field_name": field_key, "user_answer": ans}
            for field_key, ans in answers.items()
            if ans and str(ans).strip()
        ]
        st.session_state.pending_user_responses = responses
        return True

    return False
