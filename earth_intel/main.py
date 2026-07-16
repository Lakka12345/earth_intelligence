from agents.agent1 import run_agent1
from agents.agent2 import run_agent2, is_broad_location_text
from agents.agent3_discovery import run_agent3
from agents.agent3_interactive_runner import run_agent3_interactive
from agents.agent4_orchestrator import run_agent4

from security.input_validator import validate_input
from security.injection_guard import detect_prompt_injection

# SECURITY INTEGRATION — central imports for all security modules consumed
# by main.py's final risk-assessment gate.  Modules used only inside Agent 3
# or Agent 4 are imported locally at their call sites to keep dependency
# surface minimal here.
from security.provider_trust import generate_provider_trust_report
from security.security_risk_assessment import (
    assess_security_risk,
    save_security_assessment,
    RecommendedAction,
)

from models.retrieval_request import (
    build_retrieval_request,
)

from models.plan_reconciliation import reconcile_agent1_plan

from models.schemas import UserResponse, ClarificationRound

from formatting.understanding_summary import render_understanding_summary_banner


MAX_CLARIFICATION_ROUNDS = 3
MAX_REVISION_ROUNDS = 5


def _ask_single_question(question, round_number, idx):
    """
    Renders ONE Agent2ClarificationQuestion as a numbered list (when it
    has options) and collects the user's answer, resolving a numeric
    choice back to the actual option text. The free-text catch-all
    ("None of the above - type your own answer") is always the last
    numbered option and, when chosen, prompts for a follow-up free-text
    line instead of accepting the literal catch-all text as the answer.

    Falls back to a plain free-text prompt when the question has no
    options at all (the rare, genuinely open-ended case).

    Returns the resolved answer string.
    """
    options = list(getattr(question, "options", None) or [])

    print(f"\nQuestion {idx}: {question.question}")

    if not options:
        # No options -- genuinely open-ended question.
        return input("\nYour answer: ").strip()

    for opt_idx, option_text in enumerate(options, start=1):
        print(f"  {opt_idx}. {option_text}")

    catchall_index = len(options)  # last option, 1-indexed
    is_catchall_last = "none of the above" in options[-1].lower()

    while True:
        raw = input("\nYour choice (enter a number): ").strip()

        if not raw:
            print("Please enter a number or your answer.")
            continue

        if raw.isdigit():
            choice_num = int(raw)

            if 1 <= choice_num <= len(options):
                if is_catchall_last and choice_num == catchall_index:
                    follow_up = input("\nPlease type your answer: ").strip()
                    if follow_up:
                        return follow_up
                    print("Please type an answer.")
                    continue

                return options[choice_num - 1]

            print(f"Please enter a number between 1 and {len(options)}.")
            continue

        # User typed free text directly instead of a number -- accept
        # it as-is rather than forcing them to pick the catch-all first.
        return raw


def _maybe_narrow_broad_place(place: str) -> str:
    """
    Catches broad regions (an ocean, sea, gulf, or whole country) the
    instant the user TYPES them as an answer -- e.g. picking "I'll type
    the name of the place" and then typing "Arabian Sea".

    This has to happen here, synchronously, in the same turn: that
    typed answer never passes back through Agent 1, and the
    clarification loop is hard-capped at one round, so there is no
    later LLM round in which Agent 2 could catch it and ask again. If
    we didn't check it here, a broad region typed as an ANSWER would
    silently slip through even though a broad region stated in the
    ORIGINAL query gets caught (see _ensure_scope_question_present in
    agent2.py). Uses the same marker list as that check, via
    is_broad_location_text, so both paths stay consistent.

    Returns a final, already-narrowed answer string -- never the bare
    broad-region name on its own.
    """
    if not is_broad_location_text(place):
        return place

    print(f"\n\"{place}\" is a very large area. Pulling data for all of "
          "it could be far more than you actually need.")
    print(f"  1. The entire {place}")
    print("  2. Just a specific country's coastline/waters within it")
    print("  3. A specific city, district, or shorter stretch")

    while True:
        raw = input("\nYour choice (enter a number): ").strip()

        if raw == "1":
            return f"{place} (entire area)"

        if raw == "2":
            sub = input("  Which country's coastline/waters? ").strip()
            if sub:
                return f"{place} - {sub}"
            print("Please enter a country name.")
            continue

        if raw == "3":
            sub = input("  Which city, district, or stretch? ").strip()
            if sub:
                return f"{place} - {sub}"
            print("Please enter a place name.")
            continue

        if raw:
            # Free text typed directly -- may itself still be broad
            # (e.g. they typed "Indian Ocean" instead of picking 1-3),
            # so recurse to keep narrowing until it isn't.
            return _maybe_narrow_broad_place(raw)

        print("Please enter a number or your answer.")


def _ask_location_question(question):
    """
    Specialized handling for the is_location_question=True case: the
    two-path lat/lon-vs-place-name pattern. Renders the same numbered
    list as _ask_single_question, but branches into a dedicated
    coordinate prompt when the user picks the lat/lon path, so the
    answer we send back to Agent 2 is already in a clean, parseable
    form. Every path that can yield a typed place name is passed
    through _maybe_narrow_broad_place so a broad region (a sea, ocean,
    or whole country) never gets accepted as-is.
    """
    options = list(getattr(question, "options", None) or [])

    print(f"\nQuestion: {question.question}")

    for opt_idx, option_text in enumerate(options, start=1):
        print(f"  {opt_idx}. {option_text}")

    while True:
        raw = input("\nYour choice (enter a number): ").strip()

        if raw == "1" or (options and raw.strip() == options[0]):
            lat = input("  Latitude: ").strip()
            lon = input("  Longitude: ").strip()
            return f"{lat}, {lon}"

        if raw == "2" or (len(options) > 1 and raw.strip() == options[1]):
            place = input("  Place name / region / coastline: ").strip()
            return _maybe_narrow_broad_place(place)

        if raw == "3" or (options and raw.strip() == options[-1]):
            follow_up = input("\nPlease type your answer: ").strip()
            if follow_up:
                return _maybe_narrow_broad_place(follow_up)
            print("Please type an answer.")
            continue

        if raw:
            # Free text typed directly.
            return _maybe_narrow_broad_place(raw)

        print("Please enter a number or your answer.")


def _ask_time_period_question(question):
    """
    Specialized handling for the is_time_period_question=True case.
    Never lets a category answer ("a range of years") stand in as the
    final answer -- every path below ends in a follow-up prompt that
    collects the actual concrete date(s), so what gets sent back to
    Agent 2 is always something like "2023-06-01 to 2023-09-30", not
    "A specific date range".
    """
    options = list(getattr(question, "options", None) or [])

    print(f"\nQuestion: {question.question}")

    for opt_idx, option_text in enumerate(options, start=1):
        print(f"  {opt_idx}. {option_text}")

    while True:
        raw = input("\nYour choice (enter a number): ").strip()

        # Option 1: specific date range -> ask for start and end.
        if raw == "1" or (options and raw.strip() == options[0]):
            start = input("  Start date (e.g. 2023-10-01, or just a month/year): ").strip()
            end = input("  End date (e.g. 2023-12-31, or just a month/year): ").strip()
            return f"{start} to {end}"

        # Option 2: single year or season -> ask which one.
        if raw == "2" or (len(options) > 1 and raw.strip() == options[1]):
            value = input("  Which year or season? ").strip()
            if value:
                return value
            print("Please enter a year or season.")
            continue

        # Option 3: named event -> ask which event.
        if raw == "3" or (len(options) > 2 and raw.strip() == options[2]):
            value = input("  Which event? (e.g. Cyclone Michaung, 2023 Chennai floods) ").strip()
            if value:
                return value
            print("Please enter an event name.")
            continue

        # Option 4 / catch-all: free text.
        if raw == "4" or (options and raw.strip() == options[-1]):
            follow_up = input("\nPlease type your answer: ").strip()
            if follow_up:
                return follow_up
            print("Please type an answer.")
            continue

        if raw:
            # Free text typed directly instead of a number.
            return raw

        print("Please enter a number or your answer.")


def _ask_scope_question(question):
    """
    Specialized handling for the is_scope_question=True case (a broad
    area like "Bay of Bengal" was named but not yet scoped). Picking
    "the entire area" is accepted as final since it's already a
    complete answer; picking either narrowing option always follows up
    for the actual name, so the answer sent back to Agent 2 is never
    just a vague "a specific part of it".
    """
    options = list(getattr(question, "options", None) or [])

    print(f"\nQuestion: {question.question}")

    for opt_idx, option_text in enumerate(options, start=1):
        print(f"  {opt_idx}. {option_text}")

    while True:
        raw = input("\nYour choice (enter a number): ").strip()

        # Option 1: the entire broad area -- already a complete answer.
        if raw == "1" or (options and raw.strip() == options[0]):
            return options[0]

        # Option 2: a specific country's coastline/waters within it.
        if raw == "2" or (len(options) > 1 and raw.strip() == options[1]):
            value = input("  Which country's coastline/waters? ").strip()
            if value:
                return value
            print("Please enter a country name.")
            continue

        # Option 3: a specific city/district/shorter stretch.
        if raw == "3" or (len(options) > 2 and raw.strip() == options[2]):
            value = input("  Which city, district, or stretch? ").strip()
            if value:
                return value
            print("Please enter a place name.")
            continue

        # Option 4 / catch-all: free text.
        if raw == "4" or (options and raw.strip() == options[-1]):
            follow_up = input("\nPlease type your answer: ").strip()
            if follow_up:
                return follow_up
            print("Please type an answer.")
            continue

        if raw:
            return raw

        print("Please enter a number or your answer.")


def _ask_clarification_questions(agent2_result, round_number):
    """
    Displays facts and assumptions from Agent 2, then prompts the user
    for each clarification question -- rendering structured options as
    a numbered list (with the free-text catch-all always available) and
    routing the location question to its specialized handler. Returns
    user_responses list.

    Facts: read from user_facing_facts if present (extra LLM field),
           otherwise built from resolved_information with source=inherited_from_agent1.
    Assumptions: read from active_assumptions (always in schema).
    """
    # ── Display Facts ────────────────────────────────────────────────
    # Try schema-stripped extra field first, fall back to resolved_info
    facts = getattr(agent2_result, "user_facing_facts", None) or []
    if not facts and agent2_result.resolved_information:
        facts = [
            f"{r.field_name}: {r.resolved_value}"
            for r in agent2_result.resolved_information
            if r.resolution_source.value in ("user", "inherited_from_agent1")
        ]

    if facts:
        print("\n" + "─" * 50)
        print("WHAT WE KNOW  (from your query)")
        print("─" * 50)
        for fact in facts:
            print(f"  • {fact}")

    # ── Display Assumptions ───────────────────────────────────────────
    # active_assumptions is always in the schema
    assumptions = agent2_result.active_assumptions or []
    if assumptions:
        print("\nASSUMPTIONS  (automatically inferred — you can correct these)")
        print("─" * 50)
        for a in assumptions:
            print(f"  • {a.assumption}")

    if facts or assumptions:
        print("─" * 50)

    # ── Ask clarification questions ───────────────────────────────────
    user_responses = []

    for idx, question in enumerate(
        agent2_result.prioritized_questions,
        start=1,
    ):
        if getattr(question, "is_location_question", False):
            answer = _ask_location_question(question)
        elif getattr(question, "is_scope_question", False):
            answer = _ask_scope_question(question)
        elif getattr(question, "is_time_period_question", False):
            answer = _ask_time_period_question(question)
        else:
            answer = _ask_single_question(question, round_number, idx)

        user_responses.append(
            UserResponse(
                field_name=f"round_{round_number}_question_{idx}",
                user_answer=answer,
            )
        )

    return user_responses


def _critical_blocking_gaps(agent2_result):
    """
    Returns only the remaining gaps that are BOTH severity=critical AND
    blocks_retrieval=True -- the ones that actually justify stopping the
    loop or synthesizing an assumption. (Agent 2 can carry non-critical
    or non-blocking gaps too; those never trigger this path.)
    """
    return [
        g for g in agent2_result.remaining_gaps
        if g.severity.value == "critical" and g.blocks_retrieval
    ]


def _synthesize_assumption_dict(gap) -> dict:
    """
    Deterministic, Python-owned fallback Assumption for a critical gap
    that survived all MAX_CLARIFICATION_ROUNDS unresolved. Only used
    once the hard round cap has actually been hit AND the user has
    chosen (or defaulted, via the safety net) to proceed anyway --
    never invented silently. Confidence is deliberately low, since this
    was never actually resolved with the user.
    """
    return {
        "assumption": (
            f'Proceeding with a default interpretation for '
            f'"{gap.gap_name}": {gap.description}'
        ),
        "reason": (
            "This could not be resolved after "
            f"{MAX_CLARIFICATION_ROUNDS} rounds of clarification, so a "
            "reasonable default is being used so retrieval can proceed."
        ),
        "confidence": 0.4,
        "risk_if_wrong": (
            "Retrieved datasets may not match what was actually "
            "intended for this part of the request -- review the "
            "results with that in mind."
        ),
    }


def _force_proceed_with_assumptions(agent2_result, gaps_to_assume):
    """
    Patches agent2_result so retrieval can proceed: synthesizes an
    Assumption for every gap in gaps_to_assume (skipping any whose
    assumption text is already present, so this is safe to call more
    than once), clears prioritized_questions, and sets
    retrieval_readiness to proceed_with_assumptions at both the top
    level and inside refined_scientific_plan (the schema requires the
    two to match).
    """
    patched = agent2_result.model_dump()

    existing_text = {
        a["assumption"].lower()
        for a in patched.get("active_assumptions", [])
    }

    for gap in gaps_to_assume:
        candidate = _synthesize_assumption_dict(gap)
        if candidate["assumption"].lower() not in existing_text:
            patched["active_assumptions"].append(candidate)
            patched["refined_scientific_plan"]["active_assumptions"].append(
                candidate
            )
            existing_text.add(candidate["assumption"].lower())

    patched["retrieval_readiness"] = "proceed_with_assumptions"
    patched["refined_scientific_plan"]["retrieval_readiness"] = (
        "proceed_with_assumptions"
    )
    patched["prioritized_questions"] = []

    from models.schemas import ClarificationAgentOutput
    return ClarificationAgentOutput.model_validate(patched)


def _handle_unresolved_critical_gaps(
    agent1_result, agent2_result, clarification_history
):
    """
    Called when the hard MAX_CLARIFICATION_ROUNDS cap has been reached
    and critical, retrieval-blocking gaps still remain -- i.e. the cap
    (a safety limit) fired before the real stopping condition ("no
    critical gaps remain") was met.

    Rather than silently forcing proceed_with_assumptions, this:
      1. Tells the user exactly which critical information is still
         missing.
      2. States plainly how that will affect retrieval quality.
      3. Lets the user choose to proceed with assumptions, or provide
         the missing information now (one more explicit round, outside
         the automatic cap since it's now an informed choice).
    """
    critical_gaps = _critical_blocking_gaps(agent2_result)

    if not critical_gaps:
        # Nothing critical actually remains (e.g. only non-critical
        # gaps were left) -- nothing to explain, just finalize.
        return _force_proceed_with_assumptions(agent2_result, [])

    print(
        f"\nAfter {MAX_CLARIFICATION_ROUNDS} rounds of clarification, "
        "the following information is still missing:"
    )
    for g in critical_gaps:
        print(f"  - {g.gap_name}: {g.description}")

    print(
        "\nWithout this, dataset discovery will have to guess at these "
        "details, which may return datasets that don't fully match "
        "what you actually need."
    )

    print("\nWould you like to:")
    print("  1. Continue anyway, using reasonable assumptions for the above")
    print("  2. Provide the missing information now")

    choice = input("\n> ").strip()

    if choice != "2":
        return _force_proceed_with_assumptions(agent2_result, critical_gaps)

    round_number = len(clarification_history) + 1

    user_responses = _ask_clarification_questions(
        agent2_result, round_number
    )

    clarification_history.append(
        ClarificationRound(
            round_number=round_number,
            questions_asked=agent2_result.prioritized_questions,
            responses_received=user_responses,
        )
    )

    print(f"\nRunning Agent 2 (Clarification Round {round_number})...\n")

    agent2_result = run_agent2(
        agent1_output=agent1_result,
        user_responses=user_responses,
        clarification_history=clarification_history,
        round_number=round_number,
    )

    if agent2_result.retrieval_readiness == "clarification_required":
        # Don't loop forever -- finalize with assumptions for whatever
        # critical gaps are still open after this one extra round.
        agent2_result = _force_proceed_with_assumptions(
            agent2_result, _critical_blocking_gaps(agent2_result)
        )

    return agent2_result


def _run_clarification_loop(agent1_result):
    """
    Repeats Agent 2 calls while retrieval_readiness == clarification_required,
    threading clarification_history across every round. Hard cap = 3 rounds.

    - Passes round_number so the LLM knows when it must stop asking.
    - If round 3 is reached and only non-critical gaps remain, the LLM is
      instructed (via prompt) to set proceed_with_assumptions; Python also
      enforces this as a fallback below.
    - Invalid-answer re-asks (Case C) are handled by the LLM within the same
      round and do not increment round_number here.
    """
    clarification_history = []

    print("Running Agent 2...\n")

    agent2_result = run_agent2(
        agent1_output=agent1_result,
        round_number=1,
    )

    print("Agent 2 completed.\n")

    round_number = 1

    while (
        agent2_result.retrieval_readiness == "clarification_required"
        and round_number <= MAX_CLARIFICATION_ROUNDS
    ):

        print(
            f"\nClarification required before retrieval "
            f"(round {round_number}/{MAX_CLARIFICATION_ROUNDS}).\n"
        )

        user_responses = _ask_clarification_questions(
            agent2_result, round_number
        )

        clarification_history.append(
            ClarificationRound(
                round_number=round_number,
                questions_asked=agent2_result.prioritized_questions,
                responses_received=user_responses,
            )
        )

        next_round = round_number + 1

        print(
            f"\nRunning Agent 2 (Clarification Round {next_round})...\n"
        )

        agent2_result = run_agent2(
            agent1_output=agent1_result,
            user_responses=user_responses,
            clarification_history=clarification_history,
            round_number=next_round,
        )

        print(
            f"\nUpdated Agent 2 Output (Clarification Round {next_round}):\n"
        )

        round_number = next_round

        # Primary stopping condition is "no critical gaps remain" --
        # MAX_CLARIFICATION_ROUNDS is only a safety cap. If the cap is
        # hit while critical gaps are still open, hand off to the
        # explain-and-choose flow instead of silently forcing a result.
        if round_number > MAX_CLARIFICATION_ROUNDS:
            if agent2_result.retrieval_readiness == "clarification_required":
                agent2_result = _handle_unresolved_critical_gaps(
                    agent1_result, agent2_result, clarification_history
                )
            break

    return agent2_result, clarification_history


def _continue_clarification_after_revision(
    agent1_result, agent2_result, clarification_history
):
    """
    Re-enters a bounded clarification loop if a revision round causes
    Agent 2 to report clarification_required again (e.g. the user's
    requested change introduced new ambiguity). Reuses the same
    question/answer mechanism and history threading as the initial
    clarification loop.
    """
    round_number = len(clarification_history) + 1
    max_round = len(clarification_history) + MAX_CLARIFICATION_ROUNDS

    while (
        agent2_result.retrieval_readiness == "clarification_required"
        and round_number <= max_round
    ):
        print(
            f"\nClarification required after revision "
            f"(round {round_number}).\n"
        )

        user_responses = _ask_clarification_questions(
            agent2_result, round_number
        )

        clarification_history.append(
            ClarificationRound(
                round_number=round_number,
                questions_asked=agent2_result.prioritized_questions,
                responses_received=user_responses,
            )
        )

        print(
            f"\nRunning Agent 2 (Clarification Round {round_number})...\n"
        )

        agent2_result = run_agent2(
            agent1_output=agent1_result,
            user_responses=user_responses,
            clarification_history=clarification_history,
            round_number=round_number,
        )

        round_number += 1

    return agent2_result, clarification_history


def _run_confirmation_and_revision_loop(
    agent1_result,
    agent2_result,
    clarification_history,
):
    """
    Implements steps 3-5 of the objective:
      - Render the Scientific Understanding summary (pure formatter).
      - Ask for confirmation: continue to discovery, or modify.
      - If modify: ask what to change, feed it back through Agent 2 as
        a UserResponse (per the approved design -- no separate Python
        merge mechanism; Agent 2's own LLM call performs the merge,
        gap re-detection, assumption updates, and readiness decision),
        regenerate the summary, and ask again.
      - Up to MAX_REVISION_ROUNDS modify rounds.

    Returns the final, user-approved agent2_result.
    """
    revision_round = 0

    while True:

        print(
            "\n"
            + render_understanding_summary_banner(
                agent2_result,
                revision_round=revision_round if revision_round else None,
            )
        )

        print(
            "\nIs this understanding correct and sufficient to begin "
            "dataset discovery?"
        )
        print("  1. Continue to Discovery")
        print("  2. Modify the understanding")

        choice = input("\n> ").strip()

        if choice == "1":
            return agent2_result

        if choice != "2":
            print(
                "\nPlease enter 1 to continue or 2 to modify the understanding."
            )
            continue

        if revision_round >= MAX_REVISION_ROUNDS:
            print(
                f"\nMaximum of {MAX_REVISION_ROUNDS} revision rounds reached. "
                "Proceeding with the current understanding."
            )
            return agent2_result

        revision_round += 1

        change_request = input(
            "\nWhat would you like to change? "
            "(Goal, Objectives, Variables, Measurements, Location, "
            "Time Period, Dataset Requirements, Assumptions, or any "
            "other scientific detail)\n> "
        )

        user_responses = [
            UserResponse(
                field_name=f"revision_{revision_round}_modification",
                user_answer=change_request,
            )
        ]

        # Thread the revision into clarification_history too, so later
        # rounds (clarification or revision) see the full sequence of
        # questions/answers AND user-requested modifications.
        clarification_history.append(
            ClarificationRound(
                round_number=len(clarification_history) + 1,
                questions_asked=[],
                responses_received=user_responses,
            )
        )

        print(
            f"\nRunning Agent 2 (Revision Round {revision_round})...\n"
        )

        agent2_result = run_agent2(
            agent1_output=agent1_result,
            user_responses=user_responses,
            clarification_history=clarification_history,
            round_number=len(clarification_history) + 1,
        )

        print(
            f"\nUpdated Agent 2 Output (Revision Round {revision_round}):\n"
        )

        # If the modification reopened a critical gap, fall back into
        # the clarification loop rather than silently looping the
        # revision prompt against an unresolved-critical-information
        # state.
        if agent2_result.retrieval_readiness == "clarification_required":
            print(
                "\nThe requested change reopened a clarification "
                "requirement. Returning to clarification questions.\n"
            )
            agent2_result, clarification_history = _continue_clarification_after_revision(
                agent1_result, agent2_result, clarification_history
            )


def main():

    query = input("Enter scientific query: ")
    query = validate_input(query)

    # FIX 5 — Capture the PromptInjectionReport object returned by
    # detect_prompt_injection() so it can be passed directly to the
    # security gate below, avoiding a round-trip through the JSON database.
    _pi_report_obj = detect_prompt_injection(query)

    print("\nRunning Agent 1...\n")

    agent1_result = run_agent1(query)

    print("Agent 1 completed.\n")

    agent2_result, clarification_history = _run_clarification_loop(
        agent1_result
    )

    if agent2_result.retrieval_readiness == "clarification_required":
        print(
            f"\nMore clarification is still required after "
            f"{MAX_CLARIFICATION_ROUNDS} rounds. Stopping here."
        )
        return

    agent2_result = _run_confirmation_and_revision_loop(
        agent1_result,
        agent2_result,
        clarification_history,
    )

    print("\nUnderstanding confirmed. Proceeding to dataset discovery.\n")

    print("\nBuilding Retrieval Request...\n")

    # Reconcile Agent 1's frozen plan with the location/time-period
    # answers the user actually gave. Without this, spatial_context,
    # temporal_context, and dataset_requirements would still carry
    # Agent 1's PRE-ANSWER values (e.g. "Global" coverage) here, even
    # though the confirmation screen just showed the user their real,
    # narrowed values -- this is the same reconciliation
    # understanding_summary.py used to display the "Effective ..."
    # lines, so what gets retrieved always matches what was confirmed.
    reconciled_plan = reconcile_agent1_plan(
        agent2_result.refined_scientific_plan.agent1_plan_preserved,
        agent2_result.resolved_information,
    )

    retrieval_request = build_retrieval_request(
        approved_plan=reconciled_plan,
        user_approved_retrieval=True,
    )

    print("Retrieval Request built.\n")

    print("Running Agent 3...\n")

    agent3_result = run_agent3(
        retrieval_request
    )

    print("Agent 3 completed.\n")

    # ---------------------------------------------------------------- #
    # SECURITY INTEGRATION — Provider Trust Assessment (post-Agent 3)  #
    #                                                                   #
    # After Agent 3 finishes ranking providers, evaluate every ranked  #
    # source for provider trust and store the reports.  The reports    #
    # are threaded through to the final Security Risk Assessment gate  #
    # at the bottom of main().  Agent 3's ranking and the Qdrant       #
    # integration are completely untouched.                            #
    # ---------------------------------------------------------------- #
    print("\nRunning Provider Trust Assessment...\n")

    # SECURITY INTEGRATION — build provider metadata dicts from Agent 3's
    # ranked ScoredSource objects using only fields already present on
    # CandidateSource (no new schema changes).
    from security.provider_trust import load_db as _load_pt_db, save_provider_trust_report, ensure_db_exists as _ensure_pt_db
    _ensure_pt_db()
    _pt_db = _load_pt_db()

    provider_trust_reports: list = []
    for scored_source in agent3_result.ranked_sources:
        c = scored_source.candidate
        sc = scored_source.score_card

        # Build a metadata dict mapping Agent 3's existing score fields onto
        # the keys provider_trust.generate_provider_trust_report() expects.
        provider_meta = {
            "provider_name":                 c.name or c.discovery_origin,
            "provider_url":                  c.url,
            "is_government_agency":          False,       # not tracked by Agent 3
            "is_international_organization": False,
            "has_regulatory_mandate":        False,
            "founding_year":                 None,
            "partner_agencies":              [],
            "last_updated":                  None,
            "update_frequency_days":         None,
            "has_recent_dataset":            True,
            "data_domains":                  getattr(c, "variable_names", []),
            "spatial_resolution_km":         None,
            "temporal_resolution_hours":     None,
            "variables_provided":            getattr(c, "variable_names", []),
            "expected_variables":            [v.variable for v in retrieval_request.variables],
            "coverage_percent":              round(sc.completeness.score * 100, 1),
            "has_documentation":             bool(c.metadata_url if hasattr(c, "metadata_url") else False),
            "agreement_with_other_providers": sc.consistency.score,
            "units_documented":              sc.metadata_quality.score > 0.5,
            "crs_documented":                sc.metadata_quality.score > 0.5,
            "has_metadata_standard":         sc.metadata_quality.score > 0.7,
            "license_type":                  "open",
            "citation_count":                int(sc.scientific_acceptance.score * 1000),
            "peer_reviewed_publications":    int(sc.scientific_acceptance.score * 300),
            "used_by_agencies":              [],
            "api_uptime_percent":            round(sc.real_time_availability.score * 100, 1),
            "avg_response_latency_ms":       None,
            "accessible":                    not c.requires_payment,
        }

        try:
            pt_report = generate_provider_trust_report(
                provider_meta,
                requested_task=retrieval_request.goal,
                db=_pt_db,
            )
            save_provider_trust_report(pt_report)
            provider_trust_reports.append(pt_report)
            print(
                f"  [{scored_source.rank}] {pt_report['provider_name']:<30} "
                f"trust={pt_report['overall_trust_score']:.4f}  "
                f"({pt_report['trust_level']})"
            )
        except Exception as _pt_exc:
            print(f"  [WARN] Provider trust assessment skipped for "
                  f"'{c.name}': {_pt_exc}")

    # Summarise the best available trust report for the final gate below.
    # Using the top-ranked provider (rank 1) as the representative report.
    _best_pt_report = provider_trust_reports[0] if provider_trust_reports else None
    _best_pt_for_gate = None
    if _best_pt_report:
        _best_pt_for_gate = {
            "trust_score":      _best_pt_report["overall_trust_score"] * 100,
            "trust_level":      _best_pt_report["trust_level"],
            "provider_name":    _best_pt_report["provider_name"],
            "historical_issues": 0,
        }

    # ---------------------------------------------------------------- #
    # Agent 3 → Agent 4 handoff                                        #
    # CHANGED: run_agent3() itself is still discovery-only and         #
    # unchanged -- it still returns ranked/auth-required/needs-eval/   #
    # rejected buckets with no user interaction inside it. What        #
    # changed is what main.py does with that result: the new           #
    # run_agent3_interactive() layer (agents/agent3_interactive_runner)#
    # takes the untouched DiscoveryOutput and adds the website-policy  #
    # analysis, ranking-preference question, adaptive re-ranking, and  #
    # override question described in the Discovery Agent spec. This is #
    # the one place real user interaction now occurs between Agent 3   #
    # and Agent 4 -- intentionally, since the new requirements call    #
    # for it; the "no user interaction" note that used to be here no   #
    # longer holds.                                                     #
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("AGENT 3 DISCOVERY SUMMARY")
    print("=" * 70)

    print("\nRanked sources (raw Phase 5 output, before adaptive ranking):")
    for scored in agent3_result.ranked_sources:
        c = scored.candidate
        tag = ""
        if c.requires_payment:
            tag = " [PAID]"
        elif c.requires_login:
            tag = " [LOGIN REQUIRED]"
        print(f"  [{scored.rank}] {c.name}{tag}  (score={scored.final_score:.3f})")

    if agent3_result.rejected_sources:
        print(f"\nRejected sources: {len(agent3_result.rejected_sources)}")

    # New interactive layer: analysis -> ranking preference -> adaptive
    # ranking -> formatted output -> override question -> final payload.
    agent3_final_payload = run_agent3_interactive(retrieval_request, agent3_result)

    print("\nPassing discovery results to Agent 4...")
    print("=" * 70)

    agent4_result = run_agent4(agent3_final_payload, retrieval_request)

    # ---------------------------------------------------------------- #
    # SECURITY INTEGRATION — Security Risk Assessment (final gate)     #
    #                                                                   #
    # This is the LAST security step before the pipeline returns its   #
    # result.  It consumes every report gathered across the pipeline:  #
    #   • Prompt Injection   — captured earlier in this run            #
    #   • Provider Trust     — generated above after Agent 3           #
    #   • Integrity          — generated inside Agent 4 per download   #
    #   • Provenance         — generated inside Agent 4 per download   #
    #   • Cross-Agent Verif. — generated inside Agent 4 after retrieval#
    #   • Dataset Validation — generated inside Agent 4 per download   #
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("FINAL SECURITY RISK ASSESSMENT")
    print("=" * 70)

    # SECURITY INTEGRATION — collect the security reports that Agent 4
    # attached to its output (integrity, provenance, cross_agent,
    # validation).  All are Optional — if Agent 4 didn't produce them
    # (e.g. nothing was downloaded) the gate degrades gracefully.
    _sec = getattr(agent4_result, "security_reports", {}) or {}

    # SECURITY INTEGRATION — build a compound integrity/provenance summary
    # from all per-download records that Agent 4 stored.
    _integrity_summary = _sec.get("integrity")
    _provenance_summary = _sec.get("provenance")
    _cross_agent_summary = _sec.get("cross_agent_verification")
    _validation_summary = _sec.get("dataset_validation")

    # FIX 5 — Use the PromptInjectionReport object captured at query time
    # rather than re-reading it from the JSON database.  This avoids the
    # DB round-trip entirely.  Fall back to the DB lookup only if the
    # object from detect_prompt_injection() is unavailable (e.g. the
    # function signature does not return it in this deployment).
    _pi_gate_report = None
    if _pi_report_obj is not None:
        try:
            # Support both dict-style and object-style report shapes.
            _pi_inner = (
                _pi_report_obj.get("report", _pi_report_obj)
                if isinstance(_pi_report_obj, dict)
                else getattr(_pi_report_obj, "report", _pi_report_obj)
            )
            _pi_gate_report = {
                "injection_detected": (
                    _pi_inner.get("injection_detected", False)
                    if isinstance(_pi_inner, dict)
                    else getattr(_pi_inner, "injection_detected", False)
                ),
                "risk_score": (
                    _pi_inner.get("overall_risk_score", 0.0)
                    if isinstance(_pi_inner, dict)
                    else getattr(_pi_inner, "overall_risk_score", 0.0)
                ),
                "confidence": (
                    _pi_inner.get("confidence_score", 1.0)
                    if isinstance(_pi_inner, dict)
                    else getattr(_pi_inner, "confidence_score", 1.0)
                ),
            }
        except Exception as _pi_exc:
            print(f"  [WARN] Could not parse prompt-injection report object: {_pi_exc}")
    if _pi_gate_report is None:
        # Fallback: read from DB only when the live object was unavailable.
        try:
            from security.prompt_injection import load_detection_database as _load_pi_db
            _pi_records = _load_pi_db()
            if _pi_records:
                _last_pi = _pi_records[-1].get("report", {})
                _pi_gate_report = {
                    "injection_detected": _last_pi.get("injection_detected", False),
                    "risk_score":         _last_pi.get("overall_risk_score", 0.0),
                    "confidence":         _last_pi.get("confidence_score", 1.0),
                }
        except Exception as _pi_exc:
            print(f"  [WARN] Could not read prompt-injection DB for gate: {_pi_exc}")

    risk_report = assess_security_risk(
        prompt_injection = _pi_gate_report,
        provider_trust   = _best_pt_for_gate,
        integrity        = _integrity_summary,
        provenance       = _provenance_summary,
        cross_agent      = _cross_agent_summary,
        validation       = _validation_summary,
    )
    save_security_assessment(risk_report)

    print(f"\n  Security Score  : {risk_report.overall_security_score:.1f} / 100")
    print(f"  Risk Level      : {risk_report.overall_risk_level.value}")
    print(f"  Decision        : {risk_report.recommended_action.value}")
    print(f"  Confidence      : {risk_report.confidence_score:.2f}")
    if risk_report.active_flags:
        print(f"  Active Flags    : {', '.join(risk_report.active_flags)}")
    print(f"\n  Reasoning:")
    for line in risk_report.risk_reasoning.splitlines():
        print(f"    {line}")
    if risk_report.recommendations:
        print(f"\n  Recommendations:")
        for i, rec in enumerate(risk_report.recommendations, 1):
            print(f"    {i}. {rec}")
    print("=" * 70)

    # SECURITY INTEGRATION — BLOCK gate: if the final assessment is BLOCK,
    # stop the pipeline here with a clear explanation rather than handing
    # off to Agent 5 or writing files.  WARN and below continue normally.
    if risk_report.recommended_action == RecommendedAction.BLOCK:
        print(
            "\n[SECURITY] Pipeline halted by Security Risk Assessment.\n"
            f"  Reason: {risk_report.risk_reasoning[:200]}\n"
            "  No data has been written to disk and no handoff to Agent 5 will occur.\n"
            "  Please review the active flags and recommendations above, then retry."
        )
        return

    if agent4_result.send_to_agent5:
        print("\nHanding downloaded data to Agent 5 for preprocessing...")
        # TODO: run_agent5(agent4_result) once Agent 5 exists.
    else:
        print(
            f"\nDone. {agent4_result.successful_download_count} validated file(s) written to: "
            f"{agent4_result.download_location}"
        )

if __name__ == "__main__":
    import sys
    if "--web" in sys.argv:
        import uvicorn
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles
        from api.routes import router
        app = FastAPI(title="Earth Intelligence Platform")
        app.include_router(router)
        app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
        print("\nStarting web server at http://localhost:8000\n")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        main()
