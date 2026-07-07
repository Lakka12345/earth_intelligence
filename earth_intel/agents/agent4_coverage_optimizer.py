"""
Agent 4 — Coverage Optimizer.

Given Agent 3's FINAL ranked source order (already adaptive-ranked per
the user's chosen criteria) and the query's requested variables, picks
the smallest combination of sources that together cover everything
requested -- always walking the list IN RANK ORDER, so a higher-ranked
source is always preferred over a lower-ranked one whenever both would
satisfy the same variable. This is a hard constraint, not a tiebreaker:
Agent 4 never substitutes a lower-ranked source for a higher-ranked one
unless the higher-ranked one was explicitly declined (wrong access
type, size rejected, etc. -- handled by the access/size resolution
steps, which call back into this module to re-run the pick with a
source excluded).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from models.website_analysis_schemas import SourceSnapshot, WebsiteAnalysisResult


@dataclass
class CoveragePlan:
    selected_source_ids: List[str] = field(default_factory=list)   # in the order they were picked (= rank order)
    covered_variables: Set[str] = field(default_factory=set)
    uncovered_variables: Set[str] = field(default_factory=set)
    per_source_new_coverage: Dict[str, List[str]] = field(default_factory=dict)  # what each source added, for the user-facing explanation

    @property
    def is_complete(self) -> bool:
        return len(self.uncovered_variables) == 0


def build_coverage_plan(
    ranked_source_ids: List[str],
    website_analyses: Dict[str, WebsiteAnalysisResult],
    requested_variables: List[str],
    excluded_source_ids: Optional[Set[str]] = None,
) -> CoveragePlan:
    """
    Greedy, rank-order-respecting set cover.

    Walks `ranked_source_ids` top to bottom (this IS the rank order --
    the caller must pass Agent 3's final adaptive-ranked list, not an
    unordered set). A source is added to the plan only if it covers at
    least one variable nothing selected so far already covers -- this
    is what prevents duplicates (a lower-ranked source whose entire
    variable set is already satisfied by higher-ranked picks is simply
    skipped, never included just to "use more sources").

    `excluded_source_ids` lets the access-resolution / size-approval
    steps say "the user declined this one, try again without it" --
    the walk resumes from the top of the remaining ranked list, so a
    still-higher-ranked alternative is always tried before a lower one.
    """
    excluded = excluded_source_ids or set()
    requested = {v.lower().strip() for v in requested_variables}
    plan = CoveragePlan(uncovered_variables=set(requested))

    if not requested:
        # Nothing specific was requested (unusual) -- fall back to just
        # the single top-ranked, non-excluded source so there's still
        # a sane default rather than selecting everything.
        for sid in ranked_source_ids:
            if sid not in excluded:
                plan.selected_source_ids.append(sid)
                plan.per_source_new_coverage[sid] = []
                break
        return plan

    for sid in ranked_source_ids:
        if sid in excluded or not plan.uncovered_variables:
            continue

        analysis = website_analyses.get(sid)
        if analysis is None:
            continue

        source_vars = {v.lower().strip() for v in analysis.availability.covered_variables}
        new_coverage = source_vars & plan.uncovered_variables
        if not new_coverage:
            # This source (even if highly ranked) adds nothing beyond
            # what higher-ranked picks already cover -- skip it. This
            # is the duplicate/redundancy guard.
            continue

        plan.selected_source_ids.append(sid)
        plan.per_source_new_coverage[sid] = sorted(new_coverage)
        plan.covered_variables |= new_coverage
        plan.uncovered_variables -= new_coverage

    return plan


def describe_plan(
    plan: CoveragePlan,
    source_snapshots: Dict[str, SourceSnapshot],
) -> str:
    """Human-readable explanation of why each source was picked, in the
    same rank order it was selected -- for the transparency the user
    asked for ('agent must intelligently decide... make it clear')."""
    lines = []
    for rank, sid in enumerate(plan.selected_source_ids, start=1):
        name = source_snapshots[sid].name if sid in source_snapshots else sid
        gained = plan.per_source_new_coverage.get(sid, [])
        lines.append(f"  {rank}. {name} -- adds: {', '.join(gained) if gained else '(fallback pick, no specific variables requested)'}")
    if plan.uncovered_variables:
        lines.append(f"  Still not covered by any ranked source: {', '.join(sorted(plan.uncovered_variables))}")
    return "\n".join(lines)
