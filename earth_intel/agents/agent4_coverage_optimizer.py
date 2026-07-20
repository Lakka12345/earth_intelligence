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

VARIABLE_ALIASES = {
    "greenhouse gas emissions": {"greenhouse gas", "greenhouse gases", "ghg", "co2", "carbon dioxide", "methane", "ch4", "emissions"},
    "land use change": {"land use", "land cover", "land_use_land_cover", "lulc", "worldcover", "landcover"},
    "temperature anomaly": {"temperature anomaly", "temp anomaly", "temperature anomalies", "anomaly"},
    # CHANGED: strict substring matching means real synonyms never matched
    # each other (e.g. a request for "rainfall intensity" vs. a provider's
    # "precipitation" hint) unless one string was literally a substring of
    # the other. This silently excluded genuinely relevant sources (e.g.
    # Open-Meteo, which lists "precipitation" but not "rainfall") from the
    # coverage plan and force-explore step alike. Aliasing them here fixes
    # both call sites at once since they share this lookup. Keys must match
    # the full variable-name strings this pipeline actually uses (as seen
    # in Agent 2's output, e.g. "rainfall intensity" not just "rainfall") --
    # a mismatched key means the alias set is never looked up at all.
    "rainfall": {"precipitation", "rainfall intensity", "rain", "precip"},
    "rainfall intensity": {"precipitation", "rainfall", "rain", "precip"},
    "river discharge": {"streamflow", "discharge", "river flow", "flow rate"},
    "water level": {"river stage", "gauge height", "stage height", "reservoir level", "sea level"},
    "soil moisture": {"volumetric soil moisture", "soil water content"},
    "flood extent": {"flood extent", "inundation extent", "flood mapping"},
    "land use/land cover": {"land use", "land cover", "lulc", "landcover", "worldcover"},
}


def _normalize_variable(value: str) -> str:
    return " ".join(str(value or "").lower().replace("_", " ").replace("-", " ").split())


def _variable_terms(value: str) -> Set[str]:
    normalized = _normalize_variable(value)
    terms = {normalized}
    terms.update(_normalize_variable(alias) for alias in VARIABLE_ALIASES.get(normalized, set()))
    return {term for term in terms if term}


def _matches_requested_variable(requested: str, candidate: str) -> bool:
    requested_terms = _variable_terms(requested)
    candidate_terms = _variable_terms(candidate)
    for req in requested_terms:
        for cand in candidate_terms:
            if req == cand or req in cand or cand in req:
                return True
    return False


def _source_variables(
    sid: str,
    website_analyses: Dict[str, WebsiteAnalysisResult],
    source_snapshots: Optional[Dict[str, SourceSnapshot]] = None,
) -> List[str]:
    analysis = website_analyses.get(sid)
    variables = []
    if analysis is not None:
        variables.extend(analysis.availability.covered_variables or [])
    snapshot = (source_snapshots or {}).get(sid)
    if snapshot is not None:
        variables.extend(snapshot.variables_available or [])
    return variables


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
    source_snapshots: Optional[Dict[str, SourceSnapshot]] = None,
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
    requested_lookup = {_normalize_variable(v): v for v in requested_variables}
    requested = set(requested_lookup.keys())
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
        snapshot = (source_snapshots or {}).get(sid)
        if analysis is None and snapshot is None:
            continue

        source_vars = _source_variables(sid, website_analyses, source_snapshots)
        new_coverage = {
            req for req in plan.uncovered_variables
            if any(_matches_requested_variable(req, source_var) for source_var in source_vars)
        }
        if not new_coverage:
            # This source (even if highly ranked) adds nothing beyond
            # what higher-ranked picks already cover -- skip it. This
            # is the duplicate/redundancy guard.
            continue

        plan.selected_source_ids.append(sid)
        plan.per_source_new_coverage[sid] = [requested_lookup.get(v, v) for v in sorted(new_coverage)]
        plan.covered_variables |= new_coverage
        plan.uncovered_variables -= new_coverage

    plan.covered_variables = {requested_lookup.get(v, v) for v in plan.covered_variables}
    plan.uncovered_variables = {requested_lookup.get(v, v) for v in plan.uncovered_variables}
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
