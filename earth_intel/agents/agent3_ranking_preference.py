"""
Ranking Preference — Discovery Agent extension.

Asks which criteria (Accuracy / Accessibility / Availability, one or
more) should drive presentation order, then re-sorts the already-
finalized ScoredSource list using the matching composite score(s) from
website_analyzer.py -- each composite is itself built from exactly the
factor list the person specified, and each composite still incorporates
the existing SourceScoreCard's underlying factors (nothing from the
original 13-factor scoring is bypassed).
"""

from dataclasses import dataclass, field
from typing import Dict, List

from models.website_analysis_schemas import RankingCriterion, RankingPreference, WebsiteAnalysisResult

_SELECTED_DIMENSIONS_WEIGHT = 0.65
_ORIGINAL_FINAL_SCORE_WEIGHT = 0.35


@dataclass
class AdaptiveRankingEntry:
    scored_source: object
    analysis: WebsiteAnalysisResult
    adaptive_score: float
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    adaptive_rank: int = 0


def ask_ranking_preference() -> RankingPreference:
    options = [
        ("Accuracy (Authority, Credibility, Scientific Acceptance, Consistency, Historical Reliability, Metadata Quality)", RankingCriterion.accuracy),
        ("Accessibility (Authentication, Registration ease, API, Rate limits, Payment, Retrieval speed, Formats)", RankingCriterion.accessibility),
        ("Availability (Relevance, Completeness, Variable/Spatial/Temporal coverage, Resolution, Historical coverage, Continuity)", RankingCriterion.availability),
    ]

    print("\nOn what basis would you like the websites to be ranked?")
    print("(You may choose one or more options, e.g. '1,3')\n")
    for idx, (label, _) in enumerate(options, start=1):
        print(f"  {idx}. {label}")

    while True:
        raw = input("\nYour choice(s): ").strip()
        if not raw:
            print("No selection made — ranking on all three criteria equally.")
            return RankingPreference(selected_criteria=[c for _, c in options])

        chosen: List[RankingCriterion] = []
        valid = True
        for token in raw.split(","):
            token = token.strip()
            if not token.isdigit() or not (1 <= int(token) <= len(options)):
                valid = False
                break
            chosen.append(options[int(token) - 1][1])

        if not valid or not chosen:
            print("Please enter one or more valid numbers separated by commas.")
            continue

        seen = set()
        deduped = [c for c in chosen if not (c in seen or seen.add(c))]
        return RankingPreference(selected_criteria=deduped)


_DIMENSION_GETTERS = {
    RankingCriterion.accuracy: lambda a: a.accuracy.accuracy_composite_score,
    RankingCriterion.accessibility: lambda a: a.accessibility.accessibility_composite_score,
    RankingCriterion.availability: lambda a: a.availability.availability_composite_score,
}


def adaptive_rank(
    scored_sources: List,
    analyses: Dict[str, WebsiteAnalysisResult],
    preference: RankingPreference,
) -> List[AdaptiveRankingEntry]:
    entries: List[AdaptiveRankingEntry] = []

    for scored in scored_sources:
        candidate = getattr(scored, "candidate", None)
        source_id = getattr(candidate, "source_id", None)
        analysis = analyses.get(source_id)
        if analysis is None:
            continue

        dimension_scores = {
            c.value: _DIMENSION_GETTERS[c](analysis) for c in preference.selected_criteria
        }
        selected_avg = sum(dimension_scores.values()) / len(dimension_scores)
        original_final_score = getattr(scored, "final_score", 0.0)

        adaptive_score = (
            _SELECTED_DIMENSIONS_WEIGHT * selected_avg
            + _ORIGINAL_FINAL_SCORE_WEIGHT * original_final_score
        )

        entries.append(AdaptiveRankingEntry(
            scored_source=scored,
            analysis=analysis,
            adaptive_score=adaptive_score,
            dimension_scores=dimension_scores,
        ))

    entries.sort(key=lambda e: e.adaptive_score, reverse=True)
    for position, entry in enumerate(entries, start=1):
        entry.adaptive_rank = position

    return entries
