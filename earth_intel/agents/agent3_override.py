"""
Website Override — Discovery Agent extension.

Credential classifications: Free/Anonymous | Requires User Credentials | Paid.
Self-registration buckets have been removed — if a site requires login, users
must supply their own credentials.
"""

from typing import Dict, List

from agents.agent3_ranking_preference import AdaptiveRankingEntry
from models.website_analysis_schemas import (
    Agent3ToAgent4Mode,
    Agent3ToAgent4Payload,
    CredentialEase,
    RankingPreference,
    SourceSnapshot,
    WebsiteAnalysisResult,
)


def _credential_buckets(entries: List[AdaptiveRankingEntry]):
    """
    Classify sources into three buckets only:
      real_credentials — requires the user's own login/API key
      paid             — requires payment
      unconfirmed      — requires login but registration ease is unknown
    Self-registerable sources are treated identically to real_credentials
    because automated bot-registration has been removed from the pipeline.
    """
    real_credentials, paid, unconfirmed = [], [], []
    for e in entries:
        sid = e.scored_source.candidate.source_id
        acc = e.analysis.accessibility
        if acc.payment_required:
            paid.append(sid)
        elif acc.credential_ease in (
            CredentialEase.agent_can_self_register,
            CredentialEase.user_must_provide_real_credentials,
        ):
            real_credentials.append(sid)
        elif acc.credential_ease == CredentialEase.unknown and acc.authentication_required:
            unconfirmed.append(sid)
    return real_credentials, paid, unconfirmed


def _build_context(entries: List[AdaptiveRankingEntry], requested_variables: List[str]):
    """Bundles the full context Agent 4 needs so it never has to
    re-derive or re-import Agent 3 internals."""
    website_analyses = {e.scored_source.candidate.source_id: e.analysis for e in entries}
    source_snapshots = {}
    for e in entries:
        c = e.scored_source.candidate
        source_snapshots[c.source_id] = SourceSnapshot(
            source_id=c.source_id,
            name=c.name,
            url=c.url,
            api_type=(c.api_type.value if getattr(c, "api_type", None) else "unknown"),
            dataset_type=str(getattr(c, "dataset_type", "unknown")),
            variables_available=list(getattr(c, "variables_available", []) or []),
            login_url=getattr(c, "login_url", None),
            price_estimate=getattr(c, "price_estimate", None),
        )
    return website_analyses, source_snapshots


def ask_override_and_build_payload(
    entries: List[AdaptiveRankingEntry],
    preference: RankingPreference,
    requested_variables: List[str] = None,
    pre_collected_credentials: Dict[str, Dict] = None,
) -> Agent3ToAgent4Payload:
    requested_variables = requested_variables or []
    pre_collected_credentials = pre_collected_credentials or {}
    real_cred, paid, unconfirmed = _credential_buckets(entries)
    website_analyses, source_snapshots = _build_context(entries, requested_variables)

    print("\nWould you like to obtain data from a specific website instead of "
          "following the ranked recommendations?")
    print("  1. Yes")
    print("  2. No")

    while True:
        raw = input("\nYour choice: ").strip().lower()
        if raw in ("1", "yes", "y"):
            website = input("\nEnter the preferred website (name or URL): ").strip()
            if not website:
                print("Please enter a website name or URL.")
                continue
            return Agent3ToAgent4Payload(
                mode=Agent3ToAgent4Mode.user_override,
                final_ranked_source_ids=[e.scored_source.candidate.source_id for e in entries],
                self_registerable_source_ids=[],   # removed — field kept for schema compat
                real_credentials_required_source_ids=real_cred,
                paid_source_ids=paid,
                unconfirmed_credential_source_ids=unconfirmed,
                website_analyses=website_analyses,
                source_snapshots=source_snapshots,
                requested_variables=requested_variables,
                pre_collected_credentials=pre_collected_credentials,
                override_website=website,
                ranking_preference=preference,
                notes=[
                    f"User overrode the ranked recommendations in favor of: {website}.",
                    "Ranked list, including credential-ease breakdown, is still included for Agent 4's reference.",
                ],
            )
        elif raw in ("2", "no", "n"):
            return Agent3ToAgent4Payload(
                mode=Agent3ToAgent4Mode.ranked_selection,
                final_ranked_source_ids=[e.scored_source.candidate.source_id for e in entries],
                self_registerable_source_ids=[],   # removed — field kept for schema compat
                real_credentials_required_source_ids=real_cred,
                paid_source_ids=paid,
                unconfirmed_credential_source_ids=unconfirmed,
                website_analyses=website_analyses,
                source_snapshots=source_snapshots,
                requested_variables=requested_variables,
                pre_collected_credentials=pre_collected_credentials,
                override_website=None,
                ranking_preference=preference,
                notes=[
                    f"User accepted the ranked recommendations "
                    f"({', '.join(c.value for c in preference.selected_criteria)} basis).",
                    f"{len(real_cred)} source(s) require the user's own real credentials; "
                    f"{len(paid)} require payment; "
                    f"{len(unconfirmed)} require login but registration ease is unconfirmed -- "
                    f"Agent 4 should prompt the user for credentials directly.",
                ],
            )
        else:
            print("Please answer 'Yes' or 'No' (or 1 / 2).")
