"""
Agent 4 — Access Resolver.

Credential states: Free/Anonymous | Requires User Credentials | Paid.
Self-registration (automated bot-registration with dummy emails) has been
removed. If a site requires login, the user is prompted for their real
credentials immediately, or may skip the source.
"""

import getpass
from typing import Dict, Optional

from connectors.base_connector import Credentials
from agents.agent4_credential_store import load_credentials, save_provider_credentials
from models.agent4_schemas import AccessDecisionType, SourceDecision
from models.website_analysis_schemas import CredentialEase, SourceSnapshot, WebsiteAnalysisResult


def resolve_access(
    snapshot: SourceSnapshot,
    analysis: WebsiteAnalysisResult,
    pre_collected_credentials: Optional[Dict[str, dict]] = None,
) -> "tuple[SourceDecision, Optional[Credentials]]":
    """
    Returns (decision, credentials). Credentials are returned separately
    from SourceDecision so secrets never ride on the serialised output path.

    Decision flow:
      0. Credentials already collected by Agent 3 this session → reuse silently.
      1. No authentication required → free_access.
      2. Stored credentials from a previous run → reuse silently.
      3. Payment required → prompt user; never automate.
      4. Any login required (free-tier, self-reg, or real creds) → prompt user.
         Self-registration is NOT attempted automatically; users supply real credentials.
      5. Unconfirmed access → prompt user for credentials or allow skip.
    """
    acc = analysis.accessibility
    sid = snapshot.source_id

    print(f"\n--- Access check: {snapshot.name} ---")

    # 0. Credentials already collected by Agent 3 this session.
    pre_collected = (pre_collected_credentials or {}).get(sid)
    if pre_collected:
        print(f"  Using credentials already collected for this source during discovery.")
        return (
            SourceDecision(
                source_id=sid, decision=AccessDecisionType.user_provided_credentials,
                credentials_used=True, credentials_persisted=False,
                notes="Reused credentials already collected by Agent 3's access gate this session.",
            ),
            Credentials(
                username=pre_collected.get("username", pre_collected.get("api_key", "")),
                password=pre_collected.get("password", pre_collected.get("token", "")),
                api_key=pre_collected.get("api_key"),
                token=pre_collected.get("token"),
                session_token=pre_collected.get("session_token"),
            ),
        )

    # 1. Free access.
    if not acc.authentication_required:
        return SourceDecision(source_id=sid, decision=AccessDecisionType.free_access, notes="No login required."), None

    # 2. Stored credentials from a previous run.
    stored = load_credentials(sid)
    if stored:
        effective_token = stored.bearer_token or stored.refresh_token or stored.token
        print(f"  Using previously saved credentials for {snapshot.name} (no login needed this run).")
        return (
            SourceDecision(
                source_id=sid, decision=AccessDecisionType.user_provided_credentials,
                credentials_used=True, credentials_persisted=True,
                notes="Reused credentials saved from an earlier run.",
            ),
            Credentials(
                username=stored.username,
                password=stored.password,
                api_key=stored.api_key,
                token=effective_token or stored.session_token,
                session_token=stored.session_token,
            ),
        )

    # 3. Paid access.
    if acc.payment_required:
        print(f"  This source requires payment.")
        print(f"  {acc.payment_notes}")
        print(f"  Pricing/payment URL: {snapshot.url}")
        print("  Agent 4 will not attempt payment. Free sources will continue while this remains pending.")
        proceed = input("  Choose: skip provider (skip) or search free alternatives (free): ").strip().lower()
        return SourceDecision(
            source_id=sid,
            decision=AccessDecisionType.skipped_declined if proceed in ("skip",) else AccessDecisionType.payment_redirect,
            notes="User declined paid access; alternate source will be substituted if available."
                  if proceed in ("skip",) else f"Waiting for user payment at {snapshot.url}.",
        ), None

    # 4. Login required (covers CredentialEase.agent_can_self_register,
    #    user_must_provide_real_credentials, and anything else that needs auth).
    #    Automated registration is NOT attempted — prompt the user directly.
    if acc.credential_ease in (
        CredentialEase.agent_can_self_register,
        CredentialEase.user_must_provide_real_credentials,
    ):
        print(f"\n  Provider:          {snapshot.name}")
        print(f"  Why login needed:  {acc.credential_ease_notes or 'Provider requires an account to download data.'}")
        if getattr(snapshot, 'registration_url', None):
            print(f"  Register here:     {snapshot.registration_url}")
        print(f"  Login here:        {getattr(snapshot, 'login_url', None) or snapshot.url}")
        print("  After registering, enter your credentials below. They will be saved and reused automatically.")
        choice = input("  Choose: provide credentials (credentials), skip this source (skip), or search alternatives (alternatives): ").strip().lower()
        if choice in ("credentials", "credential", "creds", "yes", "y"):
            return _prompt_user_credentials(sid, snapshot)
        return SourceDecision(
            source_id=sid, decision=AccessDecisionType.skipped_declined,
            notes="User declined to provide credentials; alternate source will be substituted if available.",
        ), None

    # 5. Unconfirmed access.
    print(f"\n  Provider:          {snapshot.name}")
    print(f"  Access status:     Unconfirmed (Agent 3 could not determine if login is required)")
    print(f"  Notes:             {acc.credential_ease_notes or 'Unknown access requirements.'}")
    if getattr(snapshot, 'registration_url', None):
        print(f"  Register here:     {snapshot.registration_url}")
    print(f"  Login/source URL:  {getattr(snapshot, 'login_url', None) or snapshot.url}")
    choice = input("  Provide credentials (credentials), skip source (skip), or search alternatives (alternatives)? ").strip().lower()
    if choice in ("credentials", "credential", "creds", "mine", "yes", "y"):
        return _prompt_user_credentials(sid, snapshot)
    return SourceDecision(
        source_id=sid, decision=AccessDecisionType.skipped_declined,
        notes="User skipped an unconfirmed-access source; alternate source will be substituted if available.",
    ), None


def _prompt_user_credentials(source_id: str, snapshot: SourceSnapshot) -> "tuple[SourceDecision, Credentials]":
    print("  What type of credentials does this provider use?")
    print("    1. Username + password")
    print("    2. API key")
    print("    3. Bearer token")
    print("    4. Username + password + API key (some providers need both)")
    cred_type = input("  Choice [1/2/3/4, default 1]: ").strip() or "1"

    username = ""
    password = ""
    api_key: Optional[str] = None
    bearer_token: Optional[str] = None
    refresh_token: Optional[str] = None

    if cred_type in ("1", "4"):
        username = input("  Username / email: ").strip()
        password = getpass.getpass("  Password (input hidden): ")

    if cred_type in ("2", "4"):
        api_key = getpass.getpass("  API key (input hidden): ") or None

    if cred_type == "3":
        bearer_token = getpass.getpass("  Bearer token (input hidden): ") or None
        refresh = input("  Refresh token (leave blank if none): ").strip()
        refresh_token = refresh or None

    persist = input("  Save credentials for next time? (yes/no): ").strip().lower() in ("yes", "y")
    if persist:
        save_provider_credentials(
            source_id,
            username=username,
            password=password,
            api_key=api_key,
            bearer_token=bearer_token,
            refresh_token=refresh_token,
        )

    effective_token = bearer_token or refresh_token
    return (
        SourceDecision(
            source_id=source_id, decision=AccessDecisionType.user_provided_credentials,
            credentials_used=True, credentials_persisted=persist,
            notes="User-supplied credentials.",
        ),
        Credentials(
            username=username,
            password=password,
            api_key=api_key,
            token=effective_token,
        ),
    )
