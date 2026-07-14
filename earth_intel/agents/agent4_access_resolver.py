"""
Agent 4 — Access Resolver.

For each source in the coverage plan, resolves how Agent 4 will
actually get access to it, driven entirely by the AccessibilityProfile
Agent 3 already computed. Produces one SourceDecision per source.
Reuses stored credentials silently (no need to re-ask every run) --
that's the whole point of the credential-persistence feature.
"""

import getpass
import random
import string
import time
from typing import Dict, Optional

from connectors.base_connector import Credentials
from connectors.connector_factory import get_connector
from agents.agent4_credential_store import load_credentials, save_credentials, save_provider_credentials
from models.agent4_schemas import AccessDecisionType, SourceDecision
from models.website_analysis_schemas import CredentialEase, SourceSnapshot, WebsiteAnalysisResult


def _generate_random_secret(length: int = 16) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choices(alphabet, k=length))


def _generate_registration_email(snapshot: SourceSnapshot) -> str:
    safe_provider = "".join(c.lower() for c in snapshot.source_id if c.isalnum())[:24] or "provider"
    return f"earthintel.{safe_provider}.{int(time.time())}@example.com"


def _attempt_self_registration(snapshot: SourceSnapshot) -> Optional[Credentials]:
    connector = get_connector(snapshot)
    if not connector.supports_self_registration():
        print(f"  Automated registration isn't implemented yet for this specific provider "
              f"({connector.name} connector). This is a known gap, not a guess -- "
              f"no fabricated credentials will be used.")
        return None

    generated_email = _generate_registration_email(snapshot)
    try:
        creds = connector.self_register(snapshot, generated_email)
        print(f"  Registered successfully with username '{creds.username}'.")
        return creds
    except NotImplementedError:
        print(f"  Automated registration isn't implemented yet for this specific provider ({connector.name}).")
        return None
    except Exception as exc:
        print(f"  Registration attempt failed: {exc}")
        return None


def resolve_access(
    snapshot: SourceSnapshot,
    analysis: WebsiteAnalysisResult,
    pre_collected_credentials: Optional[Dict[str, dict]] = None,
) -> "tuple[SourceDecision, Optional[Credentials]]":
    """
    Returns (decision, credentials). Credentials are returned
    separately from SourceDecision -- NEVER stored inside it -- because
    SourceDecision ends up inside Agent4Output, which may be logged,
    printed, or serialized and handed to Agent 5. Raw secrets must not
    ride along on that path. The orchestrator is responsible for using
    the returned Credentials for this run only and then discarding them
    (persistence, if the user opted in, already happened via
    save_credentials/keyring above -- that's the only place they touch
    disk, encrypted, not in-process objects that get passed around).
    """
    acc = analysis.accessibility
    sid = snapshot.source_id

    print(f"\n--- Access check: {snapshot.name} ---")

    # 0. Credentials already collected by Agent 3's own access gate
    # (DiscoveryOutput.retrieval_credentials) -- use these before asking
    # the user anything or touching the local keyring. This is data
    # Agent 3 already gathered THIS session; it takes priority.
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

    # 1. Free access -- nothing to resolve.
    if not acc.authentication_required:
        return SourceDecision(source_id=sid, decision=AccessDecisionType.free_access, notes="No login required."), None

    # 2. Already have stored credentials from a previous run.
    stored = load_credentials(sid)
    if stored:
        # Prefer the most capable token available: bearer > refresh > api_key > session > password.
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

    # 3. Paid access -- Agent 4 never automates payment.
    if acc.payment_required:
        print(f"  This source requires payment.")
        print(f"  {acc.payment_notes}")
        print(f"  Pricing/payment URL: {snapshot.url}")
        print("  Agent 4 will not attempt payment. Free sources will continue while this remains pending.")
        proceed = input("  Choose: pay now (pay), skip provider (skip), or search free alternatives (free): ").strip().lower()
        if proceed in ("skip", "free"):
            return SourceDecision(source_id=sid, decision=AccessDecisionType.skipped_declined, notes="User declined paid access; alternate source will be substituted if available."), None
        return SourceDecision(source_id=sid, decision=AccessDecisionType.payment_redirect, notes=f"Waiting for user payment at {snapshot.url}."), None

    # 4. Agent can self-register.
    if acc.credential_ease == CredentialEase.agent_can_self_register:
        print("  This source allows simple self-service registration. Agent 4 will try it automatically.")
        creds = _attempt_self_registration(snapshot)
        if creds:
            save_credentials(sid, creds.username, creds.password)
            return (
                SourceDecision(
                    source_id=sid, decision=AccessDecisionType.agent_self_registered,
                    credentials_used=True, credentials_persisted=True,
                    notes="Agent registered and authenticated automatically.",
                ),
                creds,
            )
        manual = input("  Automated registration is not available for this connector. Provide credentials, skip, or search alternatives? (credentials/skip/alternatives): ").strip().lower()
        if manual in ("credentials", "credential", "creds", "mine"):
            return _prompt_user_credentials(sid, snapshot)
        return SourceDecision(source_id=sid, decision=AccessDecisionType.skipped_declined, notes="No registration path available; alternate source will be substituted if available."), None

    # 5. Real credentials required.
    if acc.credential_ease == CredentialEase.user_must_provide_real_credentials:
        print(f"\n  Provider:          {snapshot.name}")
        print(f"  Why login needed:  {acc.credential_ease_notes or 'Provider requires an account to download data.'}")
        if snapshot.registration_url:
            print(f"  Register here:     {snapshot.registration_url}")
        print(f"  Login here:        {snapshot.login_url or snapshot.url}")
        print("  After registering, come back and enter your credentials below.")
        print("  They will be saved securely and reused automatically on every future run.")
        choice = input("  Choose: provide credentials (credentials), skip this source (skip), or search alternatives (alternatives): ").strip().lower()
        if choice in ("credentials", "credential", "creds", "yes", "y"):
            return _prompt_user_credentials(sid, snapshot)
        return SourceDecision(source_id=sid, decision=AccessDecisionType.skipped_declined, notes="User declined to provide real credentials; alternate source will be substituted if available."), None

    # 6. Unconfirmed -- be honest about the uncertainty, let the user decide.
    print(f"\n  Provider:          {snapshot.name}")
    print(f"  Access status:     Unconfirmed (Agent 3 could not determine if login is required)")
    print(f"  Notes:             {acc.credential_ease_notes or 'Unknown access requirements.'}")
    if snapshot.registration_url:
        print(f"  Register here:     {snapshot.registration_url}")
    print(f"  Login/source URL:  {snapshot.login_url or snapshot.url}")
    choice = input("  Try self-registration as a probe (yes), provide credentials (mine), skip source (skip), or search alternatives (alternatives)? [yes/mine/skip/alternatives]: ").strip().lower()
    if choice == "yes":
        creds = _attempt_self_registration(snapshot)
        if creds:
            persist = input("  Save these credentials for next time? (yes/no): ").strip().lower() in ("yes", "y")
            if persist:
                save_credentials(sid, creds.username, creds.password)
            return SourceDecision(source_id=sid, decision=AccessDecisionType.agent_self_registered, credentials_used=True, credentials_persisted=persist), creds
        return SourceDecision(source_id=sid, decision=AccessDecisionType.skipped_unresolved, notes="Self-registration probe failed and access requirement remains unconfirmed."), None
    if choice == "mine":
        return _prompt_user_credentials(sid, snapshot)
    return SourceDecision(source_id=sid, decision=AccessDecisionType.skipped_declined, notes="User skipped an unconfirmed-access source; alternate source will be substituted if available."), None


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
