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
from typing import Dict, Optional

from agents.agent4_connectors.base import Credentials
from agents.agent4_connectors.registry import get_connector
from agents.agent4_credential_store import load_credentials, save_credentials
from models.agent4_schemas import AccessDecisionType, SourceDecision
from models.website_analysis_schemas import CredentialEase, SourceSnapshot, WebsiteAnalysisResult


def _generate_random_secret(length: int = 16) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choices(alphabet, k=length))


def _attempt_self_registration(snapshot: SourceSnapshot) -> Optional[Credentials]:
    connector = get_connector(snapshot)
    if not connector.supports_self_registration():
        print(f"  Automated registration isn't implemented yet for this specific provider "
              f"({connector.name} connector). This is a known gap, not a guess -- "
              f"no fabricated credentials will be used.")
        return None

    real_email = input("  Enter a real email address to use for registration/verification: ").strip()
    if not real_email or "@" not in real_email:
        print("  A valid email is required to complete registration (most providers send a verification link).")
        return None

    try:
        creds = connector.self_register(snapshot, real_email)
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
            ),
        )

    # 1. Free access -- nothing to resolve.
    if not acc.authentication_required:
        return SourceDecision(source_id=sid, decision=AccessDecisionType.free_access, notes="No login required."), None

    # 2. Already have stored credentials from a previous run.
    stored = load_credentials(sid)
    if stored:
        print(f"  Using previously saved credentials for this source.")
        return (
            SourceDecision(
                source_id=sid, decision=AccessDecisionType.user_provided_credentials,
                credentials_used=True, credentials_persisted=True,
                notes="Reused credentials saved from an earlier run.",
            ),
            Credentials(username=stored.username, password=stored.password),
        )

    # 3. Paid access -- Agent 4 never automates payment.
    if acc.payment_required:
        print(f"  This source requires payment.")
        print(f"  {acc.payment_notes}")
        print(f"  URL: {snapshot.url}")
        print(f"  Please visit the link above and complete payment/access yourself if you'd like to use this source.")
        proceed = input("  Type 'skip' to exclude this source and use an alternate instead, or press Enter to keep it (you'll handle access manually): ").strip().lower()
        if proceed == "skip":
            return SourceDecision(source_id=sid, decision=AccessDecisionType.skipped_declined, notes="User declined paid access; alternate source will be substituted if available."), None
        return SourceDecision(source_id=sid, decision=AccessDecisionType.payment_redirect, notes=f"User will complete access manually at {snapshot.url}."), None

    # 4. Agent can self-register.
    if acc.credential_ease == CredentialEase.agent_can_self_register:
        print(f"  This source allows simple self-service registration -- I can handle this for you.")
        choice = input("  Should I register an account and retrieve the data automatically? (yes/no): ").strip().lower()
        if choice in ("yes", "y"):
            creds = _attempt_self_registration(snapshot)
            if creds:
                persist = input("  Save these credentials for next time? (yes/no): ").strip().lower() in ("yes", "y")
                if persist:
                    save_credentials(sid, creds.username, creds.password)
                return (
                    SourceDecision(
                        source_id=sid, decision=AccessDecisionType.agent_self_registered,
                        credentials_used=True, credentials_persisted=persist,
                        notes="Agent registered and authenticated automatically.",
                    ),
                    creds,
                )
            # Automated registration unavailable for this provider -- fall through to asking the user.
            manual = input("  Automated registration isn't available for this provider yet. "
                            "Do you want to register yourself and provide credentials? (yes/no): ").strip().lower()
            if manual in ("yes", "y"):
                return _prompt_user_credentials(sid, snapshot)
            return SourceDecision(source_id=sid, decision=AccessDecisionType.skipped_declined, notes="No registration path available; alternate source will be substituted if available."), None
        return SourceDecision(source_id=sid, decision=AccessDecisionType.skipped_declined, notes="User declined self-registration; alternate source will be substituted if available."), None

    # 5. Real credentials required.
    if acc.credential_ease == CredentialEase.user_must_provide_real_credentials:
        print(f"  {acc.credential_ease_notes}")
        choice = input("  Are you willing to provide your own account credentials for this source? (yes/no): ").strip().lower()
        if choice in ("yes", "y"):
            return _prompt_user_credentials(sid, snapshot)
        return SourceDecision(source_id=sid, decision=AccessDecisionType.skipped_declined, notes="User declined to provide real credentials; alternate source will be substituted if available."), None

    # 6. Unconfirmed -- be honest about the uncertainty, let the user decide.
    print(f"  {acc.credential_ease_notes}")
    choice = input("  Try self-registration as a probe (yes), provide your own credentials (mine), or skip this source (skip)? [yes/mine/skip]: ").strip().lower()
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
    username = input("  Username/email for this source: ").strip()
    password = getpass.getpass("  Password (input hidden): ")
    persist = input("  Save these credentials for next time? (yes/no): ").strip().lower() in ("yes", "y")
    if persist:
        save_credentials(source_id, username, password)
    return (
        SourceDecision(
            source_id=source_id, decision=AccessDecisionType.user_provided_credentials,
            credentials_used=True, credentials_persisted=persist,
            notes="User-supplied credentials.",
        ),
        Credentials(username=username, password=password),
    )
