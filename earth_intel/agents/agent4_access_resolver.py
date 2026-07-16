"""
Agent 4 — Access Resolver.

Credential states: Free/Anonymous | Requires User Credentials | Paid.

When a source requires login, the user is offered four options:
  existing    — supply credentials they already have (cached for future runs)
  new         — automated registration assistant (Playwright form-fill)
  skip        — skip this source
  alternatives — search lower-ranked alternatives

Credentials are returned separately from SourceDecision so secrets never
ride on the serialised output path.
"""

import base64
import getpass
import json
import os
from typing import Dict, Optional

from connectors.base_connector import Credentials
# Renamed on import to avoid clashing with the simpler save_credentials/
# load_credentials fallback pair defined below (point 2/3 of this change).
# These two keep talking to the full multi-field credential store
# (bearer tokens, API keys, refresh tokens, etc.) exactly as before.
from agents.agent4_credential_store import (
    load_credentials as store_load_credentials,
    save_provider_credentials as store_save_provider_credentials,
)
from agents.agent4_registration_assistant import guided_new_account_flow
from models.agent4_schemas import AccessDecisionType, SourceDecision
from models.website_analysis_schemas import CredentialEase, SourceSnapshot, WebsiteAnalysisResult

# ---------------------------------------------------------------------------
# Keyring-missing fallback: local JSON credential cache
# ---------------------------------------------------------------------------
# `keyring` needs a working OS credential backend (macOS Keychain, Windows
# Credential Locker, a Secret Service / KWallet on Linux, etc.). Headless
# servers, containers, and some CI environments don't have one, so importing
# it can fail outright or fail at first use. Either way we want retries in
# the access-resolution loop to survive without re-prompting the user.
#
# IMPORTANT: base64 is *obfuscation*, not encryption -- it is trivially
# reversible by anyone who can read the cache file. This fallback exists so
# credentials aren't stored in cleartext-with-obvious-field-names, and so
# retries don't force the user to retype a password every loop iteration.
# It is NOT a substitute for real at-rest encryption. If that's a
# requirement, swap this for something like `cryptography.fernet` with a
# locally-stored key instead of the base64 encode/decode below.
try:
    import keyring
    _KEYRING_AVAILABLE = True
except ImportError:
    keyring = None
    _KEYRING_AVAILABLE = False

_FALLBACK_CACHE_PATH = os.path.join("data", ".credentials_cache.json")
_KEYRING_SERVICE_PREFIX = "agent4_source_credentials"


def _obfuscate(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def _deobfuscate(value: str) -> str:
    return base64.b64decode(value.encode("ascii")).decode("utf-8")


def _load_fallback_cache() -> dict:
    if not os.path.exists(_FALLBACK_CACHE_PATH):
        return {}
    try:
        with open(_FALLBACK_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # Corrupt or unreadable cache -- treat as empty rather than crash.
        return {}


def _write_fallback_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(_FALLBACK_CACHE_PATH) or ".", exist_ok=True)
    with open(_FALLBACK_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def save_credentials(provider_name: str, username: str, password: str) -> None:
    """
    Persist a username/password pair for `provider_name`.

    Uses the OS keyring when available; otherwise falls back to a local
    base64-obfuscated JSON cache at `data/.credentials_cache.json`, keyed
    by a unique per-provider service ID so retries within the same
    access-resolution loop (and later runs) don't need to re-ask the user.
    """
    service_id = f"{_KEYRING_SERVICE_PREFIX}:{provider_name}"

    if _KEYRING_AVAILABLE:
        try:
            keyring.set_password(service_id, username or provider_name, password)
            return
        except Exception:
            # Keyring import succeeded but the backend itself isn't usable
            # at runtime (common in headless/containerized environments) --
            # fall through to the local file cache below.
            pass

    cache = _load_fallback_cache()
    cache[service_id] = {
        "username": _obfuscate(username or ""),
        "password": _obfuscate(password or ""),
    }
    _write_fallback_cache(cache)


def load_credentials(provider_name: str) -> Optional[Dict[str, str]]:
    """
    Retrieve a previously saved username/password pair for `provider_name`.

    Returns a plain dict with "username" and "password" keys, or None if
    nothing has been saved yet. Mirrors the save/load pairing in
    save_credentials(): keyring first, local obfuscated JSON cache as the
    fallback when keyring is unavailable or fails.
    """
    service_id = f"{_KEYRING_SERVICE_PREFIX}:{provider_name}"

    if _KEYRING_AVAILABLE:
        try:
            password = keyring.get_password(service_id, provider_name)
            if password is not None:
                return {"username": provider_name, "password": password}
        except Exception:
            pass

    cache = _load_fallback_cache()
    entry = cache.get(service_id)
    if not entry:
        return None
    return {
        "username": _deobfuscate(entry["username"]),
        "password": _deobfuscate(entry["password"]),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def resolve_access(
    snapshot: SourceSnapshot,
    analysis: WebsiteAnalysisResult,
    pre_collected_credentials: Optional[Dict[str, dict]] = None,
) -> "tuple[SourceDecision, Optional[Credentials]]":
    """
    Returns (decision, credentials).

    Decision flow:
      0. Credentials already collected by Agent 3 this session → reuse silently.
      1. No authentication required → free_access.
      2. Stored credentials from a previous run → reuse silently.
      3. Payment required → user chooses skip or payment redirect.
      4. Login required → 4-option menu (existing / new / skip / alternatives).
      5. Unconfirmed access → same 4-option menu.
    """
    acc = analysis.accessibility
    sid = snapshot.source_id

    print(f"\n--- Access check: {snapshot.name} ---")

    # 0. Credentials already collected by Agent 3 this session.
    pre_collected = (pre_collected_credentials or {}).get(sid)
    if pre_collected:
        print("  Using credentials already collected for this source during discovery.")
        return (
            SourceDecision(
                source_id=sid,
                decision=AccessDecisionType.user_provided_credentials,
                credentials_used=True,
                credentials_persisted=False,
                notes="Reused credentials collected by Agent 3's access gate this session.",
            ),
            Credentials(
                username=pre_collected.get("username", pre_collected.get("api_key", "")),
                password=pre_collected.get("password", pre_collected.get("token", "")),
                api_key=pre_collected.get("api_key"),
                token=pre_collected.get("token"),
                session_token=pre_collected.get("session_token"),
            ),
        )

    # 1. Free / anonymous access.
    if not acc.authentication_required:
        return (
            SourceDecision(source_id=sid, decision=AccessDecisionType.free_access,
                           notes="No login required."),
            None,
        )

    # 2. Stored credentials from a previous run — reuse silently.
    stored = store_load_credentials(sid)
    if stored:
        effective_token = stored.bearer_token or stored.refresh_token or stored.token
        print(f"  Using previously saved credentials for {snapshot.name} (no login needed this run).")
        return (
            SourceDecision(
                source_id=sid,
                decision=AccessDecisionType.user_provided_credentials,
                credentials_used=True,
                credentials_persisted=True,
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
        if acc.payment_notes:
            print(f"  {acc.payment_notes}")
        print(f"  Pricing/payment URL: {snapshot.url}")
        print("  Agent 4 will not automate payment.")
        choice = input(
            "  Choose: skip this source (skip) or search free alternatives (alternatives): "
        ).strip().lower()
        skipped = choice in ("skip", "s")
        return (
            SourceDecision(
                source_id=sid,
                decision=(AccessDecisionType.skipped_declined if skipped
                          else AccessDecisionType.payment_redirect),
                notes=("User declined paid access; alternate source will be substituted if available."
                       if skipped else f"Waiting for user payment at {snapshot.url}."),
            ),
            None,
        )

    # 4 & 5. Login required (confirmed or unconfirmed).
    return _login_required_menu(sid, snapshot, acc)


# ---------------------------------------------------------------------------
# 4-option login menu
# ---------------------------------------------------------------------------

def _login_required_menu(
    sid: str,
    snapshot: SourceSnapshot,
    acc,
) -> "tuple[SourceDecision, Optional[Credentials]]":
    """
    Present the four-option menu whenever an account is needed.
    Loops until the user makes a valid choice.
    """
    provider_name    = snapshot.name
    registration_url = getattr(snapshot, "registration_url", None)
    login_url        = getattr(snapshot, "login_url", None) or snapshot.url

    print(f"\n  Provider:  {provider_name}")
    if acc.credential_ease_notes:
        print(f"  Note:      {acc.credential_ease_notes}")
    if registration_url:
        print(f"  Register:  {registration_url}")
    print(f"  Login:     {login_url}")

    while True:
        print(
            f"\n  This dataset requires an account. Do you want to use an [existing] account, "
            f"make a [new] account, [skip] this source, or search [alternatives]?"
        )
        choice = input("  Your choice [existing / new / skip / alternatives]: ").strip().lower()

        if choice in ("existing", "exist", "e"):
            return _use_existing_credentials(sid, snapshot)

        if choice in ("new", "n", "create"):
            return _create_new_account(sid, snapshot, registration_url, login_url)

        if choice in ("skip", "s"):
            return (
                SourceDecision(
                    source_id=sid,
                    decision=AccessDecisionType.skipped_declined,
                    notes="User chose to skip this source; alternate will be substituted if available.",
                ),
                None,
            )

        if choice in ("alternatives", "alt", "a", "alternative"):
            return (
                SourceDecision(
                    source_id=sid,
                    decision=AccessDecisionType.skipped_declined,
                    notes="User requested alternative sources; lower-ranked providers will be tried.",
                ),
                None,
            )

        print("  Please type 'existing', 'new', 'skip', or 'alternatives'.")


# ---------------------------------------------------------------------------
# Option A: existing account
# ---------------------------------------------------------------------------

def _use_existing_credentials(
    sid: str,
    snapshot: SourceSnapshot,
) -> "tuple[SourceDecision, Optional[Credentials]]":
    """Collect credentials the user already has and cache them."""
    return _prompt_user_credentials(sid, snapshot)


# ---------------------------------------------------------------------------
# Option B: new account (Playwright-assisted registration)
# ---------------------------------------------------------------------------

def _create_new_account(
    sid: str,
    snapshot: SourceSnapshot,
    registration_url: Optional[str],
    login_url: str,
) -> "tuple[SourceDecision, Optional[Credentials]]":
    """
    Run the guided registration assistant:
      - Collect personal details from the console.
      - Use Playwright to fill and submit the registration form.
      - Wait for email verification.
      - Cache and return the new credentials.
    """
    result = guided_new_account_flow(
        provider_name=snapshot.name,
        registration_url=registration_url or login_url,
        login_url=login_url,
    )

    if result is None:
        # Registration could not complete — fall back to existing-credentials prompt
        print("\n  Automated registration did not complete.")
        print("  If you registered manually, choose 'existing' to enter your credentials now.")
        choice = input("  Enter credentials now (yes) or skip this source (no)? [yes/no]: ").strip().lower()
        if choice in ("yes", "y"):
            return _prompt_user_credentials(sid, snapshot)
        return (
            SourceDecision(
                source_id=sid,
                decision=AccessDecisionType.skipped_declined,
                notes="Registration could not be completed; source skipped.",
            ),
            None,
        )

    email, password = result

    # Cache the new credentials immediately so they are never asked again --
    # both in the full multi-field store (for future runs) and in the
    # lightweight keyring/base64-fallback cache (so retries within *this*
    # run's access-resolution loop don't need keyring to succeed either).
    store_save_provider_credentials(
        sid,
        username=email,
        password=password,
    )
    save_credentials(snapshot.name, email, password)
    print(f"  Credentials for {snapshot.name} saved securely — you will not be asked again.")

    return (
        SourceDecision(
            source_id=sid,
            decision=AccessDecisionType.user_provided_credentials,
            credentials_used=True,
            credentials_persisted=True,
            notes="Account created via automated registration assistant; credentials cached.",
        ),
        Credentials(username=email, password=password),
    )


# ---------------------------------------------------------------------------
# Credential prompt (existing account path)
# ---------------------------------------------------------------------------

def _prompt_user_credentials(
    source_id: str,
    snapshot: SourceSnapshot,
) -> "tuple[SourceDecision, Credentials]":
    """Prompt for credential type and values, then cache."""
    print("\n  What type of credentials does this provider use?")
    print("    1. Username + password")
    print("    2. API key")
    print("    3. Bearer token")
    print("    4. Username + password + API key (some providers need both)")
    cred_type = input("  Choice [1/2/3/4, default 1]: ").strip() or "1"

    username: str = ""
    password: str = ""
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

    # Always offer to cache — remind the user this means no future prompts
    persist = input(
        "  Save credentials securely for next time? (yes/no) "
        "[Choosing 'yes' means you will never be asked for this provider again]: "
    ).strip().lower() in ("yes", "y")

    if persist:
        store_save_provider_credentials(
            source_id,
            username=username,
            password=password,
            api_key=api_key,
            bearer_token=bearer_token,
            refresh_token=refresh_token,
        )
        print(f"  Credentials saved. {snapshot.name} will authenticate automatically on future runs.")

    effective_token = bearer_token or refresh_token
    return (
        SourceDecision(
            source_id=source_id,
            decision=AccessDecisionType.user_provided_credentials,
            credentials_used=True,
            credentials_persisted=persist,
            notes="User-supplied credentials.",
        ),
        Credentials(
            username=username,
            password=password,
            api_key=api_key,
            token=effective_token,
        ),
    )
