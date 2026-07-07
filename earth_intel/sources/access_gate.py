"""
sources/access_gate.py — Human Approval Gate for Agent 3.

This module handles ALL human interactions needed before Agent 4 is allowed
to retrieve data from a source. It enforces two rules:

  Rule 1 — REGISTRATION gate:
    If a source requires login, Agent 3 MUST ask the user if they have
    an account. If they don't, it offers the registration URL and waits.
    Agent 3 NEVER stores credentials — it only records user consent.

  Rule 2 — PAYMENT gate:
    If a source requires payment, Agent 3 MUST show the price estimate
    and ask for EXPLICIT confirmation ("YES" typed) before proceeding.
    Agent 4 NEVER initiates any payment without this confirmation.

Flow called from agent3_discovery.py Phase 8:

    gate = AccessGate()
    result = gate.run(accepted_sources)
    # result.approved_sources  → pass to Agent 4
    # result.rejected_sources  → log and skip

The gate also collects and returns user credentials so Agent 4 can
use them without asking again.

SECURITY NOTE:
  Credentials are held in memory only for the duration of this run.
  They are never written to disk, never logged, and never stored in
  Qdrant. Agent 4 receives them via the AccessGateResult object which
  is discarded after the session.
"""

from dataclasses import dataclass, field
from getpass import getpass
from typing import Dict, List, Optional, Tuple

from models.discovery_schemas import CandidateSource, AccessType


# ------------------------------------------------------------------ #
# Data structures                                                       #
# ------------------------------------------------------------------ #

@dataclass
class CredentialSet:
    """
    Credentials for one provider.
    Held in-memory only. Never serialized.
    """
    source_id:  str
    username:   Optional[str] = None
    password:   Optional[str] = None
    api_key:    Optional[str] = None
    token:      Optional[str] = None     # OAuth token if provider uses OAuth
    extra:      Dict[str, str] = field(default_factory=dict)


@dataclass
class AccessGateResult:
    """
    Result of running the access gate against a list of candidates.

    approved_sources  → Agent 4 may retrieve from these
    rejected_sources  → skipped (user declined or no payment approval)
    credentials       → keyed by source_id; Agent 4 uses these for auth
    payment_confirmed → source_ids where user confirmed payment
    """
    approved_sources:   List[CandidateSource]
    rejected_sources:   List[CandidateSource]
    credentials:        Dict[str, CredentialSet]   # source_id → creds
    payment_confirmed:  List[str]                  # source_ids


# ------------------------------------------------------------------ #
# Separator helpers                                                     #
# ------------------------------------------------------------------ #

def _line(char: str = "-", width: int = 70) -> None:
    print(char * width)

def _header(title: str) -> None:
    _line("=")
    print(f"  {title}")
    _line("=")

def _ask_yes_no(prompt: str) -> bool:
    while True:
        ans = input(f"{prompt} (yes/no): ").strip().lower()
        if ans in ("yes", "y"):
            return True
        if ans in ("no", "n"):
            return False
        print("  Please type yes or no.")


# ------------------------------------------------------------------ #
# AccessGate                                                            #
# ------------------------------------------------------------------ #

class AccessGate:
    """
    Runs the human approval gate for a list of CandidateSource objects.

    For each source:
      - Free sources   → auto-approved, no interaction
      - Registration   → ask user if they have / want to create an account,
                         then collect credentials
      - Paid           → show price, ask for EXPLICIT payment confirmation,
                         then collect credentials
      - Browse-only    → auto-rejected (not downloadable)
    """

    def run(self, candidates: List[CandidateSource]) -> AccessGateResult:
        """
        Main entry point. Call this from Agent 3 Phase 8.

        Returns an AccessGateResult with approved/rejected sources
        and any credentials the user provided.
        """
        approved:          List[CandidateSource]    = []
        rejected:          List[CandidateSource]    = []
        credentials:       Dict[str, CredentialSet] = {}
        payment_confirmed: List[str]                = []

        # Separate by access type
        free_sources        = [c for c in candidates if
                               c.access_type == AccessType.free and not c.requires_payment]
        registration_sources = [c for c in candidates if
                                c.requires_login and not c.requires_payment]
        paid_sources        = [c for c in candidates if c.requires_payment]

        # ── Free: auto-approve ──────────────────────────────────────
        if free_sources:
            approved.extend(free_sources)

        # ── Registration: ask user ──────────────────────────────────
        if registration_sources:
            _header("LOGIN REQUIRED — These sources need a free account")
            for source in registration_sources:
                creds, ok = self._handle_registration(source)
                if ok:
                    approved.append(source)
                    if creds:
                        credentials[source.source_id] = creds
                else:
                    rejected.append(source)

        # ── Paid: show price, ask for explicit confirmation ─────────
        if paid_sources:
            _header("PAYMENT REQUIRED — These sources cost money")
            for source in paid_sources:
                creds, ok = self._handle_payment(source)
                if ok:
                    approved.append(source)
                    payment_confirmed.append(source.source_id)
                    if creds:
                        credentials[source.source_id] = creds
                else:
                    rejected.append(source)

        # ── Summary ────────────────────────────────────────────────
        print()
        _line()
        print(f"ACCESS GATE SUMMARY")
        _line()
        print(f"  Approved : {len(approved)} source(s)")
        print(f"  Rejected : {len(rejected)} source(s)")
        _line()

        return AccessGateResult(
            approved_sources=approved,
            rejected_sources=rejected,
            credentials=credentials,
            payment_confirmed=payment_confirmed,
        )

    # ---------------------------------------------------------------- #
    # Registration flow                                                  #
    # ---------------------------------------------------------------- #

    def _handle_registration(
        self,
        source: CandidateSource,
    ) -> Tuple[Optional[CredentialSet], bool]:
        """
        Ask the user if they have an account for this source and
        optionally collect credentials. Returns (credentials, approved).
        """
        print()
        _line()
        print(f"  SOURCE    : {source.name}")
        print(f"  URL       : {source.url}")
        if source.login_url:
            print(f"  REGISTER  : {source.login_url}")
        _line()
        print("  This source requires a FREE account to download data.")
        print()

        # Does the user have an account?
        has_account = _ask_yes_no("  Do you have an account for this source?")

        if not has_account:
            if source.login_url:
                print()
                print(f"  You can create a free account at:")
                print(f"  → {source.login_url}")
                print()
                create = _ask_yes_no("  Would you like to continue after creating the account?")
                if not create:
                    print(f"  Skipping {source.name}.")
                    return None, False
                # User says they'll create one — continue to credential collection
            else:
                print(f"  Skipping {source.name}.")
                return None, False

        # Collect credentials
        print()
        print(f"  Enter your credentials for {source.name}.")
        print(f"  (These are used only this session and never stored.)")
        print()

        creds = self._collect_credentials(source)
        if creds is None:
            print(f"  No credentials entered. Skipping {source.name}.")
            return None, False

        print(f"  ✓ Credentials accepted. {source.name} approved for retrieval.")
        return creds, True

    # ---------------------------------------------------------------- #
    # Payment flow                                                       #
    # ---------------------------------------------------------------- #

    def _handle_payment(
        self,
        source: CandidateSource,
    ) -> Tuple[Optional[CredentialSet], bool]:
        """
        Show price estimate, ask for EXPLICIT payment confirmation,
        then collect credentials. Returns (credentials, approved).
        """
        print()
        _line("!")
        print(f"  SOURCE    : {source.name}")
        print(f"  URL       : {source.url}")
        print(f"  PRICE     : {source.price_estimate or 'Unknown — check pricing page'}")
        if source.login_url:
            print(f"  PRICING   : {source.login_url}")
        _line("!")
        print()
        print("  ⚠  This source COSTS MONEY.")
        print("     Agent 4 will NOT initiate any purchase without your explicit")
        print("     confirmation here AND again before the actual transaction.")
        print()

        # Hard confirmation — must type YES in caps or mixed
        confirmed = self._ask_payment_confirmation(source.name)
        if not confirmed:
            print(f"  Payment declined. Skipping {source.name}.")
            return None, False

        # Collect credentials for the paid account
        print()
        print(f"  Enter your API key or credentials for {source.name}.")
        print(f"  (These are used only this session and never stored.)")
        print()

        creds = self._collect_credentials(source)
        if creds is None:
            print(f"  No credentials entered. Skipping {source.name}.")
            return None, False

        print(f"  ✓ Payment confirmed and credentials accepted. "
              f"{source.name} approved for retrieval.")
        print(f"    Agent 4 will ask for final confirmation before any purchase.")
        return creds, True

    def _ask_payment_confirmation(self, source_name: str) -> bool:
        """Requires the user to type YES (case-insensitive) to confirm."""
        print(f"  Type YES to authorize payment for {source_name}.")
        print(f"  Type anything else to skip this source.")
        ans = input("  → ").strip().upper()
        return ans == "YES"

    # ---------------------------------------------------------------- #
    # Credential collection                                              #
    # ---------------------------------------------------------------- #

    def _collect_credentials(
        self,
        source: CandidateSource,
    ) -> Optional[CredentialSet]:
        """
        Prompt the user for credentials appropriate to the source's api_type.
        Returns None if the user provides nothing.
        """
        from models.discovery_schemas import APIType

        creds = CredentialSet(source_id=source.source_id)

        # API-key style (e.g. Tavily, Planet, Tomorrow.io, Spire)
        if source.api_type in (APIType.rest,) and source.requires_payment:
            api_key = getpass("  API key: ").strip()
            if not api_key:
                return None
            creds.api_key = api_key
            return creds

        # CMR (NASA Earthdata — username + password)
        if source.api_type.value == "cmr" or "earthdata" in source.source_id:
            username = input("  Earthdata username: ").strip()
            if not username:
                return None
            password = getpass("  Earthdata password: ").strip()
            if not password:
                return None
            creds.username = username
            creds.password = password
            return creds

        # Generic: try username+password first, offer API key as alternative
        print("  (Press Enter to skip a field if not applicable)")
        username = input("  Username / email: ").strip()
        if username:
            password = getpass("  Password: ").strip()
            if password:
                creds.username = username
                creds.password = password
                return creds

        # Fallback — API key
        api_key = getpass("  API key (if username/password not applicable): ").strip()
        if api_key:
            creds.api_key = api_key
            return creds

        return None


# ------------------------------------------------------------------ #
# Convenience function called from agent3_discovery.py                 #
# ------------------------------------------------------------------ #

def run_access_gate(
    candidates: List[CandidateSource],
) -> AccessGateResult:
    """
    Convenience wrapper. Call this instead of instantiating AccessGate directly.

    Usage in agent3_discovery.py Phase 8:

        from sources.access_gate import run_access_gate

        gate_result = run_access_gate(accepted_sources)
        approved    = gate_result.approved_sources
        credentials = gate_result.credentials   # pass to Agent 4
    """
    gate = AccessGate()
    return gate.run(candidates)
