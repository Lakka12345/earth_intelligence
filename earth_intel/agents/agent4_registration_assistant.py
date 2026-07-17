"""
Agent 4 — Registration Assistant  (v2 — Intelligent Rewrite)
=============================================================

Handles the "new account" path when a provider requires registration.
Uses Playwright (headless browser) to:
  1. Accept personal registration strings from the orchestrator, or collect
     them once cleanly from the terminal if they are absent.
  2. Navigate to the provider's registration page entirely in the background
     (headless=True by default — no browser windows open or flicker).
  3. Intelligently locate form fields via a fuzzy, multi-strategy fallback
     cascade rather than brittle hardcoded CSS/ID selectors.
  4. Monitor the DOM continuously for CAPTCHA widgets; if one is found,
     execute a "Headful Veto": spawn a temporary visible browser context
     so the user can solve the challenge, then resume automated tracking.
  5. Submit the form using a robust fallback chain.
  6. Pause for email verification and return credentials to the caller.

Install Playwright if needed:
  pip install playwright
  playwright install chromium

Design principles
-----------------
* Headless-by-default  — no UI chrome unless a CAPTCHA veto is triggered.
* Fuzzy locator cascade — label → placeholder → attribute regex → role.
  Never uses bare #id or .class selectors for dynamic sites.
* Captcha-aware        — detects reCAPTCHA / hCaptcha / Cloudflare Turnstile
  and yields control to the user via a second visible context.
* Robust submit        — button text matching → type=submit → Enter key fallback.
* Fully annotated      — every non-trivial decision is explained inline.
"""

from __future__ import annotations

import getpass
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# § 1b  Credential broker — keyring-backed master identity
# ---------------------------------------------------------------------------
#
# Replaces local JSON/obfuscated storage. Credentials are stored in the OS
# credential vault (Windows Credential Manager / macOS Keychain / Linux
# Secret Service) under a single service namespace, "earth_intel", so every
# connector/registration flow shares one master identity instead of asking
# the user to re-enter details per site.
#
# SECURITY NOTE: this deliberately stores ONE email+password pair reused
# across every provider that gets self-registered. That's convenient, but
# it also means a credential leak at any single provider exposes the same
# password everywhere else it was used. That tradeoff is inherent to the
# "one master identity" design as specified; if that's a concern later,
# switch to a per-provider keyring username (e.g. f"site:{provider_name}")
# instead of the fixed "master_email"/"master_password" keys below.

_KEYRING_SERVICE = "earth_intel"
_KEYRING_EMAIL_KEY = "master_email"
_KEYRING_PASSWORD_KEY = "master_password"


def _keyring_available() -> bool:
    try:
        import keyring  # noqa: F401
        return True
    except ImportError:
        return False


def get_stored_master_credentials() -> Optional[Tuple[str, str]]:
    """Return (email, password) from the OS keyring, or None if unset/unavailable."""
    if not _keyring_available():
        return None
    import keyring
    try:
        email = keyring.get_password(_KEYRING_SERVICE, _KEYRING_EMAIL_KEY)
        password = keyring.get_password(_KEYRING_SERVICE, _KEYRING_PASSWORD_KEY)
    except Exception as exc:
        print(f"  ⚠  Keyring read failed ({exc}); falling back to interactive prompt.")
        return None
    if email and password:
        return email, password
    return None


def save_master_credentials(email: str, password: str) -> None:
    """Persist (email, password) to the OS keyring. Best-effort; never raises."""
    if not _keyring_available():
        print("  ⚠  'keyring' package not installed -- credentials will not be "
              "remembered for next time. Run: pip install keyring")
        return
    import keyring
    try:
        keyring.set_password(_KEYRING_SERVICE, _KEYRING_EMAIL_KEY, email)
        keyring.set_password(_KEYRING_SERVICE, _KEYRING_PASSWORD_KEY, password)
    except Exception as exc:
        print(f"  ⚠  Could not save credentials to keyring ({exc}).")


def broker_master_credentials(provider_name: str) -> Tuple[str, str]:
    """
    The credential broker workflow:
      1. Check the keyring for stored master credentials.
      2. If found, ask "Use these (y/n)?" -- 'y' reuses them, 'n' collects
         new ones and overwrites the stored pair.
      3. If the keyring is empty, collect once and save.

    Returns (email, password). Always saves back to keyring on any path
    that produces new credentials.
    """
    stored = get_stored_master_credentials()

    if stored:
        stored_email, stored_password = stored
        print(f"\n  I have stored credentials for {stored_email}.")
        while True:
            choice = input("  Use these (y/n)? ").strip().lower()
            if choice in ("y", "yes"):
                return stored_email, stored_password
            if choice in ("n", "no"):
                break
            print("  Please answer y or n.")

    # No stored credentials, or user declined the stored pair -- collect new.
    print(f"\n  Enter new master credentials to use for {provider_name} "
          f"(and future registrations):")
    email = ""
    while not email:
        email = input("  Email address (required): ").strip()
        if not email:
            print("  ✖  Email is required.")
    password = ""
    while not password:
        password = getpass.getpass("  Choose a password (hidden, required): ")
        confirm = getpass.getpass("  Confirm password (hidden):            ")
        if password != confirm:
            print("  ✖  Passwords do not match — try again.")
            password = ""
        elif len(password) < 8:
            print("  ✖  Password must be at least 8 characters.")
            password = ""

    save_master_credentials(email, password)
    return email, password


# ---------------------------------------------------------------------------
# § 1  User-detail payload
# ---------------------------------------------------------------------------

@dataclass
class RegistrationDetails:
    """
    Universal registration payload.  The orchestrator may inject this directly;
    the terminal collector below is the fallback for interactive runs.
    """
    email: str
    password: str
    first_name: str = ""
    last_name: str = ""
    organization: str = ""
    country: str = ""
    # Derived automatically from the email local-part when not supplied.
    username: str = ""
    # Arbitrary key→value pairs for site-specific fields (e.g. "job_title").
    extra: Dict[str, str] = field(default_factory=dict)


def collect_registration_details(
    provider_name: str,
    prefill: Optional[RegistrationDetails] = None,
) -> RegistrationDetails:
    """
    Prompt the user once for any details that the orchestrator did not supply.

    Parameters
    ----------
    provider_name : str
        Human-readable name shown in prompts.
    prefill : RegistrationDetails or None
        Partial or complete detail object from the orchestrator.  Any
        non-empty field is used as-is; missing fields are requested from the
        terminal.  Pass None to request everything interactively.
    """
    p = prefill  # shorthand

    print(f"\n{'─'*60}")
    print(f"  Registration assistant — {provider_name}")
    print(f"{'─'*60}")
    print("  Details already supplied by the orchestrator will be skipped.")
    print("  Press Enter to skip optional fields.\n")

    # ── Email + Password (required) — via keyring credential broker ────────
    #    Orchestrator-supplied values (prefill) still take priority, since
    #    those represent an explicit per-call override; otherwise we go
    #    through the stored-master-identity broker rather than prompting
    #    for a brand new email/password on every single site.
    email = (p.email if p and p.email else "").strip()
    password = (p.password if p and p.password else "").strip()
    if email and password:
        print("  Email and password supplied by orchestrator — skipping broker.")
    else:
        broker_email, broker_password = broker_master_credentials(provider_name)
        email = email or broker_email
        password = password or broker_password

    # ── Optional fields ────────────────────────────────────────────────────
    def _get(attr: str, prompt: str) -> str:
        if p and getattr(p, attr, ""):
            return getattr(p, attr).strip()
        return input(f"  {prompt} (optional): ").strip()

    first_name   = _get("first_name",   "First name")
    last_name    = _get("last_name",    "Last name")
    organization = _get("organization", "Organisation / institution")
    country      = _get("country",      "Country")

    # Derive a username from the email local-part unless one was provided.
    username = (p.username if p and p.username else "").strip()
    if not username:
        username = email.split("@")[0]

    return RegistrationDetails(
        email=email,
        password=password,
        first_name=first_name,
        last_name=last_name,
        organization=organization,
        country=country,
        username=username,
        extra=p.extra if p else {},
    )


# ---------------------------------------------------------------------------
# § 2  Field-hint tables
# ---------------------------------------------------------------------------

# Each entry: (RegistrationDetails attribute, [hint tokens ordered by priority])
# Tokens are matched case-insensitively against: label text, placeholder,
# name, id, aria-label, and autocomplete attributes.
_FIELD_HINTS: List[Tuple[str, List[str]]] = [
    ("email",        ["email", "e-mail", "mail"]),
    ("username",     ["username", "user_name", "user", "login", "account", "handle"]),
    ("first_name",   ["first_name", "firstname", "given_name", "givenname", "first", "fname", "forename"]),
    ("last_name",    ["last_name", "lastname", "family_name", "familyname", "surname", "last", "lname"]),
    ("organization", ["organization", "organisation", "org", "institution", "affiliation", "company", "institute"]),
    ("country",      ["country", "nation", "location", "region"]),
    ("password",     ["new_password", "newpassword", "password", "passwd", "pass", "pwd"]),
]

# Confirm-password field hints — filled with the same password value.
_CONFIRM_HINTS: List[str] = [
    "confirm_password", "confirmpassword", "password_confirm", "passwordconfirm",
    "password2", "passwd2", "retype", "repeat", "verify", "confirmation",
    "confirm", "repassword",
]

# DOM signals that indicate a CAPTCHA widget is present.
_CAPTCHA_SIGNALS: List[str] = [
    "recaptcha", "hcaptcha", "cf-turnstile", "turnstile",
    "captcha", "challenge-form", "robot", "g-recaptcha",
]

# Button / link text patterns that indicate a submit action.
_SUBMIT_LABELS: List[str] = [
    "register", "sign up", "signup", "create account", "create my account",
    "get started", "join", "submit", "continue", "next", "proceed",
]


# ---------------------------------------------------------------------------
# § 3  Playwright availability guard
# ---------------------------------------------------------------------------

def _playwright_available() -> bool:
    """Return True only when both `playwright` and `chromium` are installed."""
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
        # Probe for the chromium executable without launching it.
        with sync_playwright() as pw:
            _ = pw.chromium.executable_path  # raises if not installed
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# § 4  Core field-filling engine (fuzzy, fallback-driven)
# ---------------------------------------------------------------------------

def _verify_and_refill(page, locator, value: str, max_attempts: int = 3, timeout_ms: int = 800) -> bool:
    """
    Read the value back from the browser after filling and compare against
    the intended string. Some sites truncate or drop characters under fast
    automated fill (JS onChange handlers firing mid-keystroke, React
    controlled-input lag, etc.) -- .fill() alone does not guarantee the DOM
    ends up holding the full string. Re-fill up to max_attempts times until
    the read-back matches exactly.

    Returns True once the field's actual value matches `value`, False if it
    still doesn't match after all attempts (caller should not proceed to
    submit in that case).
    """
    first = locator.first
    for attempt in range(1, max_attempts + 1):
        try:
            actual = first.input_value(timeout=timeout_ms)
        except Exception:
            # Not an <input>/<textarea>/<select> (e.g. contenteditable) --
            # input_value() isn't applicable; treat the original fill as
            # trusted since we can't verify it this way.
            return True
        if actual == value:
            return True
        # Mismatch (likely truncation) -- clear and re-fill.
        try:
            first.fill("", timeout=timeout_ms)
            first.fill(value, timeout=timeout_ms)
        except Exception:
            return False
    try:
        return first.input_value(timeout=timeout_ms) == value
    except Exception:
        return False


def _try_fill(page, locator, value: str, timeout_ms: int = 800) -> bool:
    """
    Attempt to fill *locator* with *value*.

    Returns True if the element was found, visible, filled without error,
    AND the read-back value matches exactly (see _verify_and_refill).
    Uses a short timeout so we fail fast and try the next strategy.
    """
    try:
        # `count()` without arguments counts all matching nodes.
        if locator.count() == 0:
            return False
        first = locator.first
        # is_visible() may raise if the element is in a detached frame.
        if not first.is_visible(timeout=timeout_ms):
            return False
        first.scroll_into_view_if_needed(timeout=timeout_ms)
        first.click(timeout=timeout_ms)   # focus before fill
        first.fill(value, timeout=timeout_ms)
        return _verify_and_refill(page, locator, value, timeout_ms=timeout_ms)
    except Exception:
        return False


def _fill_field_fuzzy(page, hint_tokens: List[str], value: str) -> bool:
    """
    Try to locate and fill a single logical field using a cascade of
    Playwright strategies for the given hint tokens.

    Strategy order (from most semantic to most brittle):
      1. get_by_label()         — matches visible <label> text
      2. get_by_placeholder()   — matches placeholder= attribute
      3. Attribute regex        — name=, id=, aria-label=, autocomplete=
      4. get_by_role("textbox") filtered by accessible name

    Returns True as soon as any strategy succeeds.
    """
    for hint in hint_tokens:
        # ── Strategy 1: label text (semantic, most reliable) ──────────────
        for loc in [
            page.get_by_label(hint, exact=False),
            # Some sites wrap labels in fieldsets; also try role=group.
        ]:
            if _try_fill(page, loc, value):
                return True

        # ── Strategy 2: placeholder attribute ─────────────────────────────
        loc = page.get_by_placeholder(hint, exact=False)
        if _try_fill(page, loc, value):
            return True

        # ── Strategy 3: attribute-level CSS (case-insensitive via `i`) ────
        attr_selectors = [
            f"input[name*='{hint}' i]",
            f"input[id*='{hint}' i]",
            f"input[aria-label*='{hint}' i]",
            f"input[autocomplete='{hint}']",
            f"textarea[name*='{hint}' i]",
            f"textarea[id*='{hint}' i]",
            # Some React/Vue components use data attributes.
            f"input[data-field*='{hint}' i]",
            f"input[data-name*='{hint}' i]",
        ]
        for sel in attr_selectors:
            try:
                loc = page.locator(sel)
                if _try_fill(page, loc, value):
                    return True
            except Exception:
                continue

        # ── Strategy 4: accessible-name role match ─────────────────────────
        #    Works for custom component libraries that set aria-label but not
        #    a visible <label> element.
        try:
            loc = page.get_by_role("textbox", name=re.compile(hint, re.IGNORECASE))
            if _try_fill(page, loc, value):
                return True
        except Exception:
            pass

    return False  # All strategies exhausted for this field.


def _fill_form(page, details: RegistrationDetails) -> Tuple[List[str], List[str]]:
    """
    Drive the fuzzy field-filling engine for every logical field.

    Returns
    -------
    filled   : list of human-readable field names that were successfully filled.
    unfilled : list of fields for which no matching input was found.
    """
    filled: List[str] = []
    unfilled: List[str] = []

    values: Dict[str, str] = {
        "email":        details.email,
        "username":     details.username,
        "first_name":   details.first_name,
        "last_name":    details.last_name,
        "organization": details.organization,
        "country":      details.country,
        "password":     details.password,
    }

    for field_key, hint_tokens in _FIELD_HINTS:
        value = values.get(field_key, "").strip()
        if not value:
            # Skip optional fields the user left blank — don't inject empty strings.
            continue

        display = field_key.replace("_", " ").title()
        if _fill_field_fuzzy(page, hint_tokens, value):
            filled.append(display)
        else:
            unfilled.append(display)

    # ── Confirm-password field ────────────────────────────────────────────
    # We try confirm-password separately because it shouldn't be in _FIELD_HINTS
    # (it shares the same value as password, not a distinct payload key).
    if details.password:
        _fill_field_fuzzy(page, _CONFIRM_HINTS, details.password)
        # We don't report this in filled/unfilled — it's supplementary.

    # ── Extra / site-specific fields ─────────────────────────────────────
    for extra_key, extra_val in details.extra.items():
        if extra_val:
            _fill_field_fuzzy(page, [extra_key], extra_val)

    return filled, unfilled


# ---------------------------------------------------------------------------
# § 5  CAPTCHA detection
# ---------------------------------------------------------------------------

def _detect_captcha(page) -> bool:
    """
    Scan the page HTML *and* its iframes for known CAPTCHA signal strings.

    Checking raw HTML is fast and catches both inline and iframe-embedded
    widgets without needing to interact with cross-origin frames.
    """
    try:
        html = page.content().lower()
        if any(sig in html for sig in _CAPTCHA_SIGNALS):
            return True

        # Also inspect iframe src attributes visible from the main frame.
        frames = page.frames
        for frame in frames:
            try:
                if any(sig in (frame.url or "").lower() for sig in _CAPTCHA_SIGNALS):
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False


def _pre_submit_validation(page, details: RegistrationDetails) -> Tuple[bool, List[str]]:
    """
    Final gate immediately before clicking Submit/Register.

    Re-locates the required fields (email, password) via the same hint
    cascade used to fill them, and reads their current DOM value back one
    more time. This catches truncation/lag introduced *between* the fill
    step and now (e.g. a client-side formatter mutating the field after
    the fact) that per-field verification in _try_fill wouldn't see.

    Returns (ok, problems) -- ok is False if any required field's live
    value doesn't match what we intend to submit; problems lists which.
    """
    problems: List[str] = []
    required = [("email", details.email), ("password", details.password)]

    for field_key, expected in required:
        if not expected:
            continue
        hint_tokens = dict(_FIELD_HINTS)[field_key]
        located = False
        for hint in hint_tokens:
            for sel in [
                f"input[name*='{hint}' i]", f"input[id*='{hint}' i]",
                f"input[aria-label*='{hint}' i]",
                f"input[type='{'password' if field_key == 'password' else 'email'}']",
            ]:
                try:
                    loc = page.locator(sel)
                    if loc.count() == 0:
                        continue
                    first = loc.first
                    if not first.is_visible(timeout=500):
                        continue
                    located = True
                    if not _verify_and_refill(page, loc, expected, timeout_ms=800):
                        problems.append(field_key)
                    break
                except Exception:
                    continue
            if located:
                break
        # If we can't locate the field at all here, don't fail the gate --
        # _fill_form already logged it as unfilled; this pass only
        # re-validates fields that DO exist in the DOM right now.

    return (len(problems) == 0), problems


# ---------------------------------------------------------------------------
# § 6  Submit-button discovery and fallbacks
# ---------------------------------------------------------------------------

def _click_submit(page, details: RegistrationDetails) -> bool:
    """
    Attempt to submit the form using a fallback chain:
      1. type=submit input or button
      2. Role-based button with known submit-label text
      3. Keyboard Enter in the password field

    Returns True if a submit action was triggered (does not guarantee the
    form was accepted — page validation may still reject it).
    """
    from playwright.sync_api import TimeoutError as PWTimeout  # local import

    # ── Fallback A: semantic submit elements ──────────────────────────────
    submit_selectors = [
        "button[type='submit']",
        "input[type='submit']",
        "button[type='button'][form]",   # explicit form association
    ]
    for sel in submit_selectors:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0 and loc.is_visible(timeout=600):
                loc.scroll_into_view_if_needed(timeout=600)
                loc.click(timeout=4_000)
                return True
        except Exception:
            continue

    # ── Fallback B: button/link with submit-like label text ───────────────
    for label in _SUBMIT_LABELS:
        for strategy in [
            lambda l=label: page.get_by_role("button", name=re.compile(l, re.IGNORECASE)),
            lambda l=label: page.get_by_role("link",   name=re.compile(l, re.IGNORECASE)),
            lambda l=label: page.get_by_text(re.compile(l, re.IGNORECASE)),
        ]:
            try:
                loc = strategy()
                if loc.count() > 0 and loc.first.is_visible(timeout=600):
                    loc.first.scroll_into_view_if_needed(timeout=600)
                    loc.first.click(timeout=4_000)
                    return True
            except Exception:
                continue

    # ── Fallback C: press Enter in the password field ─────────────────────
    print("  ⚠  No submit button found — pressing Enter in the password field.")
    try:
        pw_loc = _fill_field_fuzzy.__globals__  # just for clarity — see below
        # Re-locate the password input using the same hint tokens.
        for hint in _FIELD_HINTS[-1][1]:   # "password" is the last entry
            for sel in [f"input[name*='{hint}' i]", f"input[type='password']"]:
                try:
                    loc = page.locator(sel).first
                    if loc.count() > 0 and loc.is_visible(timeout=600):
                        loc.press("Enter", timeout=2_000)
                        return True
                except Exception:
                    continue
    except Exception:
        pass

    # ── Fallback D: JavaScript form.submit() ─────────────────────────────
    print("  ⚠  Attempting JavaScript form.submit() as last resort.")
    try:
        page.evaluate("() => { const f = document.querySelector('form'); if (f) f.submit(); }")
        return True
    except Exception:
        pass

    return False   # All submit strategies failed.


# ---------------------------------------------------------------------------
# § 7  Headful Veto — bring up a visible context for CAPTCHA solving
# ---------------------------------------------------------------------------

def _headful_veto(pw, registration_url: str, details: RegistrationDetails) -> str:
    """
    Spawn a temporary *visible* (headful) Chromium context so the user can
    manually solve a CAPTCHA.  The function:
      1. Opens a new visible browser window.
      2. Re-navigates to the registration URL.
      3. Re-fills all non-sensitive fields automatically.
      4. Pauses execution until the user presses Enter.
      5. Captures and returns the final page URL.

    The visible context is closed cleanly afterwards.
    """
    print("\n" + "━"*60)
    print("  🔐  [Action Required] CAPTCHA detected!")
    print("      Pausing background flow and opening a visible browser…")
    print("━"*60)

    final_url = registration_url  # fallback if we can't read the page

    try:
        browser_hf = pw.chromium.launch(
            headless=False,
            args=["--start-maximized"],
        )
        ctx_hf = browser_hf.new_context(
            viewport=None,          # let the OS decide — maximised window
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            no_viewport=True,
        )
        page_hf = ctx_hf.new_page()
        page_hf.goto(registration_url, wait_until="domcontentloaded", timeout=30_000)
        time.sleep(2)  # allow JS-rendered content to settle

        # Pre-fill all non-CAPTCHA fields so the user only needs to tick the box.
        _fill_form(page_hf, details)
        print("\n  Fields have been pre-filled in the visible browser.")
        print("  ✅  Please solve the CAPTCHA in the browser window.")
        print("  ✅  Then SUBMIT the form in the browser.")
        input("\n  [Press Enter HERE after you have solved the CAPTCHA and submitted the form] ")

        # Capture the resulting URL / success state.
        try:
            final_url = page_hf.url
        except Exception:
            pass

        browser_hf.close()
    except Exception as exc:
        print(f"  ⚠  Headful veto browser error: {exc}")

    return final_url


# ---------------------------------------------------------------------------
# § 8  Main Playwright orchestration
# ---------------------------------------------------------------------------

def run_registration_assistant(
    provider_name: str,
    registration_url: str,
    details: RegistrationDetails,
) -> Tuple[bool, str]:
    """
    Open the registration page in a *headless* background context, fill
    the form intelligently, handle CAPTCHAs via Headful Veto, and submit.

    Parameters
    ----------
    provider_name     : Display name for logging.
    registration_url  : Full URL of the registration page.
    details           : Populated RegistrationDetails payload.

    Returns
    -------
    (success: bool, message: str)
    """
    if not _playwright_available():
        return False, (
            "Playwright is not installed.  "
            "Run: pip install playwright && playwright install chromium"
        )

    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

    print(f"\n  🤖  Launching background registration for {provider_name}…")
    print(f"  🌐  Target: {registration_url}")
    print("  👻  Running invisibly in headless mode.\n")

    try:
        with sync_playwright() as pw:
            # ── Launch headless Chromium ───────────────────────────────────
            browser = pw.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                ],
            )
            context = browser.new_context(
                viewport={"width": 1440, "height": 900},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                # Spoof navigator.webdriver to reduce bot-detection triggers.
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            )
            # Remove the `navigator.webdriver` JS property that headless sets.
            context.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
            page = context.new_page()

            # ── Navigate to registration page ─────────────────────────────
            try:
                page.goto(
                    registration_url,
                    wait_until="domcontentloaded",
                    timeout=30_000,
                )
            except PWTimeout:
                browser.close()
                return False, f"Page load timed out: {registration_url}"
            except Exception as exc:
                browser.close()
                return False, f"Navigation error: {exc}"

            # Allow JS-rendered forms to settle (SPA frameworks, lazy loaders).
            page.wait_for_load_state("networkidle", timeout=10_000)

            # ── CAPTCHA pre-check ─────────────────────────────────────────
            #    Check before filling so the veto can re-fill in the visible
            #    window rather than having two sets of filled values.
            captcha_present = _detect_captcha(page)

            final_url: str

            if captcha_present:
                # Close the headless browser — the veto opens its own.
                browser.close()
                final_url = _headful_veto(pw, registration_url, details)
                return True, f"CAPTCHA veto completed. Final URL: {final_url}"

            # ── Headless field-filling ─────────────────────────────────────
            filled, unfilled = _fill_form(page, details)

            if filled:
                print(f"  ✔  Auto-filled: {', '.join(filled)}")
            if unfilled:
                # Report but do NOT abort — submit anyway; site may not need them.
                print(f"  ⚠  Could not locate: {', '.join(unfilled)}")
                print("     (These fields may not exist on this particular form.)")

            # ── CAPTCHA post-fill check (some appear after interaction) ────
            time.sleep(1)
            captcha_present = _detect_captcha(page)
            if captcha_present:
                browser.close()
                final_url = _headful_veto(pw, registration_url, details)
                return True, f"CAPTCHA veto completed after fill. Final URL: {final_url}"

            # ── Pre-submission validation gate ──────────────────────────────
            #    Required by design: never click Submit/Register without
            #    confirming the live DOM value matches what we intended to
            #    send. Re-fills in place if a mismatch (truncation/lag) is
            #    found; only proceeds once every required field verifies.
            ok, problems = _pre_submit_validation(page, details)
            if not ok:
                print(f"  ✖  Pre-submit validation failed for: {', '.join(problems)} "
                      f"(value did not match after re-fill attempts).")
                browser.close()
                final_url = _headful_veto(pw, registration_url, details)
                return True, f"Pre-submit validation failed; handed off to headful veto. Final URL: {final_url}"

            # ── Submit the form ────────────────────────────────────────────
            submitted = _click_submit(page, details)

            if not submitted:
                print("  ✖  All submit strategies failed.")
                # Last resort: open visible window for manual submission.
                browser.close()
                final_url = _headful_veto(pw, registration_url, details)
                return True, f"Manual submit completed. Final URL: {final_url}"

            # ── Wait for post-submit page transition ───────────────────────
            try:
                page.wait_for_load_state("domcontentloaded", timeout=15_000)
            except PWTimeout:
                pass  # Form may have submitted; don't block on this.

            time.sleep(2)   # brief pause for confirmation UI to render

            # Capture the resulting URL (useful for debugging / success checks).
            try:
                final_url = page.url
            except Exception:
                final_url = registration_url

            browser.close()

        print(f"\n  ✅  Form submission attempted. Post-submit URL: {final_url}")
        return True, f"Form submitted. Post-submit URL: {final_url}"

    except Exception as exc:
        return False, f"Unexpected browser-automation error: {exc}"


# ---------------------------------------------------------------------------
# § 9  Guided new-account flow (entry point called by orchestrator)
# ---------------------------------------------------------------------------

def guided_new_account_flow(
    provider_name: str,
    registration_url: str,
    login_url: Optional[str],
    prefill_details: Optional[RegistrationDetails] = None,
) -> Optional[Tuple[str, str]]:
    """
    Orchestrate the complete new-account path.

    Steps
    -----
    1. Validate that a registration URL is available.
    2. Collect (or accept from orchestrator) personal registration details.
    3. Run Playwright to fill and submit the form in headless mode.
       — CAPTCHA triggers a Headful Veto automatically.
    4. Pause for optional email-verification step.
    5. Return (email, password) for immediate login by the caller.

    Parameters
    ----------
    provider_name      : Display name for the provider.
    registration_url   : Direct URL to the registration / sign-up form.
    login_url          : Fallback URL used when registration_url is empty.
    prefill_details    : Optional pre-populated payload from the orchestrator.
                         Any missing fields are requested from the terminal.

    Returns
    -------
    (email, password) tuple on success, or None if the user aborts.
    """
    # ── Step 1: Resolve registration URL ──────────────────────────────────
    reg_url = (registration_url or login_url or "").strip()
    if not reg_url:
        print(f"\n  ✖  No registration URL is available for {provider_name}.")
        print("     Please register manually at the provider's website, then")
        print("     choose 'existing account' on the next prompt.")
        return None

    # ── Step 2: Collect / merge registration details ──────────────────────
    #    collect_registration_details() skips any fields already in prefill.
    details = collect_registration_details(provider_name, prefill=prefill_details)

    # ── Step 3: Automated browser registration ─────────────────────────────
    #    Runs headless; CAPTCHA triggers the Headful Veto internally.
    success, message = run_registration_assistant(
        provider_name=provider_name,
        registration_url=reg_url,
        details=details,
    )

    if not success:
        print(f"\n  ✖  Automated registration could not complete:")
        print(f"     {message}")
        print(f"\n  Options:")
        print(f"    • Register manually at: {reg_url}")
        print("    • Choose 'existing account' on the next prompt to use")
        print("      credentials you registered yourself.")
        return None

    # ── Step 4: Email verification pause ──────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  📬  Registration submitted for {provider_name}!")
    print(f"      Confirmation email sent to: {details.email}")
    print(f"{'─'*60}")
    print("  Please:")
    print("    1. Open your inbox.")
    print("    2. Click the verification / activation link.")
    print("    3. Return here and press Enter to continue.\n")
    input("  [Press Enter after clicking the verification link] ")

    # ── Step 5: Return credentials to the orchestrator ────────────────────
    print(f"\n  ✅  Proceeding with account: {details.email}")
    return details.email, details.password


# ---------------------------------------------------------------------------
# § 10  Standalone smoke-test entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Quick interactive smoke-test.  Run directly to verify Playwright works
    and the fuzzy-fill logic reaches the target page.

      python agents/agent4_registration_assistant.py
    """
    import sys

    print("Agent 4 — Registration Assistant (standalone test)")
    print("=" * 60)

    test_url = input("Enter a registration URL to test (or press Enter to skip): ").strip()
    if not test_url:
        print("No URL provided — exiting smoke-test.")
        sys.exit(0)

    test_provider = input("Provider name [Test Site]: ").strip() or "Test Site"

    result = guided_new_account_flow(
        provider_name=test_provider,
        registration_url=test_url,
        login_url=None,
        prefill_details=None,
    )

    if result:
        email, _ = result
        print(f"\n  Smoke-test complete.  Account email: {email}")
    else:
        print("\n  Smoke-test ended without credentials (user aborted or error).")
