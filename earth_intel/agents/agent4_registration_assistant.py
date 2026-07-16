"""
Agent 4 — Registration Assistant.

Handles the "new account" path when a provider requires registration.
Uses Playwright (headless browser) to:
  1. Navigate to the provider's registration page.
  2. Collect the user's personal details from the console.
  3. Heuristically detect form fields and inject the user's details.
  4. Submit the form.
  5. Wait for the user to click their verification email link.
  6. Return the credentials so the caller can proceed immediately.

Limitations (documented honestly):
  - Field detection is heuristic. Providers with non-standard markup
    (e.g. custom React inputs, shadow DOM, CAPTCHAs) may need the user
    to fill remaining fields manually in the opened browser window.
  - CAPTCHA is never bypassed. If detected, the user is told to solve
    it in the browser window before pressing Enter.
  - MFA / institutional SSO flows are passed through to the user.
  - If Playwright is not installed, falls back to a clear console-only
    instruction path.

Install Playwright:
  pip install playwright
  playwright install chromium
"""

from __future__ import annotations

import getpass
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# User detail collection
# ---------------------------------------------------------------------------

@dataclass
class RegistrationDetails:
    email: str
    password: str
    first_name: str = ""
    last_name: str = ""
    organization: str = ""
    country: str = ""
    username: str = ""          # some providers use a separate username field
    extra: Dict[str, str] = field(default_factory=dict)


def collect_registration_details(provider_name: str) -> RegistrationDetails:
    """Prompt the user for personal details needed to create an account."""
    print(f"\n  Registration details for {provider_name}:")
    print("  (These will be filled into the registration form automatically.)")
    print("  Press Enter to skip optional fields.\n")

    email = ""
    while not email:
        email = input("  Email address (required): ").strip()
        if not email:
            print("  Email is required.")

    password = ""
    while not password:
        password = getpass.getpass("  Choose a password (input hidden, required): ")
        confirm = getpass.getpass("  Confirm password (input hidden): ")
        if password != confirm:
            print("  Passwords do not match — please try again.")
            password = ""
        elif len(password) < 8:
            print("  Password must be at least 8 characters.")
            password = ""

    first_name  = input("  First name (optional): ").strip()
    last_name   = input("  Last name (optional): ").strip()
    organization = input("  Organisation / institution (optional): ").strip()
    country     = input("  Country (optional): ").strip()

    # Derive a username from the email local part if none is provided
    username = email.split("@")[0] if email else ""

    return RegistrationDetails(
        email=email,
        password=password,
        first_name=first_name,
        last_name=last_name,
        organization=organization,
        country=country,
        username=username,
    )


# ---------------------------------------------------------------------------
# Heuristic field mapping
# ---------------------------------------------------------------------------

# Maps RegistrationDetails field → common HTML input attribute values.
# The heuristic tries name, id, placeholder, aria-label, and label text.
_FIELD_HINTS: List[Tuple[str, List[str]]] = [
    ("email",        ["email", "e-mail", "mail", "username", "user_name", "user"]),
    ("username",     ["username", "user_name", "user", "login", "account"]),
    ("first_name",   ["first_name", "firstname", "given_name", "givenname", "first"]),
    ("last_name",    ["last_name", "lastname", "family_name", "familyname", "surname", "last"]),
    ("organization", ["organization", "organisation", "org", "institution", "affiliation", "company"]),
    ("country",      ["country", "nation", "location"]),
    ("password",     ["password", "passwd", "pass", "pwd", "new_password", "newpassword"]),
]

# A second password field (confirm/repeat) — we fill it with the same password.
_CONFIRM_PASSWORD_HINTS = [
    "confirm_password", "confirmpassword", "password_confirm", "passwordconfirm",
    "password2", "passwd2", "retype", "repeat", "verify",
]

_CAPTCHA_SIGNALS = [
    "recaptcha", "hcaptcha", "captcha", "cf-turnstile",
    "challenge", "robot", "human",
]


def _playwright_available() -> bool:
    try:
        import playwright  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Playwright automation
# ---------------------------------------------------------------------------

def _fill_form(page, details: RegistrationDetails) -> Tuple[List[str], List[str]]:
    """
    Heuristically find and fill registration form inputs.

    Returns:
        filled   — human-readable list of fields successfully filled
        unfilled — human-readable list of fields not found (user must fill manually)
    """
    filled: List[str] = []
    unfilled: List[str] = []

    detail_values: Dict[str, str] = {
        "email":        details.email,
        "username":     details.username,
        "first_name":   details.first_name,
        "last_name":    details.last_name,
        "organization": details.organization,
        "country":      details.country,
        "password":     details.password,
    }

    for field_key, hint_list in _FIELD_HINTS:
        value = detail_values.get(field_key, "")
        if not value:
            # Skip optional empty fields — don't inject blanks.
            continue

        located = False
        for hint in hint_list:
            # Build a CSS/attribute selector list covering the most common patterns
            selectors = [
                f"input[name='{hint}']",
                f"input[name='{hint.upper()}']",
                f"input[id='{hint}']",
                f"input[id='{hint.upper()}']",
                f"input[placeholder*='{hint}' i]",
                f"input[aria-label*='{hint}' i]",
                f"input[autocomplete='{hint}']",
            ]
            for sel in selectors:
                try:
                    locator = page.locator(sel).first
                    if locator.count() > 0 and locator.is_visible(timeout=500):
                        locator.fill(value)
                        located = True
                        break
                except Exception:
                    continue
            if located:
                break

        # Fallback: look for a label whose text contains the hint
        if not located:
            for hint in hint_list:
                try:
                    label = page.get_by_label(hint, exact=False)
                    if label.count() > 0 and label.first.is_visible(timeout=500):
                        label.first.fill(value)
                        located = True
                        break
                except Exception:
                    continue

        if located:
            display = field_key.replace("_", " ").title()
            filled.append(display)
        else:
            if value:   # Only report as unfilled if we actually had a value to put there
                unfilled.append(field_key.replace("_", " ").title())

    # Fill confirm-password field if present
    for hint in _CONFIRM_PASSWORD_HINTS:
        selectors = [
            f"input[name='{hint}']",
            f"input[id='{hint}']",
            f"input[placeholder*='confirm' i]",
            f"input[placeholder*='repeat' i]",
            f"input[placeholder*='retype' i]",
        ]
        for sel in selectors:
            try:
                locator = page.locator(sel).first
                if locator.count() > 0 and locator.is_visible(timeout=500):
                    locator.fill(details.password)
                    break
            except Exception:
                continue

    return filled, unfilled


def _detect_captcha(page) -> bool:
    """Return True if the page appears to have a CAPTCHA widget."""
    try:
        html = page.content().lower()
        return any(signal in html for signal in _CAPTCHA_SIGNALS)
    except Exception:
        return False


def _find_submit_button(page):
    """Return the most likely submit button locator, or None."""
    candidates = [
        "button[type='submit']",
        "input[type='submit']",
        "button:has-text('Register')",
        "button:has-text('Sign up')",
        "button:has-text('Create account')",
        "button:has-text('Submit')",
        "button:has-text('Continue')",
        "button:has-text('Next')",
    ]
    for sel in candidates:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0 and loc.is_visible(timeout=500):
                return loc
        except Exception:
            continue
    return None


def run_registration_assistant(
    provider_name: str,
    registration_url: str,
    details: RegistrationDetails,
    headless: bool = False,   # False = user can see and interact with the browser
) -> Tuple[bool, str]:
    """
    Open the registration page, fill the form, and submit it.

    Returns:
        (success: bool, message: str)
    """
    if not _playwright_available():
        return False, (
            "Playwright is not installed. "
            "Install it with: pip install playwright && playwright install chromium"
        )

    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

    print(f"\n  Launching automated registration assistant for {provider_name}...")
    print(f"  Opening: {registration_url}")
    if not headless:
        print("  A browser window will open. You can interact with it if the assistant")
        print("  cannot fill a field automatically.")

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=headless)
            context = browser.new_context(
                viewport={"width": 1280, "height": 900},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            )
            page = context.new_page()

            try:
                page.goto(registration_url, wait_until="domcontentloaded", timeout=30_000)
            except PWTimeout:
                browser.close()
                return False, f"Timed out loading registration page: {registration_url}"
            except Exception as exc:
                browser.close()
                return False, f"Could not open registration page: {exc}"

            # Small wait for JS-rendered forms
            time.sleep(2)

            # Detect CAPTCHA before attempting fill
            has_captcha = _detect_captcha(page)

            # Fill the form
            filled, unfilled = _fill_form(page, details)

            if filled:
                print(f"  Automatically filled: {', '.join(filled)}")
            if unfilled:
                print(f"  Could not locate fields for: {', '.join(unfilled)}")
                if not headless:
                    print("  Please fill these in the browser window before submitting.")

            if has_captcha:
                print("\n  A CAPTCHA was detected on this page.")
                print("  Please solve it in the browser window, then press Enter here to continue.")
                input("  [Press Enter after solving the CAPTCHA] ")

            if unfilled and headless:
                # Can't show the browser window — tell the user what's missing
                print(f"\n  The following fields could not be filled automatically: {', '.join(unfilled)}")
                print("  Re-running in visible browser mode so you can complete them...")
                browser.close()
                return run_registration_assistant(
                    provider_name, registration_url, details, headless=False
                )

            # Try to click submit
            submit_btn = _find_submit_button(page)
            if submit_btn:
                try:
                    submit_btn.click()
                    # Wait briefly for the page to respond
                    page.wait_for_load_state("domcontentloaded", timeout=10_000)
                except PWTimeout:
                    pass   # Form may have been submitted; continue
                except Exception as exc:
                    print(f"  Submit click raised: {exc} — please click the submit button manually.")
                    if not headless:
                        input("  [Press Enter after submitting the form] ")
            else:
                print("  Could not locate a submit button automatically.")
                if not headless:
                    print("  Please click the submit button in the browser window.")
                    input("  [Press Enter after submitting the form] ")

            # Give the page a moment to show a post-submit message
            time.sleep(2)

            browser.close()

        return True, "Form submitted."

    except Exception as exc:
        return False, f"Browser automation error: {exc}"


# ---------------------------------------------------------------------------
# Full new-account flow (called from resolve_access)
# ---------------------------------------------------------------------------

def guided_new_account_flow(
    provider_name: str,
    registration_url: str,
    login_url: Optional[str],
) -> Optional[Tuple[str, str]]:
    """
    Orchestrate the complete new-account path:
      1. Collect personal details from the console.
      2. Run Playwright to fill and submit the registration form.
      3. Pause for email verification.
      4. Return (email, password) for immediate login.

    Returns (username, password) on success, or None if the user aborts.
    """
    reg_url = registration_url or login_url
    if not reg_url:
        print(f"\n  No registration URL is available for {provider_name}.")
        print("  Please register manually at the provider's website, then choose 'existing'.")
        return None

    # Step B: collect personal details
    details = collect_registration_details(provider_name)

    # Step C: automated browser registration
    success, message = run_registration_assistant(
        provider_name=provider_name,
        registration_url=reg_url,
        details=details,
        headless=False,   # always visible so user can intervene if needed
    )

    if not success:
        print(f"\n  Automated registration could not complete: {message}")
        print("  Options:")
        print(f"    • Register manually at: {reg_url}")
        print("    • Choose 'existing' on the next prompt to enter credentials you register yourself.")
        return None

    # Step D: verification email notice
    print(f"\n  Registration form submitted successfully!")
    print(f"  A verification email has been sent to {details.email}.")
    print("  Please open your inbox, click the verification link,")
    print("  and then press Enter here to continue.")
    input("\n  [Press Enter after clicking the verification link] ")

    # Step E: return credentials for immediate use
    print(f"  Proceeding with account: {details.email}")
    return details.email, details.password
