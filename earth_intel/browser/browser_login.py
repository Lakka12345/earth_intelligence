"""
Browser Login
=============
Generic, provider-agnostic login handler that drives a Playwright browser
through the most common authentication patterns encountered on scientific
data portals.

Supported patterns
------------------
- Username / password forms (any field names, auto-detected)
- OAuth / OpenID Connect redirects (follows redirects automatically)
- Cookie / consent / GDPR banners (dismisses before attempting login)
- "Accept terms and conditions" checkboxes and buttons
- Multi-page login flows (handles intermediate redirects)
- CSRF tokens embedded in forms (read and submitted automatically via the
  browser — no manual extraction needed because the form submission goes
  through the actual DOM)

What is NOT here
----------------
- No provider-specific hardcoded selectors.  If a portal has a very
  unusual layout the connector should subclass ``BrowserLogin`` and
  override ``_fill_credentials`` or ``_click_submit``.
- No email verification flow (separate task).
- No CAPTCHA solving.

Usage::

    from browser.browser_manager import BrowserManager
    from browser.browser_login import BrowserLogin
    from connectors.base_connector import Credentials

    with BrowserManager() as mgr:
        ctx = mgr.get_authenticated_context("nasa")
        page = ctx.new_page()
        session = BrowserLogin(page, ctx, "nasa").login(
            login_url="https://urs.earthdata.nasa.gov/login",
            credentials=Credentials(username="...", password="..."),
        )
        # session.is_authenticated is True on success
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

from browser.browser_session import BrowserSession
from connectors.base_connector import Credentials

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Selector heuristics
# ------------------------------------------------------------------

# Ordered lists of CSS selectors tried in sequence; first match wins.

_USERNAME_SELECTORS: List[str] = [
    'input[name="username"]',
    'input[name="user"]',
    'input[name="login"]',
    'input[name="email"]',
    'input[type="email"]',
    'input[id*="user"]',
    'input[id*="email"]',
    'input[id*="login"]',
    'input[placeholder*="username" i]',
    'input[placeholder*="email" i]',
    'input[autocomplete="username"]',
    'input[autocomplete="email"]',
]

_PASSWORD_SELECTORS: List[str] = [
    'input[name="password"]',
    'input[name="pass"]',
    'input[type="password"]',
    'input[id*="pass"]',
    'input[placeholder*="password" i]',
    'input[autocomplete="current-password"]',
]

_SUBMIT_SELECTORS: List[str] = [
    'button[type="submit"]',
    'input[type="submit"]',
    'button[id*="login" i]',
    'button[id*="sign" i]',
    'button[class*="login" i]',
    'button[class*="submit" i]',
    '[data-testid*="login" i]',
    '[data-testid*="submit" i]',
]

_CONSENT_SELECTORS: List[str] = [
    # GDPR / cookie banners
    'button[id*="accept" i]',
    'button[id*="consent" i]',
    'button[id*="agree" i]',
    'button[id*="cookie" i]',
    'button[class*="accept" i]',
    'button[class*="consent" i]',
    '[aria-label*="accept" i]',
    '[aria-label*="agree" i]',
    '[aria-label*="cookie" i]',
]

_TERMS_SELECTORS: List[str] = [
    'input[type="checkbox"][name*="agree" i]',
    'input[type="checkbox"][name*="terms" i]',
    'input[type="checkbox"][name*="accept" i]',
    'input[type="checkbox"][id*="agree" i]',
    'input[type="checkbox"][id*="terms" i]',
    'button[id*="accept-terms" i]',
    'button[class*="accept-terms" i]',
    'a[id*="accept" i][href*="terms" i]',
]

# Selectors that indicate successful login (user dashboard / profile)
_SUCCESS_INDICATORS: List[str] = [
    '[class*="logout" i]',
    '[id*="logout" i]',
    'a[href*="logout" i]',
    'a[href*="signout" i]',
    '[class*="dashboard" i]',
    '[class*="profile" i]',
    '[class*="account" i]',
    '[aria-label*="account" i]',
    '[data-testid*="user" i]',
]

# Selectors that indicate a failed login attempt
_FAILURE_INDICATORS: List[str] = [
    '[class*="error" i]',
    '[class*="alert-danger" i]',
    '[role="alert"]',
    '[id*="error" i]',
    '[class*="invalid" i]',
    '[class*="failed" i]',
]


class BrowserLogin:
    """
    Drive a Playwright Page through a generic login flow.

    Parameters
    ----------
    page:
        An open Playwright ``Page`` object.
    context:
        The ``BrowserContext`` that owns *page*.
    provider_key:
        Opaque key for this provider (used in log messages and cookie
        persistence).
    timeout_ms:
        Default navigation / wait timeout in milliseconds.
    """

    def __init__(
        self,
        page,
        context,
        provider_key: str,
        timeout_ms: int = 15_000,
    ) -> None:
        self._page = page
        self._context = context
        self._provider_key = provider_key
        self._timeout = timeout_ms

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def login(
        self,
        login_url: str,
        credentials: Credentials,
        wait_for_selector: Optional[str] = None,
    ) -> BrowserSession:
        """
        Navigate to *login_url* and authenticate using *credentials*.

        Returns a ``BrowserSession`` with ``is_authenticated`` set to
        ``True`` on success or ``False`` on failure.  Never raises — all
        exceptions are caught and stored in ``session.extra["login_error"]``.

        Parameters
        ----------
        login_url:
            Full URL of the login / sign-in page.
        credentials:
            Username + password (and optional token / API key).
        wait_for_selector:
            Optional CSS selector to wait for after submitting credentials,
            indicating a successful login (e.g. ``"#dashboard"``).  When
            omitted, the generic success-indicator heuristic is used.
        """
        session = BrowserSession(
            provider_key=self._provider_key,
            context=self._context,
            page=self._page,
        )

        try:
            logger.info(
                "[BrowserLogin] Navigating to login page: %s", login_url
            )
            self._page.goto(login_url, timeout=self._timeout, wait_until="domcontentloaded")

            # Step 1 — dismiss consent / cookie banners
            self._dismiss_banners()

            # Step 2 — accept terms if present
            self._accept_terms()

            # Step 3 — fill credentials and submit
            self._fill_and_submit(credentials)

            # Step 4 — wait for post-login page to settle
            self._page.wait_for_load_state("domcontentloaded", timeout=self._timeout)

            # Step 5 — verify success
            if wait_for_selector:
                try:
                    self._page.wait_for_selector(
                        wait_for_selector, timeout=self._timeout
                    )
                    session.is_authenticated = True
                except Exception:
                    session.is_authenticated = False
            else:
                session.is_authenticated = self._detect_success()

            if session.is_authenticated:
                session.refresh_cookies()
                # Extract bearer token from localStorage if present
                token = (
                    session.get_local_storage("token")
                    or session.get_local_storage("access_token")
                    or session.get_local_storage("auth_token")
                )
                if token:
                    session.auth_token = token
                logger.info(
                    "[BrowserLogin] Login succeeded for '%s'.", self._provider_key
                )
            else:
                error_text = self._extract_error_text()
                session.extra["login_error"] = error_text
                logger.warning(
                    "[BrowserLogin] Login failed for '%s': %s",
                    self._provider_key,
                    error_text,
                )

        except Exception as exc:
            session.is_authenticated = False
            session.extra["login_error"] = str(exc)
            logger.error(
                "[BrowserLogin] Unexpected error during login for '%s': %s",
                self._provider_key,
                exc,
            )

        return session

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _first_visible(self, selectors: List[str]) -> Optional[Tuple[str, object]]:
        """
        Return the first (selector, element) pair from *selectors* that
        exists and is visible on the current page, or ``None``.
        """
        for sel in selectors:
            try:
                el = self._page.query_selector(sel)
                if el and el.is_visible():
                    return sel, el
            except Exception:
                continue
        return None

    def _dismiss_banners(self) -> None:
        """Click cookie consent / GDPR accept buttons if present."""
        result = self._first_visible(_CONSENT_SELECTORS)
        if result:
            sel, el = result
            logger.debug("[BrowserLogin] Dismissing consent banner (%s).", sel)
            try:
                el.click()
                self._page.wait_for_load_state("domcontentloaded", timeout=3_000)
            except Exception:
                pass

    def _accept_terms(self) -> None:
        """Tick terms-and-conditions checkboxes / click accept buttons."""
        for sel in _TERMS_SELECTORS:
            try:
                el = self._page.query_selector(sel)
                if el and el.is_visible():
                    tag = el.evaluate("el => el.tagName.toLowerCase()")
                    if tag == "input":
                        if not el.is_checked():
                            el.check()
                            logger.debug("[BrowserLogin] Checked terms checkbox (%s).", sel)
                    else:
                        el.click()
                        logger.debug("[BrowserLogin] Clicked terms button (%s).", sel)
            except Exception:
                continue

    def _fill_and_submit(self, credentials: Credentials) -> None:
        """Type username / password and click submit."""
        # Username
        user_result = self._first_visible(_USERNAME_SELECTORS)
        if user_result:
            sel, el = user_result
            logger.debug("[BrowserLogin] Filling username field (%s).", sel)
            el.fill(credentials.username or credentials.email or "")

        # Password
        pass_result = self._first_visible(_PASSWORD_SELECTORS)
        if pass_result:
            sel, el = pass_result
            logger.debug("[BrowserLogin] Filling password field (%s).", sel)
            el.fill(credentials.password or "")

        # Submit
        submit_result = self._first_visible(_SUBMIT_SELECTORS)
        if submit_result:
            sel, el = submit_result
            logger.debug("[BrowserLogin] Clicking submit (%s).", sel)
            el.click()
        else:
            # Fallback: press Enter on the password field
            if pass_result:
                _, el = pass_result
                logger.debug("[BrowserLogin] No submit button found; pressing Enter.")
                el.press("Enter")

    def _detect_success(self) -> bool:
        """
        Heuristically determine whether login succeeded.

        Returns True if a success indicator is found and no failure
        indicator is visible.
        """
        # Failure takes priority
        if self._first_visible(_FAILURE_INDICATORS):
            return False
        if self._first_visible(_SUCCESS_INDICATORS):
            return True
        # If the URL changed away from the login page, assume success
        current_url = self._page.url
        suspicious_fragments = ("login", "signin", "sign-in", "auth/login")
        return not any(frag in current_url.lower() for frag in suspicious_fragments)

    def _extract_error_text(self) -> str:
        """Extract visible error message text from the page."""
        for sel in _FAILURE_INDICATORS:
            try:
                el = self._page.query_selector(sel)
                if el and el.is_visible():
                    text = (el.inner_text() or "").strip()
                    if text:
                        return text
            except Exception:
                continue
        return "Login failed (no error message found on page)."
