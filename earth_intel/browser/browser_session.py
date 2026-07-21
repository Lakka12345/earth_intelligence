"""
Browser Session
===============
A thin wrapper that groups a Playwright ``BrowserContext`` and an open
``Page`` together with session-level metadata (provider key, whether the
session is authenticated, any auth token extracted during login).

BrowserSession objects are produced by ``BrowserLogin`` and consumed by
``BrowserDownload``.  They should not be created directly.

No Playwright-specific types are used in the constructor signature so
that callers that only do type-checking and don't actually run the
browser don't need Playwright installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BrowserSession:
    """
    Groups a live Playwright BrowserContext and its primary Page.

    Attributes
    ----------
    provider_key:
        Opaque identifier for the provider (same key used by BrowserManager).
    context:
        The Playwright ``BrowserContext`` for this session.
    page:
        The primary ``Page`` within *context*.  A new page can be opened
        via ``context.new_page()`` when needed.
    is_authenticated:
        True once BrowserLogin has successfully completed authentication.
    auth_token:
        Optional bearer / API token extracted from cookies or local storage
        after login, which connectors can inject into API calls.
    cookies:
        List of cookie dicts as returned by Playwright's ``context.cookies()``.
        Kept up-to-date by BrowserLogin after each successful auth step.
    extra:
        Arbitrary key-value store for provider-specific session data
        (e.g. CSRF tokens, job IDs, download URLs discovered during auth).
    """

    provider_key: str
    context: Any          # playwright BrowserContext
    page: Any             # playwright Page
    is_authenticated: bool = False
    auth_token: Optional[str] = None
    cookies: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def refresh_cookies(self) -> List[Dict[str, Any]]:
        """Re-read cookies from the live context and cache them."""
        try:
            self.cookies = self.context.cookies()
        except Exception:
            pass
        return self.cookies

    def get_cookie(self, name: str) -> Optional[str]:
        """Return the value of a cookie by name, or None."""
        for cookie in self.cookies:
            if cookie.get("name") == name:
                return cookie.get("value")
        return None

    def get_local_storage(self, key: str) -> Optional[str]:
        """Read a value from the page's localStorage."""
        try:
            return self.page.evaluate(f"window.localStorage.getItem({key!r})")
        except Exception:
            return None

    def get_session_storage(self, key: str) -> Optional[str]:
        """Read a value from the page's sessionStorage."""
        try:
            return self.page.evaluate(f"window.sessionStorage.getItem({key!r})")
        except Exception:
            return None

    def inject_auth_header(self, token: str) -> None:
        """
        Route all subsequent requests from this context to add an
        Authorization header.  Useful when an API token is obtained
        via the browser and subsequent downloads use direct HTTP.
        """
        self.auth_token = token
        try:
            self.context.set_extra_http_headers(
                {"Authorization": f"Bearer {token}"}
            )
        except Exception:
            pass
