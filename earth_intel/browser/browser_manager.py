"""
Browser Manager
===============
Manages the Playwright browser lifecycle and authenticated browser contexts.

Responsibilities
----------------
- Launch and close the Playwright Chromium browser (headless by default).
- Cache and reuse authenticated browser contexts per provider so that a
  login performed once is not repeated for subsequent downloads from the
  same provider within the same process lifetime.
- Store / restore cookies so that a context can be pre-loaded with a
  previously obtained authenticated session.

This module is the single place that holds the ``playwright`` process and
the ``Browser`` instance.  Everything else talks to ``BrowserManager``
rather than calling Playwright directly.

Backward compatibility
----------------------
Playwright is imported lazily inside ``BrowserManager.__enter__``.  If it
is not installed the import raises ``ImportError`` at that point, which
Agent 4 catches and converts to a structured error — existing API-based
downloads are not affected.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Directory used to persist cookie files between process runs.
_COOKIE_STORE_DIR = Path(os.environ.get("BROWSER_COOKIE_DIR", ".browser_sessions"))


class BrowserManager:
    """
    Context-manager that owns one Playwright Chromium browser instance.

    Usage::

        with BrowserManager(headless=True) as mgr:
            ctx = mgr.get_context("nasa")            # new or cached context
            page = ctx.new_page()
            ...
            mgr.save_cookies("nasa", ctx)            # persist for next run

    Contexts are cached by ``provider_key`` for the lifetime of the
    ``with`` block.  Re-entering the context manager creates a fresh
    browser and empty context cache.
    """

    def __init__(
        self,
        headless: bool = True,
        slow_mo: int = 0,
        downloads_dir: Optional[str] = None,
    ) -> None:
        self.headless = headless
        self.slow_mo = slow_mo
        self.downloads_dir = downloads_dir or str(Path.cwd() / "data" / "browser_downloads")

        # Set by __enter__
        self._playwright = None
        self._browser = None
        # provider_key -> BrowserContext
        self._contexts: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "BrowserManager":
        from playwright.sync_api import sync_playwright  # lazy import

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo,
        )
        logger.info(
            "[BrowserManager] Chromium launched (headless=%s).", self.headless
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for key, ctx in list(self._contexts.items()):
            try:
                ctx.close()
            except Exception:
                pass
        self._contexts.clear()

        if self._browser:
            try:
                self._browser.close()
            except Exception:
                pass
        if self._playwright:
            try:
                self._playwright.stop()
            except Exception:
                pass

        logger.info("[BrowserManager] Browser closed.")

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def get_context(
        self,
        provider_key: str,
        storage_state: Optional[dict] = None,
    ):
        """
        Return a cached BrowserContext for *provider_key*.

        If a context for this key already exists it is returned directly,
        so any authentication performed earlier in the session is reused.

        Parameters
        ----------
        provider_key:
            Opaque string that identifies the provider (e.g. ``"nasa"``,
            ``"copernicus"``).  Used only as a cache key.
        storage_state:
            Optional Playwright storage-state dict (cookies + local-storage)
            to pre-load into a newly created context.  Ignored when a cached
            context is returned.
        """
        if self._browser is None:
            raise RuntimeError(
                "BrowserManager must be used as a context manager "
                "(call __enter__ first)."
            )

        if provider_key in self._contexts:
            logger.debug("[BrowserManager] Reusing cached context for '%s'.", provider_key)
            return self._contexts[provider_key]

        kwargs = {
            "accept_downloads": True,
            "downloads_path": self.downloads_dir,
        }
        if storage_state:
            kwargs["storage_state"] = storage_state

        ctx = self._browser.new_context(**kwargs)
        self._contexts[provider_key] = ctx
        logger.debug("[BrowserManager] Created new context for '%s'.", provider_key)
        return ctx

    def close_context(self, provider_key: str) -> None:
        """Explicitly close and remove a cached context."""
        ctx = self._contexts.pop(provider_key, None)
        if ctx:
            try:
                ctx.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Cookie / storage-state persistence
    # ------------------------------------------------------------------

    def save_cookies(self, provider_key: str, context) -> None:
        """
        Persist the storage state (cookies + local/session storage) of
        *context* to disk so it can be reloaded in a future run.
        """
        _COOKIE_STORE_DIR.mkdir(parents=True, exist_ok=True)
        path = _COOKIE_STORE_DIR / f"{provider_key}.json"
        try:
            state = context.storage_state()
            path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            logger.info("[BrowserManager] Cookies saved for '%s' → %s.", provider_key, path)
        except Exception as exc:
            logger.warning("[BrowserManager] Could not save cookies for '%s': %s.", provider_key, exc)

    def load_cookies(self, provider_key: str) -> Optional[dict]:
        """
        Load a previously saved storage state for *provider_key*.

        Returns ``None`` if no saved state exists.
        """
        path = _COOKIE_STORE_DIR / f"{provider_key}.json"
        if not path.exists():
            return None
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            logger.info("[BrowserManager] Loaded saved cookies for '%s'.", provider_key)
            return state
        except Exception as exc:
            logger.warning(
                "[BrowserManager] Could not load cookies for '%s': %s.", provider_key, exc
            )
            return None

    def get_authenticated_context(self, provider_key: str):
        """
        Return a context pre-loaded with any saved storage state.

        Convenience wrapper that calls ``load_cookies`` then ``get_context``.
        If no saved state exists, a fresh unauthenticated context is returned.
        """
        storage_state = self.load_cookies(provider_key)
        return self.get_context(provider_key, storage_state=storage_state)
