"""
Browser Download
================
Drives a Playwright browser through the interactive download steps that
scientific data portals require before a file can be obtained:

  - Clicking download buttons
  - Waiting for asynchronous dataset generation jobs
  - Monitoring download progress via Playwright's Download event
  - Detecting and returning the completed local file path

This module is only reached after the API-based download path in
``download_source()`` has failed.  It operates on an already-authenticated
``BrowserSession`` produced by ``BrowserLogin``.

All methods are non-raising: on failure they return a
``BrowserDownloadResult`` with ``success=False`` and a descriptive
``error`` message so that the existing retry / manifest logic in
``agent4_download_manager`` can handle it uniformly.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from browser.browser_session import BrowserSession

logger = logging.getLogger(__name__)

# Default timeouts
_CLICK_TIMEOUT_MS = 10_000        # wait for download button to appear
_DOWNLOAD_START_TIMEOUT_MS = 30_000  # wait for download to begin after click
_ASYNC_JOB_POLL_INTERVAL = 5     # seconds between polling async job status
_ASYNC_JOB_MAX_WAIT = 600        # seconds before giving up on async jobs

# Heuristic selectors for download buttons / links
_DOWNLOAD_BUTTON_SELECTORS: List[str] = [
    'a[download]',
    'button[id*="download" i]',
    'a[id*="download" i]',
    'button[class*="download" i]',
    'a[class*="download" i]',
    '[data-testid*="download" i]',
    '[aria-label*="download" i]',
    'button[id*="export" i]',
    'a[id*="export" i]',
    'button[class*="export" i]',
    '[data-testid*="export" i]',
    'input[type="submit"][value*="download" i]',
    'input[type="button"][value*="download" i]',
]

# Selectors that indicate an async job is running
_ASYNC_STATUS_SELECTORS: List[str] = [
    '[class*="processing" i]',
    '[class*="pending" i]',
    '[class*="preparing" i]',
    '[class*="generating" i]',
    '[id*="job-status" i]',
    '[data-testid*="job-status" i]',
]

# Selectors that indicate the async job is ready (file available)
_ASYNC_READY_SELECTORS: List[str] = [
    '[class*="complete" i]',
    '[class*="ready" i]',
    '[class*="available" i]',
    '[id*="ready" i]',
    '[data-testid*="ready" i]',
]


@dataclass
class BrowserDownloadResult:
    """
    Result returned by BrowserDownload methods.

    Attributes
    ----------
    success:
        True if a file was successfully downloaded.
    local_path:
        Absolute path to the downloaded file on disk, or ``None``.
    file_size_bytes:
        Size of the downloaded file in bytes, or ``None``.
    elapsed_seconds:
        Wall-clock time from the first click to the file being available.
    error:
        Human-readable error description when ``success`` is False.
    notes:
        Additional informational messages (e.g. "async job waited 45s").
    """

    success: bool = False
    local_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    elapsed_seconds: Optional[float] = None
    error: Optional[str] = None
    notes: List[str] = field(default_factory=list)


class BrowserDownload:
    """
    Navigate an authenticated browser session through the download flow.

    Parameters
    ----------
    session:
        An authenticated ``BrowserSession`` (``is_authenticated`` should
        be True, but the class degrades gracefully if it is not).
    dest_dir:
        Local directory where downloaded files should be moved after
        Playwright saves them to its temporary downloads directory.
    """

    def __init__(self, session: BrowserSession, dest_dir: str) -> None:
        self._session = session
        self._dest_dir = dest_dir
        os.makedirs(dest_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def download_from_url(
        self,
        download_page_url: str,
        selector: Optional[str] = None,
    ) -> BrowserDownloadResult:
        """
        Navigate to *download_page_url* and trigger the download.

        Steps:
          1. Navigate to the page.
          2. Wait for a download button (heuristic or *selector*).
          3. Click the button and capture the Playwright Download event.
          4. Wait for the file to finish downloading.
          5. Move the file to *dest_dir* and return its path.

        Parameters
        ----------
        download_page_url:
            URL of the page that contains the download button.
        selector:
            Optional explicit CSS selector for the download button.
            When omitted, the heuristic selector list is tried in order.
        """
        start = time.time()
        page = self._session.page
        result = BrowserDownloadResult()

        try:
            logger.info("[BrowserDownload] Navigating to: %s", download_page_url)
            page.goto(
                download_page_url,
                timeout=_CLICK_TIMEOUT_MS,
                wait_until="domcontentloaded",
            )

            # Resolve the button selector
            btn_selector = selector or self._find_download_selector()
            if not btn_selector:
                result.error = "No download button found on page."
                logger.warning("[BrowserDownload] %s", result.error)
                return result

            logger.debug("[BrowserDownload] Using download selector: %s", btn_selector)

            # Intercept the download event
            with page.expect_download(timeout=_DOWNLOAD_START_TIMEOUT_MS) as download_info:
                page.click(btn_selector, timeout=_CLICK_TIMEOUT_MS)

            download = download_info.value
            local_path = self._save_download(download)

            result.success = True
            result.local_path = local_path
            result.file_size_bytes = (
                os.path.getsize(local_path) if local_path and os.path.exists(local_path) else None
            )
            result.elapsed_seconds = time.time() - start
            result.notes.append(f"Downloaded via browser click in {result.elapsed_seconds:.1f}s.")
            logger.info(
                "[BrowserDownload] Download complete: %s (%s bytes).",
                local_path,
                result.file_size_bytes,
            )

        except Exception as exc:
            result.error = str(exc)
            result.elapsed_seconds = time.time() - start
            logger.error("[BrowserDownload] Download failed: %s", exc)

        return result

    def wait_for_async_job(
        self,
        status_page_url: str,
        ready_selector: Optional[str] = None,
        download_selector: Optional[str] = None,
        poll_interval: int = _ASYNC_JOB_POLL_INTERVAL,
        max_wait: int = _ASYNC_JOB_MAX_WAIT,
    ) -> BrowserDownloadResult:
        """
        Poll a status page until an asynchronous dataset generation job
        completes, then trigger the download.

        Some portals (e.g. Copernicus CDS, INCOIS order pages) generate
        datasets in the background.  This method polls *status_page_url*
        every *poll_interval* seconds until a "ready" indicator appears
        or *max_wait* seconds elapse.

        Parameters
        ----------
        status_page_url:
            URL of the job status / order page.
        ready_selector:
            CSS selector that appears when the job is complete.  Heuristics
            are used when omitted.
        download_selector:
            CSS selector of the download button that appears after the job
            completes.  Heuristics are used when omitted.
        poll_interval:
            Seconds between page refreshes.
        max_wait:
            Maximum total wait time in seconds before giving up.
        """
        start = time.time()
        page = self._session.page
        result = BrowserDownloadResult()

        elapsed = 0
        while elapsed < max_wait:
            try:
                page.goto(
                    status_page_url,
                    timeout=_CLICK_TIMEOUT_MS,
                    wait_until="domcontentloaded",
                )
            except Exception as exc:
                logger.warning("[BrowserDownload] Could not load status page: %s", exc)
                time.sleep(poll_interval)
                elapsed = time.time() - start
                continue

            # Check if job is done
            is_ready = self._check_async_ready(ready_selector)
            logger.debug(
                "[BrowserDownload] Async job status: %s (elapsed %.0fs).",
                "READY" if is_ready else "PENDING",
                elapsed,
            )

            if is_ready:
                result.notes.append(f"Async job completed after {elapsed:.0f}s.")
                # Trigger the actual download from the same page
                btn_selector = download_selector or self._find_download_selector()
                if btn_selector:
                    try:
                        with page.expect_download(timeout=_DOWNLOAD_START_TIMEOUT_MS) as dl_info:
                            page.click(btn_selector, timeout=_CLICK_TIMEOUT_MS)
                        download = dl_info.value
                        local_path = self._save_download(download)
                        result.success = True
                        result.local_path = local_path
                        result.file_size_bytes = (
                            os.path.getsize(local_path)
                            if local_path and os.path.exists(local_path)
                            else None
                        )
                        result.elapsed_seconds = time.time() - start
                        logger.info(
                            "[BrowserDownload] Async job download complete: %s", local_path
                        )
                        return result
                    except Exception as exc:
                        result.error = f"Async job ready but download failed: {exc}"
                        result.elapsed_seconds = time.time() - start
                        logger.error("[BrowserDownload] %s", result.error)
                        return result
                else:
                    result.error = "Async job completed but no download button found."
                    result.elapsed_seconds = time.time() - start
                    return result

            time.sleep(poll_interval)
            elapsed = time.time() - start

        result.error = f"Async job did not complete within {max_wait}s."
        result.elapsed_seconds = time.time() - start
        logger.warning("[BrowserDownload] %s", result.error)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_download_selector(self) -> Optional[str]:
        """Try heuristic selectors and return the first visible one."""
        page = self._session.page
        for sel in _DOWNLOAD_BUTTON_SELECTORS:
            try:
                el = page.query_selector(sel)
                if el and el.is_visible():
                    return sel
            except Exception:
                continue
        return None

    def _check_async_ready(self, ready_selector: Optional[str]) -> bool:
        """Return True if the async job appears to be complete."""
        page = self._session.page
        selectors = [ready_selector] if ready_selector else _ASYNC_READY_SELECTORS
        for sel in selectors:
            if not sel:
                continue
            try:
                el = page.query_selector(sel)
                if el and el.is_visible():
                    return True
            except Exception:
                continue
        # If no "still processing" indicator is visible either, assume done.
        for sel in _ASYNC_STATUS_SELECTORS:
            try:
                el = page.query_selector(sel)
                if el and el.is_visible():
                    return False  # explicitly still running
            except Exception:
                continue
        return False

    def _save_download(self, download) -> Optional[str]:
        """
        Wait for a Playwright Download to finish and move it to *dest_dir*.

        Returns the final local path or ``None`` on failure.
        """
        try:
            suggested = download.suggested_filename or "browser_download"
            dest_path = os.path.join(self._dest_dir, suggested)
            # Avoid overwriting existing files
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(suggested)
                dest_path = os.path.join(self._dest_dir, f"{base}_{int(time.time())}{ext}")
            download.save_as(dest_path)
            return dest_path
        except Exception as exc:
            logger.error("[BrowserDownload] Failed to save download: %s", exc)
            return None
