"""
Browser Utilities
=================
Small helpers shared across the browser automation modules.

None of these functions import Playwright directly so they are safe to
import even when Playwright is not installed.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# URL helpers
# ------------------------------------------------------------------

def is_likely_login_page(url: str) -> bool:
    """
    Return True when *url* looks like a login / authentication page.

    Used to detect when a connector has been redirected to a login wall.
    """
    patterns = (
        r"/login",
        r"/signin",
        r"/sign[-_]in",
        r"/auth(?:/|$)",
        r"/account/login",
        r"/users/sign_in",
        r"urs\.earthdata\.nasa\.gov/login",
        r"oauth",
    )
    url_lower = url.lower()
    return any(re.search(p, url_lower) for p in patterns)


def is_likely_data_file(content_type: str, filename: str) -> bool:
    """
    Return True when the response looks like an actual data file rather
    than an HTML page or error response.
    """
    html_types = ("text/html", "application/xhtml")
    if any(t in content_type.lower() for t in html_types):
        return False

    data_extensions = (
        ".nc", ".nc4", ".netcdf",
        ".tif", ".tiff", ".geotiff",
        ".csv", ".tsv",
        ".hdf", ".hdf5", ".h5",
        ".zarr", ".zip", ".tar", ".gz", ".bz2",
        ".json", ".geojson",
        ".parquet",
    )
    _, ext = os.path.splitext(filename.lower())
    return ext in data_extensions


def absolute_url(base_url: str, href: str) -> str:
    """Resolve *href* against *base_url* to produce an absolute URL."""
    return urljoin(base_url, href)


def provider_key_from_url(url: str) -> str:
    """
    Derive a short provider key from a URL for use as a cookie / context
    cache key.

    Example::

        >>> provider_key_from_url("https://urs.earthdata.nasa.gov/login")
        'earthdata.nasa.gov'
    """
    parsed = urlparse(url)
    host = parsed.hostname or parsed.netloc
    # Strip common subdomains that don't distinguish the provider
    for prefix in ("www.", "urs.", "auth.", "login.", "sso.", "portal."):
        if host.startswith(prefix):
            host = host[len(prefix):]
            break
    return host


# ------------------------------------------------------------------
# File helpers
# ------------------------------------------------------------------

def wait_for_file(
    path: str,
    timeout: float = 120.0,
    poll_interval: float = 1.0,
) -> bool:
    """
    Block until *path* exists and its size has stopped growing, or until
    *timeout* seconds elapse.

    Returns True if the file is ready, False on timeout.
    """
    deadline = time.time() + timeout
    prev_size = -1

    while time.time() < deadline:
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size > 0 and size == prev_size:
                return True
            prev_size = size
        time.sleep(poll_interval)

    return False


def move_to_dest(src: str, dest_dir: str, filename: Optional[str] = None) -> str:
    """
    Move *src* into *dest_dir*, returning the final path.

    If *filename* is omitted the original basename is used.  Overwrites
    are avoided by appending a timestamp suffix.
    """
    os.makedirs(dest_dir, exist_ok=True)
    name = filename or os.path.basename(src)
    dest = os.path.join(dest_dir, name)
    if os.path.exists(dest) and os.path.abspath(dest) != os.path.abspath(src):
        base, ext = os.path.splitext(name)
        dest = os.path.join(dest_dir, f"{base}_{int(time.time())}{ext}")
    os.replace(src, dest)
    return dest


# ------------------------------------------------------------------
# Playwright availability check
# ------------------------------------------------------------------

def playwright_available() -> bool:
    """Return True if the ``playwright`` package is importable."""
    try:
        import importlib
        importlib.import_module("playwright.sync_api")
        return True
    except ImportError:
        return False
