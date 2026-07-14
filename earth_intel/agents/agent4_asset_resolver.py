"""
Agent 4 — Asset Resolver  (production-quality revision)

Changes from v1:
  • Structured error types instead of bare `except: return None`
  • Shared persistent session pool (connection reuse, TCP keep-alive)
  • TTL metadata cache (avoids duplicate API calls within a run)
  • Parallel metadata/HEAD requests via ThreadPoolExecutor
  • Intelligent STAC asset ranking (roles + media type + eo:bands + resolution + cloud cover + variable match)
  • Multi-candidate item ranking (temporal overlap, spatial overlap, cloud cover, recency)
  • Improved size estimation with confidence score and priority chain
  • Intelligent dataset matching with scoring (temporal/spatial coverage, variable overlap, completeness)
  • Extended download safety (executables, installers, CAPTCHA, wrong MIME types)
  • ConnectorDiagnostics dataclass for structured debugging
"""

from __future__ import annotations

import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Error types ───────────────────────────────────────────────────────────────

class ConnectorError(Exception):
    """Base class for all structured connector errors."""
    def __init__(self, message: str, provider: str = "", url: str = "", recoverable: bool = True):
        super().__init__(message)
        self.provider = provider
        self.url = url
        self.recoverable = recoverable

class NetworkError(ConnectorError):
    """DNS failure, connection refused, TCP timeout."""

class AuthenticationError(ConnectorError):
    """HTTP 401 / 403, token expired, missing credentials."""
    def __init__(self, message, **kwargs):
        super().__init__(message, recoverable=False, **kwargs)

class TimeoutError(ConnectorError):
    """Request timed out."""

class RateLimitError(ConnectorError):
    """HTTP 429 — caller should back off."""
    def __init__(self, message, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after

class APIError(ConnectorError):
    """Provider API returned an unexpected error (5xx, malformed JSON, etc.)."""
    def __init__(self, message, status_code: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.status_code = status_code

class DatasetNotFoundError(ConnectorError):
    """No dataset matched the query."""
    def __init__(self, message, **kwargs):
        super().__init__(message, recoverable=False, **kwargs)

class AssetNotFoundError(ConnectorError):
    """Dataset found but no downloadable asset located."""
    def __init__(self, message, **kwargs):
        super().__init__(message, recoverable=False, **kwargs)

class MetadataError(ConnectorError):
    """Metadata could not be retrieved or parsed."""

class ValidationError(ConnectorError):
    """Downloaded file failed content validation."""
    def __init__(self, message, **kwargs):
        super().__init__(message, recoverable=False, **kwargs)


def _classify_http_error(resp: requests.Response, provider: str = "", url: str = "") -> ConnectorError:
    """Turn an HTTP error response into the correct ConnectorError subtype."""
    sc = resp.status_code
    if sc == 401:
        return AuthenticationError(f"HTTP 401 Unauthorized from {provider or url}", provider=provider, url=url)
    if sc == 403:
        return AuthenticationError(f"HTTP 403 Forbidden from {provider or url} — check credentials or data access policy", provider=provider, url=url)
    if sc == 404:
        return DatasetNotFoundError(f"HTTP 404 Not Found: {url}", provider=provider, url=url)
    if sc == 429:
        retry_after = None
        try:
            retry_after = int(resp.headers.get("Retry-After", 0)) or None
        except (ValueError, TypeError):
            pass
        return RateLimitError(f"HTTP 429 Rate Limited by {provider or url}", retry_after=retry_after, provider=provider, url=url)
    if sc >= 500:
        return APIError(f"HTTP {sc} Server Error from {provider or url}", status_code=sc, provider=provider, url=url)
    return APIError(f"HTTP {sc} from {provider or url}", status_code=sc, provider=provider, url=url)


# ── Diagnostics ───────────────────────────────────────────────────────────────

@dataclass
class SizeEstimateResult:
    bytes: Optional[float] = None
    method: str = "unavailable"
    confidence: float = 0.0   # 0.0–1.0
    human_readable: str = "Unknown"

    def __bool__(self):
        return self.bytes is not None


@dataclass
class ConnectorDiagnostics:
    provider: str = ""
    protocol: str = ""
    dataset_selected: str = ""
    candidate_count: int = 0
    ranking_score: float = 0.0
    asset_selected: str = ""
    download_endpoint: str = ""
    metadata_endpoint: str = ""
    authentication_type: str = "none"
    subset_capable: bool = False
    size_estimate: SizeEstimateResult = field(default_factory=SizeEstimateResult)
    confidence_score: float = 0.0
    errors: List[str] = field(default_factory=list)
    retry_history: List[str] = field(default_factory=list)
    validation_result: str = "not_checked"

    def log_error(self, err: Exception) -> None:
        self.errors.append(f"{type(err).__name__}: {err}")

    def log_retry(self, attempt: int, reason: str) -> None:
        self.retry_history.append(f"attempt {attempt}: {reason}")

    def summary(self) -> str:
        lines = [
            f"Provider:        {self.provider}",
            f"Protocol:        {self.protocol}",
            f"Dataset:         {self.dataset_selected}",
            f"Candidates:      {self.candidate_count}",
            f"Ranking score:   {self.ranking_score:.2f}",
            f"Asset:           {self.asset_selected}",
            f"Download:        {self.download_endpoint}",
            f"Metadata:        {self.metadata_endpoint}",
            f"Auth type:       {self.authentication_type}",
            f"Subset capable:  {self.subset_capable}",
            f"Size estimate:   {self.size_estimate.human_readable} "
                f"(method={self.size_estimate.method}, confidence={self.size_estimate.confidence:.0%})",
            f"Confidence:      {self.confidence_score:.0%}",
            f"Validation:      {self.validation_result}",
        ]
        if self.errors:
            lines.append("Errors:")
            lines.extend(f"  • {e}" for e in self.errors)
        if self.retry_history:
            lines.append("Retries:")
            lines.extend(f"  • {r}" for r in self.retry_history)
        return "\n".join(lines)


# ── Constants ─────────────────────────────────────────────────────────────────

_TIMEOUT = 20
_DATA_EXTENSIONS = (
    ".nc", ".nc4", ".cdf",
    ".grib", ".grb", ".grb2",
    ".tif", ".tiff",
    ".csv",
    ".json", ".geojson",
    ".parquet",
    ".zarr",
    ".h5", ".hdf5", ".hdf",
    ".zip",
)
_HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")
_GOOD_CONTENT_TYPES = (
    "application/netcdf",
    "application/x-netcdf",
    "application/octet-stream",
    "application/zip",
    "text/csv",
    "application/json",
    "application/geo+json",
    "image/tiff",
    "application/x-hdf",
    "application/x-hdf5",
    "application/grib",
)
# Files that must always be rejected regardless of content-type
_REJECTED_EXTENSIONS = (
    ".exe", ".dll", ".msi", ".bat", ".cmd", ".sh", ".ps1",
    ".dmg", ".pkg", ".deb", ".rpm",
    ".js", ".vbs", ".wsf",
)
_REJECTED_CONTENT_TYPES = (
    "application/x-msdownload",
    "application/x-executable",
    "application/x-dosexec",
    "application/vnd.microsoft.portable-executable",
    "text/javascript",
    "application/javascript",
)
_CAPTCHA_MARKERS = (
    b"captcha",
    b"recaptcha",
    b"hcaptcha",
    b"verify you are human",
    b"bot detection",
)
_LOGIN_MARKERS = (
    b"<form",
    b"login",
    b"sign in",
    b"username",
    b"password",
    b"unauthorized",
    b"access denied",
    b"forbidden",
)
_MIN_VALID_BYTES = 512   # files smaller than this are treated as placeholders


# ── Session pool ──────────────────────────────────────────────────────────────

class _SessionPool:
    """Thread-safe pool of persistent requests.Session objects with retry adapters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, requests.Session] = {}

    def get(self, key: str = "default", credentials=None) -> requests.Session:
        with self._lock:
            if key not in self._sessions:
                s = requests.Session()
                retry = Retry(
                    total=3,
                    backoff_factor=0.5,
                    status_forcelist=(500, 502, 503, 504),
                    allowed_methods={"GET", "HEAD", "POST"},
                    raise_on_status=False,
                )
                adapter = HTTPAdapter(
                    max_retries=retry,
                    pool_connections=10,
                    pool_maxsize=20,
                )
                s.mount("https://", adapter)
                s.mount("http://", adapter)
                s.headers.update({"User-Agent": "EarthIntelAgent4/1.0"})
                self._sessions[key] = s
            sess = self._sessions[key]

        if credentials:
            token = (
                getattr(credentials, "token", None)
                or getattr(credentials, "session_token", None)
                or getattr(credentials, "api_key", None)
            )
            if token:
                sess.headers["Authorization"] = f"Bearer {token}"
            username = getattr(credentials, "username", None)
            password = getattr(credentials, "password", None)
            if username and password:
                sess.auth = (username, password)
        return sess


_pool = _SessionPool()


def _session(credentials=None, key: str = "default") -> requests.Session:
    return _pool.get(key=key, credentials=credentials)


# ── Metadata cache ────────────────────────────────────────────────────────────

@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class _MetadataCache:
    """Simple TTL cache for API responses within a single agent run."""

    def __init__(self, default_ttl: float = 300.0) -> None:
        self._lock = threading.Lock()
        self._store: Dict[str, _CacheEntry] = {}
        self.default_ttl = default_ttl

    def get(self, key: str) -> Any:
        with self._lock:
            entry = self._store.get(key)
            if entry and time.monotonic() < entry.expires_at:
                return entry.value
            if entry:
                del self._store[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        with self._lock:
            self._store[key] = _CacheEntry(
                value=value,
                expires_at=time.monotonic() + (ttl or self.default_ttl),
            )

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


_cache = _MetadataCache(default_ttl=300.0)   # 5-minute TTL by default


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_html(resp: requests.Response) -> bool:
    ct = (resp.headers.get("Content-Type") or "").lower()
    return any(h in ct for h in _HTML_CONTENT_TYPES)


def _is_data_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in _DATA_EXTENSIONS)


def _is_rejected_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in _REJECTED_EXTENSIONS)


def _is_rejected_content_type(ct: str) -> bool:
    ct_lower = ct.lower()
    return any(r in ct_lower for r in _REJECTED_CONTENT_TYPES)


def _sample_looks_safe(path: str) -> Tuple[bool, str]:
    """
    Read first 4 KB of a downloaded file and check for HTML/login/CAPTCHA.
    Returns (is_safe, reason).
    """
    try:
        with open(path, "rb") as f:
            sample = f.read(4096).lstrip().lower()
    except Exception as exc:
        return False, f"Could not read file sample: {exc}"

    if any(m in sample for m in _CAPTCHA_MARKERS):
        return False, "File content contains CAPTCHA challenge markers."

    if sample.startswith((b"<!doctype html", b"<html")):
        if any(m in sample for m in _LOGIN_MARKERS):
            return False, "File content is an HTML login/auth page."
        return False, "File content is an HTML page, not a dataset."

    # PE/ELF/Mach-O executable magic bytes
    if sample[:2] in (b"MZ", b"\x7fE") or sample[:4] == b"\xcf\xfa\xed\xfe":
        return False, "File content is an executable binary."

    return True, ""


def _format_bytes(n: Optional[float]) -> str:
    if n is None:
        return "Unknown"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def _parse_iso(dt_str: str) -> Optional[datetime]:
    if not dt_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(dt_str[:19], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


# ── STAC intelligent asset ranking (Task 1) ───────────────────────────────────

# Scientific data media types in priority order
_STAC_MEDIA_PRIORITY = {
    "application/x-netcdf": 100,
    "application/netcdf": 100,
    "image/tiff; application=geotiff; profile=cloud-optimized": 90,
    "image/tiff; application=geotiff": 85,
    "image/tiff": 80,
    "application/x-hdf5": 75,
    "application/x-hdf": 75,
    "application/grib": 70,
    "application/zip": 40,
    "application/json": 20,
    "text/csv": 15,
}

_STAC_ROLE_PRIORITY = {
    "data": 100,
    "cog": 95,
    "analytic": 90,
    "reflectance": 85,
    "temperature": 85,
    "overview": 30,
    "visual": 25,
    "thumbnail": 0,
    "metadata": 0,
    "tilejson": 0,
    "map": 10,
}


def _score_stac_asset(
    asset: dict,
    asset_name: str,
    variables: List[str],
) -> float:
    """
    Score a single STAC asset for scientific relevance.
    Higher is better.
    """
    score = 0.0
    roles = asset.get("roles") or []
    media_type = (asset.get("type") or "").lower()

    # Role score
    for role in roles:
        score += _STAC_ROLE_PRIORITY.get(role.lower(), 5)

    # Thumbnail/overview penalty
    if any(r in ("thumbnail", "metadata", "tilejson") for r in roles):
        return -999.0

    # Media type score
    for mt, pts in _STAC_MEDIA_PRIORITY.items():
        if mt in media_type:
            score += pts
            break
    else:
        if media_type and "html" not in media_type:
            score += 5   # unknown but non-HTML gets a small boost

    # eo:bands — prefer bands matching requested variables
    eo_bands = asset.get("eo:bands") or []
    var_terms = {v.lower() for v in variables}
    for band in eo_bands:
        band_name = (band.get("common_name") or band.get("name") or "").lower()
        band_desc = (band.get("description") or "").lower()
        if any(t in band_name or t in band_desc for t in var_terms):
            score += 20

    # raster extension: resolution (prefer finer = larger pixel size in meters)
    raster_bands = asset.get("raster:bands") or []
    for rb in raster_bands:
        spatial_res = rb.get("spatial_resolution")
        if spatial_res:
            try:
                res = float(spatial_res)
                # Finer resolution → higher score (penalty is proportional to coarseness)
                score += max(0, 20 - res / 100)
            except (TypeError, ValueError):
                pass

    # proj:shape — larger array = more data content
    proj_shape = asset.get("proj:shape")
    if proj_shape and len(proj_shape) >= 2:
        try:
            pixels = proj_shape[0] * proj_shape[1]
            score += min(15, pixels / 1_000_000)
        except (TypeError, ValueError):
            pass

    # file:size — larger files are generally more complete
    file_size = asset.get("file:size") or asset.get("size")
    if file_size:
        try:
            score += min(10, float(file_size) / (10 * 1024 * 1024))
        except (TypeError, ValueError):
            pass

    # Variable name match in asset key
    for v in variables:
        v_norm = v.lower().replace(" ", "_")
        if v_norm in asset_name.lower():
            score += 15

    return score


def _rank_stac_items(
    items: List[dict],
    variables: List[str],
    bbox: Optional[Tuple[float, float, float, float]],
    time_range: Optional[Tuple[str, str]],
) -> List[Tuple[float, dict]]:
    """
    Score and rank STAC items by temporal overlap, spatial overlap,
    variable coverage, cloud cover, and recency.
    """
    scored = []
    now = datetime.now(timezone.utc)

    t_start = _parse_iso(time_range[0]) if time_range and time_range[0] else None
    t_end = _parse_iso(time_range[1]) if time_range and time_range[1] else None

    for item in items:
        score = 0.0
        props = item.get("properties") or {}

        # Recency: prefer more recent items (up to +20 pts)
        item_dt = _parse_iso(props.get("datetime") or props.get("start_datetime") or "")
        if item_dt:
            age_days = (now - item_dt).days
            score += max(0, 20 - age_days / 30)

        # Temporal overlap with requested range
        if t_start and t_end and item_dt:
            if t_start <= item_dt <= t_end:
                score += 30
            else:
                score -= 10

        # Cloud cover (lower is better; +25 for 0%, 0 for 100%)
        cc = props.get("eo:cloud_cover") or props.get("cloud_cover")
        if cc is not None:
            try:
                score += max(0, 25 * (1 - float(cc) / 100))
            except (TypeError, ValueError):
                pass

        # Spatial overlap: rough check if item bbox overlaps requested bbox
        item_bbox = item.get("bbox")
        if bbox and item_bbox and len(item_bbox) >= 4:
            try:
                ix0, iy0, ix1, iy1 = item_bbox[:4]
                rx0, ry0, rx1, ry1 = bbox
                x_overlap = max(0, min(ix1, rx1) - max(ix0, rx0))
                y_overlap = max(0, min(iy1, ry1) - max(iy0, ry0))
                if x_overlap > 0 and y_overlap > 0:
                    req_area = max(1e-9, (rx1 - rx0) * (ry1 - ry0))
                    overlap_area = x_overlap * y_overlap
                    score += min(20, 20 * overlap_area / req_area)
            except (TypeError, ValueError):
                pass

        # Asset variable relevance (aggregate across all assets)
        assets = item.get("assets") or {}
        for aname, asset in assets.items():
            for v in variables:
                if v.lower() in aname.lower():
                    score += 5

        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _select_best_stac_asset(
    item: dict,
    variables: List[str],
) -> Tuple[Optional[str], str]:
    """
    Select the best asset href from a STAC item using intelligent scoring.
    Returns (href, asset_name).
    """
    assets = item.get("assets") or {}
    if not assets:
        return None, ""

    scored = []
    for aname, asset in assets.items():
        s = _score_stac_asset(asset, aname, variables)
        if s > -500:   # filter out thumbnails etc.
            scored.append((s, aname, asset))

    if not scored:
        return None, ""

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_name, best_asset = scored[0]
    href = best_asset.get("href")
    return href, best_name


# ── Dataset scoring (Task 7) ──────────────────────────────────────────────────

def score_dataset_match(
    dataset: dict,
    variables: List[str],
    bbox: Optional[Tuple[float, float, float, float]] = None,
    time_range: Optional[Tuple[str, str]] = None,
) -> float:
    """
    Score a dataset descriptor dict for relevance to a query.
    Works on dicts from STAC collections, CKAN packages, CMR collections, etc.
    """
    score = 0.0
    text_fields = [
        dataset.get("title") or dataset.get("name") or "",
        dataset.get("description") or dataset.get("summary") or dataset.get("notes") or "",
        " ".join(dataset.get("keywords") or dataset.get("tags") or []),
        " ".join(
            t.get("name", "") if isinstance(t, dict) else str(t)
            for t in (dataset.get("tags") or [])
        ),
    ]
    full_text = " ".join(text_fields).lower()

    # Variable matching (most important signal)
    for v in variables:
        v_lower = v.lower()
        # Direct match
        if v_lower in full_text:
            score += 20
        # Partial token match
        for token in v_lower.split():
            if len(token) > 3 and token in full_text:
                score += 5

    # Temporal coverage
    t_start_str = dataset.get("extent", {}).get("temporal", {}).get("interval", [[None]])[0][0] if "extent" in dataset else dataset.get("start_date") or dataset.get("temporal_coverage_start")
    t_end_str = dataset.get("extent", {}).get("temporal", {}).get("interval", [[None, None]])[0][1] if "extent" in dataset else dataset.get("end_date") or dataset.get("temporal_coverage_end")
    if time_range and time_range[0] and t_start_str and t_end_str:
        t_req = _parse_iso(time_range[0])
        t_ds_start = _parse_iso(str(t_start_str))
        t_ds_end = _parse_iso(str(t_end_str)) if t_end_str else datetime.now(timezone.utc)
        if t_req and t_ds_start and t_ds_end:
            if t_ds_start <= t_req <= t_ds_end:
                score += 15

    # Spatial coverage
    if bbox and "extent" in dataset:
        try:
            sp = dataset["extent"].get("spatial", {}).get("bbox", [[]])[0]
            if sp and len(sp) >= 4:
                x_ov = max(0, min(sp[2], bbox[2]) - max(sp[0], bbox[0]))
                y_ov = max(0, min(sp[3], bbox[3]) - max(sp[1], bbox[1]))
                if x_ov > 0 and y_ov > 0:
                    score += 10
        except (IndexError, TypeError, ValueError):
            pass

    # Provider quality signals
    if dataset.get("license") not in (None, "", "notspecified"):
        score += 2
    if dataset.get("doi") or dataset.get("related_identifiers"):
        score += 5

    return score


# ── Parallel metadata fetch helper (Task 5) ───────────────────────────────────

def parallel_head_requests(
    urls: List[str],
    credentials=None,
    max_workers: int = 8,
    timeout: int = 15,
) -> Dict[str, Optional[requests.Response]]:
    """
    Issue HEAD requests to multiple URLs concurrently.
    Returns {url: response_or_None}.
    """
    results: Dict[str, Optional[requests.Response]] = {}

    def _fetch(url: str) -> Tuple[str, Optional[requests.Response]]:
        s = _session(credentials)
        try:
            r = s.head(url, allow_redirects=True, timeout=timeout)
            return url, r
        except Exception:
            return url, None

    with ThreadPoolExecutor(max_workers=min(max_workers, len(urls))) as ex:
        for url, resp in ex.map(_fetch, urls):
            results[url] = resp

    return results


def parallel_get_json(
    urls: List[str],
    credentials=None,
    params_list: Optional[List[Optional[dict]]] = None,
    max_workers: int = 8,
    timeout: int = 20,
) -> Dict[str, Optional[dict]]:
    """
    Issue GET requests for JSON to multiple URLs concurrently.
    Returns {url: json_or_None}.
    """
    results: Dict[str, Optional[dict]] = {}
    params_list = params_list or [None] * len(urls)

    def _fetch(url: str, params: Optional[dict]) -> Tuple[str, Optional[dict]]:
        s = _session(credentials)
        try:
            r = s.get(url, params=params, timeout=timeout)
            ct = r.headers.get("Content-Type") or ""
            if r.ok and "json" in ct:
                return url, r.json()
        except Exception:
            pass
        return url, None

    with ThreadPoolExecutor(max_workers=min(max_workers, len(urls))) as ex:
        futures = {ex.submit(_fetch, u, p): u for u, p in zip(urls, params_list)}
        for future in as_completed(futures):
            url, data = future.result()
            results[url] = data

    return results


# ── STAC ─────────────────────────────────────────────────────────────────────

def resolve_stac_asset(
    catalog_url: str,
    variables: List[str],
    bbox: Optional[Tuple[float, float, float, float]] = None,
    time_range: Optional[Tuple[str, str]] = None,
    credentials=None,
    diagnostics: Optional[ConnectorDiagnostics] = None,
) -> Optional[str]:
    """
    Search a STAC API and return the best asset href using intelligent
    multi-candidate ranking.
    """
    s = _session(credentials)
    base = catalog_url.rstrip("/")
    diag = diagnostics or ConnectorDiagnostics(protocol="stac")

    # ── Step 1: list and score collections (cached) ───────────────────────
    cache_key = f"stac:collections:{base}"
    collections = _cache.get(cache_key)
    if collections is None:
        try:
            r = s.get(f"{base}/collections", timeout=_TIMEOUT)
            if not r.ok:
                if r.status_code in (401, 403, 429):
                    err = _classify_http_error(r, url=f"{base}/collections")
                    diag.log_error(err)
                    if isinstance(err, AuthenticationError):
                        raise err
                collections = []
            else:
                collections = r.json().get("collections", []) if "json" in (r.headers.get("Content-Type") or "") else []
            _cache.set(cache_key, collections)
        except ConnectorError:
            raise
        except Exception as exc:
            diag.log_error(NetworkError(f"STAC collections fetch failed: {exc}", url=f"{base}/collections"))
            collections = []

    # Pick best collection using dataset scoring
    collection_id = None
    if collections:
        scored_cols = [
            (score_dataset_match(col, variables, bbox, time_range), col)
            for col in collections
        ]
        scored_cols.sort(key=lambda x: x[0], reverse=True)
        best_col = scored_cols[0][1] if scored_cols else None
        if best_col:
            collection_id = best_col.get("id")
            diag.metadata_endpoint = f"{base}/collections/{collection_id}"

    # ── Step 2: search for multiple candidate items ───────────────────────
    search_url = f"{base}/search"
    search_params: Dict = {"limit": 10}   # fetch multiple for ranking
    if bbox:
        search_params["bbox"] = ",".join(str(x) for x in bbox)
    if time_range and time_range[0] and time_range[1]:
        search_params["datetime"] = f"{time_range[0]}/{time_range[1]}"
    if collection_id:
        search_params["collections"] = [collection_id]

    cache_key_search = f"stac:search:{base}:{collection_id}:{bbox}:{time_range}"
    items = _cache.get(cache_key_search)
    if items is None:
        items = []
        try:
            r = s.post(search_url, json=search_params, timeout=_TIMEOUT)
            if not r.ok:
                r = s.get(search_url, params=search_params, timeout=_TIMEOUT)
            if r.ok and "json" in (r.headers.get("Content-Type") or ""):
                items = r.json().get("features", [])
        except Exception as exc:
            diag.log_error(NetworkError(f"STAC search failed: {exc}", url=search_url))

        # Fallback: items endpoint
        if not items and collection_id:
            try:
                r = s.get(
                    f"{base}/collections/{collection_id}/items",
                    params={"limit": 10},
                    timeout=_TIMEOUT,
                )
                if r.ok and "json" in (r.headers.get("Content-Type") or ""):
                    items = r.json().get("features", [])
            except Exception as exc:
                diag.log_error(NetworkError(f"STAC items endpoint failed: {exc}"))

        _cache.set(cache_key_search, items, ttl=120.0)

    diag.candidate_count = len(items)

    if not items:
        raise DatasetNotFoundError(
            f"STAC search at {base} returned no items for variables={variables}, "
            f"bbox={bbox}, time_range={time_range}",
            url=base,
        )

    # ── Step 3: rank items and select best ────────────────────────────────
    ranked = _rank_stac_items(items, variables, bbox, time_range)
    best_score, best_item = ranked[0]
    diag.ranking_score = best_score
    diag.dataset_selected = best_item.get("id", "")

    # ── Step 4: pick best asset from best item ────────────────────────────
    href, asset_name = _select_best_stac_asset(best_item, variables)
    if href is None:
        raise AssetNotFoundError(
            f"STAC item '{best_item.get('id')}' has no suitable downloadable asset.",
            url=base,
        )

    diag.asset_selected = asset_name
    diag.download_endpoint = href
    return href


# ── THREDDS ───────────────────────────────────────────────────────────────────

def resolve_thredds_asset(
    catalog_url: str,
    variables: List[str],
    credentials=None,
    diagnostics: Optional[ConnectorDiagnostics] = None,
) -> Optional[str]:
    s = _session(credentials)
    diag = diagnostics or ConnectorDiagnostics(protocol="thredds")

    xml_url = catalog_url
    if not xml_url.endswith(".xml"):
        xml_url = re.sub(r"(/catalog\.html?)?$", "/catalog.xml", catalog_url.rstrip("/"))

    cache_key = f"thredds:catalog:{xml_url}"
    content = _cache.get(cache_key)
    if content is None:
        try:
            r = s.get(xml_url, timeout=_TIMEOUT)
            if not r.ok:
                err = _classify_http_error(r, url=xml_url)
                diag.log_error(err)
                return None
            content = r.text
            _cache.set(cache_key, content)
        except requests.exceptions.Timeout:
            diag.log_error(TimeoutError(f"THREDDS catalog timed out: {xml_url}", url=xml_url))
            return None
        except requests.exceptions.ConnectionError as exc:
            diag.log_error(NetworkError(f"THREDDS connection failed: {exc}", url=xml_url))
            return None
        except Exception as exc:
            diag.log_error(MetadataError(f"THREDDS catalog error: {exc}", url=xml_url))
            return None

    dataset_pattern = re.compile(
        r'<dataset[^>]+name="([^"]*)"[^>]*urlPath="([^"]*)"', re.IGNORECASE
    )
    datasets = dataset_pattern.findall(content)
    if not datasets:
        diag.log_error(DatasetNotFoundError("THREDDS catalog contained no datasets with urlPath.", url=xml_url))
        return None

    diag.candidate_count = len(datasets)

    def _score(name: str) -> float:
        name_lower = name.lower()
        return sum(
            (10 if v.lower() in name_lower else 0) +
            (5 if any(tok in name_lower for tok in v.lower().split()) else 0)
            for v in variables
        )

    best_name, best_path = max(datasets, key=lambda x: _score(x[0]))
    diag.dataset_selected = best_name
    base = re.sub(r"/catalog.*", "", catalog_url.rstrip("/"))
    url = f"{base}/fileServer/{best_path.lstrip('/')}"
    diag.download_endpoint = url
    return url


# ── ERDDAP ────────────────────────────────────────────────────────────────────

def resolve_erddap_asset(
    base_url: str,
    variables: List[str],
    bbox: Optional[Tuple[float, float, float, float]] = None,
    time_range: Optional[Tuple[str, str]] = None,
    credentials=None,
    diagnostics: Optional[ConnectorDiagnostics] = None,
) -> Optional[str]:
    s = _session(credentials)
    diag = diagnostics or ConnectorDiagnostics(protocol="erddap")

    erddap_root = re.sub(r"/(griddap|tabledap|info|search).*", "", base_url.rstrip("/"))

    # Search with multiple candidates
    query = " ".join(variables) if variables else "temperature"
    search_url = f"{erddap_root}/search/index.json"
    cache_key = f"erddap:search:{erddap_root}:{query}"
    rows = _cache.get(cache_key)
    if rows is None:
        try:
            r = s.get(search_url, params={"searchFor": query, "page": 1, "itemsPerPage": 10}, timeout=_TIMEOUT)
            if not r.ok:
                err = _classify_http_error(r, url=search_url)
                diag.log_error(err)
                return None
            data = r.json()
            rows = data.get("table", {}).get("rows", [])
            col_names = data.get("table", {}).get("columnNames", [])
            _cache.set(cache_key, (rows, col_names))
        except requests.exceptions.Timeout:
            diag.log_error(TimeoutError(f"ERDDAP search timed out", url=search_url))
            return None
        except Exception as exc:
            diag.log_error(APIError(f"ERDDAP search error: {exc}", url=search_url))
            return None
    else:
        rows, col_names = rows

    if not rows:
        raise DatasetNotFoundError(f"ERDDAP search for '{query}' at {erddap_root} returned no datasets.", url=erddap_root)

    diag.candidate_count = len(rows)
    try:
        id_idx = col_names.index("Dataset ID")
        title_idx = col_names.index("Title") if "Title" in col_names else -1
    except (ValueError, AttributeError):
        id_idx = 1
        title_idx = -1

    # Score rows by variable match in title
    def _row_score(row):
        title = row[title_idx].lower() if title_idx >= 0 and title_idx < len(row) else ""
        return sum(1 for v in variables if v.lower() in title)

    best_row = max(rows, key=_row_score)
    dataset_id = best_row[id_idx]
    diag.dataset_selected = dataset_id

    t0, t1 = (time_range or ("", ""))
    lat_clause = f"[({bbox[1]}):1:({bbox[3]})]" if bbox else "[(last)]"
    lon_clause = f"[({bbox[0]}):1:({bbox[2]})]" if bbox else "[(last)]"
    time_clause = f"[({t0 or 'last'}):1:({t1 or 'last'})]"

    var_part = ",".join(
        f"{v.replace(' ', '_')}{time_clause}{lat_clause}{lon_clause}"
        for v in (variables or ["data"])
    )
    url = f"{erddap_root}/griddap/{dataset_id}.nc?{var_part}"
    diag.download_endpoint = url
    diag.subset_capable = True
    return url


# ── OPeNDAP ───────────────────────────────────────────────────────────────────

def resolve_opendap_asset(
    dap_url: str,
    variables: List[str],
    bbox: Optional[Tuple[float, float, float, float]] = None,
    time_range: Optional[Tuple[str, str]] = None,
    credentials=None,
    diagnostics: Optional[ConnectorDiagnostics] = None,
) -> Optional[str]:
    s = _session(credentials)
    diag = diagnostics or ConnectorDiagnostics(protocol="opendap")

    dap_base = re.sub(r"\.(dds|das|nc4?|dods|html)$", "", dap_url, flags=re.IGNORECASE)

    cache_key = f"opendap:dds:{dap_base}"
    dds_vars: List[str] = _cache.get(cache_key) or []
    if not dds_vars:
        try:
            r = s.get(f"{dap_base}.dds", timeout=_TIMEOUT)
            if r.ok and not _is_html(r):
                dds_vars = re.findall(r"^\s*\w+\s+(\w+)\s*\[", r.text, re.MULTILINE)
                _cache.set(cache_key, dds_vars)
        except requests.exceptions.Timeout:
            diag.log_error(TimeoutError(f"OPeNDAP .dds timed out: {dap_base}.dds", url=dap_base))
        except Exception as exc:
            diag.log_error(MetadataError(f"OPeNDAP .dds fetch failed: {exc}", url=dap_base))

    matched: List[str] = []
    for req in variables:
        req_norm = req.lower().replace(" ", "_")
        for dv in dds_vars:
            if req_norm in dv.lower() or dv.lower() in req_norm:
                if dv not in matched:
                    matched.append(dv)
                break

    if not matched and dds_vars:
        matched = dds_vars[:3]

    diag.subset_capable = bool(matched)

    if matched:
        ce = ",".join(matched)
        url = f"{dap_base}.nc4?{ce}"
    else:
        url = f"{dap_base}.nc4"

    diag.download_endpoint = url
    return url


# ── CKAN ─────────────────────────────────────────────────────────────────────

def resolve_ckan_asset(
    portal_url: str,
    variables: List[str],
    credentials=None,
    diagnostics: Optional[ConnectorDiagnostics] = None,
) -> Optional[str]:
    s = _session(credentials)
    diag = diagnostics or ConnectorDiagnostics(protocol="ckan")
    base = re.sub(r"/api/.*", "", portal_url.rstrip("/"))

    query = " ".join(variables) if variables else "dataset"
    search_url = f"{base}/api/3/action/package_search"
    cache_key = f"ckan:search:{base}:{query}"
    packages = _cache.get(cache_key)

    if packages is None:
        try:
            r = s.get(search_url, params={"q": query, "rows": 10}, timeout=_TIMEOUT)
            if not r.ok:
                err = _classify_http_error(r, url=search_url)
                diag.log_error(err)
                return None
            result = r.json().get("result", {})
            packages = result.get("results", [])
            _cache.set(cache_key, packages)
        except requests.exceptions.Timeout:
            diag.log_error(TimeoutError(f"CKAN search timed out", url=search_url))
            return None
        except Exception as exc:
            diag.log_error(APIError(f"CKAN search failed: {exc}", url=search_url))
            return None

    if not packages:
        raise DatasetNotFoundError(f"CKAN search for '{query}' returned no packages.", url=base)

    diag.candidate_count = len(packages)

    # Score packages
    scored = [(score_dataset_match(pkg, variables), pkg) for pkg in packages]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_pkg = scored[0]
    diag.dataset_selected = best_pkg.get("name") or best_pkg.get("title") or ""
    diag.ranking_score = best_score

    resources = best_pkg.get("resources", [])
    _DATA_FMTS = {"csv", "netcdf", "nc", "geotiff", "tiff", "json", "geojson", "parquet", "hdf", "hdf5"}
    for res in resources:
        fmt = (res.get("format") or "").lower()
        url = res.get("url") or res.get("download_url") or ""
        if fmt in _DATA_FMTS or _is_data_url(url):
            diag.download_endpoint = url
            return url

    # Fallback: first resource
    if resources:
        url = resources[0].get("url") or resources[0].get("download_url") or ""
        diag.download_endpoint = url
        return url or None

    return None


# ── CMR / NASA EarthData ──────────────────────────────────────────────────────

def resolve_cmr_asset(
    short_name: str,
    variables: List[str],
    bbox: Optional[Tuple[float, float, float, float]] = None,
    time_range: Optional[Tuple[str, str]] = None,
    credentials=None,
    provider: str = "",
    diagnostics: Optional[ConnectorDiagnostics] = None,
) -> Optional[str]:
    s = _session(credentials)
    diag = diagnostics or ConnectorDiagnostics(protocol="cmr")

    params: Dict = {
        "short_name": short_name,
        "page_size": 5,
        "sort_key": "-start_date",
    }
    if bbox:
        params["bounding_box"] = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    if time_range and time_range[0]:
        params["temporal"] = f"{time_range[0]},{time_range[1] or ''}"
    if provider:
        params["provider"] = provider

    cache_key = f"cmr:granules:{short_name}:{provider}:{bbox}:{time_range}"
    entries = _cache.get(cache_key)
    if entries is None:
        try:
            r = s.get(
                "https://cmr.earthdata.nasa.gov/search/granules.json",
                params=params,
                timeout=_TIMEOUT,
            )
            if not r.ok:
                err = _classify_http_error(r, provider="NASA CMR", url=r.url)
                diag.log_error(err)
                if isinstance(err, (AuthenticationError, RateLimitError)):
                    raise err
                return None
            entries = r.json().get("feed", {}).get("entry", [])
            _cache.set(cache_key, entries, ttl=60.0)
        except ConnectorError:
            raise
        except requests.exceptions.Timeout:
            diag.log_error(TimeoutError("CMR granule search timed out", provider="NASA CMR"))
            return None
        except Exception as exc:
            diag.log_error(APIError(f"CMR search error: {exc}", provider="NASA CMR"))
            return None

    if not entries:
        raise DatasetNotFoundError(
            f"CMR granule search for {short_name} returned no results.",
            provider="NASA CMR",
        )

    diag.candidate_count = len(entries)

    # Rank entries: prefer most recent with spatial overlap
    def _entry_score(entry: dict) -> float:
        sc = 0.0
        # Recency
        begin = _parse_iso(entry.get("time_start") or "")
        if begin:
            age_days = (datetime.now(timezone.utc) - begin).days
            sc += max(0, 20 - age_days / 30)
        # Prefer OPeNDAP links
        for link in entry.get("links", []):
            if "opendap" in (link.get("title") or link.get("href") or "").lower():
                sc += 30
        return sc

    entries.sort(key=_entry_score, reverse=True)
    best = entries[0]
    diag.dataset_selected = best.get("title") or best.get("id") or ""
    links = best.get("links", [])

    # Prefer OPeNDAP
    for link in links:
        href = link.get("href") or ""
        if "opendap" in (link.get("title") or href).lower() and href:
            diag.download_endpoint = href
            diag.subset_capable = True
            return href

    # HTTP data download
    for link in links:
        rel = (link.get("rel") or "").lower()
        href = link.get("href") or ""
        if ("data#" in rel or _is_data_url(href)) and href:
            diag.download_endpoint = href
            return href

    # First non-metadata link
    for link in links:
        href = link.get("href") or ""
        if href and not href.endswith((".html", ".xml", ".json")):
            diag.download_endpoint = href
            return href

    raise AssetNotFoundError(f"CMR granule '{best.get('id')}' has no usable download links.", provider="NASA CMR")


# ── Generic URL resolution ─────────────────────────────────────────────────────

def resolve_generic_asset(
    url: str,
    credentials=None,
    reject_html: bool = True,
    diagnostics: Optional[ConnectorDiagnostics] = None,
) -> Optional[str]:
    diag = diagnostics or ConnectorDiagnostics(protocol="generic_http")
    s = _session(credentials)

    if _is_rejected_url(url):
        diag.log_error(ValidationError(f"URL has a rejected file extension: {url}", url=url))
        return None

    try:
        r = s.head(url, allow_redirects=True, timeout=_TIMEOUT)
        final_url = r.url
        ct = (r.headers.get("Content-Type") or "").lower()
        disposition = (r.headers.get("Content-Disposition") or "").lower()

        if r.status_code in (401, 403, 429):
            err = _classify_http_error(r, url=url)
            diag.log_error(err)
            raise err

        if _is_rejected_content_type(ct):
            diag.log_error(ValidationError(f"Rejected content type '{ct}' from {url}", url=url))
            return None

        if "attachment" in disposition:
            diag.download_endpoint = final_url
            return final_url

        if reject_html and any(h in ct for h in _HTML_CONTENT_TYPES):
            diag.log_error(ValidationError(f"URL returns HTML (landing page?): {final_url}", url=url))
            return None

        if any(g in ct for g in _GOOD_CONTENT_TYPES) or _is_data_url(final_url):
            diag.download_endpoint = final_url
            return final_url

        if r.ok and not any(h in ct for h in _HTML_CONTENT_TYPES):
            diag.download_endpoint = final_url
            return final_url

    except ConnectorError:
        raise
    except requests.exceptions.Timeout:
        diag.log_error(TimeoutError(f"Generic HEAD timed out: {url}", url=url))
    except requests.exceptions.ConnectionError as exc:
        diag.log_error(NetworkError(f"Connection failed: {exc}", url=url))
    except Exception as exc:
        diag.log_error(NetworkError(f"Unexpected error: {exc}", url=url))

    return None


# ── Size estimation (Task 6 priority chain) ────────────────────────────────────

def estimate_size_from_url(url: str, credentials=None) -> Optional[float]:
    """Priority 1: Content-Length from HEAD."""
    if not url:
        return None
    cache_key = f"head:content-length:{url}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached
    s = _session(credentials)
    try:
        r = s.head(url, allow_redirects=True, timeout=15)
        cl = r.headers.get("Content-Length")
        if cl and str(cl).isdigit():
            result = float(cl)
            _cache.set(cache_key, result, ttl=120.0)
            return result
    except Exception:
        pass
    return None


def estimate_size_from_stac_item(item: dict) -> Optional[float]:
    """Priority 2: STAC file:size."""
    total = 0.0
    found = False
    for asset in item.get("assets", {}).values():
        sz = asset.get("file:size") or asset.get("size")
        if sz:
            try:
                total += float(sz)
                found = True
            except (TypeError, ValueError):
                pass
    return total if found else None


def estimate_size_from_erddap_metadata(
    erddap_root: str,
    dataset_id: str,
    variables: List[str],
    credentials=None,
) -> Optional[float]:
    """Priority 3: ERDDAP /info metadata (uses dimension sizes × variable count × dtype size)."""
    s = _session(credentials)
    cache_key = f"erddap:info:{erddap_root}:{dataset_id}"
    info = _cache.get(cache_key)
    if info is None:
        try:
            r = s.get(f"{erddap_root}/info/{dataset_id}/index.json", timeout=15)
            if r.ok:
                info = r.json()
                _cache.set(cache_key, info)
        except Exception:
            pass
    if not info:
        return None

    # Extract dimension sizes from rows
    rows = info.get("table", {}).get("rows", [])
    dims = {}
    for row in rows:
        if len(row) > 4 and str(row[0]).lower() == "dimension":
            name = str(row[1])
            try:
                dims[name] = int(row[4])
            except (ValueError, IndexError):
                pass
    if not dims:
        return None

    # Estimate: product of all dimension sizes × bytes per value × variable count
    total_elements = 1
    for sz in dims.values():
        total_elements *= sz
    n_vars = max(1, len(variables))
    return float(total_elements * n_vars * 4)  # 4 bytes/float32


def estimate_size_full(
    url: str,
    stac_item: Optional[dict] = None,
    erddap_root: Optional[str] = None,
    erddap_dataset_id: Optional[str] = None,
    variables: Optional[List[str]] = None,
    credentials=None,
) -> SizeEstimateResult:
    """
    Try all size estimation strategies in priority order.
    Returns SizeEstimateResult with confidence score.
    """
    variables = variables or []

    # Priority 1: Content-Length (exact)
    size = estimate_size_from_url(url, credentials)
    if size and size > 0:
        return SizeEstimateResult(
            bytes=size, method="Content-Length header",
            confidence=0.95, human_readable=_format_bytes(size),
        )

    # Priority 2: STAC file:size
    if stac_item:
        size = estimate_size_from_stac_item(stac_item)
        if size and size > 0:
            return SizeEstimateResult(
                bytes=size, method="STAC file:size",
                confidence=0.90, human_readable=_format_bytes(size),
            )

    # Priority 3: ERDDAP metadata
    if erddap_root and erddap_dataset_id:
        size = estimate_size_from_erddap_metadata(erddap_root, erddap_dataset_id, variables, credentials)
        if size and size > 0:
            return SizeEstimateResult(
                bytes=size, method="ERDDAP dimension estimate",
                confidence=0.50, human_readable=_format_bytes(size),
            )

    # All strategies failed
    return SizeEstimateResult(method="unavailable", confidence=0.0, human_readable="Unknown")


# ── Download safety validation (Task 9) ──────────────────────────────────────

def validate_downloaded_file(
    path: str,
    expected_content_type: Optional[str] = None,
    min_bytes: int = _MIN_VALID_BYTES,
) -> Tuple[bool, str]:
    """
    Validate a downloaded file for safety and correctness.
    Returns (is_valid, reason).
    """
    import os
    if not path or not os.path.exists(path):
        return False, "Downloaded file does not exist."

    size = os.path.getsize(path)
    if size == 0:
        return False, "Downloaded file is empty (0 bytes)."
    if size < min_bytes:
        return False, f"Downloaded file is suspiciously small ({size} bytes < {min_bytes} minimum)."

    ext = os.path.splitext(path)[1].lower()
    if ext in _REJECTED_EXTENSIONS:
        return False, f"Downloaded file has a rejected extension: {ext}"

    safe, reason = _sample_looks_safe(path)
    if not safe:
        return False, reason

    return True, ""
