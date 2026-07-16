"""
Agent 4 — Download Manager.

Handles location/format choice, then executes the actual fetch per
approved source: subset-at-source first (locked-in decision), falling
back to full download when a connector doesn't support subsetting.

HONEST SCOPE NOTE: local trimming of a fully-downloaded file (to
recover the storage savings subsetting would have given) is only
implemented here for NetCDF via xarray, since that's the dominant
format among the oceanographic/climate sources this system targets.
Other formats fall back to keeping the full file as-is -- extend
_trim_local_file for additional formats as needed, don't assume it's
handled for everything.

HONEST SCOPE NOTE 2: bounding-box/time-range subsetting parameters
require actual coordinates, not the free-text location strings
RetrievalRequest currently carries ("chennai", "last 5 years", etc.).
Geocoding/date-parsing to turn those into real (lon/lat) and ISO date
tuples is a separate piece of work, not yet wired here -- until it is,
fetch_request.bounding_box / time_range will be None and connectors
that need them for subsetting will fall back to full download. This is
called out explicitly rather than silently downloading full data and
claiming it was subsetted.
"""

import os
import shutil
import time
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional

from connectors.base_connector import Credentials, FetchRequest
from connectors.connector_factory import get_connector
from connectors.connector_types import CapabilityFlags
from models.agent4_schemas import (
    DatasetMetadata,
    DownloadFormat,
    DownloadLocationMode,
    DownloadManifestEntry,
    FetchMethod,
)
from models.website_analysis_schemas import SourceSnapshot

def playwright_available() -> bool:
    """
    Browser-assisted download support (agents/browser_utils.py,
    browser_manager.py, browser_login.py, browser_download.py,
    browser_session.py) has not been built yet in this project.

    Returning False here keeps the whole Agent 4 pipeline importable
    and runnable today -- every connector that only needs the API path
    works exactly as before. The only thing disabled is the fallback
    for connectors that declare requires_browser_download /
    requires_browser_auth, which will now report a clear "browser
    download not available" error instead of crashing on import.

    Once those browser_*.py files exist, delete this stub and restore:
        from agents.browser_utils import playwright_available, provider_key_from_url
    """
    return False


DEFAULT_MANAGED_FOLDER = os.path.join(os.getcwd(), "data")
MAX_DOWNLOAD_ATTEMPTS = 3


@dataclass
class DownloadPlanItem:
    source_id: str
    batch_id: int
    can_parallelize: bool
    dependency_ids: List[str]


class DownloadRetryError(RuntimeError):
    def __init__(self, message: str, retries_attempted: int):
        super().__init__(message)
        self.retries_attempted = retries_attempted


def ask_download_location() -> str:
    print("\nWhere should downloaded data go?")
    print("  1. Managed project data/ folder (recommended, auto-organized)")
    print("  2. A custom path you specify")
    print("  3. Ask me each time per source")
    choice = input("Choice [1/2/3]: ").strip()

    if choice == "2":
        path = input("Enter the full folder path: ").strip()
        os.makedirs(path, exist_ok=True)
        return path
    if choice == "3":
        return DownloadLocationMode.ask_each_time.value
    os.makedirs(DEFAULT_MANAGED_FOLDER, exist_ok=True)
    return DEFAULT_MANAGED_FOLDER


def ask_download_format() -> DownloadFormat:
    print("\nWhat format should the data be downloaded in?")
    print("  1. Native format (recommended -- smallest, no conversion loss)")
    print("  2. CSV")
    print("  3. NetCDF")
    print("  4. Parquet")
    print("  5. GeoTIFF")
    mapping = {"1": DownloadFormat.native, "2": DownloadFormat.csv, "3": DownloadFormat.netcdf,
               "4": DownloadFormat.parquet, "5": DownloadFormat.geotiff}
    choice = input("Choice [1-5]: ").strip()
    fmt = mapping.get(choice, DownloadFormat.native)
    if fmt != DownloadFormat.native:
        print(f"  Note: converting to {fmt.value} may increase file size versus the source's native format "
              f"(e.g. NetCDF -> CSV often bloats significantly). Conversion is not yet implemented in this "
              f"version -- data will be saved in native format regardless, and this is flagged so it's not silent.")
    return fmt


def _trim_local_file(local_path: str, variables: List[str]) -> bool:
    """Best-effort local trim for NetCDF via xarray. Returns True if
    trimming succeeded (raw file reduced in place), False otherwise --
    caller keeps the untrimmed file rather than losing data."""
    if not local_path.endswith((".nc", ".nc4")):
        return False
    try:
        import xarray as xr
        ds = xr.open_dataset(local_path)
        if variables:
            available = [v for v in variables if v in ds.variables]
            if available:
                ds = ds[available]
        tmp_path = local_path + ".trimmed"
        ds.to_netcdf(tmp_path)
        ds.close()
        os.replace(tmp_path, local_path)
        return True
    except Exception as exc:
        print(f"  [Download Manager] Local trim skipped (non-fatal): {exc}")
        return False


def _looks_like_html(local_path: str) -> bool:
    try:
        with open(local_path, "rb") as f:
            sample = f.read(2048).lstrip().lower()
        html_markers = (
            b"<!doctype html",
            b"<html",
            b"<title>login",
            b"sign in",
            b"unauthorized",
            b"access denied",
            b"forbidden",
            b"error page",
        )
        return any(marker in sample for marker in html_markers)
    except Exception:
        return False


def _hash_file(local_path: str, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with open(local_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checksum_matches(local_path: str, expected: str) -> bool:
    value = (expected or "").strip().strip('"')
    if not value:
        return True
    hex_chars = set("0123456789abcdefABCDEF")
    if len(value) == 64 and set(value) <= hex_chars:
        return _hash_file(local_path, "sha256").lower() == value.lower()
    if len(value) == 40 and set(value) <= hex_chars:
        return _hash_file(local_path, "sha1").lower() == value.lower()
    if len(value) == 32 and set(value) <= hex_chars:
        return _hash_file(local_path, "md5").lower() == value.lower()
    # Weak ETags and provider-specific checksum strings are reported by
    # the engine, but cannot be safely recomputed here.
    return True


def _validate_download(
    local_path: str,
    expected_size: Optional[float] = None,
    metadata: Optional[DatasetMetadata] = None,
) -> List[str]:
    notes = []
    if not local_path:
        notes.append("No local path returned by connector.")
        return notes
    if not os.path.exists(local_path):
        notes.append("Downloaded file does not exist.")
        return notes
    actual_size = os.path.getsize(local_path)
    if actual_size <= 0:
        notes.append("Downloaded file is empty.")
    if expected_size is not None and actual_size > expected_size * 1.10:
        notes.append("Downloaded file is larger than the size estimate by more than 10 percent.")
    if expected_size is not None and actual_size < expected_size * 0.90:
        notes.append("Downloaded file is smaller than the size estimate by more than 10 percent.")
    if metadata and metadata.content_type:
        notes.append(f"Expected content type from metadata: {metadata.content_type}.")
        if "text/html" in metadata.content_type.lower():
            notes.append("Metadata content type indicates HTML, not a dataset file.")
    if metadata and metadata.file_format:
        lower_format = metadata.file_format.lower()
        if "html" in lower_format:
            notes.append("Metadata file format indicates HTML, not a dataset file.")
    if metadata and metadata.checksum and not _checksum_matches(local_path, metadata.checksum):
        notes.append("Downloaded file checksum does not match expected metadata checksum.")
    if _looks_like_html(local_path):
        notes.append("Downloaded content looks like an HTML page, not a dataset file.")
    if not notes:
        notes.append("File exists and has non-zero size.")
    return notes


def _is_valid_download(local_path: str, validation_notes: List[str]) -> bool:
    if not local_path or not os.path.exists(local_path) or os.path.getsize(local_path) <= 0:
        return False
    fatal_markers = (
        "does not exist",
        "empty",
        "smaller than the expected complete size",
        "smaller than the size estimate",
        "HTML",
        "login",
        "error page",
        "indicates HTML",
        "MIME type",
        "dataset format",
        "checksum does not match",
    )
    return not any(any(marker in note for marker in fatal_markers) for note in validation_notes)


def _download_hints(metadata: Optional[DatasetMetadata]) -> Dict[str, object]:
    if not metadata:
        return {}
    return {
        "expected_size": metadata.file_size_bytes,
        "expected_content_type": metadata.content_type,
        "expected_format": metadata.file_format,
        "checksum": metadata.checksum,
    }


def _engine_report(fetch_request: FetchRequest):
    return fetch_request.metadata.get("download_result")


def _checksum_status(local_path: str, metadata: Optional[DatasetMetadata], result) -> str:
    if result is not None and getattr(result, "checksum_status", None):
        return result.checksum_status
    if metadata and metadata.checksum:
        if local_path and os.path.exists(local_path):
            return "verified" if _checksum_matches(local_path, metadata.checksum) else "failed"
        return "not_checked"
    return "unavailable"


def _validation_status(success: bool, notes: List[str]) -> str:
    return "passed" if success else ("failed" if notes else "not_checked")


def plan_download_batches(source_ids: List[str], source_snapshots) -> List[List[DownloadPlanItem]]:
    """
    Build connector-aware batches. The current Agent 3 handoff does not
    expose explicit dataset dependencies, so each source defaults to no
    dependencies. If a future snapshot adds dependency_ids/depends_on,
    this planner will honor them without changing the flow.
    """
    remaining = []
    seen = set()
    for sid in source_ids:
        if sid in seen:
            continue
        seen.add(sid)
        snapshot = source_snapshots.get(sid)
        if snapshot is None:
            continue
        connector = get_connector(snapshot)
        deps = list(getattr(snapshot, "dependency_ids", None) or getattr(snapshot, "depends_on", None) or [])
        remaining.append((sid, snapshot, connector, deps))

    completed = set()
    batches: List[List[DownloadPlanItem]] = []
    batch_id = 0
    while remaining:
        ready = [item for item in remaining if all(dep in completed or dep not in source_ids for dep in item[3])]
        if not ready:
            ready = [remaining[0]]
        parallel_ready = [
            item for item in ready
            if CapabilityFlags.supports_parallel_download in item[2].capabilities
        ]
        if len(parallel_ready) > 1:
            current = parallel_ready
        else:
            current = [ready[0]]
        batch = []
        for sid, _snapshot, connector, deps in current:
            batch.append(
                DownloadPlanItem(
                    source_id=sid,
                    batch_id=batch_id,
                    can_parallelize=CapabilityFlags.supports_parallel_download in connector.capabilities,
                    dependency_ids=deps,
                )
            )
            completed.add(sid)
        batches.append(batch)
        current_ids = {item[0] for item in current}
        remaining = [item for item in remaining if item[0] not in current_ids]
        batch_id += 1
    return batches


def _is_non_retryable_error(exc: Exception) -> bool:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403, 404):
        return True
    text = str(exc).lower()
    permanent_markers = (
        "401",
        "403",
        "404",
        "unauthorized",
        "forbidden",
        "permission denied",
        "authentication",
        "invalid credentials",
        "not found",
    )
    return any(marker in text for marker in permanent_markers)


def _run_with_retries(label: str, fetch_callable):
    last_error = None
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            return fetch_callable(), attempt - 1
        except NotImplementedError:
            raise
        except Exception as exc:
            last_error = exc
            if _is_non_retryable_error(exc):
                raise DownloadRetryError(str(exc), attempt - 1) from exc
            if attempt == MAX_DOWNLOAD_ATTEMPTS:
                break
            delay = 2 ** (attempt - 1)
            print(f"  {label} attempt {attempt} failed: {exc}. Retrying in {delay}s...")
            time.sleep(delay)
    retries_attempted = max(0, MAX_DOWNLOAD_ATTEMPTS - 1)
    raise DownloadRetryError(str(last_error), retries_attempted) from last_error


def _move_to_dest(local_path: str, dest_dir: str, filename_hint: str) -> str:
    """
    Ensures a successfully-downloaded file ends up in dest_dir.

    Many connectors (open_meteo, CKAN, generic HTTP) write to the
    system temp folder and return that temp path rather than the
    dest_path we supplied in FetchRequest.  Without this step the
    file lives in C:\\Users\\<user>\\AppData\\Local\\Temp\\... (or /tmp/...)
    forever and the manifest points at a path the user never sees.

    Strategy:
      1. If local_path is already inside dest_dir — nothing to do.
      2. Try shutil.move() first (rename on the same drive, copy+delete
         across drives / filesystems).
      3. If move raises (e.g. cross-device with a locked source file),
         fall back to shutil.copy2() and leave the original in place —
         the user still gets their data and we log a warning.

    Returns the final path where the file now lives.
    """
    if not local_path or not os.path.exists(local_path):
        return local_path  # nothing to move; caller will report the error

    real_dest_dir = os.path.realpath(dest_dir)
    real_src      = os.path.realpath(local_path)

    # Already in the right place — skip the move entirely
    if real_src.startswith(real_dest_dir + os.sep) or real_src == real_dest_dir:
        return local_path

    os.makedirs(dest_dir, exist_ok=True)

    # Determine the final filename: prefer the connector's own filename
    # over the generic .dat placeholder we generated in dest_path.
    src_basename = os.path.basename(local_path)
    final_name   = src_basename if src_basename else filename_hint
    final_path   = os.path.join(dest_dir, final_name)

    # Avoid silently overwriting a pre-existing file with the same name
    if os.path.exists(final_path) and os.path.realpath(final_path) != real_src:
        base, ext   = os.path.splitext(final_name)
        final_name  = f"{base}_{int(time.time())}{ext}"
        final_path  = os.path.join(dest_dir, final_name)

    try:
        shutil.move(local_path, final_path)
        print(f"  [Download Manager] Moved to destination: {final_path}")
    except Exception as move_exc:
        # Cross-device or locked-file edge case — copy and keep the original
        try:
            shutil.copy2(local_path, final_path)
            print(
                f"  [Download Manager] move() failed ({move_exc}); "
                f"copy2() succeeded — original remains at {local_path}"
            )
        except Exception as copy_exc:
            # Both failed — return the original temp path so the manifest
            # is honest about where the file actually is
            print(
                f"  [Download Manager] WARNING: could not move or copy file to "
                f"{dest_dir}: move={move_exc}, copy={copy_exc}. "
                f"File remains at {local_path}"
            )
            return local_path

    return final_path


def download_source(
    snapshot: SourceSnapshot,
    variables: List[str],
    location: str,
    credentials: Optional[Credentials],
    bounding_box=None,
    time_range=None,
    dataset_metadata: Optional[DatasetMetadata] = None,
) -> DownloadManifestEntry:
    connector = get_connector(snapshot)
    dest_dir = location if location != DownloadLocationMode.ask_each_time.value else input(f"  Download path for {snapshot.name}: ").strip()
    os.makedirs(dest_dir, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in snapshot.name)[:80]
    dest_path = os.path.join(dest_dir, f"{safe_name}_{snapshot.source_id}.dat")

    fetch_request = FetchRequest(
        variables=variables,
        dest_path=dest_path,
        bounding_box=bounding_box,
        time_range=time_range,
        metadata=_download_hints(dataset_metadata),
    )
    total_retries = 0

    try:
        local_path, retries = _run_with_retries(
            "Server-side subset",
            lambda: connector.fetch_subset(snapshot, fetch_request, credentials),
        )
        total_retries += retries
        result = _engine_report(fetch_request)
        local_path = _move_to_dest(local_path, dest_dir, os.path.basename(dest_path))
        expected_size = dataset_metadata.file_size_bytes if dataset_metadata else None
        validation_notes = list(getattr(result, "validation_notes", None) or _validate_download(local_path, expected_size=expected_size, metadata=dataset_metadata))
        success = _is_valid_download(local_path, validation_notes)
        return DownloadManifestEntry(
            source_id=snapshot.source_id, source_name=snapshot.name, local_path=local_path,
            fetch_method=FetchMethod.server_side_subset, variables_included=variables, success=success,
            size_bytes=os.path.getsize(local_path) if os.path.exists(local_path) else None,
            estimated_size_bytes=expected_size,
            retries_attempted=total_retries, validation_notes=validation_notes,
            dataset_metadata=dataset_metadata,
            provider=dataset_metadata.product if dataset_metadata and dataset_metadata.product else snapshot.name,
            connector_used=connector.name,
            protocol_used=connector.descriptor.connector_type.value,
            download_time_seconds=getattr(result, "elapsed_seconds", None),
            checksum_status=_checksum_status(local_path, dataset_metadata, result),
            validation_status=_validation_status(success, validation_notes),
            error=None if success else "; ".join(validation_notes),
        )
    except NotImplementedError:
        print(f"  {connector.name} connector doesn't support subsetting for this source -- "
              f"downloading full file and attempting a local trim.")
    except DownloadRetryError as exc:
        total_retries += exc.retries_attempted
        print(f"  Server-side subset failed after retries: {exc}")
        print("  Trying the official full-download endpoint next.")
    except Exception as exc:
        print(f"  Server-side subset failed after retries: {exc}")
        print("  Trying the official full-download endpoint next.")

    try:
        local_path, retries = _run_with_retries(
            "Full download",
            lambda: connector.fetch_full(snapshot, fetch_request, credentials),
        )
        total_retries += retries
        result = _engine_report(fetch_request)
        local_path = _move_to_dest(local_path, dest_dir, os.path.basename(dest_path))
        trimmed = _trim_local_file(local_path, variables)
        expected_size = dataset_metadata.file_size_bytes if dataset_metadata else None
        if result is not None and trimmed:
            validation_notes = _validate_download(local_path, expected_size=None, metadata=dataset_metadata)
        else:
            validation_notes = list(getattr(result, "validation_notes", None) or _validate_download(local_path, expected_size=expected_size, metadata=dataset_metadata))
        success = _is_valid_download(local_path, validation_notes)
        return DownloadManifestEntry(
            source_id=snapshot.source_id, source_name=snapshot.name, local_path=local_path,
            fetch_method=FetchMethod.full_download_then_trimmed if trimmed else FetchMethod.full_download_untrimmed,
            variables_included=variables, success=success,
            size_bytes=os.path.getsize(local_path) if os.path.exists(local_path) else None,
            estimated_size_bytes=expected_size,
            retries_attempted=total_retries, validation_notes=validation_notes,
            dataset_metadata=dataset_metadata,
            provider=dataset_metadata.product if dataset_metadata and dataset_metadata.product else snapshot.name,
            connector_used=connector.name,
            protocol_used=connector.descriptor.connector_type.value,
            download_time_seconds=getattr(result, "elapsed_seconds", None),
            checksum_status=_checksum_status(local_path, dataset_metadata, result),
            validation_status=_validation_status(success, validation_notes),
            error=None if success else "; ".join(validation_notes),
        )
    except DownloadRetryError as exc:
        total_retries += exc.retries_attempted
        return DownloadManifestEntry(source_id=snapshot.source_id, source_name=snapshot.name,
                                      fetch_method=FetchMethod.full_download_untrimmed, success=False,
                                      error=str(exc), retries_attempted=total_retries,
                                      estimated_size_bytes=dataset_metadata.file_size_bytes if dataset_metadata else None,
                                      dataset_metadata=dataset_metadata,
                                      provider=dataset_metadata.product if dataset_metadata and dataset_metadata.product else snapshot.name,
                                      connector_used=connector.name,
                                      protocol_used=connector.descriptor.connector_type.value,
                                      checksum_status="unavailable" if not dataset_metadata or not dataset_metadata.checksum else "not_checked",
                                      validation_status="failed")
    except Exception as exc:
        api_error = str(exc)
        # ── Browser fallback ─────────────────────────────────────────────────
        # Only activated when the connector declares it needs browser support
        # AND Playwright is available.  All existing API-only connectors skip
        # this block entirely (requires_browser_download defaults to False).
        requires_browser = (
            getattr(connector, "requires_browser_download", False)
            or getattr(connector, "requires_browser_auth", False)
        )
        if requires_browser and playwright_available():
            print(
                f"  API download failed for {connector.name} "
                f"({api_error}). Attempting browser-assisted download."
            )
            browser_result = _browser_download_fallback(
                snapshot=snapshot,
                connector=connector,
                fetch_request=fetch_request,
                credentials=credentials,
                dest_dir=dest_dir,
                dataset_metadata=dataset_metadata,
            )
            if browser_result.success and browser_result.local_path:
                local_path = _move_to_dest(browser_result.local_path, dest_dir, os.path.basename(dest_path))
                validation_notes = _validate_download(
                    local_path,
                    expected_size=dataset_metadata.file_size_bytes if dataset_metadata else None,
                    metadata=dataset_metadata,
                )
                success = _is_valid_download(local_path, validation_notes)
                return DownloadManifestEntry(
                    source_id=snapshot.source_id,
                    source_name=snapshot.name,
                    local_path=local_path,
                    fetch_method=FetchMethod.full_download_untrimmed,
                    variables_included=variables,
                    success=success,
                    size_bytes=os.path.getsize(local_path) if os.path.exists(local_path) else None,
                    estimated_size_bytes=dataset_metadata.file_size_bytes if dataset_metadata else None,
                    retries_attempted=total_retries,
                    validation_notes=validation_notes + browser_result.notes,
                    dataset_metadata=dataset_metadata,
                    provider=dataset_metadata.product if dataset_metadata and dataset_metadata.product else snapshot.name,
                    connector_used=connector.name,
                    protocol_used=connector.descriptor.connector_type.value,
                    checksum_status=_checksum_status(local_path, dataset_metadata, None),
                    validation_status=_validation_status(success, validation_notes),
                    error=None if success else "; ".join(validation_notes),
                )
            # Browser also failed — fall through to the structured error below
            api_error = (
                f"{api_error} | Browser fallback: {browser_result.error or 'failed'}"
            )

        return DownloadManifestEntry(source_id=snapshot.source_id, source_name=snapshot.name,
                                      fetch_method=FetchMethod.full_download_untrimmed, success=False,
                                      error=api_error, retries_attempted=total_retries,
                                      estimated_size_bytes=dataset_metadata.file_size_bytes if dataset_metadata else None,
                                      dataset_metadata=dataset_metadata,
                                      provider=dataset_metadata.product if dataset_metadata and dataset_metadata.product else snapshot.name,
                                      connector_used=connector.name,
                                      protocol_used=connector.descriptor.connector_type.value,
                                      checksum_status="unavailable" if not dataset_metadata or not dataset_metadata.checksum else "not_checked",
                                      validation_status="failed")


def _browser_download_fallback(
    snapshot,
    connector,
    fetch_request: FetchRequest,
    credentials: Optional[Credentials],
    dest_dir: str,
    dataset_metadata: Optional[DatasetMetadata],
):
    """
    Attempt a browser-assisted download when the API path has failed.

    This function is only called when:
      a) The connector declares ``requires_browser_download = True`` or
         ``requires_browser_auth = True``, AND
      b) Playwright is importable.

    The function asks the connector for a browser-specific download URL
    (via the optional ``browser_download_url`` attribute or method) and
    then drives a BrowserDownload session through the login + download
    flow.

    Returns a ``BrowserDownloadResult`` — the caller checks ``.success``.
    """
    from agents.browser_manager import BrowserManager
    from agents.browser_login import BrowserLogin
    from agents.browser_download import BrowserDownload, BrowserDownloadResult
    from agents.browser_utils import provider_key_from_url

    provider_key = provider_key_from_url(snapshot.url)
    login_url = getattr(connector, "login_url", None)
    download_page_url = (
        getattr(connector, "browser_download_url", None)
        or (dataset_metadata.download_endpoint if dataset_metadata else None)
        or snapshot.url
    )
    download_selector = getattr(connector, "browser_download_selector", None)
    requires_async_wait = getattr(connector, "requires_async_job_wait", False)
    headless = not bool(os.environ.get("BROWSER_HEADFUL"))

    print(f"  [Browser] Launching Chromium for '{provider_key}' (headless={headless}).")

    with BrowserManager(headless=headless, downloads_dir=dest_dir) as mgr:
        # Restore any previously saved session for this provider
        ctx = mgr.get_authenticated_context(provider_key)
        page = ctx.new_page()

        # Login if the connector provides a login URL and credentials exist
        session = None
        if login_url and credentials and (credentials.username or credentials.email):
            login_handler = BrowserLogin(page, ctx, provider_key)
            session = login_handler.login(
                login_url=login_url,
                credentials=credentials,
            )
            if session.is_authenticated:
                mgr.save_cookies(provider_key, ctx)
                print(f"  [Browser] Login succeeded for '{provider_key}'.")
            else:
                error = session.extra.get("login_error", "Login failed.")
                print(f"  [Browser] Login failed for '{provider_key}': {error}")
                return BrowserDownloadResult(success=False, error=f"Browser login failed: {error}")
        else:
            # No login required or no credentials — create a plain session wrapper
            from agents.browser_session import BrowserSession
            session = BrowserSession(provider_key=provider_key, context=ctx, page=page, is_authenticated=True)

        downloader = BrowserDownload(session=session, dest_dir=dest_dir)

        if requires_async_wait:
            result = downloader.wait_for_async_job(
                status_page_url=download_page_url,
                download_selector=download_selector,
            )
        else:
            result = downloader.download_from_url(
                download_page_url=download_page_url,
                selector=download_selector,
            )

    if result.success:
        print(
            f"  [Browser] Download complete: {result.local_path} "
            f"({result.file_size_bytes or '?'} bytes, "
            f"{result.elapsed_seconds:.1f}s)."
        )
    else:
        print(f"  [Browser] Download failed: {result.error}")

    return result
