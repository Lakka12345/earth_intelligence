"""Provider-independent download helpers for Agent 4."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import base64
import hashlib
import os
import time
from typing import Callable, Iterable, Optional, TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from connectors.base_connector import Credentials
else:
    Credentials = object

CHUNK_SIZE = 1024 * 1024
HTML_TYPES = ("text/html", "application/xhtml+xml")
HTML_MARKERS = (
    b"<!doctype html",
    b"<html",
    b"<title>login",
    b"sign in",
    b"unauthorized",
    b"access denied",
    b"forbidden",
    b"error page",
)
DATA_FORMAT_EXTENSIONS = {
    "csv": (".csv",),
    "json": (".json", ".geojson"),
    "netcdf": (".nc", ".nc4", ".cdf"),
    "geotiff": (".tif", ".tiff"),
    "tiff": (".tif", ".tiff"),
    "parquet": (".parquet",),
    "grib": (".grib", ".grb", ".grb2"),
    "hdf5": (".h5", ".hdf5", ".hdf"),
    "zarr": (".zarr",),
}


@dataclass
class DownloadTask:
    url: str
    dest_path: str
    expected_size: Optional[float] = None
    expected_content_type: Optional[str] = None
    expected_format: Optional[str] = None
    checksum: Optional[str] = None
    source_id: str = ""
    provider: str = ""
    connector_id: str = ""
    protocol: str = ""
    allow_resume: bool = True
    max_retries: int = 3
    progress_callback: Optional[Callable[[str, int, Optional[float]], None]] = None


@dataclass
class DownloadResult:
    url: str
    dest_path: str
    success: bool
    size_bytes: int = 0
    elapsed_seconds: float = 0.0
    error: str = ""
    status_code: Optional[int] = None
    content_type: Optional[str] = None
    retries_attempted: int = 0
    resumed: bool = False
    checksum_status: str = "not_checked"
    checksum_algorithm: Optional[str] = None
    validation_status: str = "not_checked"
    validation_notes: Optional[list[str]] = None


class DownloadEngine:
    def __init__(self, max_workers: int = 4, timeout: int = 45) -> None:
        self.max_workers = max_workers
        self.timeout = timeout

    def _headers(self, credentials: Optional[Credentials], task: DownloadTask) -> dict:
        headers = {}
        if credentials:
            token = credentials.token or credentials.session_token or credentials.api_key
            if token:
                headers["Authorization"] = f"Bearer {token}"
        if task.allow_resume and os.path.exists(task.dest_path):
            size = os.path.getsize(task.dest_path)
            if size > 0:
                headers["Range"] = f"bytes={size}-"
        return headers

    def _auth(self, credentials: Optional[Credentials]):
        if credentials and credentials.username and credentials.password:
            return (credentials.username, credentials.password)
        return None

    def _looks_like_html(self, path: str) -> bool:
        try:
            with open(path, "rb") as f:
                sample = f.read(4096).lstrip().lower()
            return any(marker in sample for marker in HTML_MARKERS)
        except Exception:
            return False

    def _content_type_ok(self, actual: Optional[str], expected: Optional[str]) -> bool:
        if not expected or not actual:
            return True
        expected = expected.lower()
        actual = actual.lower()
        if any(html in expected for html in HTML_TYPES):
            return False
        return expected.split(";")[0].strip() in actual or actual.split(";")[0].strip() in expected

    def _format_ok(self, path: str, expected_format: Optional[str]) -> bool:
        if not expected_format:
            return True
        expected = expected_format.lower()
        if expected in ("unknown", "native"):
            return True
        for name, extensions in DATA_FORMAT_EXTENSIONS.items():
            if name in expected:
                return path.lower().endswith(extensions)
        return True

    def _hash_file(self, path: str, algorithm: str) -> str:
        digest = hashlib.new(algorithm)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _checksum_status(self, path: str, expected: Optional[str]) -> tuple[str, Optional[str]]:
        value = (expected or "").strip().strip('"')
        if not value:
            return "unavailable", None
        lower = value.lower()
        hex_chars = set("0123456789abcdef")
        if len(lower) == 64 and set(lower) <= hex_chars:
            return ("verified" if self._hash_file(path, "sha256") == lower else "failed", "sha256")
        if len(lower) == 40 and set(lower) <= hex_chars:
            return ("verified" if self._hash_file(path, "sha1") == lower else "failed", "sha1")
        if len(lower) == 32 and set(lower) <= hex_chars:
            return ("verified" if self._hash_file(path, "md5") == lower else "failed", "md5")
        try:
            decoded = base64.b64decode(value, validate=True).hex()
            if len(decoded) == 32:
                return ("verified" if self._hash_file(path, "md5") == decoded else "failed", "md5")
        except Exception:
            pass
        if value.startswith("W/"):
            return "weak_etag_reported", "etag"
        return "reported_unverified", "etag"

    def validate_download(
        self,
        path: str,
        *,
        expected_size: Optional[float] = None,
        expected_content_type: Optional[str] = None,
        actual_content_type: Optional[str] = None,
        expected_format: Optional[str] = None,
        checksum: Optional[str] = None,
    ) -> tuple[bool, list[str], str, Optional[str]]:
        notes: list[str] = []
        if not path:
            return False, ["No local path returned by connector."], "not_checked", None
        if not os.path.exists(path):
            return False, ["Downloaded file does not exist."], "not_checked", None
        size = os.path.getsize(path)
        if size <= 0:
            notes.append("Downloaded file is empty.")
        if expected_size is not None and size < int(expected_size):
            notes.append("Downloaded file is smaller than the expected complete size.")
        if actual_content_type and any(html in actual_content_type.lower() for html in HTML_TYPES):
            notes.append("HTTP content type indicates HTML, not dataset content.")
        if not self._content_type_ok(actual_content_type, expected_content_type):
            notes.append("Downloaded content type does not match expected MIME type.")
        if not self._format_ok(path, expected_format):
            notes.append("Downloaded file extension does not match expected dataset format.")
        if self._looks_like_html(path):
            notes.append("Downloaded content looks like an HTML, login, or error page.")
        checksum_status, checksum_algorithm = self._checksum_status(path, checksum)
        if checksum_status == "failed":
            notes.append("Downloaded file checksum does not match expected checksum.")
        if checksum_status == "unavailable":
            notes.append("Checksum unavailable from provider metadata.")
        else:
            notes.append(f"Checksum status: {checksum_status}.")
        fatal = (
            "empty",
            "smaller than the expected complete size",
            "HTML",
            "login",
            "error page",
            "does not match expected MIME type",
            "does not match expected dataset format",
            "checksum does not match",
        )
        valid = not any(any(marker in note for marker in fatal) for note in notes)
        if valid:
            notes.insert(0, "File exists and has non-zero size.")
        return valid, notes, checksum_status, checksum_algorithm

    def download_one(self, task: DownloadTask, credentials: Optional[Credentials] = None) -> DownloadResult:
        start = time.time()
        os.makedirs(os.path.dirname(task.dest_path) or ".", exist_ok=True)
        last_error = ""
        retries_attempted = 0
        status_code = None
        content_type = None
        resumed = False
        try:
            for attempt in range(1, max(1, task.max_retries) + 1):
                retries_attempted = attempt - 1
                headers = self._headers(credentials, task)
                mode = "ab" if "Range" in headers else "wb"
                try:
                    with requests.get(task.url, stream=True, timeout=self.timeout, headers=headers, auth=self._auth(credentials)) as resp:
                        status_code = resp.status_code
                        content_type = resp.headers.get("Content-Type")
                        if resp.status_code in (401, 403, 404):
                            return DownloadResult(
                                task.url,
                                task.dest_path,
                                False,
                                error=f"HTTP {resp.status_code}",
                                elapsed_seconds=time.time() - start,
                                status_code=resp.status_code,
                                content_type=content_type,
                                retries_attempted=retries_attempted,
                                validation_status="failed",
                                validation_notes=[f"HTTP {resp.status_code}"],
                            )
                        if "Range" in headers and resp.status_code == 200:
                            mode = "wb"
                        elif "Range" in headers and resp.status_code == 206:
                            resumed = True
                        elif "Range" in headers and resp.status_code == 416:
                            break
                        resp.raise_for_status()
                        if content_type and any(html in content_type.lower() for html in HTML_TYPES):
                            return DownloadResult(
                                task.url,
                                task.dest_path,
                                False,
                                error="HTML response rejected",
                                elapsed_seconds=time.time() - start,
                                status_code=resp.status_code,
                                content_type=content_type,
                                retries_attempted=retries_attempted,
                                validation_status="failed",
                                validation_notes=["HTTP content type indicates HTML, not dataset content."],
                            )
                        total = task.expected_size
                        with open(task.dest_path, mode) as f:
                            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                                if chunk:
                                    f.write(chunk)
                                    if task.progress_callback:
                                        task.progress_callback(task.dest_path, f.tell(), total)
                    break
                except requests.TooManyRedirects as exc:
                    last_error = f"Redirect loop: {exc}"
                    return DownloadResult(
                        task.url,
                        task.dest_path,
                        False,
                        error=last_error,
                        elapsed_seconds=time.time() - start,
                        status_code=status_code,
                        content_type=content_type,
                        retries_attempted=retries_attempted,
                        validation_status="failed",
                        validation_notes=["Redirect loop detected."],
                    )
                except Exception as exc:
                    last_error = str(exc)
                    if attempt >= max(1, task.max_retries):
                        raise
                    time.sleep(2 ** (attempt - 1))
            size = os.path.getsize(task.dest_path) if os.path.exists(task.dest_path) else 0
            valid, notes, checksum_status, checksum_algorithm = self.validate_download(
                task.dest_path,
                expected_size=task.expected_size,
                expected_content_type=task.expected_content_type,
                actual_content_type=content_type,
                expected_format=task.expected_format,
                checksum=task.checksum,
            )
            return DownloadResult(
                task.url,
                task.dest_path,
                valid,
                size,
                time.time() - start,
                "" if valid else "; ".join(notes),
                status_code=status_code,
                content_type=content_type,
                retries_attempted=retries_attempted,
                resumed=resumed,
                checksum_status=checksum_status,
                checksum_algorithm=checksum_algorithm,
                validation_status="passed" if valid else "failed",
                validation_notes=notes,
            )
        except Exception as exc:
            return DownloadResult(
                task.url,
                task.dest_path,
                False,
                error=str(exc) or last_error,
                elapsed_seconds=time.time() - start,
                status_code=status_code,
                content_type=content_type,
                retries_attempted=retries_attempted,
                resumed=resumed,
                validation_status="failed",
                validation_notes=[str(exc) or last_error],
            )

    def download_many(
        self,
        tasks: Iterable[DownloadTask],
        credentials: Optional[Credentials] = None,
        allow_parallel: bool = True,
    ) -> list[DownloadResult]:
        task_list = list(tasks)
        if not task_list:
            return []
        if not allow_parallel:
            return [self.download_one(task, credentials) for task in task_list]
        workers = min(self.max_workers, len(task_list))
        results: list[DownloadResult] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(self.download_one, task, credentials): task for task in task_list}
            for future in as_completed(future_map):
                results.append(future.result())
        return results
