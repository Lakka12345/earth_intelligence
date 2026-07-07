"""
security/integrity.py — Dataset Integrity Verification.

A standalone, agent-independent module for computing, storing, and
verifying SHA-256 checksums of dataset files. Intended for later use by
Agent 4 (dataset download) and Agent 5 (dataset processing), but has no
import-time dependency on either — every function here takes plain
values (Path, str, dict) so it can be imported and used in isolation,
including from this file's own demo/test block.

Manifest storage:
    All integrity records are appended to a single JSON file at
    security/manifests/integrity_manifest.json, stored as a JSON array of
    record objects. Records are never deleted or overwritten by this
    module — store_integrity_record only appends.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TypedDict


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"
MANIFEST_PATH = MANIFEST_DIR / "integrity_manifest.json"

DEFAULT_CHUNK_SIZE = 65536  # 64 KB — large enough to be efficient, small
                            # enough to keep memory flat for very large files


class IntegrityRecord(TypedDict):
    dataset_id: str
    dataset_name: str
    provider: str
    filename: str
    file_path: str
    file_size: str
    algorithm: str
    checksum: str
    timestamp: str


# --------------------------------------------------------------------------- #
# Checksum computation
# --------------------------------------------------------------------------- #

def compute_sha256(file_path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """
    Compute the SHA-256 hash of a file.

    Reads the file in fixed-size chunks rather than loading it entirely
    into memory, so this is safe to use on very large dataset files
    (multi-GB satellite products, etc.).

    Args:
        file_path: path to the file to hash.
        chunk_size: number of bytes to read per chunk (default 64 KB).

    Returns:
        The lowercase hex-encoded SHA-256 digest of the file's contents.

    Raises:
        FileNotFoundError: if `file_path` does not exist or is not a file.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Cannot compute checksum — file not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Cannot compute checksum — not a file: {path}")

    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Manifest read/write helpers
# --------------------------------------------------------------------------- #

def _ensure_manifest_dir() -> None:
    """Create the manifests directory if it doesn't exist yet."""
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)


def _load_manifest() -> list[dict]:
    """
    Load the integrity manifest as a list of record dicts.

    Handles a missing manifest file (returns an empty list, does not
    error) and invalid/corrupt JSON (backs up the corrupt file alongside
    itself with a .corrupt suffix, prints a warning, and returns an empty
    list rather than crashing or silently losing the corrupt data).
    """
    if not MANIFEST_PATH.exists():
        return []

    try:
        raw_text = MANIFEST_PATH.read_text(encoding="utf-8")
        if not raw_text.strip():
            return []
        data = json.loads(raw_text)
    except (json.JSONDecodeError, OSError) as exc:
        backup_path = MANIFEST_PATH.with_suffix(".json.corrupt")
        try:
            MANIFEST_PATH.replace(backup_path)
            print(
                f"[integrity] Warning: manifest at {MANIFEST_PATH} was invalid "
                f"({exc}). Backed up to {backup_path} and starting a fresh manifest."
            )
        except OSError:
            print(
                f"[integrity] Warning: manifest at {MANIFEST_PATH} was invalid "
                f"({exc}) and could not be backed up. Starting a fresh in-memory manifest."
            )
        return []

    if not isinstance(data, list):
        print(
            f"[integrity] Warning: manifest at {MANIFEST_PATH} did not contain "
            "a JSON array as expected. Starting a fresh manifest."
        )
        return []

    return data


def _save_manifest(records: list[dict]) -> None:
    """
    Write the full list of records back to the manifest file.

    Writes to a temporary file first and then replaces the real manifest
    atomically, so a crash or interruption mid-write can never leave the
    manifest half-written or corrupted.
    """
    _ensure_manifest_dir()

    tmp_path = MANIFEST_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    tmp_path.replace(MANIFEST_PATH)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def store_integrity_record(
    dataset_id: str,
    dataset_name: str,
    provider: str,
    file_path: Path,
    checksum: Optional[str] = None,
    algorithm: str = "SHA-256",
) -> IntegrityRecord:
    """
    Compute (if needed) and store a new integrity record for a dataset
    file, appending it to security/manifests/integrity_manifest.json
    without deleting or overwriting any existing records.

    Args:
        dataset_id: stable identifier for the dataset (e.g. source_id
            from Agent 3's CandidateSource, or a download-specific ID).
        dataset_name: human-readable dataset name.
        provider: the provider/catalog the dataset came from.
        file_path: path to the downloaded file to checksum.
        checksum: precomputed SHA-256 hex digest. If omitted, it is
            computed from `file_path` via compute_sha256.
        algorithm: hash algorithm label stored in the record. Defaults to
            "SHA-256" — the only algorithm this module currently computes.

    Returns:
        The newly stored record, as a dict matching IntegrityRecord.

    Raises:
        FileNotFoundError: if `checksum` is not provided and `file_path`
            does not exist (propagated from compute_sha256).
    """
    if checksum is None:
        checksum = compute_sha256(Path(file_path))

    path = Path(file_path)

    record: IntegrityRecord = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "provider": provider,
        "filename": path.name,
        "file_path": str(Path(file_path)),
        "file_size": path.stat().st_size,
        "algorithm": algorithm,
        "checksum": checksum,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    records = _load_manifest()
    records.append(record)
    _save_manifest(records)

    return record


def get_integrity_record(dataset_id: str) -> Optional[IntegrityRecord]:
    """
    Return the most recently stored integrity record for `dataset_id`.

    If a dataset has been checksummed more than once (e.g. re-downloaded),
    the most recent record (last appended) is returned, since it reflects
    the current expected state of the file.

    Args:
        dataset_id: the dataset identifier to look up.

    Returns:
        The matching record dict, or None if no record exists for this
        dataset_id (including the case where the manifest itself is
        missing or empty).
    """
    records = _load_manifest()

    matches = [r for r in records if r.get("dataset_id") == dataset_id]
    if not matches:
        return None

    return matches[-1]  # most recent


def _get_record_for_file(file_path: Path) -> Optional[dict]:
    """Internal helper: find the most recent record matching a file_path."""
    records = _load_manifest()
    target = str(Path(file_path))

    matches = [r for r in records if r.get("file_path") == target]
    if not matches:
        return None

    return matches[-1]


def verify_integrity(file_path: Path) -> dict:
    """
    Verify that a file on disk still matches its stored integrity record.

    Recomputes the file's SHA-256 checksum and compares it against the
    checksum most recently stored for that file_path via
    store_integrity_record.

    Args:
        file_path: path to the file to verify.

    Returns:
        A dict with keys "passed", "expected", "current", and "message".

        - If no integrity record exists for this file_path at all,
          "passed" is False, "expected" is None, and "message" explains
          that no record was found (this is distinct from a verification
          failure — there is nothing to compare against yet).
        - If the file is missing on disk, "passed" is False, "current" is
          None, and "message" explains the file could not be read.
        - Otherwise, the checksums are compared and the standard
          passed/failed shape is returned.
    """
    expected_record = _get_record_for_file(file_path)

    if expected_record is None:
        return {
            "passed": False,
            "expected": None,
            "current": None,
            "message": (
                f"No integrity record found for file: {file_path}. "
                "Nothing to verify against — store a record first."
            ),
        }

    expected_checksum = expected_record["checksum"]

    try:
        current_checksum = compute_sha256(Path(file_path))
    except FileNotFoundError:
        return {
            "passed": False,
            "expected": expected_checksum,
            "current": None,
            "message": f"Cannot verify — file not found on disk: {file_path}",
        }

    if current_checksum == expected_checksum:
        return {
            "passed": True,
            "expected": expected_checksum,
            "current": current_checksum,
            "message": "Integrity verification passed.",
        }

    return {
        "passed": False,
        "expected": expected_checksum,
        "current": current_checksum,
        "message": "Dataset integrity verification failed.",
    }


# --------------------------------------------------------------------------- #
# Demo / self-test — runnable independently of any agent
# --------------------------------------------------------------------------- #

def run_demo() -> None:
    """
    Demonstrates, end to end and independently of any agent:
      1. Computing a checksum for a sample file.
      2. Storing it as an integrity record.
      3. Verifying the unmodified file (expected: passes).
      4. Modifying the file and verifying again (expected: fails).

    Uses a temporary file under the manifests directory so this can be
    run with no other setup (`python -m security.integrity` or
    `python security/integrity.py`).
    """
    import tempfile

    print("=" * 60)
    print("security.integrity — demo")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        demo_file = Path(tmp_dir) / "demo_dataset.txt"
        demo_file.write_text("original dataset content — version 1\n", encoding="utf-8")

        print(f"\n1. Computing checksum for {demo_file} ...")
        checksum = compute_sha256(demo_file)
        print(f"   SHA-256: {checksum}")

        print("\n2. Storing integrity record ...")
        record = store_integrity_record(
            dataset_id="demo_dataset_001",
            dataset_name="Demo Dataset",
            provider="Demo Provider",
            file_path=demo_file,
            checksum=checksum,
        )
        print(f"   Stored record: {record}")

        print("\n3. Verifying unmodified file ...")
        result = verify_integrity(demo_file)
        print(f"   Result: {result}")
        assert result["passed"] is True, "Expected verification to pass on unmodified file."

        print("\n4. Modifying file and verifying again ...")
        demo_file.write_text("TAMPERED dataset content — version 2\n", encoding="utf-8")
        result = verify_integrity(demo_file)
        print(f"   Result: {result}")
        assert result["passed"] is False, "Expected verification to fail on modified file."

        print("\n5. get_integrity_record() lookup by dataset_id ...")
        looked_up = get_integrity_record("demo_dataset_001")
        print(f"   Stored record retrieved by dataset_id: {looked_up}")
        assert looked_up is not None and looked_up["checksum"] == checksum

    print("\nDemo completed successfully — all expected outcomes matched.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
