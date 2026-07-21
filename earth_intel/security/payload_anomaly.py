# security/payload_anomaly.py
"""
Detects structurally-valid but scientifically-empty payloads (all-zero or
all-NaN arrays) and returns a warning severity rather than a fatal error.
Intended to be called after file-level integrity passes.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# Fraction of values that must be zero/NaN before we flag it.
_FLATLINE_THRESHOLD = 0.95   # 95 % zeros/NaNs in a single array → flatline
_MIN_ARRAY_LENGTH   = 3      # ignore tiny arrays (e.g. 3-element bbox)


@dataclass
class PayloadAnomalyResult:
    file_path: str
    severity: str           # "ok" | "warning" | "error"
    flatline_arrays: List[str] = field(default_factory=list)
    anomaly_description: str = ""
    is_fatal: bool = False  # always False for flatline — let orchestrator decide


def _is_flatline(values: list) -> bool:
    """True if ≥ FLATLINE_THRESHOLD fraction of entries are 0 or NaN."""
    if len(values) < _MIN_ARRAY_LENGTH:
        return False
    bad = sum(
        1 for v in values
        if v == 0 or v == 0.0 or (isinstance(v, float) and math.isnan(v))
    )
    return (bad / len(values)) >= _FLATLINE_THRESHOLD


def _scan_json(data: Any, path: str = "") -> List[str]:
    """Recursively walk a decoded JSON structure; return dotted paths of
    flatline numeric arrays."""
    flatlines: List[str] = []
    if isinstance(data, dict):
        for k, v in data.items():
            flatlines.extend(_scan_json(v, f"{path}.{k}" if path else k))
    elif isinstance(data, list):
        # Is this a leaf numeric array?
        if data and all(isinstance(v, (int, float)) for v in data):
            if _is_flatline(data):
                flatlines.append(path)
        else:
            for i, item in enumerate(data):
                flatlines.extend(_scan_json(item, f"{path}[{i}]"))
    return flatlines


def _scan_csv(text: str) -> List[str]:
    """Return column names whose numeric values are flatline."""
    flatlines: List[str] = []
    reader = csv.DictReader(io.StringIO(text))
    columns: Dict[str, List[float]] = {}
    for row in reader:
        for col, raw in row.items():
            try:
                columns.setdefault(col, []).append(float(raw))
            except (ValueError, TypeError):
                pass
    for col, values in columns.items():
        if _is_flatline(values):
            flatlines.append(col)
    return flatlines


def check_payload_anomalies(file_path: str | Path) -> PayloadAnomalyResult:
    """
    Inspect a downloaded file for flatline (all-zero / all-NaN) arrays.

    Returns a PayloadAnomalyResult with:
      severity = "warning"  → structurally valid but scientifically empty
      severity = "ok"         → no anomalies detected
      severity = "error"     → file could not be parsed (structural failure;
                                handled separately by integrity/validation stages)

    NEVER returns is_fatal=True — the orchestrator decides whether to rely
    on a fallback source; this function only classifies.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    flatlines: List[str] = []

    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return PayloadAnomalyResult(
            file_path=str(path),
            severity="error",
            anomaly_description=f"Could not read file: {exc}",
        )

    if suffix == ".json":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            return PayloadAnomalyResult(
                file_path=str(path),
                severity="error",
                anomaly_description=f"Invalid JSON: {exc}",
            )
        flatlines = _scan_json(data)

    elif suffix in (".csv", ".tsv"):
        flatlines = _scan_csv(raw)

    else:
        # DAT / NetCDF / HDF5 / binary — skip array scan; return ok so
        # downstream binary validators handle those formats.
        return PayloadAnomalyResult(file_path=str(path), severity="ok")

    if flatlines:
        desc = (
            f"Empty Payload Warning: {len(flatlines)} array(s) contain "
            f"≥{int(_FLATLINE_THRESHOLD*100)}% zeros/NaNs — "
            f"possible missing or masked data rather than a real signal. "
            f"Affected field(s): {', '.join(flatlines[:10])}"
            + (" …" if len(flatlines) > 10 else "")
        )
        return PayloadAnomalyResult(
            file_path=str(path),
            severity="warning",
            flatline_arrays=flatlines,
            anomaly_description=desc,
            is_fatal=False,
        )

    return PayloadAnomalyResult(file_path=str(path), severity="ok")