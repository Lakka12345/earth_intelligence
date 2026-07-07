"""
Generic CKAN provider connector — Web Retrieval Layer.

Retrieval-only. Complements (does NOT duplicate) Agent 3's existing
Phase 3 CKAN probe. Phase 3's _enrich_from_ckan already tries
GET {base}/api/3/action/package_show?id={source_id}, which requires
knowing the exact package id. This connector instead tries
/api/3/action/package_search?q=..., which surfaces results by
free-text name when the id doesn't match exactly.

Does NOT score, rank, or filter — only returns metadata or None.
"""

from typing import Any, Dict, Optional

import requests

_TIMEOUT = 8
_HEADERS = {"User-Agent": "EarthIntelligenceAgent/1.0"}


def fetch(base_url: str, candidate_name: str) -> Optional[Dict[str, Any]]:
    """
    Queries {base}/api/3/action/package_search?q={candidate_name}.

    Returns a plain dict of whatever fields were found, or None.
    Possible returned keys: description, temporal_coverage,
    spatial_coverage, metadata_url.
    """
    if not base_url or not candidate_name:
        return None

    base = base_url.rstrip("/")
    if "/dataset/" in base:
        base = base.split("/dataset/")[0]
    search_url = f"{base}/api/3/action/package_search"

    try:
        resp = requests.get(
            search_url,
            params={"q": candidate_name, "rows": 1},
            timeout=_TIMEOUT,
            headers=_HEADERS,
        )
        if resp.status_code != 200:
            return None

        results = resp.json().get("result", {}).get("results", [])
        if not results:
            return None

        pkg = results[0]
        result: Dict[str, Any] = {}

        notes = pkg.get("notes") or pkg.get("title")
        if notes:
            result["description"] = notes[:400]

        start = pkg.get("temporal_start") or pkg.get("data_collection_start_date")
        end = pkg.get("temporal_end") or pkg.get("data_collection_end_date") or "present"
        if start:
            result["temporal_coverage"] = f"{start} – {end}"

        spatial = pkg.get("spatial") or pkg.get("bbox")
        if spatial:
            result["spatial_coverage"] = str(spatial)[:120]

        pkg_name = pkg.get("name")
        if pkg_name:
            result["metadata_url"] = f"{base}/dataset/{pkg_name}"

        return result or None

    except Exception:
        return None
