"""
Retrieval Manager — Web Retrieval Layer for Agent 3 (plug-in / extension).

WHAT THIS IS
  A self-contained add-on. Agent 3's existing discovery, ranking,
  filtering, scoring, and Agent-4 hand-off logic are completely
  untouched. This module's only job is: given a candidate that already
  came out of Agent 3's own pipeline, try to fetch a little more
  metadata about it from official provider APIs / catalogs, and hand
  that metadata back. Nothing here decides whether a candidate is good,
  relevant, or should be kept -- that is 100% still Agent 3's job.

WHY IT EXISTS (what it adds that Agent 3 doesn't already do)
  Agent 3's Phase 3 (_enrich_from_html / _enrich_from_stac /
  _enrich_from_ckan / _enrich_from_erddap / _enrich_from_metadata_url,
  all in agent3_discovery.py) already does protocol-guessing enrichment
  directly against the candidate's own URL. This module does NOT
  reimplement any of that. It adds a different, complementary source of
  metadata: official provider search APIs (NASA CMR, Copernicus Data
  Space, NOAA NCEI, USGS ScienceBase) plus free-text STAC/CKAN *search*
  (as opposed to Phase 3's direct id lookups), which can surface
  metadata Phase 3's direct-URL approach cannot reach.

RETRIEVAL PRIORITY (per spec)
  1. Official provider API   (nasa / copernicus / noaa / usgs)
  2. STAC / CKAN / JSON catalog search
  3. Structured webpage metadata -- intentionally SKIPPED here, because
     Agent 3's own Phase 3 (_enrich_from_html) already covers this tier.
     Reimplementing it here would violate the "no duplicate logic" rule.
  No browser automation is used, per spec.

SAFETY / NON-INVASIVENESS GUARANTEES
  - Only fills fields that are still empty/"Unknown". Never overwrites
    anything Agent 3's own pipeline already found.
  - Every network call is wrapped in try/except inside the provider
    modules themselves; this manager adds a second layer of try/except
    so a bug in routing logic can never propagate into Agent 3.
  - If this whole package is deleted, agent3_discovery.py's defensive
    import (see the try/except around `from retrieval import
    retrieval_manager`) makes Agent 3 fall back to running exactly as
    it did before this layer existed.
"""

from typing import Any, Dict, List, Optional

from retrieval.providers import OFFICIAL_API_PROVIDERS, stac as stac_provider, ckan as ckan_provider


# Fields this layer is allowed to fill -- intentionally the same set
# Agent 3's own Phase 3 enrichment already treats as "fillable" (see
# _enrich_from_html/_enrich_from_stac/_enrich_from_ckan in
# agent3_discovery.py). Kept as an explicit whitelist so this module can
# never accidentally set a field Agent 3's own logic doesn't expect.
_FILLABLE_FIELDS = (
    "description",
    "spatial_coverage",
    "temporal_coverage",
    "metadata_url",
    "variables_available",
)


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in ("", "Unknown")
    if isinstance(value, list):
        return len(value) == 0
    return False


def _needs_more_metadata(candidate) -> bool:
    """A candidate is worth an extra retrieval call only if Agent 3's
    own Phase 3 enrichment left real gaps behind."""
    return (
        _is_empty(getattr(candidate, "description", None))
        or _is_empty(getattr(candidate, "spatial_coverage", None))
        or _is_empty(getattr(candidate, "temporal_coverage", None))
    )


def _match_official_provider(candidate):
    """Tier 1 -- match candidate against a known official-API provider
    by keyword in its name / url / discovery_origin. Returns the
    provider module, or None."""
    haystack = " ".join(filter(None, [
        getattr(candidate, "name", "") or "",
        getattr(candidate, "url", "") or "",
        getattr(candidate, "discovery_origin", "") or "",
        getattr(candidate, "source_id", "") or "",
    ])).lower()

    for keyword, provider_module in OFFICIAL_API_PROVIDERS.items():
        if keyword in haystack:
            return provider_module
    return None


def _fetch_extra_metadata(candidate, request) -> Optional[Dict[str, Any]]:
    """
    Tries providers in priority order for a single candidate.
    Returns a plain metadata dict, or None if nothing was found.
    Never raises.
    """
    try:
        # Tier 1 — official provider API
        provider_module = _match_official_provider(candidate)
        if provider_module is not None:
            data = provider_module.fetch(
                getattr(candidate, "name", ""),
                getattr(candidate, "source_id", ""),
            )
            if data:
                return data

        # Tier 2 — STAC / CKAN catalog search
        api_type = str(getattr(candidate, "api_type", "")).lower()
        base_url = getattr(candidate, "url", "")
        name = getattr(candidate, "name", "")

        if "stac" in api_type:
            data = stac_provider.fetch(base_url, name)
            if data:
                return data

        if "ckan" in api_type:
            data = ckan_provider.fetch(base_url, name)
            if data:
                return data

        # Tier 3 (structured webpage metadata) intentionally skipped —
        # already handled by Agent 3's own Phase 3 (_enrich_from_html).
        return None

    except Exception:
        return None


def _apply_metadata(candidate, data: Dict[str, Any]) -> None:
    """Fills only currently-empty fields on the candidate. Never
    overwrites anything Agent 3's own pipeline already populated."""
    for field in _FILLABLE_FIELDS:
        if field not in data:
            continue
        current = getattr(candidate, field, None)
        if _is_empty(current):
            try:
                setattr(candidate, field, data[field])
            except Exception:
                # Defensive only — a schema mismatch here must never
                # break Agent 3's pipeline.
                continue


def enrich_candidates(candidates: List, request) -> List:
    """
    THE single entry point Agent 3 calls.

    Given the candidate list Agent 3 already produced (after its own
    Phase 3 metadata probe), tries to fill remaining metadata gaps from
    official provider APIs and catalog search. Returns the same list,
    mutated in place, for a one-line call-site integration.

    Guaranteed non-fatal: any failure, for any candidate, at any stage,
    is swallowed and that candidate is simply left as Agent 3 already
    had it.
    """
    filled_count = 0
    for candidate in candidates:
        try:
            if not _needs_more_metadata(candidate):
                continue
            data = _fetch_extra_metadata(candidate, request)
            if data:
                _apply_metadata(candidate, data)
                filled_count += 1
        except Exception:
            continue

    print(
        f"[Retrieval Manager] Web Retrieval Layer: filled additional "
        f"metadata for {filled_count}/{len(candidates)} candidate(s)."
    )
    return candidates
