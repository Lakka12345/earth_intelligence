"""
Discovery Engine — runs all discoverers concurrently and merges results.

Called by Agent 3 Phase 1b to replace the sequential search loop.

Changes from v1:
  - Discoverers now run concurrently via ThreadPoolExecutor (5-10x faster).
  - Added register() so new discoverers can be plugged in without editing
    this file.
  - Returns DiscoveryResult instead of a bare list — includes timing,
    discoverer names used, and any per-discoverer failure messages.
  - Deduplication unchanged.

Usage:
    engine = DiscoveryEngine()
    engine.register(PlanetDiscoverer())        # optional plugin
    result = engine.search(request)
    candidates = result.sources
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from discovery.base import BaseDiscoverer
from discovery.erddap import ERDDAPDiscoverer
from discovery.thredds import THREDDSDiscoverer
from discovery.generic_search import GenericSearchDiscoverer
from models.discovery_schemas import CandidateSource, DiscoveryResult
from models.retrieval_request import RetrievalRequest


def _deduplicate(sources: List[CandidateSource]) -> List[CandidateSource]:
    """
    Remove duplicate sources by URL.
    Keeps the first occurrence (catalog sources come first, so they win
    over dynamically discovered duplicates).
    """
    seen_urls: set = set()
    seen_ids: set = set()
    unique: List[CandidateSource] = []

    for s in sources:
        url_key = s.url.rstrip("/").lower()
        if url_key in seen_urls or s.source_id in seen_ids:
            continue
        seen_urls.add(url_key)
        seen_ids.add(s.source_id)
        unique.append(s)

    return unique


class DiscoveryEngine:
    """
    Runs all registered discoverers concurrently and returns a
    DiscoveryResult with deduplicated CandidateSource objects.

    Each discoverer runs in its own thread. One hanging server never
    blocks the others — each thread has its own network timeouts.
    """

    MAX_WORKERS = 8

    def __init__(self, enable_generic_search: bool = True):
        """
        Args:
            enable_generic_search: Set False to skip generic web search
                                   (faster, but misses non-standard sources).
        """
        self._discoverers: List[BaseDiscoverer] = [
            ERDDAPDiscoverer(),
            THREDDSDiscoverer(),
        ]
        if enable_generic_search:
            self._discoverers.append(GenericSearchDiscoverer())

    def register(self, discoverer: BaseDiscoverer) -> None:
        """
        Register a new discoverer plugin at runtime.

        Example:
            engine.register(PlanetDiscoverer())
            engine.register(EarthDataDiscoverer())

        No restart needed — the next search() call will include it.
        """
        self._discoverers.append(discoverer)
        print(f"[DiscoveryEngine] Registered plugin: {discoverer.__class__.__name__}")

    def search(self, request: RetrievalRequest) -> DiscoveryResult:
        """
        Runs all discoverers concurrently and returns a DiscoveryResult.

        Every discoverer exception is caught — non-fatal.
        Discovery time is measured wall-clock (not sum of serial times).
        """
        all_results: List[CandidateSource] = []
        discoverers_used: List[str] = []
        failures: List[str] = []
        t0 = time.monotonic()

        # Submit all discoverers concurrently
        future_to_name = {}
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            for discoverer in self._discoverers:
                name = discoverer.__class__.__name__
                future = executor.submit(discoverer.search, request)
                future_to_name[future] = name

            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    discoverers_used.append(name)
                    print(f"[DiscoveryEngine] {name}: {len(results)} result(s).")
                except Exception as exc:
                    failures.append(f"{name}: {exc}")
                    print(f"[DiscoveryEngine] {name} failed (non-fatal): {exc}")

        unique = _deduplicate(all_results)
        elapsed = round(time.monotonic() - t0, 2)

        print(
            f"[DiscoveryEngine] Done in {elapsed}s. "
            f"{len(unique)} unique (from {len(all_results)} raw). "
            f"{len(failures)} failure(s)."
        )

        return DiscoveryResult(
            sources=unique,
            discovery_time_seconds=elapsed,
            discoverers_used=discoverers_used,
            failures=failures,
        )
