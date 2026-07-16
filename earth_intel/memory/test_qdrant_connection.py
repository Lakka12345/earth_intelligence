"""
Standalone Qdrant diagnostic — run this directly to isolate exactly
which piece is broken, instead of debugging through the full Agent 3
pipeline.

    python test_qdrant_connection.py
"""

import sys


def check(label, fn):
    print(f"\n[{label}]")
    try:
        result = fn()
        print(f"  OK: {result}")
        return True
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return False


def main():
    import qdrant_store as qs

    ok_client = check("1. Client can connect to Qdrant server", lambda: qs.get_client().get_collections())
    if not ok_client:
        print("\n-> Qdrant server itself is unreachable. Check `docker ps` and `curl http://localhost:6333/` first.")
        sys.exit(1)

    ok_embed = check("2. Embedding model loads and encodes", lambda: len(qs.embed_text("sea surface temperature")))
    if not ok_embed:
        print("\n-> Qdrant server is fine, but sentence-transformers isn't working.")
        print("   Run: pip install sentence-transformers")
        sys.exit(1)

    ok_collections = check("3. Collections can be created/verified", qs.ensure_collections_exist)

    ok_store = check(
        "4. Round-trip store",
        lambda: qs.store_discovery_result(
            query_goal="diagnostic test query",
            dataset_types=["test"],
            scored_sources=[{
                "source_id": "diagnostic_test_source",
                "name": "Diagnostic Test Source",
                "url": "https://example.com",
                "dataset_type": "test",
                "final_score": 0.9,
                "recommendation": "use",
            }],
        ),
    )

    ok_search = check(
        "5. Round-trip search (should find the point just stored)",
        lambda: qs.search_similar_discoveries("diagnostic test query", ["test"], top_k=1),
    )

    ok_reliability = check(
        "6. Reliability update round-trip",
        lambda: qs.update_source_reliability("diagnostic_test_source", succeeded=True),
    )

    print("\n" + "=" * 50)
    if all([ok_client, ok_embed, ok_collections, ok_store, ok_search, ok_reliability]):
        print("ALL CHECKS PASSED — Qdrant integration is fully working.")
    else:
        print("Some checks failed — see details above.")
    print("=" * 50)


if __name__ == "__main__":
    main()
