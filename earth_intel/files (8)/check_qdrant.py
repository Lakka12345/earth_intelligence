"""
check_qdrant.py — Diagnostic script for Earth Intelligence Qdrant memory.

Run from project root:
    python check_qdrant.py

Shows:
  1. Whether Qdrant + embedding model are reachable
  2. Point counts in both collections
  3. Sample payloads from source_discoveries (so you can see what's stored)
  4. A live similarity search against a test query, with and without the
     dataset_type filter, so you can see exactly why Phase 1c returns 0.
  5. All unique dataset_type values stored, so you can check for mismatches.
"""

import sys

# ------------------------------------------------------------------ #
# 1. Connectivity                                                      #
# ------------------------------------------------------------------ #

print("\n" + "=" * 60)
print("QDRANT DIAGNOSTIC")
print("=" * 60)

try:
    from memory.qdrant_store import (
        get_client,
        is_qdrant_available,
        embed_text,
        search_similar_discoveries,
        COLLECTION_DISCOVERIES,
        COLLECTION_RELIABILITY,
        SIMILARITY_THRESHOLD,
    )
except ImportError as e:
    print(f"\n[FATAL] Could not import qdrant_store: {e}")
    print("Make sure you are running from the project root.")
    sys.exit(1)

print("\n[1] Checking connectivity...")
available = is_qdrant_available()
print(f"    is_qdrant_available() → {available}")
if not available:
    print("    Qdrant is not reachable or sentence-transformers is missing.")
    print("    Check: is Qdrant running on localhost:6333?")
    print("    Check: pip install sentence-transformers qdrant-client")
    sys.exit(1)

client = get_client()

# ------------------------------------------------------------------ #
# 2. Collection existence + point counts                              #
# ------------------------------------------------------------------ #

print("\n[2] Collections and point counts...")
existing = {c.name for c in client.get_collections().collections}
print(f"    Collections in Qdrant: {sorted(existing)}")

for col in [COLLECTION_DISCOVERIES, COLLECTION_RELIABILITY]:
    if col not in existing:
        print(f"    {col}: DOES NOT EXIST — run Agent 3 once to create it.")
    else:
        count = client.count(collection_name=col, exact=True)
        print(f"    {col}: {count.count} point(s)")

# ------------------------------------------------------------------ #
# 3. Sample payloads from source_discoveries                          #
# ------------------------------------------------------------------ #

print(f"\n[3] Sample payloads from '{COLLECTION_DISCOVERIES}'...")
if COLLECTION_DISCOVERIES in existing:
    sample, _ = client.scroll(
        collection_name=COLLECTION_DISCOVERIES,
        limit=5,
        with_payload=True,
        with_vectors=False,
    )
    if not sample:
        print("    No points found — storage has never run successfully.")
    else:
        print(f"    Showing up to 5 of {client.count(collection_name=COLLECTION_DISCOVERIES, exact=True).count} point(s):\n")
        for i, point in enumerate(sample, 1):
            p = point.payload or {}
            print(f"    [{i}] source_id     : {p.get('source_id')}")
            print(f"         source_name   : {p.get('source_name')}")
            print(f"         dataset_type  : {p.get('dataset_type')}")
            print(f"         final_score   : {p.get('final_score')}")
            print(f"         access_type   : {p.get('access_type')}")
            print(f"         stored_at     : {p.get('stored_at')}")
            print(f"         query_goal    : {p.get('query_goal', '')[:80]}")
            print()

# ------------------------------------------------------------------ #
# 4. All unique dataset_type values                                   #
# ------------------------------------------------------------------ #

print(f"\n[4] All unique dataset_type values in '{COLLECTION_DISCOVERIES}'...")
if COLLECTION_DISCOVERIES in existing:
    all_points, _ = client.scroll(
        collection_name=COLLECTION_DISCOVERIES,
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )
    types_seen = sorted({p.payload.get("dataset_type", "MISSING") for p in all_points if p.payload})
    print(f"    Stored dataset_type values: {types_seen}")
    print()
    print("    If Phase 1c is passing dataset_types that don't appear above,")
    print("    the filter will exclude all stored points.")
    print("    The fixed qdrant_store.py now also matches 'unknown' as a fallback.")

# ------------------------------------------------------------------ #
# 5. Live similarity search — with and without filter                 #
# ------------------------------------------------------------------ #

TEST_QUERY = "sea surface temperature satellite observations Indian Ocean"
TEST_TYPES = ["satellite", "ocean"]

print(f"\n[5] Live similarity search...")
print(f"    Test query : '{TEST_QUERY}'")
print(f"    Threshold  : {SIMILARITY_THRESHOLD}")

if COLLECTION_DISCOVERIES in existing and client.count(collection_name=COLLECTION_DISCOVERIES, exact=True).count > 0:
    # 5a. No filter — should always return results if anything is stored
    print(f"\n    [5a] Search WITHOUT filter (baseline):")
    try:
        raw = client.search(
            collection_name=COLLECTION_DISCOVERIES,
            query_vector=embed_text(TEST_QUERY),
            limit=5,
            score_threshold=0.0,   # no threshold — show everything
        )
        if raw:
            for hit in raw:
                p = hit.payload or {}
                print(f"         score={hit.score:.3f}  type={p.get('dataset_type')}  name={p.get('source_name')}")
        else:
            print("         No results — collection may be empty or vector dim mismatch.")
    except Exception as e:
        print(f"         Search failed: {e}")

    # 5b. With threshold — same as Phase 1c
    print(f"\n    [5b] Search WITH threshold={SIMILARITY_THRESHOLD} (same as Phase 1c):")
    try:
        thresholded = client.search(
            collection_name=COLLECTION_DISCOVERIES,
            query_vector=embed_text(TEST_QUERY),
            limit=5,
            score_threshold=SIMILARITY_THRESHOLD,
        )
        if thresholded:
            for hit in thresholded:
                p = hit.payload or {}
                print(f"         score={hit.score:.3f}  type={p.get('dataset_type')}  name={p.get('source_name')}")
        else:
            print(f"         No results above {SIMILARITY_THRESHOLD}.")
            print(f"         → Try lowering SIMILARITY_THRESHOLD in qdrant_store.py (currently {SIMILARITY_THRESHOLD}).")
            print(f"           0.60 is a reasonable starting value for scientific queries.")
    except Exception as e:
        print(f"         Search failed: {e}")

    # 5c. Via the actual search_similar_discoveries function (full pipeline)
    print(f"\n    [5c] Via search_similar_discoveries() with types={TEST_TYPES}:")
    results = search_similar_discoveries(TEST_QUERY, TEST_TYPES)
    print(f"         Returned {len(results)} payload(s).")
    for r in results:
        print(f"           name={r.get('source_name')}  type={r.get('dataset_type')}  score=n/a (payload only)")

else:
    print("    Skipping search — collection is empty or doesn't exist.")
    print("    Run Agent 3 once first, then re-run this script.")

# ------------------------------------------------------------------ #
# 6. Summary                                                          #
# ------------------------------------------------------------------ #

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print()
if COLLECTION_DISCOVERIES not in existing:
    print("  ✗ source_discoveries collection does not exist.")
    print("    → ensure_collections_exist() has never been called.")
    print("    → Check that Agent 3 calls qdrant_store.ensure_collections_exist().")
elif client.count(collection_name=COLLECTION_DISCOVERIES, exact=True).count == 0:
    print("  ✗ source_discoveries exists but is empty.")
    print("    → phase6_store_to_qdrant() is either not being called,")
    print("      or store_discovery_result() is silently failing.")
    print("    → Check the Agent 3 log for '[Phase 6] Stored N source(s) to Qdrant'.")
    print("    → If that line is missing, Phase 6 is being skipped.")
else:
    print("  ✓ source_discoveries has data.")
    print("    → If Phase 1c still returns 0, the issue is:")
    print("       (a) SIMILARITY_THRESHOLD too high — lower it to 0.60")
    print("       (b) dataset_type filter mismatch — check [4] above")
    print("       (c) access_type missing from stored payload — update qdrant_store.py")

print()
