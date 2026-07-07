"""
Qdrant memory store for the Earth Intelligence system.

All Qdrant operations are isolated here.
No other file should import qdrant_client directly.

What gets stored:
- Source discovery results (scored sources per query)
- Source reliability history (updated when retrieval succeeds/fails)
- Query embeddings (for semantic similarity search)

When a new query comes in, Agent 3:
1. Embeds the query goal
2. Searches Qdrant for semantically similar past discoveries
3. If found — adjusts historical_reliability scores from memory
4. After scoring — stores new results back

Collections:
- "source_discoveries"  : one point per (query, source) pair
- "source_reliability"  : one point per source_id (updated over time)
"""

import json
import time
import uuid
from typing import Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models


def _stable_point_id(*parts: str) -> str:
    """
    CHANGED (bug fix): previously point IDs were built with Python's
    built-in hash(), which is randomized per-process by default
    (PYTHONHASHSEED). That meant the same source_id/query produced a
    DIFFERENT point ID on every restart, so upsert() silently created a
    new duplicate point each run instead of updating the existing one --
    reliability history never actually accumulated, and
    search_similar_discoveries would fill up with near-duplicate points
    from repeat runs of the same query.

    uuid5 is deterministic: the same input string always produces the
    same UUID, in this process or any other, forever. Qdrant accepts
    UUID strings as point IDs natively.
    """
    key = "_".join(parts)
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
EMBEDDING_DIM = 384            # matches sentence-transformers all-MiniLM-L6-v2

COLLECTION_DISCOVERIES = "source_discoveries"
COLLECTION_RELIABILITY = "source_reliability"

SIMILARITY_THRESHOLD = 0.75    # minimum cosine similarity to use cached result
TOP_K_RESULTS = 5              # how many similar past queries to retrieve


# ------------------------------------------------------------------ #
# Embedding — lightweight local model, no API cost                     #
# ------------------------------------------------------------------ #

_embedding_model = None


def _get_embedding_model():
    """
    Lazy-loads sentence-transformers model.
    Uses all-MiniLM-L6-v2 — 80MB, runs on CPU, no GPU needed.
    """
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def embed_text(text: str) -> List[float]:
    """Embed a string into a vector. Used for query similarity search."""
    model = _get_embedding_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


# ------------------------------------------------------------------ #
# Client                                                               #
# ------------------------------------------------------------------ #

_qdrant_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    """Returns a singleton Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _qdrant_client


# ------------------------------------------------------------------ #
# Collection setup                                                     #
# ------------------------------------------------------------------ #

def ensure_collections_exist() -> None:
    """
    Creates Qdrant collections if they don't exist.
    Safe to call multiple times — idempotent.
    """
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}

    if COLLECTION_DISCOVERIES not in existing:
        client.create_collection(
            collection_name=COLLECTION_DISCOVERIES,
            vectors_config=qdrant_models.VectorParams(
                size=EMBEDDING_DIM,
                distance=qdrant_models.Distance.COSINE,
            ),
        )
        print(f"Created Qdrant collection: {COLLECTION_DISCOVERIES}")

    if COLLECTION_RELIABILITY not in existing:
        client.create_collection(
            collection_name=COLLECTION_RELIABILITY,
            vectors_config=qdrant_models.VectorParams(
                size=EMBEDDING_DIM,
                distance=qdrant_models.Distance.COSINE,
            ),
        )
        print(f"Created Qdrant collection: {COLLECTION_RELIABILITY}")


# ------------------------------------------------------------------ #
# Search — find similar past discoveries                               #
# ------------------------------------------------------------------ #

def search_similar_discoveries(
    query_goal: str,
    dataset_types: List[str],
    top_k: int = TOP_K_RESULTS,
) -> List[Dict]:
    """
    Searches Qdrant for past discoveries similar to the current query.

    Returns a list of payloads from the most similar past queries
    above the similarity threshold.

    Returns empty list if Qdrant is unavailable or no matches found.
    """
    try:
        client = get_client()
        query_vector = embed_text(query_goal)

        results = client.search(
            collection_name=COLLECTION_DISCOVERIES,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=SIMILARITY_THRESHOLD,
            query_filter=qdrant_models.Filter(
                should=[
                    qdrant_models.FieldCondition(
                        key="dataset_type",
                        match=qdrant_models.MatchValue(value=dt),
                    )
                    for dt in dataset_types
                ]
            ) if dataset_types else None,
        )

        return [hit.payload for hit in results if hit.payload]

    except Exception as exc:
        print(f"[Qdrant] Search failed (non-fatal): {exc}")
        return []


def get_source_reliability_history(source_id: str) -> Optional[Dict]:
    """
    Returns historical reliability data for a specific source_id.
    Returns None if source has never been used before.
    """
    try:
        client = get_client()
        source_vector = embed_text(source_id)

        results = client.search(
            collection_name=COLLECTION_RELIABILITY,
            query_vector=source_vector,
            limit=1,
            score_threshold=0.99,    # near-exact match on source_id
        )

        if results and results[0].payload:
            return results[0].payload

        return None

    except Exception as exc:
        print(f"[Qdrant] Reliability lookup failed (non-fatal): {exc}")
        return None


# ------------------------------------------------------------------ #
# Store — save new discovery results                                   #
# ------------------------------------------------------------------ #

def store_discovery_result(
    query_goal: str,
    dataset_types: List[str],
    scored_sources: List[Dict],
) -> bool:
    """
    Stores discovery results in Qdrant after Agent 3 completes scoring.

    Each scored source is stored as a separate point with:
    - Vector: embedding of query_goal
    - Payload: source metadata + scores

    Returns True if successful, False if Qdrant unavailable.
    """
    try:
        client = get_client()
        query_vector = embed_text(query_goal)

        points = []
        for i, source in enumerate(scored_sources):
            point_id = _stable_point_id(query_goal, str(source.get("source_id", i)))

            points.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=query_vector,
                    payload={
                        "query_goal": query_goal,
                        "dataset_type": source.get("dataset_type", "unknown"),
                        "source_id": source.get("source_id"),
                        "source_name": source.get("name"),
                        "source_url": source.get("url"),
                        "final_score": source.get("final_score"),
                        "recommendation": source.get("recommendation"),
                        "variables_available": source.get("variables_available", []),
                        "spatial_coverage": source.get("spatial_coverage"),
                        "temporal_coverage": source.get("temporal_coverage"),
                        "authority_score": source.get("catalog_authority_score"),
                        "health_score": source.get("health_score", 1.0),
                        "stored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "dataset_types_requested": dataset_types,
                    },
                )
            )

        if points:
            client.upsert(
                collection_name=COLLECTION_DISCOVERIES,
                points=points,
            )

        return True

    except Exception as exc:
        print(f"[Qdrant] Store failed (non-fatal): {exc}")
        return False


def update_source_reliability(
    source_id: str,
    succeeded: bool,
) -> bool:
    """
    Updates the reliability history for a source after retrieval.
    Called by Agent 4 (retrieval agent) after a download attempt.

    Increments times_used and times_succeeded accordingly.
    Recalculates reliability score as success_rate.

    Returns True if successful.
    """
    try:
        client = get_client()
        source_vector = embed_text(source_id)

        existing = get_source_reliability_history(source_id)

        if existing:
            times_used = existing.get("times_used", 0) + 1
            times_succeeded = existing.get("times_succeeded", 0) + (1 if succeeded else 0)
        else:
            times_used = 1
            times_succeeded = 1 if succeeded else 0

        reliability_score = times_succeeded / times_used if times_used > 0 else 0.5

        point_id = _stable_point_id("reliability", source_id)

        client.upsert(
            collection_name=COLLECTION_RELIABILITY,
            points=[
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=source_vector,
                    payload={
                        "source_id": source_id,
                        "times_used": times_used,
                        "times_succeeded": times_succeeded,
                        "reliability_score": round(reliability_score, 4),
                        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                )
            ],
        )

        return True

    except Exception as exc:
        print(f"[Qdrant] Reliability update failed (non-fatal): {exc}")
        return False


# ------------------------------------------------------------------ #
# Health check                                                         #
# ------------------------------------------------------------------ #

def is_qdrant_available() -> bool:
    """
    Returns True if Qdrant is reachable AND the embedding model works.
    False otherwise. Agent 3 uses this to decide whether to use memory
    or skip it.

    CHANGED (bug fix): previously this only checked server connectivity
    via get_collections(). That meant a missing sentence-transformers
    install (embed_text's lazy import) would still report "available",
    and every subsequent search/store call would then fail silently
    inside its own try/except with a "(non-fatal)" print -- from the
    caller's perspective this looked identical to "everything is
    working", just quietly doing nothing. Now both preconditions are
    checked up front so the failure surfaces at the one health-check
    call site instead of being rediscovered separately in every
    function that happens to touch embeddings.
    """
    try:
        client = get_client()
        client.get_collections()
    except Exception:
        return False

    try:
        embed_text("healthcheck")
    except Exception as exc:
        print(f"[Qdrant] Server is reachable but the embedding model failed to load: {exc}")
        print("[Qdrant] Run: pip install sentence-transformers")
        return False

    return True
