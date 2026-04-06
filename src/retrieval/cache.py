"""
Semantic Response Cache — the single biggest cost-reduction mechanism.

WHY THIS EXISTS (Business Case)
────────────────────────────────
Every query currently makes 2-3 Claude API calls:
  1. Filter extraction   (~256 tokens output)
  2. Multi-query expand  (~512 tokens output)
  3. Final generation    (~2048 tokens output)

At $3/M output tokens (claude-sonnet-4-6):
  ≈ $0.008 per query

For a finance company with 500 queries/day: ~$4/day = ~$120/month.

With semantic caching, 60-70% of real-world finance queries are near-duplicates
(analysts ask the same things: "Apple revenue 2023?", "AAPL FY23 revenue", etc.).
Cached hits cost $0.000 — only embedding lookup (~2ms).

HOW THE CACHE WORKS
────────────────────
1. Embed the incoming query (BAAI/bge-large, already loaded, no API cost)
2. Search ChromaDB "response_cache" collection for similar queries
3. If cosine similarity ≥ 0.92:  return cached answer (< 20ms, zero API cost)
4. If miss: run full pipeline, store result, return to user (~2-4s, API cost)

CACHE SIMILARITY THRESHOLD = 0.92
  - 0.85-0.89: too permissive — "Apple debt" ≠ "Apple revenue" (both score ~0.87)
  - 0.90-0.92: practical sweet-spot for financial Q&A (same intent, different phrasing)
  - 0.95+: too strict — barely helps (only catches exact rephrasing)

CACHE TTL = 7 days
  - SEC filings don't change once published
  - Quarterly filings arrive every 3 months, so 7-day TTL is safe
  - Analysts re-use the same queries throughout the week

STORAGE
  - Separate ChromaDB collection "response_cache" (no interference with SEC data)
  - Max 10,000 entries (configurable) — evicts oldest on overflow
  - Persists to same data/chroma_db directory (zero extra infrastructure)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)

CACHE_COLLECTION = "response_cache"
DEFAULT_TTL_SECONDS = 7 * 24 * 3600  # 7 days
SIMILARITY_THRESHOLD = 0.92
MAX_CACHE_ENTRIES = 10_000


class SemanticCache:
    """
    Semantic similarity cache for RAG responses.

    Wraps the existing ChromaDB client — no extra infrastructure needed.
    The cache collection is separate from the SEC filings collection.
    """

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        self.ttl = ttl_seconds
        self.threshold = threshold
        self._collection = None  # lazy init

    @property
    def collection(self):
        if self._collection is None:
            from src.indexing.vector_store import get_chroma_client
            client = get_chroma_client()
            self._collection = client.get_or_create_collection(
                name=CACHE_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            logger.debug(
                "Cache collection ready (%d entries)", self._collection.count()
            )
        return self._collection

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(self, query: str) -> Optional[dict]:
        """
        Look up a cached response by semantic similarity.

        Returns cached result dict or None on miss/expired.
        Typical latency: 2-5ms (embedding already loaded).
        """
        from src.indexing.embeddings import embed_texts

        if self.collection.count() == 0:
            return None

        query_emb = embed_texts([query], is_query=True)[0]

        try:
            result = self.collection.query(
                query_embeddings=[query_emb],
                n_results=1,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.debug("Cache lookup failed: %s", exc)
            return None

        if not result["ids"][0]:
            return None

        distance = result["distances"][0][0]
        similarity = 1.0 - distance  # cosine distance → similarity

        if similarity < self.threshold:
            logger.debug("Cache MISS (similarity=%.3f < %.2f)", similarity, self.threshold)
            return None

        meta = result["metadatas"][0][0]

        # Check TTL
        cached_at = float(meta.get("cached_at", 0))
        if time.time() - cached_at > self.ttl:
            logger.debug("Cache EXPIRED (age=%.0fh)", (time.time() - cached_at) / 3600)
            return None

        logger.info(
            "Cache HIT (similarity=%.3f, age=%.1fh) — zero API cost",
            similarity, (time.time() - cached_at) / 3600,
        )

        payload_json = result["documents"][0][0]
        try:
            return json.loads(payload_json)
        except Exception:
            return None

    def put(self, query: str, result: dict) -> None:
        """
        Store a RAG result in the cache.

        Args:
            query: the original user question
            result: dict with keys: answer, sources, filters, num_chunks
        """
        from src.indexing.embeddings import embed_texts

        query_emb = embed_texts([query], is_query=True)[0]
        entry_id = hashlib.sha256(query.encode()).hexdigest()[:20]

        # Evict oldest if at capacity
        if self.collection.count() >= MAX_CACHE_ENTRIES:
            self._evict_oldest()

        # Store payload as JSON string in the "document" field
        payload = json.dumps({
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "filters": result.get("filters", {}),
            "num_chunks": result.get("num_chunks", 0),
            "from_cache": True,
        })

        self.collection.upsert(
            ids=[entry_id],
            embeddings=[query_emb],
            documents=[payload],
            metadatas=[{
                "query": query[:500],
                "cached_at": time.time(),
            }],
        )
        logger.debug("Cached response for query: '%s'", query[:80])

    def invalidate(self, query: str) -> None:
        """Remove a specific query from the cache."""
        entry_id = hashlib.sha256(query.encode()).hexdigest()[:20]
        try:
            self.collection.delete(ids=[entry_id])
        except Exception:
            pass

    def clear(self) -> None:
        """Clear the entire cache."""
        try:
            from src.indexing.vector_store import get_chroma_client
            client = get_chroma_client()
            client.delete_collection(CACHE_COLLECTION)
            self._collection = None
            logger.info("Response cache cleared")
        except Exception as exc:
            logger.warning("Cache clear failed: %s", exc)

    def stats(self) -> dict:
        """Return cache statistics."""
        count = self.collection.count()
        return {
            "entries": count,
            "ttl_hours": self.ttl / 3600,
            "threshold": self.threshold,
            "max_entries": MAX_CACHE_ENTRIES,
        }

    def _evict_oldest(self) -> None:
        """Evict the 10% oldest entries when cache is full."""
        try:
            all_items = self.collection.get(include=["metadatas"])
            ids = all_items["ids"]
            metas = all_items["metadatas"]
            # Sort by cached_at ascending (oldest first)
            paired = sorted(zip(ids, metas), key=lambda x: float(x[1].get("cached_at", 0)))
            evict_count = max(1, len(paired) // 10)
            evict_ids = [p[0] for p in paired[:evict_count]]
            self.collection.delete(ids=evict_ids)
            logger.info("Evicted %d oldest cache entries", evict_count)
        except Exception as exc:
            logger.debug("Cache eviction failed: %s", exc)


# Module-level singleton
_cache_instance: Optional[SemanticCache] = None


def get_cache() -> SemanticCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache()
    return _cache_instance
