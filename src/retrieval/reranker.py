"""
Reranker: Cohere reranking (primary) with cosine-similarity fallback.

Cohere free tier: 1,000 API calls/month.
When COHERE_API_KEY is not set or the monthly limit is exhausted,
falls back to cosine similarity between query embedding and chunk embeddings.

Pipeline position: called after hybrid + multi-query retrieval,
before contextual compression and generation.
"""

from __future__ import annotations

import logging
from typing import Optional

from config.settings import settings
from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cohere reranker
# ---------------------------------------------------------------------------

def cohere_rerank(
    query: str,
    chunks: list[Chunk],
    top_n: Optional[int] = None,
) -> list[Chunk]:
    """
    Rerank chunks using Cohere Rerank API.

    Args:
        query: user query
        chunks: candidate chunks to rerank
        top_n: number of top chunks to return (default: settings.cohere_rerank_top_n)

    Returns:
        Reranked list of chunks (top_n length)
    """
    top_n = top_n or settings.cohere_rerank_top_n
    if not chunks:
        return []

    try:
        import cohere
        client = cohere.Client(api_key=settings.cohere_api_key)

        documents = [c.text[:2000] for c in chunks]  # Cohere has a 2048-char limit per doc

        response = client.rerank(
            model=settings.cohere_rerank_model,
            query=query,
            documents=documents,
            top_n=min(top_n, len(chunks)),
            return_documents=False,
        )

        reranked = [chunks[r.index] for r in response.results]
        logger.info(
            "Cohere rerank: %d → %d chunks (query: '%s')",
            len(chunks), len(reranked), query[:60],
        )
        return reranked

    except Exception as exc:
        logger.warning("Cohere reranking failed (%s); falling back to cosine similarity", exc)
        return _cosine_fallback_rerank(query, chunks, top_n)


# ---------------------------------------------------------------------------
# Cosine similarity fallback
# ---------------------------------------------------------------------------

def _cosine_fallback_rerank(
    query: str,
    chunks: list[Chunk],
    top_n: int,
) -> list[Chunk]:
    """
    Rerank by cosine similarity between query embedding and chunk embeddings.
    Used when Cohere API is unavailable.
    """
    import numpy as np
    from src.indexing.embeddings import embed_texts

    try:
        texts = [query] + [c.text[:512] for c in chunks]
        embeddings = embed_texts(texts, is_query=False)
        query_emb = np.array(embeddings[0])
        chunk_embs = np.array(embeddings[1:])

        # Normalise (BGE embeddings should already be normalised)
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        chunk_norms = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-9)

        scores = chunk_norms @ query_norm
        ranked_indices = np.argsort(scores)[::-1][:top_n]

        reranked = [chunks[i] for i in ranked_indices]
        logger.info("Cosine fallback rerank: %d → %d chunks", len(chunks), len(reranked))
        return reranked

    except Exception as exc:
        logger.warning("Cosine fallback rerank failed: %s — returning unranked top_n", exc)
        return chunks[:top_n]


# ---------------------------------------------------------------------------
# Contextual compression (embedding-based)
# ---------------------------------------------------------------------------

def embedding_filter_compress(
    query: str,
    chunks: list[Chunk],
    threshold: Optional[float] = None,
) -> list[Chunk]:
    """
    Filter out chunks whose embedding similarity to the query is below threshold.
    This is a cheap alternative/complement to Cohere reranking.

    Args:
        query: user query
        chunks: chunks to filter
        threshold: cosine similarity threshold (default: settings.similarity_threshold)

    Returns:
        Filtered list of chunks
    """
    import numpy as np
    from src.indexing.embeddings import embed_texts

    threshold = threshold or settings.similarity_threshold
    if not chunks:
        return []

    try:
        texts = [query] + [c.text[:512] for c in chunks]
        embeddings = embed_texts(texts, is_query=False)
        query_emb = np.array(embeddings[0])
        chunk_embs = np.array(embeddings[1:])

        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        chunk_norms = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-9)
        scores = chunk_norms @ query_norm

        filtered = [c for c, s in zip(chunks, scores) if s >= threshold]
        logger.debug(
            "Embedding filter: %d → %d chunks (threshold=%.2f)",
            len(chunks), len(filtered), threshold,
        )
        return filtered if filtered else chunks[:3]  # always return at least 3

    except Exception as exc:
        logger.warning("Embedding filter failed: %s", exc)
        return chunks


# ---------------------------------------------------------------------------
# Combined reranker (public API)
# ---------------------------------------------------------------------------

class Reranker:
    """
    Applies reranking to a list of candidate chunks.

    Uses Cohere if API key is set, otherwise cosine similarity fallback.
    Optionally applies embedding-based contextual compression after reranking.
    """

    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_n: Optional[int] = None,
        apply_compression: bool = True,
    ) -> list[Chunk]:
        """
        Full reranking pipeline.

        1. Cohere rerank (or cosine fallback) → top_n
        2. Optional embedding-based compression filter

        Returns:
            Final ranked + filtered chunks
        """
        top_n = top_n or settings.cohere_rerank_top_n

        if settings.reranking_enabled:
            reranked = cohere_rerank(query, chunks, top_n=top_n)
        else:
            reranked = _cosine_fallback_rerank(query, chunks, top_n=top_n)

        if apply_compression and len(reranked) > top_n:
            reranked = embedding_filter_compress(query, reranked)

        return reranked
