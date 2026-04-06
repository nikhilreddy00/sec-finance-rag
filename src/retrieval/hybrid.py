"""
Hybrid retriever: BM25 (sparse) + ChromaDB/Qdrant (dense).

Uses Reciprocal Rank Fusion (RRF) to merge results from both retrievers,
weighted by settings.bm25_weight / settings.dense_weight.

Returns up to settings.retrieval_k candidate chunks before reranking.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from config.settings import settings
from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Financial text tokenizer (shared by BM25 index and query)
# ---------------------------------------------------------------------------

# Common financial stopwords that add noise to BM25
_FINANCIAL_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "this",
    "that", "these", "those", "it", "its", "not", "no", "we", "our",
    "us", "their", "they", "them", "such", "other", "which", "who",
    "also", "than", "any", "each", "all", "more", "some",
})

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:[.'&-][a-zA-Z0-9]+)*")


def tokenize_financial(text: str) -> list[str]:
    """
    Tokenize text for BM25 with financial-domain awareness.

    - Strips punctuation but keeps internal dots/hyphens (e.g. "10-K", "R&D")
    - Lowercases
    - Removes common stopwords
    - Handles dollar amounts and percentages
    """
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _FINANCIAL_STOPWORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# RRF merge
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    bm25_results: list[tuple[Chunk, float]],
    dense_results: list[tuple[Chunk, float]],
    bm25_weight: float,
    dense_weight: float,
    k: int = 60,              # RRF k constant (default 60 per literature)
) -> list[tuple[Chunk, float]]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    Each chunk's RRF score = bm25_weight / (k + rank_bm25)
                            + dense_weight / (k + rank_dense)

    Returns list of (Chunk, rrf_score) sorted descending.
    """
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, Chunk] = {}

    for rank, (chunk, _) in enumerate(bm25_results, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + bm25_weight / (k + rank)
        chunks_by_id[chunk.chunk_id] = chunk

    for rank, (chunk, _) in enumerate(dense_results, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + dense_weight / (k + rank)
        chunks_by_id[chunk.chunk_id] = chunk

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [(chunks_by_id[cid], scores[cid]) for cid in sorted_ids]


# ---------------------------------------------------------------------------
# BM25 retriever
# ---------------------------------------------------------------------------

def bm25_search(query: str, top_k: int, filters: Optional[dict] = None) -> list[tuple[Chunk, float]]:
    """
    Search the persisted BM25 index.

    Args:
        query: natural language query
        top_k: number of results to return
        filters: optional dict of metadata field→value to filter results

    Returns:
        List of (Chunk, score) sorted descending
    """
    from src.indexing.pipeline import load_bm25_index

    try:
        bm25, chunks = load_bm25_index()
    except FileNotFoundError:
        logger.warning("BM25 index not found — skipping sparse retrieval")
        return []

    tokens = tokenize_financial(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)

    results: list[tuple[Chunk, float]] = []
    for idx, score in enumerate(scores):
        chunk = chunks[idx]
        if filters and not _matches_filters(chunk, filters):
            continue
        results.append((chunk, float(score)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def _matches_filters(chunk: Chunk, filters: dict) -> bool:
    """Return True if chunk metadata matches all filter key-value pairs.

    Special filters:
    - section_prefix: matches if the chunk's section starts with the value
    """
    meta = chunk.metadata()
    for key, value in filters.items():
        if key == "section_prefix":
            # Section prefix matching (e.g. "Item 1A" matches "Item 1A — Risk Factors")
            chunk_section = meta.get("section", "")
            if not chunk_section.startswith(str(value)):
                return False
            continue
        chunk_val = meta.get(key)
        if chunk_val is None:
            return False
        if isinstance(value, list):
            if chunk_val not in value:
                return False
        elif str(chunk_val).lower() != str(value).lower():
            return False
    return True


# ---------------------------------------------------------------------------
# Dense retriever
# ---------------------------------------------------------------------------

def dense_search(
    query: str,
    top_k: int,
    filters: Optional[dict] = None,
) -> list[tuple[Chunk, float]]:
    """
    Search the vector store using dense embeddings.

    Args:
        query: natural language query
        top_k: number of results to return
        filters: optional ChromaDB/Qdrant where-clause dict

    Returns:
        List of (Chunk, score) sorted descending
    """
    from src.indexing.embeddings import embed_texts

    query_vec = embed_texts([query], is_query=True)[0]

    if settings.vector_store == "chroma":
        return _chroma_dense_search(query_vec, top_k, filters)
    else:
        return _qdrant_dense_search(query_vec, top_k, filters)


def _chroma_dense_search(
    query_vec: list[float],
    top_k: int,
    filters: Optional[dict],
) -> list[tuple[Chunk, float]]:
    from src.indexing.vector_store import get_chroma_collection

    collection = get_chroma_collection()
    where = _build_chroma_where(filters) if filters else None

    result = collection.query(
        query_embeddings=[query_vec],
        n_results=min(top_k, collection.count() or 1),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks_scores: list[tuple[Chunk, float]] = []
    for doc_id, doc_text, meta, dist in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        score = 1.0 - dist   # cosine distance → similarity
        chunk = _meta_to_chunk(doc_text, meta, doc_id=doc_id)
        chunks_scores.append((chunk, score))

    return sorted(chunks_scores, key=lambda x: x[1], reverse=True)


def _qdrant_dense_search(
    query_vec: list[float],
    top_k: int,
    filters: Optional[dict],
) -> list[tuple[Chunk, float]]:
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from src.indexing.vector_store import get_qdrant_client

    client = get_qdrant_client()
    qdrant_filter = None
    if filters:
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
            if not isinstance(v, list)
        ]
        if conditions:
            qdrant_filter = Filter(must=conditions)

    hits = client.search(
        collection_name=settings.qdrant_collection_name,
        query_vector=query_vec,
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True,
    )

    results = []
    for hit in hits:
        payload = hit.payload or {}
        chunk = _meta_to_chunk(payload.get("text", ""), payload)
        results.append((chunk, hit.score))
    return results


def _build_chroma_where(filters: dict) -> dict:
    """Convert simple key-value filter dict to ChromaDB where clause.

    Handles the custom 'section_prefix' filter by converting it to
    a string-contains check (ChromaDB doesn't support startswith).
    """
    conditions = []
    for k, v in filters.items():
        if k == "section_prefix":
            # Use $contains for prefix matching on section field
            conditions.append({"section": {"$contains": str(v)}})
        else:
            conditions.append({k: {"$eq": v}})

    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _meta_to_chunk(text: str, meta: dict, doc_id: str = "") -> Chunk:
    """Reconstruct a Chunk from stored metadata.

    Args:
        text: chunk text from document store
        meta: metadata dict from vector store
        doc_id: document ID from vector store (used as chunk_id)
    """
    return Chunk(
        chunk_id=doc_id or meta.get("chunk_id", ""),
        text=text,
        ticker=meta.get("ticker", ""),
        company_name=meta.get("company_name", ""),
        cik=meta.get("cik", ""),
        form_type=meta.get("form_type", ""),
        filing_date=meta.get("filing_date", ""),
        fiscal_year=int(meta.get("fiscal_year", 0)),
        section=meta.get("section", ""),
        chunk_type=meta.get("chunk_type", "text"),
        table_name=meta.get("table_name", ""),
        page_number=int(meta.get("page_number", 0)),
    )


# ---------------------------------------------------------------------------
# Hybrid retriever (public API)
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Combines BM25 + dense retrieval via RRF.

    Usage::

        retriever = HybridRetriever()
        chunks = retriever.retrieve("Apple revenue 2023", filters={"ticker": "AAPL"})
    """

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None,
    ) -> list[Chunk]:
        k = top_k or settings.retrieval_k

        bm25_results = bm25_search(query, top_k=k, filters=filters)
        dense_results = dense_search(query, top_k=k, filters=filters)

        merged = reciprocal_rank_fusion(
            bm25_results,
            dense_results,
            bm25_weight=settings.bm25_weight,
            dense_weight=settings.dense_weight,
        )

        logger.debug(
            "Hybrid search '%s': %d BM25 + %d dense → %d merged",
            query[:60], len(bm25_results), len(dense_results), len(merged),
        )
        return [chunk for chunk, _ in merged[:k]]
