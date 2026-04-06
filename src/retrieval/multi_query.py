"""
Multi-query retriever: expands a single query into N variations using Claude,
retrieves documents for each, then deduplicates and merges results.

This improves recall for ambiguous or multi-faceted financial questions.

Example:
    Original: "What are Apple's long-term growth strategies?"
    Generated:
        1. "Apple future revenue growth plans and investments"
        2. "Apple strategic initiatives expansion markets"
        3. "AAPL management outlook capital allocation"
"""

from __future__ import annotations

import logging
from typing import Optional

import anthropic

from config.settings import settings
from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

MULTI_QUERY_PROMPT = """You are a financial research analyst helping improve search recall.

Given a question about SEC filings or financial data, generate {n} alternative
search queries that would retrieve different but relevant documents.
Each query should approach the topic from a distinct angle.

Output ONLY a numbered list, one query per line. No explanations.

Original question: {question}

Alternative queries:"""


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

def generate_query_variants(query: str, n: Optional[int] = None) -> list[str]:
    """
    Use Claude to generate query variants for improved recall.

    Args:
        query: original user query
        n: number of variants to generate (default: settings.multi_query_count)

    Returns:
        List of alternative queries (may be empty on failure)
    """
    n = n or settings.multi_query_count
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    try:
        message = client.messages.create(
            model=settings.claude_model,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": MULTI_QUERY_PROMPT.format(question=query, n=n),
            }],
        )
        raw = message.content[0].text.strip()
        variants = _parse_numbered_list(raw)
        logger.debug("Generated %d query variants for: %s", len(variants), query[:60])
        return variants[:n]
    except Exception as exc:
        logger.warning("Multi-query expansion failed: %s", exc)
        return []


def _parse_numbered_list(text: str) -> list[str]:
    """Parse numbered list from LLM output."""
    import re
    lines = text.strip().split("\n")
    queries: list[str] = []
    for line in lines:
        # Remove leading numbers/bullets
        cleaned = re.sub(r"^\s*[\d]+[.)]\s*", "", line).strip()
        cleaned = re.sub(r"^\s*[-*•]\s*", "", cleaned).strip()
        if cleaned and len(cleaned) > 10:
            queries.append(cleaned)
    return queries


# ---------------------------------------------------------------------------
# Multi-query retriever
# ---------------------------------------------------------------------------

class MultiQueryRetriever:
    """
    Expands query → N variants → retrieves for each → deduplicates by chunk_id.

    Usage::

        retriever = MultiQueryRetriever()
        chunks = retriever.retrieve("Apple revenue 2023")
    """

    def __init__(self) -> None:
        from src.retrieval.hybrid import HybridRetriever
        self._hybrid = HybridRetriever()

    def retrieve(
        self,
        query: str,
        filters: Optional[dict] = None,
        top_k: Optional[int] = None,
    ) -> list[Chunk]:
        """
        Retrieve with multi-query expansion.

        1. Generate N query variants
        2. Retrieve top_k chunks for original + each variant
        3. Deduplicate by chunk_id (first occurrence wins for ordering)
        4. Return union limited to top_k

        Returns:
            Deduplicated list of chunks, original query results ranked first
        """
        k = top_k or settings.retrieval_k

        # Always include original query
        all_queries = [query] + generate_query_variants(query)

        seen_ids: set[str] = set()
        merged: list[Chunk] = []

        for q in all_queries:
            chunks = self._hybrid.retrieve(q, top_k=k, filters=filters)
            for chunk in chunks:
                if chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    merged.append(chunk)

        logger.info(
            "Multi-query: %d queries → %d unique chunks (from %d total)",
            len(all_queries), len(merged), len(all_queries) * k,
        )

        # Return up to 2× k before reranking — reranker will trim to top_k
        return merged[: k * 2]
