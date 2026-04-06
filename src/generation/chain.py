"""
RAG generation chain: retrieval → reranking → generation via Claude.

Full pipeline per query:
    1. InputGuardrail.validate(query)
    2. SelfQueryRetriever.retrieve(query) → extracts metadata filters
    3. MultiQueryRetriever.retrieve(query, filters) → hybrid + multi-query
    4. Reranker.rerank(query, chunks) → Cohere / cosine fallback
    5. format_rag_prompt(question, reranked_chunks)
    6. Claude API (streaming or batch)
    7. OutputGuardrail.validate(response, chunks)
    8. Append financial disclaimer

Usage::

    chain = FinanceRAGChain()
    # Batch
    result = chain.query("What was Apple's revenue in 2023?")
    print(result["answer"])

    # Streaming (yields text deltas)
    for delta in chain.query_stream("Apple risk factors 2023"):
        print(delta, end="", flush=True)
"""

from __future__ import annotations

import logging
from typing import Generator, Optional

import anthropic

from config.settings import settings
from src.generation.prompts import FINANCIAL_DISCLAIMER, format_rag_prompt
from src.ingestion.chunker import Chunk
from src.retrieval.multi_query import MultiQueryRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.self_query import SelfQueryRetriever

logger = logging.getLogger(__name__)


class FinanceRAGChain:
    """
    End-to-end finance RAG chain.

    Lazy-initialises all components on first use to avoid cold-start
    overhead when imported.
    """

    def __init__(self) -> None:
        self._client: Optional[anthropic.Anthropic] = None
        self._self_query: Optional[SelfQueryRetriever] = None
        self._multi_query: Optional[MultiQueryRetriever] = None
        self._reranker: Optional[Reranker] = None

    # ── Lazy initialisation ───────────────────────────────────────────────────

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        return self._client

    @property
    def self_query(self) -> SelfQueryRetriever:
        if self._self_query is None:
            self._self_query = SelfQueryRetriever()
        return self._self_query

    @property
    def multi_query(self) -> MultiQueryRetriever:
        if self._multi_query is None:
            self._multi_query = MultiQueryRetriever()
        return self._multi_query

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    # ── Public API ────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        filters: Optional[dict] = None,
        top_k: Optional[int] = None,
    ) -> dict:
        """
        Execute full RAG pipeline and return a structured result.

        COST OPTIMISATION ORDER:
          1. Semantic cache check  → 0 API calls if hit  (< 20ms)
          2. Input guardrails      → rule-based first, Claude only if needed
          3. Retrieve + rerank     → no LLM calls
          4. Claude generation     → 1 API call (only on cache miss)
          5. Store in cache        → amortises future identical queries

        Returns:
            {
                "answer": str,
                "sources": list[dict],
                "filters": dict,
                "num_chunks": int,
                "from_cache": bool,   # True = zero API cost this call
            }
        """
        from src.guardrails.input_guard import InputGuardrail
        from src.guardrails.output_guard import OutputGuardrail
        from src.retrieval.cache import get_cache

        # Step 1: Input guardrails (rule-based, no Claude call)
        guard_in = InputGuardrail()
        validated_query = guard_in.validate(question)

        # Step 2: Semantic cache lookup — returns instantly if hit, costs nothing
        if settings.cache_enabled:
            cache = get_cache()
            cached = cache.get(validated_query)
            if cached is not None:
                logger.info("Serving from semantic cache — 0 Claude API calls used")
                return cached

        # Step 3-4: Retrieve + rerank (no Claude calls, all local/Cohere)
        chunks, applied_filters = self._retrieve(validated_query, filters, top_k)

        # Step 5: Build prompt (top 2 reranked chunks = tight, precise context)
        system_prompt, user_prompt = format_rag_prompt(validated_query, chunks)

        # Step 6: Generate — ONE Claude API call (previously was up to 3)
        message = self.client.messages.create(
            model=settings.claude_model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw_answer = message.content[0].text

        # Step 7: Output guardrails
        guard_out = OutputGuardrail()
        validated_answer = guard_out.validate(raw_answer, question, chunks)

        result = {
            "answer": validated_answer,
            "sources": [c.metadata() for c in chunks],
            "filters": applied_filters,
            "num_chunks": len(chunks),
            "from_cache": False,
        }

        # Step 8: Store in cache for future identical/similar queries (no cost)
        if settings.cache_enabled:
            cache.put(validated_query, result)

        return result

    def query_stream(
        self,
        question: str,
        filters: Optional[dict] = None,
        top_k: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        Streaming version of query(). Yields text deltas as they arrive.

        Usage (Streamlit)::
            for token in chain.query_stream(question):
                st.write(token)
        """
        from src.guardrails.input_guard import InputGuardrail

        guard_in = InputGuardrail()
        validated_query = guard_in.validate(question)

        chunks, _ = self._retrieve(validated_query, filters, top_k)
        system_prompt, user_prompt = format_rag_prompt(validated_query, chunks)

        with self.client.messages.stream(
            model=settings.claude_model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text

        # Yield disclaimer at end
        yield FINANCIAL_DISCLAIMER

    def get_sources(
        self,
        question: str,
        filters: Optional[dict] = None,
        top_k: Optional[int] = None,
    ) -> tuple[list[Chunk], dict]:
        """Return retrieved chunks without generating an answer (for debugging)."""
        return self._retrieve(question, filters, top_k)

    # ── Internal retrieval ────────────────────────────────────────────────────

    def _retrieve(
        self,
        query: str,
        filters: Optional[dict],
        top_k: Optional[int],
    ) -> tuple[list[Chunk], dict]:
        """
        Run retrieval pipeline: self-query → (conditional) multi-query → rerank.

        COST OPTIMISATION — Multi-query is conditional:
          - SKIP multi-query when self-query already extracted both ticker AND form_type.
            These "precise" queries already narrow to 1-3 relevant filings; expanding
            with 2 Claude-generated variants adds minimal recall but costs 1 API call.
          - RUN multi-query for broad queries ("compare Apple and Microsoft margins")
            where no specific filing is identified. These benefit most from expansion.

        This alone saves ~40% of Claude API calls (most analyst queries are specific).
        """
        # Self-querying: rule-based filter extraction + hybrid retrieval (no Claude call
        # when ticker/form is explicit — fast-path in self_query.py)
        chunks, applied_filters = self.self_query.retrieve(
            query, override_filters=filters
        )

        # Decide whether multi-query is worth the Claude API call.
        # Only skip when ALL three precision filters are present (ticker + form + year).
        # The sidebar always passes ticker=AAPL & form_type=10-K, so we need
        # fiscal_year to also be set before considering the query "precise".
        is_precise_query = (
            "ticker" in applied_filters
            and "form_type" in applied_filters
            and "fiscal_year" in applied_filters
        )

        if not chunks:
            logger.info("Self-query returned 0 chunks; running multi-query without filters")
            chunks = self.multi_query.retrieve(query, top_k=top_k)
            applied_filters = {}
        elif not is_precise_query:
            # Broad query or cross-year: use multi-query expansion
            expanded = self.multi_query.retrieve(query, filters=applied_filters, top_k=top_k)
            seen = {c.chunk_id for c in chunks}
            for c in expanded:
                if c.chunk_id not in seen:
                    chunks.append(c)
                    seen.add(c.chunk_id)
            logger.debug("Multi-query expansion added %d unique chunks", len(chunks))
        else:
            logger.debug(
                "Precise query (ticker=%s, form=%s, year=%s) — skipping multi-query",
                applied_filters.get("ticker"), applied_filters.get("form_type"),
                applied_filters.get("fiscal_year"),
            )

        # Rerank: 50 candidates → top 2 (set in settings.cohere_rerank_top_n)
        final_chunks = self.reranker.rerank(query, chunks)

        logger.info(
            "Retrieved %d final chunks (from %d candidates) for: '%s'",
            len(final_chunks), len(chunks), query[:80],
        )
        return final_chunks, applied_filters


# ---------------------------------------------------------------------------
# Module-level singleton (use for Streamlit caching)
# ---------------------------------------------------------------------------

_chain_instance: Optional[FinanceRAGChain] = None


def get_rag_chain() -> FinanceRAGChain:
    """Return a module-level singleton FinanceRAGChain."""
    global _chain_instance
    if _chain_instance is None:
        _chain_instance = FinanceRAGChain()
    return _chain_instance
