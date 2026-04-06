"""
Self-querying retriever: extracts structured metadata filters from the user's
natural language query using Claude, then passes those filters to hybrid search.

Example:
    Query: "What were Microsoft's risk factors in their 2022 annual report?"
    Extracted filters: {"ticker": "MSFT", "form_type": "10-K", "fiscal_year": 2022}

The extracted filters narrow the search space before hybrid retrieval,
dramatically improving precision for company/year/form-specific questions.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import anthropic

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt for filter extraction
# ---------------------------------------------------------------------------

FILTER_EXTRACTION_PROMPT = """You are a financial document search assistant.
The knowledge base contains ONLY Apple Inc. (AAPL) 10-K annual filings from FY2020 to FY2025.

Given a user query, extract any metadata filters present.
Output ONLY a JSON object (no explanation, no markdown) with any of these optional fields:
{{
  "ticker": "AAPL",          // always AAPL — the only company in this dataset
  "form_type": "10-K",       // always 10-K — the only form type in this dataset
  "fiscal_year": 2023        // integer year (2020-2025), only if explicitly mentioned
}}

Rules:
- Always include "ticker": "AAPL" and "form_type": "10-K" unless the query is clearly non-specific.
- Only include "fiscal_year" if the query explicitly mentions a year (e.g. "2023", "FY2022", "last year" = 2025).
- If the query spans multiple years (e.g. "from 2020 to 2023") omit fiscal_year so all years are searched.
- Output only the JSON object, nothing else.

Query: {query}"""

# Apple is the only company in this dataset
TICKER_ALIASES: dict[str, str] = {
    "apple": "AAPL",
    "apple inc": "AAPL",
    "aapl": "AAPL",
    "iphone": "AAPL",
    "ipad": "AAPL",
    "mac": "AAPL",
    "tim cook": "AAPL",
}

FORM_ALIASES: dict[str, str] = {
    "annual report": "10-K",
    "annual": "10-K",
    "10k": "10-K",
    "10-k": "10-K",
}


# ---------------------------------------------------------------------------
# Rule-based fast extraction (avoids LLM call for simple queries)
# ---------------------------------------------------------------------------

def _fast_extract(query: str) -> dict:
    """
    Quick rule-based extraction for Apple 10-K queries.

    Since the dataset contains only Apple 10-K filings (FY2020-FY2025),
    ticker and form_type are always pre-filled. Only fiscal_year and section
    are extracted dynamically from the query text.
    """
    filters: dict = {
        "ticker": "AAPL",
        "form_type": "10-K",
    }
    q_lower = query.lower()

    # Year (4-digit, 2020–2025 only — our dataset range)
    year_match = re.search(r'\b(202[0-5])\b', query)
    if year_match:
        filters["fiscal_year"] = int(year_match.group(1))
    # Also handle "FY20XX" format
    fy_match = re.search(r'\bfy\s*(20[0-9]{2})\b', q_lower)
    if fy_match:
        yr = int(fy_match.group(1))
        if 2020 <= yr <= 2025:
            filters["fiscal_year"] = yr

    # If query spans multiple years or is comparative, drop fiscal_year filter
    multi_year = re.search(r'\b(202[0-5])\b.*\b(202[0-5])\b', query)
    compare_words = any(w in q_lower for w in ("compare", "across", "trend", "over the years", "history", "evolution", "from 20", "all years"))
    if multi_year or compare_words:
        filters.pop("fiscal_year", None)

    # Section-based filtering for common financial query patterns
    section_patterns = {
        r"risk\s*factor": "Item 1A",
        r"\b(revenue|sales|net\s*sales|income|margin|operating|financial\s*(result|performance|statement))": "Item 7",
        r"(business\s*description|overview|products?\s*and\s*services)": "Item 1",
        r"(management.?s?\s*discussion|md&?a)": "Item 7",
        r"(market\s*risk|interest\s*rate\s*risk|currency\s*risk)": "Item 7A",
        r"(control|procedure|internal\s*control)": "Item 9A",
        r"(legal\s*proceed|litigation)": "Item 3",
        r"(compensation|executive\s*pay)": "Item 11",
    }
    for pattern, item_prefix in section_patterns.items():
        if re.search(pattern, q_lower):
            filters["section_prefix"] = item_prefix
            break

    return filters


# ---------------------------------------------------------------------------
# Claude-powered extraction
# ---------------------------------------------------------------------------

def extract_filters_with_claude(query: str) -> dict:
    """
    Use Claude to extract metadata filters from query.
    Falls back to empty dict on any error.
    """
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    prompt = FILTER_EXTRACTION_PROMPT.format(query=query)

    try:
        message = client.messages.create(
            model=settings.claude_model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        # Extract JSON from response (sometimes wrapped in backticks)
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            filters = json.loads(json_match.group())
            return filters
    except Exception as exc:
        logger.debug("Claude filter extraction failed: %s", exc)

    return {}


# ---------------------------------------------------------------------------
# Self-querying retriever
# ---------------------------------------------------------------------------

class SelfQueryRetriever:
    """
    Wraps HybridRetriever with automatic metadata filter extraction.

    Workflow:
    1. Fast rule-based extraction
    2. If fast extraction is incomplete, use Claude to extract remaining filters
    3. Apply filters to hybrid retrieval
    """

    def __init__(self) -> None:
        from src.retrieval.hybrid import HybridRetriever
        self._hybrid = HybridRetriever()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        override_filters: Optional[dict] = None,
    ) -> tuple[list, dict]:
        """
        Retrieve chunks with automatic filter extraction.

        section_prefix is treated as a soft filter: if it produces 0 results,
        retry without it (SEC iXBRL HTML often lacks clear section headers).

        Returns:
            (chunks, filters) — the chunks and the filters that were applied
        """
        if override_filters is not None:
            filters = override_filters
        else:
            filters = self._extract_filters(query)

        logger.info("Self-query filters for '%s': %s", query[:60], filters)

        from src.ingestion.chunker import Chunk
        chunks = self._hybrid.retrieve(query, top_k=top_k, filters=filters or None)

        # Soft fallback: if section_prefix filter returned 0 results, retry without it
        if not chunks and "section_prefix" in filters:
            relaxed = {k: v for k, v in filters.items() if k != "section_prefix"}
            logger.info("section_prefix filter returned 0 results; retrying without it")
            chunks = self._hybrid.retrieve(query, top_k=top_k, filters=relaxed or None)
            filters = relaxed

        return chunks, filters

    def _extract_filters(self, query: str) -> dict:
        """
        Extract filters for Apple-only dataset.

        Fast rule-based extraction always provides ticker=AAPL and form_type=10-K.
        Claude is only invoked when fiscal_year is ambiguous (e.g. "last year",
        "most recent", "previous filing") to resolve relative year references.
        """
        fast = _fast_extract(query)

        # If fiscal_year already extracted, we're done — no Claude call needed
        if "fiscal_year" in fast:
            return fast

        # Check for relative year references that need Claude to resolve
        q_lower = query.lower()
        relative_refs = ("last year", "most recent", "latest", "previous year", "current year", "this year")
        needs_claude = any(ref in q_lower for ref in relative_refs)

        if needs_claude:
            claude_filters = extract_filters_with_claude(query)
            # Merge: fast (ticker/form_type) takes precedence over Claude output
            merged = {**claude_filters, **fast}
            return {k: v for k, v in merged.items() if v is not None}

        # Default: return ticker + form_type (no year filter = search all 6 filings)
        return fast
