"""
Output guardrails for the finance chatbot.

Checks performed on every LLM response:
1. Prohibited phrase detection (guaranteed returns, etc.)
2. Source citation check — response must reference at least one source
3. Financial advice disclaimer injection
4. Grounding check — response should not contradict context (lightweight heuristic)

Does NOT raise exceptions — returns a safe modified response instead.
"""

from __future__ import annotations

import logging
import re

from src.generation.prompts import FINANCIAL_DISCLAIMER
from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prohibited phrases (investment fraud / misleading claims)
# ---------------------------------------------------------------------------

PROHIBITED_PHRASES = [
    r"guaranteed (return|profit|gain|income)",
    r"sure (bet|thing|fire)",
    r"can('t| not) lose",
    r"risk[ -]free investment",
    r"definitely (will|going to) (increase|rise|go up)",
    r"you should (buy|sell|invest in)",
    r"I recommend (buying|selling|investing)",
    r"insider (information|tip|knowledge)",
]

# Phrases that trigger disclaimer injection
ADVICE_INDICATORS = [
    "invest", "buy", "sell", "portfolio", "recommend", "should consider",
    "opportunity", "return", "growth potential", "undervalued", "overvalued",
]

# Minimum number of source citations expected in response
MIN_SOURCE_CITATIONS = 1


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_prohibited_phrases(response: str) -> str:
    """
    Replace prohibited investment advice phrases with safe alternatives.
    Returns modified response.
    """
    modified = response
    for pattern in PROHIBITED_PHRASES:
        match = re.search(pattern, modified, re.IGNORECASE)
        if match:
            logger.warning("Prohibited phrase detected in response: %s", match.group())
            # Wrap prohibited phrase with a disclaimer note
            modified = re.sub(
                pattern,
                "[Note: Forward-looking statements involve uncertainty]",
                modified,
                flags=re.IGNORECASE,
            )
    return modified


def check_source_citations(response: str, chunks: list[Chunk]) -> str:
    """
    Verify the response contains at least one source citation.
    If not, append a source summary block.
    """
    # Look for [Source ...] pattern
    has_citation = bool(re.search(r"\[Source\s*\d*\s*:", response, re.IGNORECASE))

    if not has_citation and chunks:
        # Auto-append a sources block
        source_lines = []
        for i, chunk in enumerate(chunks[:5], start=1):
            source_lines.append(
                f"[Source {i}: {chunk.ticker} | {chunk.form_type} | "
                f"{chunk.filing_date} | {chunk.section}]"
            )
        sources_block = "\n\n**Sources consulted:**\n" + "\n".join(source_lines)
        response = response + sources_block
        logger.debug("Auto-appended source citations to response")

    return response


def inject_disclaimer(response: str, question: str) -> str:
    """
    Inject the financial disclaimer if the response or question contains
    advice-adjacent language.
    """
    # Always inject disclaimer (it's short and important)
    if FINANCIAL_DISCLAIMER not in response:
        response = response + FINANCIAL_DISCLAIMER
    return response


def check_grounding(response: str, chunks: list[Chunk]) -> str:
    """
    Lightweight heuristic: if response mentions a specific number/percentage
    but no chunk contains it, flag as potentially ungrounded.

    This is a soft check — it logs a warning but does NOT modify the response.
    """
    if not chunks:
        return response

    # Extract numbers from response (percentages, dollar amounts)
    response_numbers = set(re.findall(r"\b\d+\.?\d*%?\b", response))
    context_text = " ".join(c.text for c in chunks)
    context_numbers = set(re.findall(r"\b\d+\.?\d*%?\b", context_text))

    hallucinated_numbers = response_numbers - context_numbers
    if len(hallucinated_numbers) > 3:
        logger.warning(
            "Possible hallucination: %d numbers in response not found in context: %s",
            len(hallucinated_numbers),
            list(hallucinated_numbers)[:5],
        )

    return response


# ---------------------------------------------------------------------------
# Main guardrail class
# ---------------------------------------------------------------------------

class OutputGuardrail:
    """
    Validates and enriches LLM responses before returning to the user.

    Unlike InputGuardrail, this NEVER blocks a response — it modifies it
    to be safer and more compliant.
    """

    def validate(
        self,
        response: str,
        question: str,
        chunks: list[Chunk],
    ) -> str:
        """
        Run all output checks and return the (possibly modified) response.

        Args:
            response: raw LLM response text
            question: original user question
            chunks: retrieved context chunks

        Returns:
            Safe, compliant response string
        """
        # 1. Remove prohibited phrases
        response = check_prohibited_phrases(response)

        # 2. Ensure source citations are present
        response = check_source_citations(response, chunks)

        # 3. Grounding check (logs warning, no modification)
        response = check_grounding(response, chunks)

        # 4. Always inject disclaimer
        response = inject_disclaimer(response, question)

        return response
