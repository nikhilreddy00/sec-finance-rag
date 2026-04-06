"""
Input guardrails for the finance chatbot.

Checks performed (in order):
1. Query length limit (500 chars)
2. Prompt injection detection
3. PII detection (SSN, credit card, phone numbers)
4. Finance topic filter — redirects off-topic queries

Raises GuardrailError on hard violations; returns sanitised query otherwise.
"""

from __future__ import annotations

import logging
import re

import anthropic

from config.settings import settings

logger = logging.getLogger(__name__)


class GuardrailError(Exception):
    """Raised when a query violates a hard guardrail rule."""


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore (previous|above|prior) instructions",
    r"disregard (your|all) (previous|prior|system)",
    r"system\s*:\s*",
    r"<\|im_start\|>",
    r"<\|endoftext\|>",
    r"\[INST\]",
    r"###\s*(System|Instruction)",
    r"forget (everything|all|your instructions)",
    r"new instructions?:",
    r"override (your|the) (system|instructions?)",
    r"you are now",
    r"act as (an? )?(unrestricted|jailbroken|evil|DAN)",
]

# PII patterns
PII_PATTERNS = {
    "SSN": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
    "Credit Card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "Phone": r"\b(?:\+1[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}\b",
    "Email": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
}

# Finance topic keywords — queries containing ANY of these are likely on-topic
FINANCE_KEYWORDS = [
    "revenue", "profit", "loss", "earnings", "income", "ebitda",
    "cash flow", "balance sheet", "debt", "equity", "assets", "liabilities",
    "market cap", "stock", "share", "dividend", "eps", "pe ratio",
    "annual report", "quarterly", "10-k", "10-q", "8-k", "sec", "edgar",
    "filing", "risk", "management", "ceo", "cfo", "financial", "fiscal",
    "revenue growth", "gross margin", "operating margin", "net margin",
    "r&d", "capex", "free cash flow", "guidance", "outlook", "acquisition",
    "merger", "ipo", "bankruptcy", "restructuring", "segment", "division",
    "company", "corporation", "industry", "sector", "competitor",
    "inflation", "interest rate", "federal reserve", "gdp", "economy",
    "investment", "portfolio", "hedge", "derivative", "bond", "yield",
]


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_length(query: str) -> None:
    """Raise GuardrailError if query is too long."""
    if len(query) > settings.max_query_length:
        raise GuardrailError(
            f"Query is too long ({len(query)} chars). "
            f"Please limit to {settings.max_query_length} characters."
        )


def check_injection(query: str) -> None:
    """Raise GuardrailError if prompt injection is detected."""
    q_lower = query.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            raise GuardrailError(
                "I'm unable to process this request. "
                "Please ask a question about SEC filings or financial data."
            )


def check_pii(query: str) -> str:
    """Replace detected PII with placeholders and log a warning."""
    sanitised = query
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, sanitised)
        if matches:
            logger.warning("PII detected in query (%s): redacting", pii_type)
            sanitised = re.sub(pattern, f"[{pii_type} REDACTED]", sanitised)
    return sanitised


def check_finance_topic(query: str) -> bool:
    """
    Lightweight keyword check. Returns True if likely finance-related.
    Used as a fast pre-filter before calling Claude.
    """
    q_lower = query.lower()
    return any(kw in q_lower for kw in FINANCE_KEYWORDS)


def check_finance_topic_with_claude(query: str) -> bool:
    """
    Use Claude to classify if query is finance-related.
    Only called when keyword check is inconclusive.
    """
    from src.generation.prompts import TOPIC_CLASSIFIER_PROMPT

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    prompt = TOPIC_CLASSIFIER_PROMPT.format(question=query)

    try:
        message = client.messages.create(
            model=settings.claude_model,
            max_tokens=5,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = message.content[0].text.strip().lower()
        return answer.startswith("yes")
    except Exception as exc:
        logger.warning("Topic classification failed: %s — allowing query", exc)
        return True


# ---------------------------------------------------------------------------
# Main guardrail class
# ---------------------------------------------------------------------------

class InputGuardrail:
    """
    Validates and sanitises user queries before they enter the RAG pipeline.

    Usage::

        guard = InputGuardrail()
        try:
            clean_query = guard.validate(raw_query)
        except GuardrailError as e:
            return str(e)  # return error message to user
    """

    def validate(self, query: str) -> str:
        """
        Run all input checks. Returns sanitised query or raises GuardrailError.

        Args:
            query: raw user input

        Returns:
            Sanitised query string

        Raises:
            GuardrailError: if a hard rule is violated
        """
        query = query.strip()

        # 1. Length check
        check_length(query)

        # 2. Injection check
        check_injection(query)

        # 3. PII redaction
        query = check_pii(query)

        # 4. Topic check (fast keyword first, Claude fallback)
        is_finance = check_finance_topic(query)
        if not is_finance:
            is_finance = check_finance_topic_with_claude(query)
            if not is_finance:
                raise GuardrailError(
                    "This chatbot specialises in SEC filings and financial data analysis. "
                    "Please ask a question related to company financials, SEC filings, "
                    "or financial markets."
                )

        logger.debug("Input guardrail passed for query: '%s'", query[:80])
        return query
