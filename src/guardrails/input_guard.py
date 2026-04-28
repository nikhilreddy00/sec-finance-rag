"""
Input guardrails for the Apple 10-K finance chatbot.

Checks performed in order (cheapest first):
1. Query length limit
2. Prompt injection detection
3. PII redaction
4. Hard-block rules (real-time data, investment advice, other companies)
5. Dataset-scope check — keyword fast path, then Claude classifier

The Claude classifier asks specifically: "Can this be answered from
Apple's 10-K filings (FY2020-FY2025)?" — not the broad "is this finance?"
question that lets unrelated topics slip through.

Raises GuardrailError on hard violations.
Returns sanitised query otherwise.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import anthropic

from config.settings import settings

logger = logging.getLogger(__name__)


class GuardrailError(Exception):
    """Raised when a query violates a hard guardrail rule."""


# ---------------------------------------------------------------------------
# Prompt injection patterns
# ---------------------------------------------------------------------------

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
    r"pretend (you are|to be)",
    r"jailbreak",
    r"do anything now",
]

# ---------------------------------------------------------------------------
# PII patterns
# ---------------------------------------------------------------------------

PII_PATTERNS = {
    "SSN": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
    "Credit Card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "Phone": r"\b(?:\+1[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}\b",
    "Email": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
}

# ---------------------------------------------------------------------------
# Hard-block 1: Real-time / live data requests
# These cannot be answered by any historical SEC filing — block immediately.
# ---------------------------------------------------------------------------

REALTIME_PATTERNS = [
    r"\b(current|today'?s?|live|right now|at the moment|this (minute|hour|second))\b.{0,30}(price|stock|share|value|market cap|trading)",
    r"\b(stock|share)\s*price\b",
    r"\bmarket cap(itali[sz]ation)?\b.{0,20}(now|today|current)",
    r"\b(buy|sell|short)\s+(signal|now|today|immediately)",
    r"\blatest\s+(earnings|revenue|profit)\b(?!.{0,30}(20[0-9]{2}|fy|fiscal))",
    r"\breal[ -]?time\b",
    r"\bstock\s+screener\b",
    r"\bprice.{0,10}prediction\b",
    r"\bforecast.{0,15}(price|stock|share)\b",
]

REALTIME_MESSAGE = (
    "I only have access to Apple's historical 10-K annual SEC filings "
    "(FY2020–FY2025). I cannot provide real-time stock prices, current "
    "market data, or live financial information. "
    "Please use a market data provider like Yahoo Finance or Bloomberg for that."
)

# ---------------------------------------------------------------------------
# Hard-block 2: Investment advice / buy-sell recommendations
# ---------------------------------------------------------------------------

ADVICE_PATTERNS = [
    r"\b(should|would you recommend).{0,20}(buy|sell|invest|purchase|hold|short)\b",
    r"\b(buy|sell|invest in|purchase).{0,20}(stock|share|equity)\b",
    r"\bwhat.{0,20}(stock|share|investment).{0,20}(buy|pick|choose|recommend)\b",
    r"\b(is it (worth|good|smart|wise)).{0,20}(buy|invest|purchase)\b",
    r"\b(best|top|good).{0,20}(stock|investment|portfolio)\b",
    r"\bshould i (invest|buy|sell|hold|short)\b",
    r"\bwhat.{0,20}i.{0,10}(invest|buy|sell)\b",
    r"\bportfolio (recommend|suggest|advice|allocat)\b",
    r"\btrade(ing)? (setup|signal|tip|recommendation)\b",
    r"\bentry (point|price).{0,15}(stock|share)\b",
    r"\btarget price\b",
]

ADVICE_MESSAGE = (
    "I'm a financial document research assistant — I analyse SEC filings, "
    "not provide investment advice. "
    "I cannot recommend buying, selling, or holding any security. "
    "Please consult a licensed financial advisor for investment decisions. "
    "I can answer questions about Apple's actual disclosures in their "
    "10-K annual reports (FY2020–FY2025)."
)

# ---------------------------------------------------------------------------
# Hard-block 3: Companies outside the dataset
# Dataset contains ONLY Apple Inc. (AAPL) 10-K filings.
# ---------------------------------------------------------------------------

# Companies outside the dataset — checked with word-boundary regex to prevent
# single-letter tickers ("v", "c", "t") from matching inside regular words.
# Multi-word / longer names use substring matching (safe — no false positives).
_OUT_OF_SCOPE_WORD_TOKENS = [
    # Unambiguous multi-char tickers / names
    "tsla", "tesla",
    "msft", "microsoft",
    "googl", "goog", "alphabet", "google",
    "amzn", "amazon",
    "meta", "facebook",
    "nvda", "nvidia",
    "jpm", "jpmorgan",
    "visa",                # "v" alone is too short — match "visa" instead
    "jnj",
    "wmt", "walmart",
    "exxon", "xom",
    "coca-cola", "cocacola",
    "disney",
    "nflx", "netflix",
    "boeing",
    "goldman",
    "morgan stanley",
    "bank of america",
    "wells fargo",
    "citigroup", "citibank",
    "intc", "intel",
    "amd",
    "crm", "salesforce",
    "orcl", "oracle",
    "ibm",
    "csco", "cisco",
    "qualcomm", "qcom",
    "broadcom", "avgo",
    "at&t",
    "verizon",
    "pfizer",
    "merck",
    "abbvie",
    "unitedhealth",
    "home depot",
    "starbucks", "sbux",
    "nike",
    "caterpillar",
    "fedex",
    "shopify",
    "uber", "lyft",
    "twitter", "snapchat",
    "paypal", "pypl",
    "coinbase",
    "berkshire",
    "warren buffett",
    "microsoft",
    "samsung",
    "huawei",
]

# These need word-boundary matching (they're ≤4 chars and would match substrings)
_OUT_OF_SCOPE_TICKER_WORDS = {
    "tsla", "msft", "googl", "goog", "amzn", "nvda", "jpm", "jnj",
    "wmt", "xom", "nflx", "intc", "amd", "crm", "orcl", "csco",
    "qcom", "avgo", "pfe", "mrk", "abbv", "unh", "sbux", "nke",
    "pypl",
}

_OUT_OF_SCOPE_TICKER_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(t) for t in sorted(_OUT_OF_SCOPE_TICKER_WORDS, key=len, reverse=True)) + r')\b',
    re.IGNORECASE,
)

_OUT_OF_SCOPE_NAME_TOKENS = [
    t for t in _OUT_OF_SCOPE_WORD_TOKENS
    if t not in _OUT_OF_SCOPE_TICKER_WORDS and len(t) > 4
]

OUT_OF_SCOPE_MESSAGE = (
    "My knowledge base contains only Apple Inc. (AAPL) 10-K annual reports "
    "(FY2020–FY2025). I cannot answer questions about other companies. "
    "Ask me about Apple's business, financials, risks, or strategy instead."
)

# ---------------------------------------------------------------------------
# Hard-block 4: Topics completely outside this dataset
# ---------------------------------------------------------------------------

OFF_TOPIC_PATTERNS = [
    # Macro topics
    r"\b(federal reserve|fed rate|interest rate policy|monetary policy|quantitative easing)\b",
    r"\b(inflation rate|cpi|ppi|gdp growth|unemployment rate)\b",
    r"\b(crypto|bitcoin|ethereum|nft|blockchain|defi|web3)\b",
    r"\b(forex|currency (exchange|pair|trading))\b",
    r"\b(commodity|oil price|gold price|silver price)\b",
    r"\b(real estate|mortgage rate|housing market)\b",
    # Personal finance
    r"\b(401k|ira|roth|retirement (plan|fund|savings))\b",
    r"\b(credit score|personal loan|student loan|tax (return|filing|deduction))\b",
    # General coding / non-finance
    r"\b(write (a |me a )?(?:function|class|script|code|program))\b",
    r"\b(python|javascript|typescript|sql|html|css)\b(?!.{0,20}(apple|aapl|10-k|filing))",
    r"\b(machine learning|deep learning|neural network|training (a |the )?model)\b",
    r"\b(recipe|cook|food|restaurant)\b",
    r"\b(weather|forecast|temperature|climate)\b",
    r"\b(politics|election|president|senator|congress)\b",
    r"\b(sport|football|basketball|baseball|soccer)\b",
    r"\b(movie|music|song|album|artist|celebrity)\b",
    r"\b(travel|hotel|flight|vacation|tourist)\b",
    r"\b(medical|diagnosis|symptom|treatment|drug)\b",
    r"\b(legal advice|lawsuit|attorney|lawyer)\b(?!.{0,30}(apple|aapl|filing|sec))",
]

OFF_TOPIC_MESSAGE = (
    "I'm a specialised assistant for Apple Inc.'s SEC 10-K filings (FY2020–FY2025). "
    "This topic is outside my dataset. "
    "I can answer questions about Apple's financial results, risk factors, "
    "business segments, MD&A, executive compensation, and other disclosures "
    "from their annual SEC filings."
)

# ---------------------------------------------------------------------------
# Apple-specific keyword fast path (scope check)
# A query that contains ANY of these is likely in-scope.
# ---------------------------------------------------------------------------

APPLE_KEYWORDS = frozenset({
    # Apple products / segments
    "iphone", "ipad", "mac", "macbook", "imac", "apple watch", "airpods",
    "apple tv", "vision pro",
    # Apple financial / corporate terms
    "aapl", "apple", "tim cook", "luca maestri", "jeff williams",
    # SEC filing terms
    "10-k", "10k", "annual report", "sec filing", "edgar", "proxy",
    # Generic financial terms specific enough to be in-scope
    "revenue", "net sales", "gross margin", "operating income", "net income",
    "earnings", "eps", "ebitda", "cash flow", "free cash flow",
    "balance sheet", "income statement", "cash and equivalents",
    "debt", "equity", "assets", "liabilities",
    "dividend", "buyback", "share repurchase", "capital return",
    "r&d", "research and development", "capex", "capital expenditure",
    "segment", "services", "wearables", "americas", "europe", "greater china",
    "japan", "rest of asia pacific",
    "risk factor", "risk factors", "item 1a", "item 7", "mda", "md&a",
    "fiscal year", "fy2020", "fy2021", "fy2022", "fy2023", "fy2024", "fy2025",
    "management discussion", "management's discussion",
    "supply chain", "manufacturing", "contract manufacturer",
    "foxconn", "tsmc",
    "services segment", "app store", "apple music", "icloud",
    "product revenue", "service revenue",
    "operating margin", "gross margin", "net margin",
    "stock-based compensation", "sbc",
    "legal proceedings", "litigation",
    "audit", "auditor", "accounting",
    "executive compensation", "ceo compensation",
})


def _contains_apple_keyword(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in APPLE_KEYWORDS)


def _contains_out_of_scope_company(query: str) -> bool:
    q = query.lower()
    # Word-boundary match for short tickers (prevents matching inside words)
    if _OUT_OF_SCOPE_TICKER_PATTERN.search(q):
        return True
    # Substring match for longer company names (safe — no false positives)
    return any(name in q for name in _OUT_OF_SCOPE_NAME_TOKENS)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_length(query: str) -> None:
    if len(query) > settings.max_query_length:
        raise GuardrailError(
            f"Query is too long ({len(query)} chars). "
            f"Please limit to {settings.max_query_length} characters."
        )


def check_injection(query: str) -> None:
    q_lower = query.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            raise GuardrailError(
                "This request cannot be processed. "
                "Please ask a question about Apple's SEC 10-K filings."
            )


def check_pii(query: str) -> str:
    sanitised = query
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, sanitised)
        if matches:
            logger.warning("PII detected in query (%s): redacting", pii_type)
            sanitised = re.sub(pattern, f"[{pii_type} REDACTED]", sanitised)
    return sanitised


def check_realtime_request(query: str) -> None:
    q_lower = query.lower()
    for pattern in REALTIME_PATTERNS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            raise GuardrailError(REALTIME_MESSAGE)


def check_investment_advice(query: str) -> None:
    q_lower = query.lower()
    for pattern in ADVICE_PATTERNS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            raise GuardrailError(ADVICE_MESSAGE)


def check_out_of_scope_company(query: str) -> None:
    if _contains_out_of_scope_company(query):
        raise GuardrailError(OUT_OF_SCOPE_MESSAGE)


def check_off_topic_pattern(query: str) -> None:
    q_lower = query.lower()
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            raise GuardrailError(OFF_TOPIC_MESSAGE)


def check_dataset_scope_with_claude(query: str) -> bool:
    """
    Use Claude to verify the query is answerable from Apple's 10-K filings.

    Only called when fast-path keyword checks are inconclusive. The question
    is dataset-specific: NOT "is this finance?" but "is this answerable from
    Apple Inc.'s annual 10-K reports for FY2020-2025?"
    """
    from src.generation.prompts import SCOPE_CLASSIFIER_PROMPT

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    prompt = SCOPE_CLASSIFIER_PROMPT.format(question=query)

    try:
        message = client.messages.create(
            model=settings.claude_model,
            max_tokens=5,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = message.content[0].text.strip().lower()
        return answer.startswith("yes")
    except Exception as exc:
        logger.warning("Scope classification failed: %s — allowing query", exc)
        return True  # fail open on API error


# ---------------------------------------------------------------------------
# Main guardrail class
# ---------------------------------------------------------------------------

class InputGuardrail:
    """
    Validates and sanitises user queries before they enter the RAG pipeline.

    Checks (cheapest → most expensive):
    1. Length limit
    2. Prompt injection
    3. PII redaction
    4. Hard-block: real-time data requests
    5. Hard-block: investment advice / buy-sell recommendations
    6. Hard-block: companies outside the dataset (non-Apple)
    7. Hard-block: off-topic patterns (crypto, macro, coding, etc.)
    8. Soft-pass: Apple keyword fast path (→ allow without Claude)
    9. Claude scope classifier (only when keyword check is inconclusive)
    """

    def validate(self, query: str) -> str:
        """
        Run all checks. Returns sanitised query or raises GuardrailError.
        """
        query = query.strip()

        # 1. Length
        check_length(query)

        # 2. Injection
        check_injection(query)

        # 3. PII redaction
        query = check_pii(query)

        # 4. Real-time data (hard block — can never be answered from static filings)
        check_realtime_request(query)

        # 5. Investment advice (hard block)
        check_investment_advice(query)

        # 6. Out-of-scope company (hard block)
        check_out_of_scope_company(query)

        # 7. Off-topic patterns (hard block for clearly unrelated domains)
        check_off_topic_pattern(query)

        # 8. Fast pass: if Apple-specific keywords present, allow without LLM call
        if _contains_apple_keyword(query):
            logger.debug("In-scope fast pass: Apple keyword detected")
            return query

        # 9. Claude scope classifier for ambiguous queries
        in_scope = check_dataset_scope_with_claude(query)
        if not in_scope:
            raise GuardrailError(
                "I'm a specialised assistant for Apple Inc.'s SEC 10-K annual reports "
                "(FY2020–FY2025). This question doesn't appear to be about Apple's "
                "financial disclosures. Try asking about Apple's revenue, risk factors, "
                "business segments, MD&A, or other 10-K topics."
            )

        logger.debug("Input guardrail passed for query: '%s'", query[:80])
        return query
