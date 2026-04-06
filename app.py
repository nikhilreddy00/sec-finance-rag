"""
Streamlit chatbot frontend for the Finance RAG system.

Features:
- Chat interface with streaming responses
- Sidebar: company/form/year filters
- Source citation expander showing retrieved chunks
- Session-persistent chat history
- System stats panel

Deploy to Streamlit Community Cloud (free):
    1. Push repo to GitHub
    2. Connect at share.streamlit.io
    3. Add ANTHROPIC_API_KEY in Secrets panel
"""

from __future__ import annotations

import logging
import os

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Apple 10-K Chatbot",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"))

# ---------------------------------------------------------------------------
# Load environment from .env (local dev)
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Lazy-load the RAG chain (cached across Streamlit reruns)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading FinBot pipeline... (first load may take ~60s)")
def load_chain():
    from src.generation.chain import get_rag_chain
    return get_rag_chain()


@st.cache_data(ttl=300)
def get_stats() -> dict:
    try:
        from src.indexing.vector_store import get_indexed_count
        from config.settings import settings
        return {
            "indexed_chunks": get_indexed_count(),
            "vector_store": settings.vector_store,
            "embedding_model": settings.embedding_model.split("/")[-1],
            "claude_model": settings.claude_model,
            "reranking": "Cohere" if settings.reranking_enabled else "Cosine (no API key)",
        }
    except Exception:
        return {}

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🍎 Apple 10-K Chatbot")
    st.caption("Annual Report Analysis — FY2020 to FY2025")
    st.divider()

    st.subheader("Search Filters")
    st.caption("Narrow results to a specific fiscal year")

    year_input = st.selectbox(
        "Fiscal Year",
        options=["All Years", 2020, 2021, 2022, 2023, 2024, 2025],
        help="Filter to a specific Apple annual report",
    )
    fiscal_year_filter = None if year_input == "All Years" else int(year_input)

    # Apple is the only company — ticker and form_type are always AAPL / 10-K
    active_filters: dict = {"ticker": "AAPL", "form_type": "10-K"}
    if fiscal_year_filter:
        active_filters["fiscal_year"] = fiscal_year_filter

    if fiscal_year_filter:
        st.info(f"Filtering: Apple 10-K FY{fiscal_year_filter}")
    else:
        st.info("Searching across all 6 Apple 10-K filings")

    st.divider()

    # System stats
    st.subheader("System Info")
    stats = get_stats()
    if stats:
        st.metric("Indexed Chunks", f"{stats.get('indexed_chunks', 'N/A'):,}" if isinstance(stats.get('indexed_chunks'), int) else "N/A")
        st.caption(f"Vector store: **{stats.get('vector_store', '—')}**")
        st.caption(f"Embeddings: **{stats.get('embedding_model', '—')}**")
        st.caption(f"LLM: **{stats.get('claude_model', '—')}**")
        st.caption(f"Reranking: **{stats.get('reranking', '—')}**")

    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.sources = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("Apple Inc. — 10-K Annual Report Chatbot")
st.caption(
    "Ask questions about Apple's annual SEC 10-K filings from FY2020 to FY2025. "
    "Answers are grounded in Apple's actual filings with source citations."
)

# Display chat history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources for assistant messages
        if msg["role"] == "assistant" and i < len(st.session_state.sources):
            sources = st.session_state.sources[i]
            if sources:
                with st.expander(f"Sources ({len(sources)} filing excerpts)"):
                    for j, src in enumerate(sources, start=1):
                        st.markdown(
                            f"**{j}.** `{src.get('ticker', '')}` — "
                            f"{src.get('form_type', '')} ({src.get('filing_date', '')}) — "
                            f"*{src.get('section', '')}*"
                        )

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

EXAMPLE_QUESTIONS = [
    "What were Apple's total net sales in FY2024?",
    "How did Apple's gross margin change from 2020 to 2024?",
    "What are the main risk factors Apple disclosed in their FY2023 10-K?",
    "What did Apple say about its Services segment growth strategy?",
    "Summarise Apple's capital return program — share buybacks and dividends.",
    "How did Apple's R&D spending evolve from FY2020 to FY2025?",
    "What were Apple's key supply chain risks mentioned in the annual reports?",
    "Compare Apple's iPhone revenue across the last three fiscal years.",
]

if not st.session_state.messages:
    st.info("Try one of these example questions about Apple's 10-K filings:")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % 2].button(q, key=f"example_{i}"):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

if prompt := st.chat_input("Ask about Apple's 10-K annual reports..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and stream assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        current_sources = []

        try:
            chain = load_chain()

            # Streaming response
            with st.spinner("Searching SEC filings..."):
                # First do retrieval to get sources (non-streaming)
                chunks, applied_filters = chain.get_sources(
                    prompt,
                    filters=active_filters or None,
                )
                current_sources = [c.metadata() for c in chunks]

            # Now stream the answer
            for delta in chain.query_stream(
                question=prompt,
                filters=active_filters or None,
            ):
                full_response += delta
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        except Exception as e:
            error_msg = str(e)
            if "GuardrailError" in type(e).__name__ or "guardrail" in error_msg.lower():
                full_response = f"⚠️ {error_msg}"
            else:
                full_response = (
                    "I encountered an error processing your question. "
                    "Please check that the ingestion pipeline has been run "
                    "and try again."
                )
                logging.exception("Query error: %s", e)
            response_placeholder.markdown(full_response)

        # Show sources
        if current_sources:
            with st.expander(f"Sources ({len(current_sources)} filing excerpts)"):
                for j, src in enumerate(current_sources, start=1):
                    st.markdown(
                        f"**{j}.** `{src.get('ticker', '')}` — "
                        f"{src.get('form_type', '')} ({src.get('filing_date', '')}) — "
                        f"*{src.get('section', '')}*"
                    )

    # Persist to session state
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.sources.append(current_sources)
