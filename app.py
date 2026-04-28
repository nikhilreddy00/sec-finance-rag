from __future__ import annotations

import logging
import os

import streamlit as st

st.set_page_config(
    page_title="Apple 10-K FinBot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@st.cache_resource(show_spinner="Loading FinBot pipeline... (~30s on first load)")
def load_chain():
    from src.generation.chain import get_rag_chain
    return get_rag_chain()


@st.cache_data(ttl=60)
def get_stats() -> dict:
    try:
        from src.indexing.vector_store import get_indexed_count
        from config.settings import settings
        return {
            "indexed_chunks": get_indexed_count(),
            "vector_store": settings.vector_store,
            "embedding_model": settings.embedding_model.split("/")[-1],
            "claude_model": settings.claude_model,
            "reranking": "Cohere" if settings.reranking_enabled else "Cosine similarity",
        }
    except Exception:
        return {}


if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources_per_msg" not in st.session_state:
    st.session_state.sources_per_msg = []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📊 Apple 10-K FinBot")
    st.caption("Powered by Claude · BGE · Hybrid RAG")
    st.divider()

    st.subheader("Search Filters")

    year_input = st.selectbox(
        "Fiscal Year",
        options=["All Years", 2020, 2021, 2022, 2023, 2024, 2025],
        help="Filter to a specific Apple annual report",
    )
    fiscal_year_filter = None if year_input == "All Years" else int(year_input)

    active_filters: dict = {"ticker": "AAPL", "form_type": "10-K"}
    if fiscal_year_filter:
        active_filters["fiscal_year"] = fiscal_year_filter
        st.info(f"Filtering: AAPL 10-K FY{fiscal_year_filter}")
    else:
        st.info("Searching all 6 Apple 10-K filings (FY2020–FY2025)")

    st.divider()
    st.subheader("System Info")
    stats = get_stats()
    if stats:
        count = stats.get("indexed_chunks")
        st.metric("Indexed Chunks", f"{count:,}" if isinstance(count, int) else "N/A")
        st.caption(f"Vector store: **{stats.get('vector_store', '—')}**")
        st.caption(f"Embeddings: **{stats.get('embedding_model', '—')}**")
        st.caption(f"LLM: **{stats.get('claude_model', '—')}**")
        st.caption(f"Reranking: **{stats.get('reranking', '—')}**")

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.sources_per_msg = []
        st.rerun()

    st.divider()
    st.caption(
        "Built by [Nikhil Kumar Reddy](https://github.com/nikhilreddy00) · "
        "[GitHub](https://github.com/nikhilreddy00/sec-finance-rag)"
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Apple Inc. — 10-K Annual Report Chatbot")
st.caption(
    "Ask questions about Apple's SEC 10-K filings (FY2020–FY2025). "
    "All answers are grounded in Apple's actual filings with source citations."
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.markdown("**Stage 1**\n\nBGE-large Embeddings")
col2.markdown("**Stage 2**\n\nHybrid BM25 + Dense")
col3.markdown("**Stage 3**\n\nCohere Reranking")
col4.markdown("**Stage 4**\n\nClaude Generation")
col5.markdown("**Stage 5**\n\nOutput Guardrails")
st.divider()

# ---------------------------------------------------------------------------
# Chat history display
# ---------------------------------------------------------------------------

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and i // 2 < len(st.session_state.sources_per_msg):
            sources = st.session_state.sources_per_msg[i // 2]
            if sources:
                with st.expander(f"📎 Sources ({len(sources)} filing excerpts)"):
                    for j, src in enumerate(sources, start=1):
                        st.markdown(
                            f"**{j}.** `{src.get('ticker', '')}` — "
                            f"{src.get('form_type', '')} (FY{src.get('fiscal_year', '')}) — "
                            f"*{src.get('section', '')}*"
                        )

# ---------------------------------------------------------------------------
# Example questions (shown on empty chat)
# ---------------------------------------------------------------------------

EXAMPLE_QUESTIONS = [
    "What were Apple's total net sales in FY2024?",
    "How did Apple's gross margin change from FY2020 to FY2024?",
    "What are the main risk factors Apple disclosed in FY2023?",
    "What did Apple say about its Services segment growth strategy?",
    "Summarise Apple's capital return program — share buybacks and dividends.",
    "How did Apple's R&D spending evolve from FY2020 to FY2025?",
    "What were Apple's key supply chain risks mentioned in the annual reports?",
    "Compare Apple's iPhone revenue across the last three fiscal years.",
]

if not st.session_state.messages:
    st.info("💡 **Try one of these questions** about Apple's 10-K filings:")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % 2].button(q, key=f"example_{i}"):
            st.session_state["_pending_prompt"] = q

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------

prompt = st.chat_input(
    "Ask about Apple's 10-K annual reports...",
) or st.session_state.pop("_pending_prompt", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching Apple's SEC filings..."):
            try:
                chain = load_chain()
                result = chain.query(
                    question=prompt,
                    filters=active_filters or None,
                )
                full_response = result["answer"]
                current_sources = result["sources"]

            except Exception as e:
                err = str(e)
                if "guardrail" in err.lower() or "GuardrailError" in type(e).__name__:
                    full_response = f"⚠️ {err}"
                    current_sources = []
                else:
                    full_response = (
                        "⚠️ Something went wrong processing your question. "
                        "Please ensure the index is loaded and try again."
                    )
                    current_sources = []
                    logging.exception("Query error: %s", e)

        st.markdown(full_response)

        if current_sources:
            with st.expander(f"📎 Sources ({len(current_sources)} filing excerpts)"):
                for j, src in enumerate(current_sources, start=1):
                    st.markdown(
                        f"**{j}.** `{src.get('ticker', '')}` — "
                        f"{src.get('form_type', '')} (FY{src.get('fiscal_year', '')}) — "
                        f"*{src.get('section', '')}*"
                    )

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.sources_per_msg.append(current_sources)
