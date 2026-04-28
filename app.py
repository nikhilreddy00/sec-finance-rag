"""
Streamlit chatbot frontend — Apple 10-K Finance RAG Demo.

Demo limit: 10 total queries across all visitors.
After the limit, visitors are prompted to run the project locally.

Deploy to Streamlit Community Cloud:
  1. Push repo to GitHub (indices are committed)
  2. Connect at share.streamlit.io
  3. Add ANTHROPIC_API_KEY (and optionally COHERE_API_KEY) in Secrets
"""

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

# ---------------------------------------------------------------------------
# Inject API key from Streamlit secrets (Streamlit Community Cloud)
# ---------------------------------------------------------------------------
if "ANTHROPIC_API_KEY" in st.secrets:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
if "COHERE_API_KEY" in st.secrets:
    os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]

# ---------------------------------------------------------------------------
# Global query counter (persists across sessions within one server process)
# ---------------------------------------------------------------------------
from src.demo_counter import QUERY_LIMIT, get_count, queries_remaining, try_increment

# ---------------------------------------------------------------------------
# Load RAG chain once across all sessions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

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

    remaining = queries_remaining()
    used = get_count()

    # Query budget meter
    st.subheader("Demo Query Budget")
    progress_color = "normal" if remaining > 3 else ("off" if remaining == 0 else "normal")
    st.progress(used / QUERY_LIMIT, text=f"{used} / {QUERY_LIMIT} queries used")
    if remaining > 0:
        st.success(f"**{remaining}** quer{'y' if remaining == 1 else 'ies'} remaining")
    else:
        st.error("Demo limit reached — see instructions below")

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
# Main area header
# ---------------------------------------------------------------------------

st.title("Apple Inc. — 10-K Annual Report Chatbot")
st.caption(
    "Ask questions about Apple's SEC 10-K filings (FY2020–FY2025). "
    "All answers are grounded in Apple's actual filings with source citations."
)

# Pipeline architecture badge row
col1, col2, col3, col4, col5 = st.columns(5)
col1.markdown("**Stage 1**\n\nBGE-large Embeddings")
col2.markdown("**Stage 2**\n\nHybrid BM25 + Dense")
col3.markdown("**Stage 3**\n\nCohere Reranking")
col4.markdown("**Stage 4**\n\nClaude Generation")
col5.markdown("**Stage 5**\n\nOutput Guardrails")
st.divider()

# ---------------------------------------------------------------------------
# "Limit reached" banner — shown once limit is hit
# ---------------------------------------------------------------------------

if queries_remaining() == 0 and get_count() >= QUERY_LIMIT:
    st.error("## 🚫 Demo Query Limit Reached")
    st.markdown(
        """
The **10 public demo queries** have been used up. Thank you to everyone who tried it!

### Run it yourself with your own API key — it's free to set up:

```bash
# 1. Clone the repo
git clone https://github.com/nikhilreddy00/sec-finance-rag.git
cd sec-finance-rag

# 2. Create virtual environment
python -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > .env

# 5. Launch the chatbot (index is pre-built — no ingestion needed)
streamlit run app.py
```

Get a free Anthropic API key at **[console.anthropic.com](https://console.anthropic.com)**.
Source code: **[github.com/nikhilreddy00/sec-finance-rag](https://github.com/nikhilreddy00/sec-finance-rag)**
        """
    )
    st.stop()

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
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------

prompt = st.chat_input(
    "Ask about Apple's 10-K annual reports...",
    disabled=(queries_remaining() == 0),
)

if prompt:
    # Check demo limit before processing
    allowed = try_increment()
    if not allowed:
        st.error("Demo query limit reached. See instructions above to run locally.")
        st.stop()

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
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

    # Show "queries remaining" toast after each query
    rem = queries_remaining()
    if rem == 0:
        st.warning("That was the last demo query. Others can run it locally — see the sidebar.")
    elif rem <= 3:
        st.info(f"⏳ {rem} demo quer{'y' if rem == 1 else 'ies'} remaining for all visitors.")
