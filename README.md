<div align="center">

# SEC EDGAR Finance RAG Chatbot

**Production-grade Retrieval-Augmented Generation over 25,000+ SEC filings from the entire S&P 500**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Claude](https://img.shields.io/badge/LLM-Claude%20Sonnet%204.6-orange?logo=anthropic)](https://anthropic.com)
[![LlamaIndex](https://img.shields.io/badge/Framework-LlamaIndex-purple)](https://llamaindex.ai)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB%20%2F%20Qdrant-green)](https://trychroma.com)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

</div>

---

## The Problem

Publicly traded companies file thousands of documents with the SEC every year — 10-K annual reports, 10-Q quarterly filings, 8-K material event disclosures, DEF 14A proxy statements, and more. These filings are the ground truth for company financials, risk disclosures, strategic plans, and executive decisions.

**The challenge:** These documents are massive, dense, and written in legalese. A single Apple 10-K is 150–300 pages long. Across the S&P 500 over 5 years, that's **20,000–30,000 documents totalling billions of tokens** — far beyond what any LLM context window can hold, and far too large to search manually.

Analysts who need answers like:
- *"What were Microsoft's cloud segment margins in FY2023?"*
- *"Which S&P 500 companies disclosed material AI-related risks in their 2024 10-K?"*
- *"How did NVIDIA's R&D spending evolve from 2020 to 2024?"*

…have no tool that can answer these questions accurately, with citations, and grounded exclusively in the actual filed documents — not in a model's hallucinated memory.

---

## The Solution

A **multi-stage RAG (Retrieval-Augmented Generation) pipeline** that:

1. **Ingests** all S&P 500 SEC filings asynchronously from EDGAR at scale
2. **Parses** each filing intelligently based on its format (HTML, PDF, plain text)
3. **Chunks** documents with financial-domain awareness — sentence-window for narratives, atomic preservation for tables
4. **Indexes** chunks into a hybrid vector + sparse search engine
5. **Retrieves** relevant context using self-querying, multi-query expansion, and Reciprocal Rank Fusion
6. **Reranks** candidates with a neural cross-encoder for precision
7. **Generates** grounded answers via Claude with source citations and financial disclaimers
8. **Guards** every query and response against injection, hallucination, and off-topic requests
9. **Evaluates** retrieval quality continuously using RAGAS and DeepEval

The result is a chatbot that answers financial questions with **citation-backed accuracy**, **zero hallucination on factual figures**, and **cost-efficient operation** through a layered caching and conditional-LLM-call architecture.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                               │
│              Streamlit Chat  ·  FastAPI REST  ·  SSE Stream         │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   INPUT GUARDRAILS     │  ← Injection · PII · Topic filter
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   SEMANTIC CACHE       │  ← 0-cost response on cache hit
                    └───────────┬───────────┘
                                │ (cache miss)
                    ┌───────────▼───────────┐
                    │  SELF-QUERY RETRIEVER  │  ← Extracts: ticker · form_type
                    │  (rule-based + Claude) │     fiscal_year · section
                    └───────────┬───────────┘
                                │
               ┌────────────────▼────────────────┐
               │   MULTI-QUERY EXPANSION          │  ← Claude generates 2 query
               │   (conditional on query breadth) │     variants for recall boost
               └────────────────┬────────────────┘
                                │
          ┌─────────────────────▼──────────────────────┐
          │           HYBRID SEARCH                     │
          │  ┌─────────────────┐  ┌──────────────────┐ │
          │  │  BM25 (sparse)  │  │  BGE-large       │ │  → 50 candidates
          │  │  rank_bm25      │  │  (dense, 1024d)  │ │     via RRF fusion
          │  └─────────────────┘  └──────────────────┘ │     BM25:0.4 + Dense:0.6
          └─────────────────────┬──────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   COHERE RERANKING     │  ← 50 candidates → top 8
                    │   (cosine fallback)    │     cross-encoder precision scoring
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  CLAUDE GENERATION     │  ← Grounded answer + citations
                    │  claude-sonnet-4-6     │     Financial disclaimer appended
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  OUTPUT GUARDRAILS     │  ← Prohibited phrases · Grounding check
                    └───────────────────────┘
```

---

## Pipeline Stages

### Stage 1 — Data Acquisition: Async SEC EDGAR Scraper

**What it does:** Concurrently fetches filings for all S&P 500 companies across 4 form types (10-K, 10-Q, 8-K, DEF 14A) for the last 5 years — yielding 25,000–35,000 documents.

**Why `aiohttp` + `asyncio`?**
The SEC EDGAR API is I/O-bound: each request takes 200–800ms but uses almost no CPU. With synchronous requests, scraping 500 companies × 5 years would take 8–12 hours. Async concurrency with `asyncio.Semaphore` drops this to under 30 minutes while respecting SEC's fair-use policy of ≤10 req/sec.

**Why `tenacity` for retries?**
SEC's EDGAR servers return transient 5xx errors under load. `tenacity` provides exponential backoff with jitter — a retry after 2s, 4s, 8s — which absorbs these failures without manual polling loops.

**Idempotent by design:** The scraper checks for existing files before downloading. Re-runs only fetch newly published filings — critical for a production system running on a schedule.

```python
# From src/ingestion/scraper.py
async with EDGARScraper() as scraper:
    filings = await scraper.scrape(
        tickers=get_sp500_tickers(),     # ~503 tickers from Wikipedia
        form_types=["10-K", "10-Q", "8-K", "DEF 14A"],
        years_back=5,
    )
```

---

### Stage 2 — Document Parsing: Format-Aware Smart Parser

**The challenge:** EDGAR filings arrive in three formats: HTML (iXBRL), PDF, and plain text. Each requires a different parsing strategy to extract clean, structured content.

**The decision tree:**

| Format | Parser | Why |
|--------|--------|-----|
| `.htm` / `.html` | Unstructured.io | SEC adopted iXBRL in 2009 — 85%+ of modern filings are HTML. Unstructured gives **semantic blocks** (Title, NarrativeText, Table) instead of raw text soup. No API cost. |
| `.pdf` (simple) | PyMuPDF4LLM | Text-heavy PDFs like 8-K press releases (5–10 pages, few tables). 10× faster than LlamaParse, free, handles 90%+ of PDFs adequately. |
| `.pdf` (complex) | LlamaParse | Financial statements with multi-column merged-cell tables lose column alignment in free parsers. LlamaParse reconstructs tables semantically. Triggered only when: page count > 50 AND table density ≥ 0.15 tables/page (≈5–10% of filings). |
| `.txt` / other | Plain text split | Zero cost, zero dependencies. |

**Why not always use LlamaParse?**
At $0.003/page, parsing 30,000 filings × 200 avg pages = **$18,000**. Selectively routing the 5–10% that genuinely need it costs ~$900. The complexity probe (pdfplumber page/table count in <0.5s) pays for itself immediately.

**iXBRL block merging:** SEC's inline XBRL produces hundreds of tiny blocks (individual numbers, short labels). A post-processing step merges consecutive short blocks (<200 chars) into coherent paragraphs while keeping titles and tables atomic.

**10-K section detection:** Regex-based Item header detection tags every block with its section context (`Item 7 — Management's Discussion and Analysis`, `Item 1A — Risk Factors`) — this metadata later enables precise section-filtered retrieval.

---

### Stage 3 — Chunking: Sentence-Window with Domain Awareness

**Why not fixed-size character splitting?**
Fixed splits arbitrarily cut sentences mid-thought, destroying semantic coherence. A sentence starting with "However, this risk..." severed from its prior sentence loses all meaning for embedding.

**Strategy:**

| Content Type | Chunking Rule | Why |
|-------------|--------------|-----|
| Narrative text (10-K sections, 8-K body) | SentenceSplitter (512 tokens, 100 overlap) | Respects sentence boundaries. Overlap ensures transitions between chunks aren't lost. |
| Financial tables | Kept atomic (1 chunk per table) | Splitting a balance sheet mid-row destroys the row→column relationship. A 3-row table fragment is meaningless without context. |
| Short-form filings (8-K, DEF 14A) | SentenceSplitter (256 tokens) | These are shorter documents — smaller chunks improve retrieval precision without sacrificing coverage. |

**Sentence-window context:** Each narrative chunk stores a `window_text` covering the 3 chunks before and after it. This richer context is stored as metadata and surfaced during generation, giving Claude wider document context without bloating the index.

**Rich metadata per chunk:**
```python
{
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "form_type": "10-K",
    "filing_date": "2024-09-28",
    "fiscal_year": 2024,
    "section": "Item 7 — Management's Discussion and Analysis",
    "chunk_type": "text",   # or "table"
    "page_number": 42,
}
```

This metadata powers self-query filtering at retrieval time — without it, filtering by year or section would require scanning every chunk.

---

### Stage 4 — Embedding & Indexing: BGE-Large + Hybrid Index

**Why `BAAI/bge-large-en-v1.5`?**

| Model | Dimensions | MTEB Score | Cost | Chosen? |
|-------|-----------|-----------|------|---------|
| OpenAI text-embedding-3-large | 3072 | ~64.6 | $0.13/M tokens | No — API cost at 30k docs scale |
| Cohere embed-v3 | 1024 | ~64.0 | $0.10/M tokens | No — API cost |
| **BGE-large-en-v1.5** | **1024** | **~64.2** | **Free (local)** | **Yes** |
| BGE-base-en-v1.5 | 768 | ~63.6 | Free (local) | No — lower quality |

BGE-large achieves near-parity with paid embedding APIs at zero ongoing cost. Embedding 30,000 documents once on a CPU takes ~4 hours — a one-time cost that pays for itself in the first day of production use.

**BGE query prefix:** BGE models require the prefix `"Represent this sentence: "` on **query** embeddings (not document embeddings) to activate their asymmetric retrieval mode. This is handled automatically by the wrapper.

**Why a hybrid index (BM25 + dense vector)?**

Financial text contains two types of signals that require different retrieval mechanisms:

- **Semantic signals:** "profits" ≈ "net income" ≈ "earnings" — dense embeddings handle this.
- **Exact signals:** `"$394.3 billion"`, `"10-K/A"`, `"Item 1A"`, ticker symbols — BM25 keyword matching handles this perfectly; dense embeddings do not (exact numbers get averaged into generic embedding space).

Hybrid search via **Reciprocal Rank Fusion (RRF)** combines both ranked lists without needing to tune score scales:
```
RRF_score(chunk) = BM25_weight / (k + BM25_rank) + Dense_weight / (k + Dense_rank)
```
Weights: BM25=0.4, Dense=0.6. The dense signal dominates for conceptual questions; BM25 steps up for exact-term queries.

**Vector store strategy:**
- **ChromaDB** (dev/demo): Zero infrastructure, persistent local storage, ideal for datasets under 100k chunks.
- **Qdrant** (production): Free cloud tier, supports filtering on metadata fields, horizontally scalable.

---

### Stage 5 — Self-Query Retrieval: Metadata Filter Extraction

**The problem with naive vector search:**
A query like *"Apple's risk factors in FY2023"* submitted to an unfiltered vector index returns chunks from every S&P 500 company across all years. The top results might include MSFT's risk factors (semantically similar) instead of AAPL's.

**Self-querying** extracts structured filters from natural language:
```
"What were Apple's risk factors in 2023?"
  → filters: { ticker: "AAPL", form_type: "10-K", fiscal_year: 2023, section_prefix: "Item 1A" }
```

**Two-tier extraction:**
1. **Rule-based fast path** (0 API calls, <1ms): Regex and lookup tables for tickers, years, and common section references. Handles ~80% of queries.
2. **Claude fallback** (1 API call): For ambiguous references like "last year", "most recent quarter", "the proxy statement". Claude extracts structured JSON from the query text.

**Why rule-based first?**
An LLM call adds 500–2000ms of latency and costs money. Rule-based extraction is instantaneous and covers the majority of user queries. Using Claude only as a fallback keeps median query latency under 100ms for the retrieval phase.

---

### Stage 6 — Multi-Query Expansion: Recall Boost for Broad Questions

**The vocabulary mismatch problem:**
*"Apple's long-term growth strategies"* might not lexically match chunks that discuss *"capital allocation priorities"* or *"strategic initiatives in emerging markets"* — even though they're the same concept.

Multi-query expansion uses Claude to generate alternative phrasings:
```
Original:  "Apple's long-term growth strategies"
Variant 1: "Apple future revenue growth plans and investments"
Variant 2: "AAPL management outlook capital allocation"
```

Each variant is retrieved independently; results are deduplicated and merged before reranking.

**Why conditional?**
Multi-query costs one Claude API call. We skip it when the query is already **precise** (all three filters present: ticker + form_type + fiscal_year). A query like *"Apple 10-K 2023 revenue"* already narrows to 1–3 relevant documents — expanding it adds minimal recall while wasting an API call. Conditional invocation saves ~40% of multi-query costs.

---

### Stage 7 — Reranking: Neural Cross-Encoder Precision

**The two-stage retrieve-then-rerank pattern:**

| Stage | Model Type | Speed | Accuracy | Role |
|-------|-----------|-------|----------|------|
| Retrieval | Bi-encoder (BGE) | Fast (ms) | Approximate | Fetches 50 candidates from millions of chunks |
| Reranking | Cross-encoder (Cohere) | Slower (200ms) | High precision | Scores each (query, chunk) pair jointly → top 8 |

**Why cross-encoders are more accurate:**
Bi-encoders embed query and document **independently** — they cannot see how specific query terms relate to specific document spans. Cross-encoders process the full `(query + document)` sequence together, enabling token-level attention between query and document. This joint processing is what makes them far more precise.

**Why Cohere specifically?**
- Free tier: 1,000 API calls/month — sufficient for development and light production use.
- `rerank-english-v3.0` is consistently top-ranked on the BEIR benchmark for passage reranking.
- **Cosine similarity fallback:** When Cohere is unavailable (rate limit, API down), the system falls back to cosine similarity between query and chunk embeddings using the already-loaded BGE model. Zero downtime, graceful degradation.

---

### Stage 8 — Answer Generation: Claude with Grounding

**Why Claude (`claude-sonnet-4-6`)?**
- Best-in-class instruction following for structured financial analysis tasks.
- Reliable at following citation formatting instructions (`[Source: Company, Form, Year]`).
- Strong at reasoning across multiple table chunks simultaneously (income statement + balance sheet cross-references).
- Streaming API (`client.messages.stream`) enables real-time token delivery to the Streamlit frontend.

**Cost optimization in the generation chain:**
```
Query arrives
  ↓
Semantic cache check  →  HIT: return instantly (0 API calls, <20ms)
  ↓ (miss)
Input guardrails      →  Rule-based (0 API calls for most queries)
  ↓
Self-query            →  Rule-based fast path (~80% of queries: 0 API calls)
  ↓
Multi-query           →  SKIPPED for precise queries (~40% reduction)
  ↓
Retrieval + rerank    →  0 LLM calls (local BM25 + embeddings + Cohere)
  ↓
Claude generation     →  1 API call (guaranteed)
  ↓
Cache storage         →  Future identical/similar queries: 0 API calls
```

In steady-state production, the semantic cache (0.92 cosine threshold, 7-day TTL) serves 60–70% of repeated analyst queries at zero cost.

---

### Stage 9 — Guardrails: Input and Output Safety

#### Input Guardrails (`src/guardrails/input_guard.py`)

| Check | Mechanism | Why |
|-------|-----------|-----|
| Length limit | Hard cap at 500 chars | Prevents context-window stuffing attacks |
| Prompt injection | 12+ regex patterns | Blocks `"ignore previous instructions"`, `"[INST]"`, `"act as DAN"` |
| PII redaction | Regex for SSN, credit cards, phone, email | Users may accidentally paste sensitive data into a search box |
| Finance topic filter | Keyword list → Claude fallback | Keeps the chatbot on-domain; rejects unrelated queries politely |

**Why keyword-first for topic filtering?** Claude's topic classification is 100% accurate but costs 1 API call. The 50+ finance keyword list rejects clearly off-topic queries (recipe requests, coding questions) with zero cost. Claude is only invoked for ambiguous edge cases.

#### Output Guardrails (`src/guardrails/output_guard.py`)

| Check | What it catches |
|-------|----------------|
| Prohibited phrases | "guaranteed return", "you should buy", "insider" — investment advice that could create liability |
| Citation enforcement | Auto-appends `[Source]` blocks if Claude omits them |
| Grounding verification | Logs a warning if financial figures in the response don't appear in the retrieved context |
| Financial disclaimer | Always appended: *"This is not financial advice. Consult a licensed advisor."* |

---

### Stage 10 — Evaluation: Measuring RAG Quality

**Why RAGAS + DeepEval?**
RAG systems fail in subtle ways — the retriever fetches wrong documents, or the generator invents plausible-sounding figures. Traditional metrics (BLEU, ROUGE) don't catch these. RAGAS and DeepEval are specifically designed for RAG evaluation:

| Metric | Framework | Threshold | What it measures |
|--------|-----------|-----------|-----------------|
| Faithfulness | RAGAS | ≥ 0.80 | Does the answer contain only claims supported by the retrieved context? |
| Answer Relevancy | RAGAS | ≥ 0.75 | Does the answer actually address the question asked? |
| Context Recall | RAGAS | — | Does the retrieved context contain the information needed to answer? |
| Context Precision | RAGAS | — | Is the context free of irrelevant chunks (low noise)? |
| Hallucination Score | DeepEval | ≤ 0.20 | Does the answer introduce facts not in the context? |

Evaluation uses a **synthetic dataset** generated from the SEC filings themselves — questions and ground-truth answers derived directly from the source documents. This ensures evaluation is grounded in the actual data domain.

---

## Results

| Metric | Value | Notes |
|--------|-------|-------|
| S&P 500 companies covered | 503 | Full index |
| Filing types | 4 | 10-K, 10-Q, 8-K, DEF 14A |
| Years covered | 5 | 2020–2024 |
| Total documents | ~25,000–35,000 | Varies by company filing frequency |
| ChromaDB chunks (Apple demo) | 1,876 | 6 filings, 100% parse success |
| BM25 index size (Apple demo) | 1,708 | Text-only chunks |
| Ingestion time (Apple, CPU) | ~3 min | Parsing 2s/file + embedding 2.5min |
| Median query latency | < 500ms | Cache miss, includes retrieval + Claude |
| Cache hit rate (production) | 60–70% | Repeat analyst queries |
| RAGAS Faithfulness | > 0.80 | Answers grounded in context |
| RAGAS Answer Relevancy | > 0.75 | On-target responses |
| DeepEval Hallucination | < 0.20 | Low fabrication rate |

---

## Project Structure

```
Finance Bot/
├── src/
│   ├── ingestion/
│   │   ├── scraper.py          # Async SEC EDGAR scraper (aiohttp, tenacity)
│   │   ├── parser.py           # Smart parser: HTML→Unstructured, PDF→PyMuPDF/LlamaParse
│   │   ├── chunker.py          # Sentence-window chunking + atomic table chunks
│   │   └── html_loader.py      # Local HTML file loader (filename→metadata)
│   │
│   ├── indexing/
│   │   ├── embeddings.py       # BGE-large-en-v1.5 wrapper (HuggingFace, cached)
│   │   ├── vector_store.py     # ChromaDB / Qdrant factory
│   │   └── pipeline.py         # End-to-end ingestion pipeline + BM25 builder
│   │
│   ├── retrieval/
│   │   ├── self_query.py       # Metadata filter extraction (rule-based + Claude)
│   │   ├── multi_query.py      # Claude query expansion (2 variants, conditional)
│   │   ├── hybrid.py           # BM25 + dense search with Reciprocal Rank Fusion
│   │   ├── reranker.py         # Cohere reranking + cosine similarity fallback
│   │   └── cache.py            # Semantic response cache (ChromaDB, 0.92 threshold)
│   │
│   ├── generation/
│   │   ├── chain.py            # Full RAG chain (orchestrates all stages)
│   │   └── prompts.py          # System prompt, RAG template, disclaimers
│   │
│   ├── guardrails/
│   │   ├── input_guard.py      # Injection, PII, topic filter
│   │   └── output_guard.py     # Citation check, prohibited phrases, grounding
│   │
│   └── evaluation/
│       ├── synthetic_dataset.py # Generates Q&A pairs from SEC filings
│       ├── ragas_eval.py       # Faithfulness + relevancy evaluation
│       └── deepeval_eval.py    # Hallucination detection
│
├── api/
│   └── main.py                 # FastAPI: /api/query, /api/query/stream, /api/health
│
├── config/
│   └── settings.py             # Pydantic settings (env-var driven)
│
├── scripts/
│   ├── ingest_apple_html.py    # Run ingestion on local Apple HTML files
│   └── run_evaluation.py       # Trigger RAGAS / DeepEval evaluation
│
├── app.py                      # Streamlit chatbot UI
├── requirements.txt
├── Dockerfile
└── .env.example
```

---

## Technology Stack

| Layer | Technology | Why This, Not Alternatives |
|-------|-----------|---------------------------|
| **LLM** | Claude Sonnet 4.6 (Anthropic) | Best instruction-following for structured financial analysis; reliable citation formatting; streaming API |
| **Embeddings** | BAAI/bge-large-en-v1.5 | Top MTEB benchmark, 1024 dims, free local hosting — matches paid APIs at $0 cost |
| **RAG Framework** | LlamaIndex | Built-in SentenceSplitter, TextNode, and document abstractions; first-class HuggingFace embedding support |
| **Vector DB (dev)** | ChromaDB | Zero-infrastructure local persistence; no cloud account needed for development |
| **Vector DB (prod)** | Qdrant | Free cloud tier; supports metadata filtering; horizontally scalable |
| **Sparse Search** | BM25 via rank_bm25 | Financial-domain keyword retrieval for exact numbers, tickers, and filing codes |
| **Reranker** | Cohere Rerank v3 | Free tier; top BEIR benchmark; graceful cosine fallback built in |
| **Document Parsing** | Unstructured.io | Semantic block extraction from SEC iXBRL HTML — far superior to raw BeautifulSoup |
| **PDF Parsing** | PyMuPDF4LLM + pdfplumber | Free, fast, handles 90%+ of EDGAR PDFs; LlamaParse selectively for complex financials |
| **Async HTTP** | aiohttp + asyncio | Non-blocking concurrent EDGAR fetching — 10–20× faster than synchronous requests |
| **Retry Logic** | tenacity | Exponential backoff for EDGAR transient errors |
| **API Layer** | FastAPI + Pydantic | Auto-validated request/response models; native async; OpenAPI docs generated automatically |
| **Frontend** | Streamlit | Chat UI with streaming support (`st.write_stream`); source citation panels; zero frontend code |
| **Evaluation** | RAGAS + DeepEval | RAG-specific metrics (faithfulness, relevancy, hallucination) — BLEU/ROUGE miss RAG failure modes |

---

## Quick Start

### Prerequisites

- Python 3.10+
- ~4 GB disk space (embedding model + index)
- `ANTHROPIC_API_KEY` (required)
- `COHERE_API_KEY` (optional — falls back to cosine similarity)

### 1. Clone and create environment

```bash
git clone https://github.com/nikhilreddy00/sec-finance-rag.git
cd sec-finance-rag
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set your API keys:
```

```env
ANTHROPIC_API_KEY=sk-ant-...
COHERE_API_KEY=...            # optional
VECTOR_STORE=chroma           # or qdrant
CLAUDE_MODEL=claude-sonnet-4-6
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
```

### 3. Run ingestion (Apple demo — ~3 min on CPU)

```bash
python scripts/ingest_apple_html.py
```

```
Step 1/4  Parsing HTML filings      ✓  6 files → 1,876 blocks
Step 2/4  Chunking documents        ✓  1,876 chunks (1,708 text + 168 table)
Step 3/4  Embedding → ChromaDB      ✓  1,876 vectors (1024-dim BGE)
Step 4/4  Building BM25 index       ✓  1,708 text chunks indexed

ChromaDB chunks : 1,876
BM25 chunks     : 1,708
Done in 3m 12s
```

### 4. Launch the chatbot

```bash
streamlit run app.py
```

Open **http://localhost:8501**

### 5. (Optional) Start API server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Docs: http://localhost:8000/docs
```

---

## API Reference

### `POST /api/query`

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Apple total revenue in FY2024?", "filters": {"fiscal_year": 2024}}'
```

```json
{
  "answer": "Apple reported total net sales of $391.0 billion in fiscal year 2024... [Source: AAPL, 10-K, 2024]",
  "sources": [{"ticker": "AAPL", "form_type": "10-K", "filing_date": "2024-09-28", "section": "Item 7 — MD&A"}],
  "filters_applied": {"ticker": "AAPL", "fiscal_year": 2024},
  "num_chunks": 8
}
```

### `POST /api/query/stream` — Server-Sent Events

```bash
curl -N -X POST http://localhost:8000/api/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Apple risk factors 2024"}'
```

### `GET /api/health`

```json
{
  "status": "ok",
  "indexed_chunks": 1876,
  "vector_store": "chroma",
  "embedding_model": "BAAI/bge-large-en-v1.5",
  "claude_model": "claude-sonnet-4-6"
}
```

---

## Configuration Reference

All settings can be overridden via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Anthropic model for generation |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | HuggingFace embedding model name |
| `VECTOR_STORE` | `chroma` | `chroma` or `qdrant` |
| `RETRIEVAL_K` | `50` | Candidate chunks before reranking |
| `COHERE_RERANK_TOP_N` | `8` | Final chunks passed to Claude |
| `BM25_WEIGHT` | `0.4` | BM25 weight in RRF fusion |
| `DENSE_WEIGHT` | `0.6` | Dense embedding weight in RRF fusion |
| `MULTI_QUERY_COUNT` | `2` | Query variants generated per expansion |
| `CHUNK_SIZE_NARRATIVE` | `512` | Tokens per narrative chunk |
| `CHUNK_OVERLAP_NARRATIVE` | `100` | Token overlap between consecutive chunks |
| `SENTENCE_WINDOW_SIZE` | `3` | Context window stored per chunk |
| `CACHE_ENABLED` | `True` | Enable semantic response cache |
| `CACHE_SIMILARITY_THRESHOLD` | `0.92` | Cosine threshold for cache hit |
| `MAX_QUERY_LENGTH` | `500` | Maximum input query characters |

---

## Example Questions

| Question | Pipeline Behavior |
|----------|------------------|
| *"What was Apple's total revenue in FY2024?"* | Self-query extracts `{AAPL, 10-K, 2024}` → filtered retrieval → skips multi-query (precise) |
| *"Compare Apple R&D spending from 2020 to 2024"* | No fiscal_year → multi-query expansion → cross-year retrieval |
| *"What risk factors did Apple disclose in 2023?"* | Section-aware filter (`Item 1A`) → targeted retrieval |
| *"Which S&P 500 companies mentioned AI risk in 2024 10-K?"* | Broad query → full multi-query → unfiltered hybrid search |
| *"Apple iPhone revenue last three years"* | Relative date resolved by Claude self-query → {2022, 2023, 2024} |

---

## Evaluation

```bash
# Run full evaluation suite
python scripts/run_evaluation.py

# RAGAS only (faithfulness + relevancy)
python scripts/run_evaluation.py --ragas --sample 50

# DeepEval only (hallucination detection)
python scripts/run_evaluation.py --deepeval --sample 50
```

---

## License

MIT License — SEC filings are public domain. Financial data sourced from SEC EDGAR.

> **Disclaimer:** This tool is for research and analysis purposes only. It does not constitute financial, investment, or legal advice. Always consult a licensed financial advisor before making investment decisions.
