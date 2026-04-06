"""
Central configuration via Pydantic Settings.
All values can be overridden with environment variables or a .env file.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(..., description="Anthropic API key (required)")
    claude_model: str = Field("claude-sonnet-4-6", description="Claude model ID")

    # ── Reranking ─────────────────────────────────────────────────────────────
    cohere_api_key: str = Field("", description="Cohere API key (optional; free tier)")
    cohere_rerank_model: str = Field("rerank-english-v3.0")
    # top_n=8: from 50 candidates, reranker identifies top 8 most relevant chunks.
    # WHY 8 not 2: Multi-year comparison queries (revenue, product sales across years)
    # require one table chunk per filing year. Dense search correctly ranks financial
    # tables #1-6, but BM25 inflates qualitative text chunks via RRF. With top_n=2
    # Claude only received narrative text without actual dollar figures. 8 ensures
    # both table chunks (numbers) and narrative chunks (context) reach Claude.
    cohere_rerank_top_n: int = Field(8, ge=1, le=20)

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = Field(
        "BAAI/bge-large-en-v1.5",
        description="HuggingFace sentence-transformers model",
    )
    embedding_batch_size: int = Field(32, ge=1)

    # ── Vector Store ──────────────────────────────────────────────────────────
    vector_store: Literal["chroma", "qdrant"] = Field("chroma")
    chroma_persist_dir: Path = Field(Path("./data/chroma_db"))
    chroma_collection_name: str = Field("sec_filings")
    qdrant_url: str = Field("", description="Qdrant cloud URL (prod)")
    qdrant_api_key: str = Field("")
    qdrant_collection_name: str = Field("sec_filings")

    # ── Data Directories ──────────────────────────────────────────────────────
    raw_data_dir: Path = Field(Path("./data/raw"))
    processed_data_dir: Path = Field(Path("./data/processed"))
    bm25_index_path: Path = Field(Path("./data/bm25_index.pkl"))
    eval_dataset_path: Path = Field(Path("./data/eval_dataset.json"))

    # ── SEC EDGAR ─────────────────────────────────────────────────────────────
    sec_user_agent: str = Field(
        "Finance Bot admin@example.com",
        description="Required by SEC EDGAR fair-use policy",
    )
    # Max concurrent requests to EDGAR (SEC limit: 10 req/sec)
    sec_max_concurrency: int = Field(8, ge=1, le=10)
    sec_request_delay: float = Field(0.12, description="Seconds between requests")

    # ── Ingestion ─────────────────────────────────────────────────────────────
    filing_types: list[str] = Field(
        default=["10-K", "10-Q", "8-K", "DEF 14A"],
        description="SEC form types to scrape",
    )
    years_back: int = Field(5, ge=1, le=20, description="How many years of filings")
    ingestion_batch_size: int = Field(100, ge=1)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    # retrieval_k=50: cast a wide net; reranker then selects the top 2 precisely.
    # More candidates = better reranker input quality. 50 is the sweet spot
    # for financial Q&A (diminishing returns beyond 50, latency increases).
    retrieval_k: int = Field(50, description="Candidates before reranking")
    bm25_weight: float = Field(0.4, description="BM25 weight in hybrid search")
    dense_weight: float = Field(0.6, description="Dense weight in hybrid search")
    similarity_threshold: float = Field(0.76, description="Contextual compression threshold")
    # multi_query_count=2: was 3. Saves one Claude call per query.
    # 2 variants captures most recall improvement; 3rd variant adds marginal gain.
    multi_query_count: int = Field(2, description="Number of query variants to generate")

    # ── Semantic Cache ────────────────────────────────────────────────────────
    # threshold=0.92: financial queries are precise — "Apple revenue 2023" and
    # "AAPL FY2023 revenue" should hit cache; "Apple revenue 2022" should not.
    cache_enabled: bool = Field(True, description="Enable semantic response caching")
    cache_similarity_threshold: float = Field(0.92, description="Cosine similarity for cache hit")
    cache_ttl_days: int = Field(7, description="Cache TTL in days (SEC filings don't change)")
    cache_max_entries: int = Field(10_000)

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size_narrative: int = Field(512)
    chunk_overlap_narrative: int = Field(100)
    chunk_size_short: int = Field(256, description="For short docs like 8-K")
    sentence_window_size: int = Field(3)

    # ── Guardrails ────────────────────────────────────────────────────────────
    max_query_length: int = Field(500)

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_sample_size: int = Field(200, description="Number of synthetic QA pairs")
    ragas_faithfulness_threshold: float = Field(0.80)
    ragas_relevancy_threshold: float = Field(0.75)
    deepeval_hallucination_threshold: float = Field(0.20)

    # ── API Server ────────────────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0")
    api_port: int = Field(8000, ge=1, le=65535)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO")

    @field_validator("chroma_persist_dir", "raw_data_dir", "processed_data_dir", mode="after")
    @classmethod
    def ensure_dirs_exist(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def reranking_enabled(self) -> bool:
        return bool(self.cohere_api_key)


# Singleton — import and use anywhere
settings = Settings()
