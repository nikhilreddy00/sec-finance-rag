"""
FastAPI backend for the Finance RAG Chatbot.

Endpoints:
    POST /api/query          → full RAG pipeline (batch)
    POST /api/query/stream   → streaming RAG response (SSE)
    GET  /api/health         → liveness check + indexed doc count
    GET  /api/stats          → retrieval and index statistics
    POST /api/evaluate       → trigger evaluation run (background task)

Run locally:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config.settings import settings

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Finance RAG API",
    description="SEC EDGAR financial data chatbot powered by Claude",
    version="1.0.0",
)

# CORS — allow Streamlit frontend (any origin in development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, description="User question")
    filters: Optional[dict] = Field(None, description="Optional metadata filters")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve")


class SourceInfo(BaseModel):
    ticker: str
    company_name: str
    form_type: str
    filing_date: str
    fiscal_year: int
    section: str
    chunk_type: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    filters_applied: dict
    num_chunks: int


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
    vector_store: str
    embedding_model: str
    claude_model: str


# ---------------------------------------------------------------------------
# Lazy-load the RAG chain (avoids slow startup on import)
# ---------------------------------------------------------------------------

_chain = None


def get_chain():
    global _chain
    if _chain is None:
        from src.generation.chain import get_rag_chain
        _chain = get_rag_chain()
    return _chain


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Liveness + readiness check."""
    from src.indexing.vector_store import get_indexed_count
    return HealthResponse(
        status="ok",
        indexed_chunks=get_indexed_count(),
        vector_store=settings.vector_store,
        embedding_model=settings.embedding_model,
        claude_model=settings.claude_model,
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Execute RAG pipeline and return answer with sources.

    Full pipeline: guardrails → self-query → hybrid → multi-query → rerank → Claude
    """
    from src.guardrails.input_guard import GuardrailError

    try:
        chain = get_chain()
        result = chain.query(
            question=request.question,
            filters=request.filters,
            top_k=request.top_k,
        )
    except GuardrailError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Query failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error. Please try again.")

    sources = [
        SourceInfo(
            ticker=s.get("ticker", ""),
            company_name=s.get("company_name", ""),
            form_type=s.get("form_type", ""),
            filing_date=s.get("filing_date", ""),
            fiscal_year=s.get("fiscal_year", 0),
            section=s.get("section", ""),
            chunk_type=s.get("chunk_type", "text"),
        )
        for s in result["sources"]
    ]

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        filters_applied=result["filters"],
        num_chunks=result["num_chunks"],
    )


@app.post("/api/query/stream")
async def query_stream(request: QueryRequest):
    """
    Streaming RAG response via Server-Sent Events.
    Useful for Streamlit st.write_stream().
    """
    from src.guardrails.input_guard import GuardrailError

    try:
        chain = get_chain()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    def generate():
        try:
            for delta in chain.query_stream(
                question=request.question,
                filters=request.filters,
                top_k=request.top_k,
            ):
                yield f"data: {delta}\n\n"
        except GuardrailError as e:
            yield f"data: [ERROR] {e}\n\n"
        except Exception as e:
            logger.exception("Streaming query failed: %s", e)
            yield "data: [ERROR] An error occurred. Please try again.\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/evaluate")
async def run_evaluation(background_tasks: BackgroundTasks, framework: str = "ragas"):
    """
    Trigger RAG evaluation in the background.

    Args:
        framework: "ragas" or "deepeval"
    """
    if framework not in ("ragas", "deepeval"):
        raise HTTPException(status_code=400, detail="framework must be 'ragas' or 'deepeval'")

    def _run_eval():
        try:
            if framework == "ragas":
                from src.evaluation.ragas_eval import RAGASEvaluator
                RAGASEvaluator().evaluate()
            else:
                from src.evaluation.deepeval_eval import DeepEvalEvaluator
                DeepEvalEvaluator().evaluate()
        except Exception as e:
            logger.exception("Evaluation failed: %s", e)

    background_tasks.add_task(_run_eval)
    return {"status": "evaluation started", "framework": framework}


@app.get("/api/stats")
async def stats():
    """Index and retrieval statistics."""
    from src.indexing.vector_store import get_indexed_count
    from src.indexing.pipeline import load_bm25_index

    bm25_count = 0
    try:
        _, chunks = load_bm25_index()
        bm25_count = len(chunks)
    except FileNotFoundError:
        pass

    return {
        "vector_store": settings.vector_store,
        "indexed_chunks": get_indexed_count(),
        "bm25_chunks": bm25_count,
        "embedding_model": settings.embedding_model,
        "reranking_enabled": settings.reranking_enabled,
        "cohere_model": settings.cohere_rerank_model if settings.reranking_enabled else None,
        "retrieval_k": settings.retrieval_k,
        "rerank_top_n": settings.cohere_rerank_top_n,
    }
