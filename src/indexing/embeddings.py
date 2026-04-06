"""
Embedding wrapper for BAAI/bge-large-en-v1.5 via sentence-transformers.

This is a free, locally-hosted model with strong MTEB benchmark performance
(1024 dimensions). No API key required.

BGE models use a query prefix for retrieval — this is handled automatically:
- Documents: plain text
- Queries:   "Represent this sentence: " prefix (improves retrieval quality)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config.settings import settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# BGE models recommend this prefix for query embeddings
BGE_QUERY_PREFIX = "Represent this sentence: "


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbedding:
    """
    Return a cached HuggingFace embedding model (BAAI/bge-large-en-v1.5).

    Loaded once at startup; subsequent calls return the same instance.
    """
    logger.info("Loading embedding model: %s", settings.embedding_model)
    model = HuggingFaceEmbedding(
        model_name=settings.embedding_model,
        embed_batch_size=settings.embedding_batch_size,
        query_instruction=BGE_QUERY_PREFIX,
        # text_instruction is empty — documents are embedded as-is
        text_instruction="",
        device=None,  # auto-detect CPU/CUDA
    )
    logger.info("Embedding model loaded (dim=%d)", _get_dim(model))
    return model


def _get_dim(model: HuggingFaceEmbedding) -> int:
    """Probe embedding dimension by encoding a dummy sentence."""
    try:
        vec = model._model.encode(["test"])
        return vec.shape[1]
    except Exception:
        return -1


def embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """
    Embed a list of texts.

    Args:
        texts: texts to embed
        is_query: if True, apply BGE query prefix
    """
    model = get_embedding_model()
    if is_query:
        texts = [BGE_QUERY_PREFIX + t for t in texts]
    return model._model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,
    ).tolist()
