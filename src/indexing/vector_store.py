"""
Vector store factory — supports ChromaDB (local/dev) and Qdrant (prod free cloud).

Usage:
    from src.indexing.vector_store import get_vector_store, get_chroma_collection

Set VECTOR_STORE=chroma (default) or VECTOR_STORE=qdrant in your .env.

ChromaDB:
  - Persistent local storage in data/chroma_db/
  - No storage limits (bound only by disk)
  - Default for development and local runs

Qdrant:
  - Free cloud tier: 1 GB / cluster (sufficient for ~5-10k dense chunks)
  - Set QDRANT_URL and QDRANT_API_KEY in .env
  - Recommended for Streamlit Cloud deployment
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Union

from config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_chroma_client():
    """Return a persistent ChromaDB client (cached)."""
    import chromadb
    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    logger.info("ChromaDB client initialised at %s", settings.chroma_persist_dir)
    return client


def get_chroma_collection():
    """Return (or create) the SEC filings ChromaDB collection."""
    from chromadb.utils import embedding_functions
    client = get_chroma_client()
    # Use cosine distance — better for normalised BGE embeddings
    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        "ChromaDB collection '%s' ready (%d docs)",
        settings.chroma_collection_name,
        collection.count(),
    )
    return collection


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

def get_qdrant_client():
    """Return a Qdrant client (cloud or local)."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    if settings.qdrant_url:
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        logger.info("Qdrant client connected to %s", settings.qdrant_url)
    else:
        # In-memory Qdrant for testing
        client = QdrantClient(":memory:")
        logger.warning("Qdrant URL not set — using in-memory mode (not persistent)")

    # Ensure collection exists (BGE-large: 1024 dims)
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection_name not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s'", settings.qdrant_collection_name)

    return client


# ---------------------------------------------------------------------------
# LlamaIndex VectorStore wrappers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_llama_vector_store():
    """
    Return a LlamaIndex VectorStore backed by the configured backend.
    Cached after first call.
    """
    if settings.vector_store == "qdrant":
        return _build_qdrant_vector_store()
    return _build_chroma_vector_store()


def _build_chroma_vector_store():
    from llama_index.vector_stores.chroma import ChromaVectorStore
    collection = get_chroma_collection()
    return ChromaVectorStore(chroma_collection=collection)


def _build_qdrant_vector_store():
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    client = get_qdrant_client()
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection_name,
    )


# ---------------------------------------------------------------------------
# LlamaIndex StorageContext & Index
# ---------------------------------------------------------------------------

def build_storage_context(vector_store=None):
    """Build a LlamaIndex StorageContext with the configured vector store."""
    from llama_index.core import StorageContext
    vs = vector_store or get_llama_vector_store()
    return StorageContext.from_defaults(vector_store=vs)


def get_vector_store_index(storage_context=None, embed_model=None):
    """
    Load (or create) a LlamaIndex VectorStoreIndex from the configured backend.
    This is the top-level index used by query engines.
    """
    from llama_index.core import VectorStoreIndex
    from src.indexing.embeddings import get_embedding_model

    sc = storage_context or build_storage_context()
    em = embed_model or get_embedding_model()

    index = VectorStoreIndex.from_vector_store(
        vector_store=sc.vector_store,
        embed_model=em,
    )
    logger.info("VectorStoreIndex loaded from %s backend", settings.vector_store)
    return index


# ---------------------------------------------------------------------------
# Document count helper
# ---------------------------------------------------------------------------

def get_indexed_count() -> int:
    """Return approximate number of indexed chunks."""
    try:
        if settings.vector_store == "chroma":
            return get_chroma_collection().count()
        elif settings.vector_store == "qdrant":
            client = get_qdrant_client()
            info = client.get_collection(settings.qdrant_collection_name)
            return info.points_count or 0
    except Exception:
        return -1
