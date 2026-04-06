"""
Re-index Apple 10-K filings from scratch.

Clears the existing ChromaDB collection, parsed JSON cache, and semantic cache,
then re-runs the full Apple HTML ingestion pipeline with the fixed parser.

Run from project root:
    python scripts/reindex_apple.py
"""

import asyncio
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings

logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def clear_chroma_collection() -> None:
    """Delete and recreate the ChromaDB collection so section metadata is fresh."""
    import chromadb
    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    try:
        client.delete_collection(settings.chroma_collection_name)
        logger.info("Deleted ChromaDB collection '%s'", settings.chroma_collection_name)
    except Exception:
        logger.info("ChromaDB collection did not exist — nothing to delete")


def clear_parsed_cache() -> None:
    """Remove all parsed.json files so the parser re-runs with the fixed regex."""
    parsed_dir = settings.processed_data_dir / "AAPL"
    if parsed_dir.exists():
        shutil.rmtree(parsed_dir)
        logger.info("Cleared parsed cache at %s", parsed_dir)
    else:
        logger.info("No parsed cache found — skipping")


def clear_bm25_index() -> None:
    """Remove the BM25 index so it is rebuilt cleanly."""
    if settings.bm25_index_path.exists():
        settings.bm25_index_path.unlink()
        logger.info("Removed BM25 index at %s", settings.bm25_index_path)


def clear_semantic_cache() -> None:
    """Clear the semantic response cache to remove stale 'insufficient information' answers."""
    cache_dir = Path("./data/cache")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        logger.info("Cleared semantic cache at %s", cache_dir)

    # Also clear in-memory cache module state
    try:
        from src.retrieval import cache as cache_mod
        if hasattr(cache_mod, "_cache_instance"):
            cache_mod._cache_instance = None
    except Exception:
        pass


async def main() -> None:
    logger.info("=== Apple Re-indexing Script ===")
    logger.info("This will clear all existing indexed data and re-index from HTML source files.")

    logger.info("Step 1/5: Clearing semantic response cache")
    clear_semantic_cache()

    logger.info("Step 2/5: Clearing BM25 index")
    clear_bm25_index()

    logger.info("Step 3/5: Clearing ChromaDB collection")
    clear_chroma_collection()

    logger.info("Step 4/5: Clearing parsed JSON cache")
    clear_parsed_cache()

    logger.info("Step 5/5: Running Apple HTML ingestion pipeline (parse → chunk → embed → index → BM25)")
    from src.indexing.pipeline import run_apple_html_pipeline
    await run_apple_html_pipeline()

    logger.info("=== Re-indexing complete ===")

    # Summary
    import chromadb
    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    col = client.get_collection(settings.chroma_collection_name)
    print(f"\nChromaDB collection '{settings.chroma_collection_name}': {col.count()} chunks indexed")

    import pickle
    with open(settings.bm25_index_path, "rb") as f:
        bm25_data = pickle.load(f)
    print(f"BM25 index: {len(bm25_data['chunks'])} text chunks indexed")


if __name__ == "__main__":
    asyncio.run(main())
