"""
Ingest Apple 10-K filings from local HTML files in Apple_Dataset/.

This script replaces the old scraping-based ingestion for the Apple dataset.
It reads aapl-YYYYMMDD.html files directly, parses them with BeautifulSoup /
Unstructured, chunks them, and indexes them into ChromaDB + BM25.

Usage:
    # From project root (activate venv first)
    python scripts/ingest_apple_html.py

    # Specify a custom dataset directory
    python scripts/ingest_apple_html.py --dataset-dir /path/to/Apple_Dataset

    # Force re-index even if data already exists
    python scripts/ingest_apple_html.py --force

    # BM25-only rebuild (fast, no re-embedding)
    python scripts/ingest_apple_html.py --bm25-only
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest Apple 10-K HTML filings from Apple_Dataset/"
    )
    parser.add_argument(
        "--dataset-dir",
        default="./Apple_Dataset",
        metavar="DIR",
        help="Path to folder containing aapl-YYYYMMDD.html files (default: ./Apple_Dataset)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing ChromaDB and BM25 index before re-ingesting",
    )
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="Only rebuild BM25 index from already-parsed documents (fast)",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    # ── Force re-index: wipe existing data ────────────────────────────────────
    if args.force:
        import shutil
        logger.warning("--force: deleting existing ChromaDB and BM25 index")
        chroma_dir = settings.chroma_persist_dir
        if chroma_dir.exists():
            shutil.rmtree(chroma_dir)
            logger.info("Deleted ChromaDB at %s", chroma_dir)
        bm25_path = settings.bm25_index_path
        if bm25_path.exists():
            bm25_path.unlink()
            logger.info("Deleted BM25 index at %s", bm25_path)

    # ── BM25-only rebuild ──────────────────────────────────────────────────────
    if args.bm25_only:
        logger.info("BM25-only rebuild from parsed documents in data/processed/")
        from src.ingestion.parser import iter_parsed_documents
        from src.ingestion.chunker import FilingChunker
        from src.indexing.pipeline import build_bm25_index

        chunker = FilingChunker()
        all_chunks = []
        for doc in iter_parsed_documents(settings.processed_data_dir):
            all_chunks.extend(chunker.chunk(doc))

        if not all_chunks:
            logger.error(
                "No parsed documents found in %s. Run ingestion first.",
                settings.processed_data_dir,
            )
            sys.exit(1)

        logger.info("Building BM25 index over %d chunks", len(all_chunks))
        build_bm25_index(all_chunks)
        print(f"\nBM25 index rebuilt: {len(all_chunks):,} chunks.")
        return

    # ── Full ingestion pipeline ────────────────────────────────────────────────
    logger.info("Starting Apple 10-K HTML ingestion from: %s", args.dataset_dir)

    from src.indexing.pipeline import run_apple_html_pipeline
    await run_apple_html_pipeline(dataset_dir=args.dataset_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    from src.indexing.vector_store import get_indexed_count
    from src.indexing.pipeline import load_bm25_index

    indexed = get_indexed_count()
    bm25_chunks = 0
    try:
        _, bm25_data = load_bm25_index()
        bm25_chunks = len(bm25_data)
    except FileNotFoundError:
        pass

    print("\n" + "=" * 55)
    print("  Apple 10-K Ingestion Complete")
    print("=" * 55)
    print(f"  ChromaDB chunks indexed : {indexed:>8,}")
    print(f"  BM25 index chunks       : {bm25_chunks:>8,}")
    print(f"  Vector store            : {settings.vector_store}")
    print(f"  Embedding model         : {settings.embedding_model.split('/')[-1]}")
    print("=" * 55)
    print("\nRun the chatbot:")
    print("  streamlit run app.py")
    print("\nRun the API server:")
    print("  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")


if __name__ == "__main__":
    asyncio.run(main())
