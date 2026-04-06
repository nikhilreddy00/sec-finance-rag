"""
Async ingestion pipeline: scrape → parse → chunk → embed → index.

Also builds and persists the BM25 index used for hybrid retrieval.

Usage (from scripts/run_ingestion.py):
    asyncio.run(run_ingestion_pipeline(tickers=["AAPL"], form_types=["10-K"]))
"""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from config.settings import settings
from src.ingestion.chunker import FilingChunker, Chunk
from src.ingestion.parser import FilingParser, ParsedDocument, save_parsed_document
from src.ingestion.scraper import FilingMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BM25 index builder
# ---------------------------------------------------------------------------

def build_bm25_index(chunks: list[Chunk]) -> None:
    """
    Build a BM25Retriever over all text chunks and persist to disk.

    Only text chunks are indexed (tables are excluded — keyword search on
    tabular data is less useful than vector similarity).
    """
    from rank_bm25 import BM25Okapi
    from src.retrieval.hybrid import tokenize_financial

    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    logger.info("Building BM25 index over %d text chunks", len(text_chunks))

    tokenised = [tokenize_financial(c.text) for c in text_chunks]
    bm25 = BM25Okapi(tokenised)

    payload = {
        "bm25": bm25,
        "chunks": text_chunks,          # keep chunks for score→chunk mapping
    }
    settings.bm25_index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.bm25_index_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("BM25 index saved to %s", settings.bm25_index_path)


def load_bm25_index() -> tuple:
    """Load persisted BM25 index. Returns (bm25, chunks)."""
    if not settings.bm25_index_path.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {settings.bm25_index_path}. "
            "Run ingestion pipeline first."
        )
    with open(settings.bm25_index_path, "rb") as f:
        payload = pickle.load(f)
    return payload["bm25"], payload["chunks"]


# ---------------------------------------------------------------------------
# Async ingestion pipeline
# ---------------------------------------------------------------------------

async def run_apple_html_pipeline(dataset_dir: Optional[str] = None) -> None:
    """
    Streamlined ingestion pipeline for Apple 10-K HTML files.

    Reads directly from Apple_Dataset/*.html — no scraping required.
    Steps:
      1. Load + parse all HTML files via html_loader
      2. Save parsed JSON to data/processed/
      3. Chunk all documents
      4. Embed and index into ChromaDB
      5. Build BM25 index

    Args:
        dataset_dir: path to Apple_Dataset folder (defaults to ./Apple_Dataset)
    """
    from src.ingestion.html_loader import load_apple_html_filings
    from src.ingestion.parser import save_parsed_document

    # ── Step 1: Load & Parse HTML files ───────────────────────────────────────
    logger.info("=== Step 1/4: Parsing Apple 10-K HTML files ===")
    from pathlib import Path as _Path
    docs = load_apple_html_filings(_Path(dataset_dir) if dataset_dir else None)
    if not docs:
        logger.error("No documents parsed. Check Apple_Dataset folder.")
        return

    # Save parsed JSON for caching
    for doc in docs:
        save_parsed_document(doc, settings.processed_data_dir)
    logger.info("Parsed and saved %d documents", len(docs))

    # ── Step 2: Chunk ─────────────────────────────────────────────────────────
    logger.info("=== Step 2/4: Chunking documents ===")
    all_chunks = _chunk_all_documents(docs)
    logger.info("Total chunks: %d", len(all_chunks))

    # ── Step 3: Embed & Index ─────────────────────────────────────────────────
    logger.info("=== Step 3/4: Embedding and indexing into ChromaDB ===")
    await _index_chunks(all_chunks)

    # ── Step 4: BM25 index ────────────────────────────────────────────────────
    logger.info("=== Step 4/4: Building BM25 index ===")
    build_bm25_index(all_chunks)

    logger.info("=== Apple HTML ingestion pipeline complete ===")
    logger.info(
        "Indexed %d chunks from %d Apple 10-K filings (FY2020–FY2025)",
        len(all_chunks), len(docs),
    )


async def run_ingestion_pipeline(
    tickers: Optional[list[str]] = None,
    form_types: Optional[list[str]] = None,
    years_back: Optional[int] = None,
    skip_scraping: bool = False,
) -> None:
    """
    Full ingestion pipeline:
    1. Scrape SEC EDGAR (unless skip_scraping=True)
    2. Parse each filing
    3. Chunk each document
    4. Embed and index chunks into vector store
    5. Build BM25 index

    Args:
        tickers: S&P 500 subset; defaults to all 500.
        form_types: defaults to settings.filing_types.
        years_back: defaults to settings.years_back.
        skip_scraping: if True, parse from already-downloaded files.
    """
    # ── Step 1: Scrape ────────────────────────────────────────────────────────
    if not skip_scraping:
        from src.ingestion.scraper import scrape_sec_filings
        logger.info("=== Step 1/5: Scraping SEC EDGAR ===")
        await scrape_sec_filings(tickers, form_types, years_back)
    else:
        logger.info("=== Step 1/5: Skipping scraping (using existing files) ===")

    # ── Step 2: Parse ─────────────────────────────────────────────────────────
    logger.info("=== Step 2/5: Parsing filings ===")
    all_parsed = await _parse_all_filings()

    # ── Step 3: Chunk ─────────────────────────────────────────────────────────
    logger.info("=== Step 3/5: Chunking documents ===")
    all_chunks = _chunk_all_documents(all_parsed)
    logger.info("Total chunks: %d", len(all_chunks))

    # ── Step 4: Embed & Index ─────────────────────────────────────────────────
    logger.info("=== Step 4/5: Embedding and indexing chunks ===")
    await _index_chunks(all_chunks)

    # ── Step 5: BM25 index ────────────────────────────────────────────────────
    logger.info("=== Step 5/5: Building BM25 index ===")
    build_bm25_index(all_chunks)

    logger.info("=== Ingestion pipeline complete ===")


# ---------------------------------------------------------------------------
# Internal steps
# ---------------------------------------------------------------------------

async def _parse_all_filings() -> list[ParsedDocument]:
    """
    Walk data/raw/, parse each filing, save to data/processed/.
    Already-parsed documents are loaded from cache (skip re-parsing).
    """
    parser = FilingParser()
    parsed_docs: list[ParsedDocument] = []

    # Find all metadata.json files (one per filing)
    meta_files = list(settings.raw_data_dir.rglob("metadata.json"))
    logger.info("Found %d raw filings to parse", len(meta_files))

    for meta_file in tqdm(meta_files, desc="Parsing filings"):
        try:
            meta_dict = json.loads(meta_file.read_text())
        except Exception as exc:
            logger.warning("Bad metadata.json at %s: %s", meta_file, exc)
            continue

        local_path = meta_dict.get("local_path", "")

        # Check if parsed.json already exists
        form_dir = meta_file.parent.name  # e.g. "2023-01-28"
        form_type_dir = meta_file.parent.parent.name  # e.g. "10-K"
        ticker = meta_file.parent.parent.parent.name

        parsed_path = (
            settings.processed_data_dir
            / ticker
            / form_type_dir
            / form_dir
            / "parsed.json"
        )
        if parsed_path.exists():
            from src.ingestion.parser import load_parsed_document
            try:
                doc = load_parsed_document(parsed_path)
                parsed_docs.append(doc)
                continue
            except Exception:
                pass  # re-parse on corrupt cache

        doc = parser.parse(meta_dict, local_path)
        if doc is not None:
            save_parsed_document(doc, settings.processed_data_dir)
            parsed_docs.append(doc)

    logger.info("Parsed %d documents", len(parsed_docs))
    return parsed_docs


def _chunk_all_documents(docs: list[ParsedDocument]) -> list[Chunk]:
    """Chunk all parsed documents. Returns flat list of all chunks."""
    chunker = FilingChunker()
    all_chunks: list[Chunk] = []
    for doc in tqdm(docs, desc="Chunking documents"):
        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)
    return all_chunks


async def _index_chunks(chunks: list[Chunk]) -> None:
    """
    Embed and upsert chunks into ChromaDB directly.

    Uses ChromaDB's native add() API — bypasses LlamaIndex VectorStoreIndex
    abstraction for reliability. Processes in batches, skips already-indexed chunks.
    """
    from src.indexing.embeddings import embed_texts
    from src.indexing.vector_store import get_chroma_collection

    collection = get_chroma_collection()

    # Get already-indexed IDs to avoid duplicates
    indexed_ids = _get_existing_chunk_ids()
    new_chunks = [c for c in chunks if c.chunk_id not in indexed_ids]
    logger.info(
        "%d chunks total, %d new (skipping %d already indexed)",
        len(chunks), len(new_chunks), len(indexed_ids),
    )

    if not new_chunks:
        logger.info("All chunks already indexed — nothing to do.")
        return

    batch_size = settings.ingestion_batch_size
    batches = [new_chunks[i:i + batch_size] for i in range(0, len(new_chunks), batch_size)]

    for batch in tqdm(batches, desc="Indexing batches"):
        texts = [c.text for c in batch]
        embeddings = embed_texts(texts, is_query=False)
        metadatas = [c.metadata() for c in batch]
        # Ensure all metadata values are ChromaDB-compatible (str/int/float/bool)
        for meta in metadatas:
            for k, v in list(meta.items()):
                if v is None:
                    meta[k] = ""
                elif not isinstance(v, (str, int, float, bool)):
                    meta[k] = str(v)
        ids = [c.chunk_id for c in batch]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    logger.info("Indexed %d new chunks into ChromaDB", len(new_chunks))


def _get_existing_chunk_ids() -> set[str]:
    """Return set of chunk IDs already in the vector store."""
    try:
        if settings.vector_store == "chroma":
            from src.indexing.vector_store import get_chroma_collection
            col = get_chroma_collection()
            result = col.get(include=[])
            return set(result["ids"])
        elif settings.vector_store == "qdrant":
            from src.indexing.vector_store import get_qdrant_client
            # Qdrant scroll to get all IDs (may be slow for very large collections)
            client = get_qdrant_client()
            ids: set[str] = set()
            offset = None
            while True:
                records, offset = client.scroll(
                    collection_name=settings.qdrant_collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                for r in records:
                    ids.add(str(r.id))
                if offset is None:
                    break
            return ids
    except Exception as exc:
        logger.warning("Could not fetch existing IDs: %s — will re-index all", exc)
    return set()
