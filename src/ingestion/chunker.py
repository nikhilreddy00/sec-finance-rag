"""
Chunking strategies for SEC EDGAR parsed documents.

Strategy:
- Narrative text (10-K Item 7, Risk Factors, etc.): SentenceWindowNodeParser
  with window_size=3, chunk_size=512 tokens, overlap=100 tokens.
- Financial tables: kept as single atomic chunks (never split).
- Short-form filings (8-K, DEF 14A): smaller chunk_size=256.

All chunks receive rich metadata for self-querying retrieval:
    company_name, ticker, cik, form_type, filing_date, fiscal_year,
    section, chunk_type (text|table), table_name, page_number, chunk_id
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from config.settings import settings
from src.ingestion.parser import ParsedDocument, TextBlock

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single indexable chunk from a filing."""
    chunk_id: str             # SHA-256 of (ticker + form_type + filing_date + text)
    text: str
    # Metadata for vector store and self-querying
    ticker: str
    company_name: str
    cik: str
    form_type: str
    filing_date: str          # YYYY-MM-DD
    fiscal_year: int
    section: str              # e.g. "Item 7 — Management's Discussion and Analysis"
    chunk_type: str           # "text" | "table"
    table_name: str = ""
    page_number: int = 0
    # SentenceWindow context (stored for MetadataReplacementPostProcessor)
    window_text: str = ""

    def metadata(self) -> dict:
        """Return flat metadata dict for vector store."""
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "cik": self.cik,
            "form_type": self.form_type,
            "filing_date": self.filing_date,
            "fiscal_year": self.fiscal_year,
            "section": self.section,
            "chunk_type": self.chunk_type,
            "table_name": self.table_name,
            "page_number": self.page_number,
        }

    def to_llama_node(self) -> TextNode:
        """Convert to LlamaIndex TextNode for indexing."""
        node = TextNode(
            id_=self.chunk_id,
            text=self.text,
            metadata=self.metadata(),
        )
        if self.window_text:
            node.metadata["window"] = self.window_text
        return node


def _chunk_id(ticker: str, form_type: str, filing_date: str, text: str, idx: int = 0) -> str:
    raw = f"{ticker}|{form_type}|{filing_date}|{idx}|{text}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class FilingChunker:
    """
    Converts a ParsedDocument into a list of Chunk objects.

    Chunking rules:
    1. Table blocks → always one chunk per table (never split).
    2. Short-form docs (8-K, DEF 14A) → SentenceSplitter(chunk_size=256).
    3. All other narrative text → SentenceSplitter(chunk_size=512) with
       sliding context windows built manually for the window_text metadata.
    """

    SHORT_FORM_TYPES = {"8-K", "8-K/A", "DEF 14A", "DEFA14A"}

    def __init__(self) -> None:
        self._narrative_splitter = SentenceSplitter(
            chunk_size=settings.chunk_size_narrative,
            chunk_overlap=settings.chunk_overlap_narrative,
            include_metadata=True,
            include_prev_next_rel=True,
        )
        self._short_splitter = SentenceSplitter(
            chunk_size=settings.chunk_size_short,
            chunk_overlap=50,
            include_metadata=True,
            include_prev_next_rel=True,
        )

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        """Return all chunks for a parsed filing."""
        is_short = doc.form_type in self.SHORT_FORM_TYPES
        chunks: list[Chunk] = []
        global_idx = 0

        for block in doc.blocks:
            if block.block_type == "table":
                chunks.append(self._chunk_table(block, doc, global_idx))
                global_idx += 1
            else:
                new = self._chunk_narrative(block, doc, is_short=is_short, start_idx=global_idx)
                chunks.extend(new)
                global_idx += len(new)

        logger.debug(
            "%s %s %s → %d chunks",
            doc.ticker, doc.form_type, doc.filing_date, len(chunks),
        )
        return chunks

    # ── Table chunking ────────────────────────────────────────────────────────

    def _chunk_table(self, block: TextBlock, doc: ParsedDocument, idx: int = 0) -> Chunk:
        """Keep an entire table as one atomic chunk."""
        cid = _chunk_id(doc.ticker, doc.form_type, doc.filing_date, block.text, idx)
        return Chunk(
            chunk_id=cid,
            text=block.text,
            ticker=doc.ticker,
            company_name=doc.company_name,
            cik=doc.cik,
            form_type=doc.form_type,
            filing_date=doc.filing_date,
            fiscal_year=doc.fiscal_year,
            section=block.section,
            chunk_type="table",
            table_name=block.table_name,
            page_number=block.page_number,
        )

    # ── Narrative chunking ────────────────────────────────────────────────────

    def _chunk_narrative(
        self,
        block: TextBlock,
        doc: ParsedDocument,
        *,
        is_short: bool,
        start_idx: int = 0,
    ) -> list[Chunk]:
        """Split narrative text into sentence-window chunks."""
        if len(block.text.strip()) < 20:
            return []

        # Wrap the block text as a LlamaIndex Document so parsers can operate
        llama_doc = Document(
            text=block.text,
            metadata={
                "section": block.section,
                "page_number": block.page_number,
                **doc_base_metadata(doc),
            },
        )

        splitter = self._short_splitter if is_short else self._narrative_splitter
        nodes = splitter.get_nodes_from_documents([llama_doc])

        # Build sliding context windows manually (mimics SentenceWindowNodeParser)
        node_texts = [n.get_content() for n in nodes]
        w = settings.sentence_window_size

        chunks: list[Chunk] = []
        for idx, node in enumerate(nodes):
            text = node_texts[idx]
            if not text.strip():
                continue
            # Surrounding window for richer retrieval context
            start = max(0, idx - w)
            end = min(len(node_texts), idx + w + 1)
            window_text = " ".join(node_texts[start:end])

            cid = _chunk_id(doc.ticker, doc.form_type, doc.filing_date, text, start_idx + idx)
            chunk = Chunk(
                chunk_id=cid,
                text=text,
                ticker=doc.ticker,
                company_name=doc.company_name,
                cik=doc.cik,
                form_type=doc.form_type,
                filing_date=doc.filing_date,
                fiscal_year=doc.fiscal_year,
                section=block.section,
                chunk_type="text",
                page_number=block.page_number,
                window_text=window_text,
            )
            chunks.append(chunk)

        return chunks


def doc_base_metadata(doc: ParsedDocument) -> dict:
    return {
        "ticker": doc.ticker,
        "company_name": doc.company_name,
        "cik": doc.cik,
        "form_type": doc.form_type,
        "filing_date": doc.filing_date,
        "fiscal_year": doc.fiscal_year,
    }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def chunk_document(doc: ParsedDocument) -> list[Chunk]:
    """Chunk a single ParsedDocument. Thread-safe (stateless chunker)."""
    chunker = FilingChunker()
    return chunker.chunk(doc)
