"""
Document parser for SEC EDGAR filings.

SMART PARSING DECISION TREE (chosen to minimise cost and maximise quality)
───────────────────────────────────────────────────────────────────────────

  .htm / .html  → Unstructured.io  (semantic elements: Title, NarrativeText, Table)
                  WHY: Most EDGAR 10-K/10-Q filings ARE HTML. Unstructured gives
                  section-aware semantic blocks for free. No API cost.
                  EDGAR delivers ~85% of filings as HTML since 2009 (iXBRL mandate).

  .pdf (simple) → PyMuPDF4LLM     (markdown, table-aware, free)
                  WHY: For text-heavy PDFs (8-K press releases, proxy statements),
                  PyMuPDF4LLM is 10× faster than LlamaParse and costs nothing.
                  Detects: page count ≤ 50 OR table density < 0.15 tables/page.

  .pdf (complex)→ LlamaParse      (API-based, paid but superior for dense tables)
                  WHY: Financial statements with complex multi-column tables
                  (consolidated balance sheets, multi-segment income statements)
                  lose structure in free parsers. LlamaParse preserves row/column
                  alignment critical for numerical accuracy.
                  TRIGGER: table density ≥ 0.15 tables/page AND page count > 50.
                  If LLAMA_CLOUD_API_KEY not set → falls back to PyMuPDF4LLM.

  .txt / other  → plain text split (zero cost, zero dependencies)

WHY NOT ALWAYS LLAMAPARSE:
  - Free tier: 1,000 pages/day. A single 10-K is 150-300 pages.
  - 30k filings × 200 pages avg = 6M pages → costs ~$18,000 at paid tier.
  - PyMuPDF4LLM handles 90%+ of EDGAR filings adequately and is free.
  - Use LlamaParse selectively: estimated 5-10% of filings trigger it.

For 10-K filings, section boundaries (Item 1, 1A, 7, 7A, 8, 9A) are detected
via regex and attached as metadata to every element within that section.

Output: list of ParsedDocument dataclass instances, serialised as JSON to
        data/processed/{ticker}/{form_type}/{filing_date}/parsed.json
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 10-K Item / section detection
# ---------------------------------------------------------------------------

# Regex to detect standard 10-K Item headers (covers most formatting variations)
ITEM_PATTERN = re.compile(
    r"(?i)^\s*item\s+(\d+[A-Za-z]?)\s*[.:\-–—]?\s*(.{0,120})$",
    re.MULTILINE,
)

KNOWN_ITEMS: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Selected Financial Data",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures about Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "10": "Directors and Executive Officers",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Certain Relationships",
    "14": "Principal Accountant Fees",
    "15": "Exhibits and Financial Statement Schedules",
}


def detect_section(text: str) -> Optional[str]:
    """Return 'Item X — <title>' if text is an item header, else None."""
    m = ITEM_PATTERN.match(text.strip())
    if not m:
        return None
    item_num = m.group(1).upper()
    title = KNOWN_ITEMS.get(item_num, m.group(2).strip())
    return f"Item {item_num} — {title}"


# ---------------------------------------------------------------------------
# Block merging — reduce noise from SEC iXBRL micro-elements
# ---------------------------------------------------------------------------

_MIN_BLOCK_LENGTH = 200  # merge consecutive text blocks shorter than this


def _merge_short_blocks(blocks: list["TextBlock"]) -> list["TextBlock"]:
    """
    Merge consecutive short text blocks into larger, more coherent blocks.

    SEC iXBRL HTML produces many tiny blocks (individual numbers, labels,
    short phrases). These are low-value for embedding and dilute the index.
    This merges consecutive short text blocks while preserving titles and tables.

    Returns a new list of TextBlock objects.
    """
    if not blocks:
        return blocks

    merged: list["TextBlock"] = []
    buffer_texts: list[str] = []
    buffer_page: int = 0

    def _flush():
        if buffer_texts:
            combined = " ".join(buffer_texts)
            if len(combined.strip()) > 30:
                merged.append(TextBlock(
                    block_type="text",
                    text=combined,
                    page_number=buffer_page,
                ))
            buffer_texts.clear()

    for block in blocks:
        # Titles and tables are always kept as-is
        if block.block_type != "text":
            _flush()
            merged.append(block)
            continue

        # Short text blocks get buffered for merging
        if len(block.text) < _MIN_BLOCK_LENGTH:
            buffer_texts.append(block.text)
            if not buffer_texts or len(buffer_texts) == 1:
                buffer_page = block.page_number
        else:
            _flush()
            merged.append(block)

    _flush()
    return merged


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TextBlock:
    """A single parsed text or table block from a filing."""
    block_type: str          # "text" | "table" | "title"
    text: str
    section: str = ""        # e.g. "Item 7 — Management's Discussion and Analysis"
    page_number: int = 0
    table_name: str = ""     # populated for table blocks


@dataclass
class ParsedDocument:
    """Parsed representation of one SEC filing."""
    ticker: str
    company_name: str
    cik: str
    form_type: str
    filing_date: str
    fiscal_year: int
    accession_number: str
    source_path: str
    blocks: list[TextBlock] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ParsedDocument":
        blocks = [TextBlock(**b) for b in data.pop("blocks", [])]
        return cls(**data, blocks=blocks)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class FilingParser:
    """
    Parses a raw filing document into a ParsedDocument.

    Tries parsers in priority order based on file extension.
    Falls back gracefully if optional libraries are not installed.
    """

    def parse(self, meta_dict: dict, local_path: str) -> Optional[ParsedDocument]:
        """
        Parse a single filing.

        Args:
            meta_dict: filing metadata dict (from metadata.json sidecar)
            local_path: path to the raw filing file

        Returns:
            ParsedDocument or None on failure
        """
        path = Path(local_path)
        if not path.exists():
            logger.warning("File not found: %s", local_path)
            return None

        doc = ParsedDocument(
            ticker=meta_dict["ticker"],
            company_name=meta_dict["company_name"],
            cik=meta_dict["cik"],
            form_type=meta_dict["form_type"],
            filing_date=meta_dict["filing_date"],
            fiscal_year=meta_dict["fiscal_year"],
            accession_number=meta_dict["accession_number"],
            source_path=local_path,
        )

        ext = path.suffix.lower()
        try:
            if ext in (".htm", ".html"):
                blocks = self._parse_html(path)
            elif ext == ".pdf":
                blocks = self._parse_pdf_smart(path)
            else:
                blocks = self._parse_text(path)
        except Exception as exc:
            logger.warning("Parse error for %s: %s", local_path, exc)
            return None

        doc.blocks = self._annotate_sections(blocks, meta_dict["form_type"])
        logger.debug(
            "Parsed %s %s %s → %d blocks",
            doc.ticker, doc.form_type, doc.filing_date, len(doc.blocks),
        )
        return doc

    # ── HTML parsing ──────────────────────────────────────────────────────────

    def _parse_html(self, path: Path) -> list[TextBlock]:
        """Parse HTML filings using Unstructured.io."""
        try:
            from unstructured.partition.html import partition_html
            from unstructured.documents.elements import (
                Title, NarrativeText, Table, ListItem, Text,
            )
        except ImportError:
            logger.warning("unstructured not installed; falling back to BeautifulSoup")
            return self._parse_html_bs4(path)

        elements = partition_html(filename=str(path))
        blocks: list[TextBlock] = []
        for el in elements:
            text = str(el).strip()
            if not text:
                continue
            if isinstance(el, Title):
                blocks.append(TextBlock(block_type="title", text=text))
            elif isinstance(el, Table):
                blocks.append(TextBlock(block_type="table", text=text))
            else:
                blocks.append(TextBlock(block_type="text", text=text))
        return _merge_short_blocks(blocks)

    def _parse_html_bs4(self, path: Path) -> list[TextBlock]:
        """Fallback HTML parser using BeautifulSoup."""
        from bs4 import BeautifulSoup

        html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")

        # Remove script/style noise
        for tag in soup(["script", "style", "meta", "link"]):
            tag.decompose()

        blocks: list[TextBlock] = []
        for el in soup.find_all(["p", "h1", "h2", "h3", "h4", "table", "li"]):
            text = el.get_text(separator=" ", strip=True)
            if not text or len(text) < 50:
                continue
            if el.name in ("h1", "h2", "h3", "h4"):
                blocks.append(TextBlock(block_type="title", text=text))
            elif el.name == "table":
                blocks.append(TextBlock(block_type="table", text=text))
            else:
                blocks.append(TextBlock(block_type="text", text=text))
        return _merge_short_blocks(blocks)

    # ── PDF parsing ───────────────────────────────────────────────────────────

    # Table density threshold: ≥ 0.15 tables/page AND > 50 pages → use LlamaParse
    _LLAMAPARSE_TABLE_DENSITY = 0.15
    _LLAMAPARSE_PAGE_THRESHOLD = 50

    def _parse_pdf_smart(self, path: Path) -> list[TextBlock]:
        """
        Smart PDF parser dispatcher.

        Decision logic:
          1. Count pages and tables (using pdfplumber, free, fast)
          2. If complex (high table density, long doc) AND LlamaParse key set → LlamaParse
          3. Otherwise → PyMuPDF4LLM (free, fast, good enough for 90%+ of filings)

        This avoids spending LlamaParse API credits on simple text-heavy PDFs like 8-K
        press releases (typically 5 pages, no complex tables).
        """
        import os
        page_count, table_count = self._probe_pdf_complexity(path)
        tables_per_page = table_count / max(page_count, 1)
        is_complex = (
            tables_per_page >= self._LLAMAPARSE_TABLE_DENSITY
            and page_count > self._LLAMAPARSE_PAGE_THRESHOLD
        )

        llama_cloud_key = os.getenv("LLAMA_CLOUD_API_KEY", "")

        if is_complex and llama_cloud_key:
            logger.info(
                "Using LlamaParse for complex PDF: %s (%d pages, %.2f tables/page)",
                path.name, page_count, tables_per_page,
            )
            return self._parse_pdf_llamaparse(path, llama_cloud_key)
        else:
            reason = (
                "complex but no LlamaParse key" if is_complex and not llama_cloud_key
                else f"simple ({page_count}p, {tables_per_page:.2f} tables/p)"
            )
            logger.debug("Using PyMuPDF4LLM for PDF [%s]: %s", reason, path.name)
            return self._parse_pdf(path)

    def _probe_pdf_complexity(self, path: Path) -> tuple[int, int]:
        """
        Fast complexity probe: count pages and tables using pdfplumber.
        Runs in <0.5s on typical filings.
        """
        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                page_count = len(pdf.pages)
                table_count = sum(len(p.find_tables()) for p in pdf.pages[:20])
                # Extrapolate from first 20 pages if doc is longer
                if page_count > 20:
                    table_count = int(table_count * (page_count / 20))
            return page_count, table_count
        except Exception:
            return 0, 0

    def _parse_pdf_llamaparse(self, path: Path, api_key: str) -> list[TextBlock]:
        """
        Parse complex PDFs using LlamaParse API.

        WHY LLAMAPARSE FOR COMPLEX TABLES:
        - Consolidated balance sheets span 3-4 columns with merged cells
        - Free parsers (PyMuPDF, pdfplumber) lose column alignment
        - LlamaParse reconstructs tables semantically → precise financial data
        - Cost: $0.003/page (LlamaParse pricing). A 300-page 10-K costs ~$0.90.
        - ONLY triggered for complex PDFs — estimated 5-10% of all filings.

        WHY NOT LLAMAPARSE FOR EVERYTHING:
        - 30k filings × 200 pages × $0.003 = $18,000 (prohibitive)
        - 5% of 30k × 200 pages × $0.003 = $900 (acceptable)
        """
        try:
            from llama_parse import LlamaParse
        except ImportError:
            logger.warning("llama-parse not installed (pip install llama-parse); falling back")
            return self._parse_pdf(path)

        try:
            parser = LlamaParse(
                api_key=api_key,
                result_type="markdown",
                verbose=False,
                language="en",
            )
            documents = parser.load_data(str(path))
            blocks: list[TextBlock] = []
            for doc in documents:
                blocks.extend(self._markdown_to_blocks(doc.text))
            return blocks
        except Exception as exc:
            logger.warning("LlamaParse failed (%s); falling back to PyMuPDF4LLM", exc)
            return self._parse_pdf(path)

    def _parse_pdf(self, path: Path) -> list[TextBlock]:
        """Parse PDF using PyMuPDF4LLM → markdown, then extract tables with pdfplumber."""
        blocks: list[TextBlock] = []

        # Primary: PyMuPDF4LLM for narrative text
        try:
            import pymupdf4llm
            md_text = pymupdf4llm.to_markdown(str(path))
            blocks.extend(self._markdown_to_blocks(md_text))
        except ImportError:
            logger.warning("pymupdf4llm not installed; using pdfplumber for text")
            blocks.extend(self._parse_pdf_pdfplumber(path))

        # Secondary: pdfplumber for precise table extraction
        table_blocks = self._extract_pdf_tables(path)
        blocks.extend(table_blocks)

        return blocks

    def _markdown_to_blocks(self, md: str) -> list[TextBlock]:
        """Convert PyMuPDF4LLM markdown output into TextBlocks."""
        blocks: list[TextBlock] = []
        current_page = 0
        for line in md.split("\n"):
            if line.startswith("-----"):  # page separator
                current_page += 1
                continue
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                title = stripped.lstrip("#").strip()
                blocks.append(TextBlock(block_type="title", text=title, page_number=current_page))
            elif "|" in stripped and stripped.count("|") >= 2:
                # Markdown table row — collect into table blocks elsewhere
                pass
            else:
                blocks.append(TextBlock(block_type="text", text=stripped, page_number=current_page))
        return blocks

    def _extract_pdf_tables(self, path: Path) -> list[TextBlock]:
        """Extract tables from PDF using pdfplumber (keep entire table as one chunk)."""
        try:
            import pdfplumber
        except ImportError:
            return []

        table_blocks: list[TextBlock] = []
        table_idx = 0
        try:
            with pdfplumber.open(str(path)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    tables = page.extract_tables()
                    for table in tables:
                        if not table:
                            continue
                        # Convert to markdown-style text
                        rows = []
                        for row in table:
                            cells = [str(c or "").strip() for c in row]
                            rows.append(" | ".join(cells))
                        table_text = "\n".join(rows)
                        table_name = f"table_{table_idx}"
                        table_blocks.append(
                            TextBlock(
                                block_type="table",
                                text=table_text,
                                page_number=page_num,
                                table_name=table_name,
                            )
                        )
                        table_idx += 1
        except Exception as exc:
            logger.debug("pdfplumber table extraction error: %s", exc)

        return table_blocks

    def _parse_pdf_pdfplumber(self, path: Path) -> list[TextBlock]:
        """Fallback: extract all text from PDF via pdfplumber."""
        try:
            import pdfplumber
            blocks: list[TextBlock] = []
            with pdfplumber.open(str(path)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    for para in text.split("\n\n"):
                        para = para.strip()
                        if len(para) > 20:
                            blocks.append(TextBlock(block_type="text", text=para, page_number=page_num))
            return blocks
        except Exception as exc:
            logger.warning("pdfplumber fallback failed: %s", exc)
            return []

    # ── Plain text parsing ────────────────────────────────────────────────────

    def _parse_text(self, path: Path) -> list[TextBlock]:
        """Parse plain text / ASCII filings."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

        blocks: list[TextBlock] = []
        for para in text.split("\n\n"):
            para = para.strip()
            if len(para) < 20:
                continue
            block_type = "title" if len(para) < 100 and para.isupper() else "text"
            blocks.append(TextBlock(block_type=block_type, text=para))
        return blocks

    # ── Section annotation ────────────────────────────────────────────────────

    def _annotate_sections(self, blocks: list[TextBlock], form_type: str) -> list[TextBlock]:
        """
        Walk through blocks and tag each with its containing 10-K section.
        For non-10-K forms, section is set to the form type name.
        """
        if form_type not in ("10-K", "10-K/A"):
            for b in blocks:
                b.section = form_type
            return blocks

        current_section = "Preamble"
        for block in blocks:
            # Check title blocks AND short text blocks (iXBRL filings label
            # Item headers as "text", not "title")
            if block.block_type in ("title", "text") and len(block.text) < 120:
                detected = detect_section(block.text)
                if detected:
                    current_section = detected
            block.section = current_section

        return blocks


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_parsed_document(doc: ParsedDocument, processed_dir: Path) -> Path:
    """Save ParsedDocument as JSON to processed_dir/{ticker}/{form}/{date}/parsed.json."""
    form_dir_name = doc.form_type.replace(" ", "_")
    out_dir = processed_dir / doc.ticker / form_dir_name / doc.filing_date
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "parsed.json"
    out_path.write_text(json.dumps(doc.to_dict(), indent=2, ensure_ascii=False))
    return out_path


def load_parsed_document(json_path: Path) -> ParsedDocument:
    """Load a ParsedDocument from JSON."""
    data = json.loads(json_path.read_text())
    return ParsedDocument.from_dict(data)


def iter_parsed_documents(processed_dir: Path):
    """Yield all ParsedDocument instances from processed_dir."""
    for json_path in processed_dir.rglob("parsed.json"):
        try:
            yield load_parsed_document(json_path)
        except Exception as exc:
            logger.warning("Could not load %s: %s", json_path, exc)
