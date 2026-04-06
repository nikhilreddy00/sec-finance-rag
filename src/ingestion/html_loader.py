"""
Apple 10-K HTML Loader — reads directly from Apple_Dataset/*.html files.

Bypasses SEC EDGAR scraping entirely. Extracts all metadata from the
filename convention used by EDGAR: aapl-YYYYMMDD.html

Filename → metadata mapping:
    aapl-20200926.html  →  fiscal_year=2020, filing_date=2020-09-26
    aapl-20210925.html  →  fiscal_year=2021, filing_date=2021-09-25
    aapl-20220924.html  →  fiscal_year=2022, filing_date=2022-09-24
    aapl-20230930.html  →  fiscal_year=2023, filing_date=2023-09-30
    aapl-20240928.html  →  fiscal_year=2024, filing_date=2024-09-28
    aapl-20250927.html  →  fiscal_year=2025, filing_date=2025-09-27

Output: list[ParsedDocument] ready for chunking and indexing.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.ingestion.parser import FilingParser, ParsedDocument

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Apple Inc. static metadata (Apple's CIK on SEC EDGAR)
# ---------------------------------------------------------------------------

APPLE_STATIC = {
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "cik": "0000320193",
    "form_type": "10-K",
}

# Default dataset directory (relative to project root)
DEFAULT_APPLE_DATASET_DIR = Path("./Apple_Dataset")


# ---------------------------------------------------------------------------
# Filename parser
# ---------------------------------------------------------------------------

def _parse_filename(path: Path) -> dict:
    """
    Extract filing metadata from EDGAR-style filename.

    Accepts patterns:
        aapl-20230930.html
        aapl-20230930.htm
        AAPL-20230930.html  (case-insensitive)

    Returns a metadata dict ready to be passed to FilingParser.parse().
    Raises ValueError if the date cannot be extracted.
    """
    stem = path.stem  # e.g. "aapl-20230930"
    match = re.match(r'(?i)aapl-(\d{4})(\d{2})(\d{2})$', stem)
    if not match:
        raise ValueError(
            f"Cannot extract date from filename '{path.name}'. "
            "Expected format: aapl-YYYYMMDD.html"
        )
    year, month, day = match.groups()
    filing_date = f"{year}-{month}-{day}"

    return {
        **APPLE_STATIC,
        "filing_date": filing_date,
        "fiscal_year": int(year),
        "accession_number": f"aapl-{year}{month}{day}",  # synthetic, not from EDGAR
        "local_path": str(path.resolve()),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_apple_html_filings(
    dataset_dir: Path | str | None = None,
) -> list[ParsedDocument]:
    """
    Parse all Apple 10-K HTML files in dataset_dir.

    Args:
        dataset_dir: path to the folder containing aapl-YYYYMMDD.html files.
                     Defaults to ./Apple_Dataset.

    Returns:
        List of ParsedDocument instances, one per HTML file found.
        Files that fail to parse are skipped with a warning.
    """
    dataset_dir = Path(dataset_dir) if dataset_dir else DEFAULT_APPLE_DATASET_DIR

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Apple_Dataset directory not found at '{dataset_dir}'. "
            "Ensure the Apple_Dataset folder with aapl-YYYYMMDD.html files exists."
        )

    html_files = sorted(
        [p for p in dataset_dir.iterdir() if p.suffix.lower() in (".html", ".htm")],
        key=lambda p: p.stem,  # sort by filename → chronological order
    )

    if not html_files:
        raise FileNotFoundError(
            f"No .html or .htm files found in '{dataset_dir}'."
        )

    logger.info(
        "Found %d Apple 10-K HTML files in '%s'", len(html_files), dataset_dir
    )

    parser = FilingParser()
    docs: list[ParsedDocument] = []

    for html_path in html_files:
        try:
            meta = _parse_filename(html_path)
        except ValueError as e:
            logger.warning("Skipping '%s': %s", html_path.name, e)
            continue

        logger.info(
            "Parsing %s  (fiscal_year=%s, filing_date=%s)",
            html_path.name, meta["fiscal_year"], meta["filing_date"],
        )

        doc = parser.parse(meta, str(html_path))
        if doc is not None:
            docs.append(doc)
            logger.info(
                "  → %d blocks extracted from %s", len(doc.blocks), html_path.name
            )
        else:
            logger.warning("  → parse returned None for %s", html_path.name)

    logger.info(
        "Apple HTML loader: successfully parsed %d / %d files",
        len(docs), len(html_files),
    )
    return docs
