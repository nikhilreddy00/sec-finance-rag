"""
Async SEC EDGAR scraper for S&P 500 companies.

Respects SEC fair-use policy:
- Max 10 requests/second (enforced via asyncio.Semaphore + delay)
- Identifies itself via User-Agent header
- Idempotent: skips already-downloaded filings

Output structure:
    data/raw/{ticker}/{form_type}/{filing_date}/
        document.{ext}          ← primary filing document
        metadata.json           ← filing metadata sidecar
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import ssl

import aiohttp
import aiofiles
import certifi
import pandas as pd
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.asyncio import tqdm as async_tqdm

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# S&P 500 ticker list
# ---------------------------------------------------------------------------

SP500_TICKERS_URL = (
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
)

_SP500_CACHE: list[str] = []


def get_sp500_tickers() -> list[str]:
    """Fetch S&P 500 tickers from Wikipedia (cached after first call)."""
    global _SP500_CACHE
    if _SP500_CACHE:
        return _SP500_CACHE
    try:
        tables = pd.read_html(SP500_TICKERS_URL)
        tickers = tables[0]["Symbol"].tolist()
        # Normalize BRK.B → BRK-B (EDGAR format)
        tickers = [t.replace(".", "-") for t in tickers]
        _SP500_CACHE = tickers
        logger.info("Loaded %d S&P 500 tickers", len(tickers))
        return tickers
    except Exception as exc:
        logger.warning("Could not fetch S&P 500 list: %s. Using fallback sample.", exc)
        return ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "JNJ", "V"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FilingMetadata:
    ticker: str
    company_name: str
    cik: str
    form_type: str
    filing_date: str          # YYYY-MM-DD
    fiscal_year: int
    accession_number: str
    primary_document: str
    document_url: str
    local_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core scraper
# ---------------------------------------------------------------------------

class EDGARScraper:
    """
    Async SEC EDGAR scraper.

    Usage::

        scraper = EDGARScraper()
        await scraper.scrape(tickers=["AAPL", "MSFT"], form_types=["10-K"])
    """

    BASE_URL = "https://data.sec.gov"
    SUBMISSIONS_URL = BASE_URL + "/submissions/CIK{cik}.json"

    def __init__(self) -> None:
        self._semaphore = asyncio.Semaphore(settings.sec_max_concurrency)
        self._headers = {"User-Agent": settings.sec_user_agent}
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "EDGARScraper":
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_ctx)
        self._session = aiohttp.ClientSession(headers=self._headers, connector=connector)
        return self

    async def __aexit__(self, *_) -> None:
        if self._session:
            await self._session.close()

    # ── Public API ────────────────────────────────────────────────────────────

    async def scrape(
        self,
        tickers: Optional[list[str]] = None,
        form_types: Optional[list[str]] = None,
        years_back: Optional[int] = None,
    ) -> list[FilingMetadata]:
        """
        Scrape filings for the given tickers and form types.

        Args:
            tickers: list of ticker symbols; defaults to full S&P 500.
            form_types: list of form types; defaults to settings.filing_types.
            years_back: how many years back; defaults to settings.years_back.

        Returns:
            List of FilingMetadata for all downloaded filings.
        """
        tickers = tickers or get_sp500_tickers()
        form_types = form_types or settings.filing_types
        years_back = years_back or settings.years_back

        logger.info(
            "Starting scrape: %d tickers × %s form types × %d years",
            len(tickers), form_types, years_back,
        )

        tasks = [
            self._scrape_ticker(ticker, form_types, years_back)
            for ticker in tickers
        ]
        results: list[list[FilingMetadata]] = await async_tqdm.gather(
            *tasks, desc="Scraping tickers"
        )
        all_filings = [m for sublist in results for m in sublist]
        logger.info("Scraped %d filings total", len(all_filings))
        return all_filings

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _scrape_ticker(
        self,
        ticker: str,
        form_types: list[str],
        years_back: int,
    ) -> list[FilingMetadata]:
        """Fetch submission history for one ticker, download matching filings."""
        try:
            cik, company_name, filings_json = await self._get_submissions(ticker)
        except Exception as exc:
            logger.warning("Skipping %s — submissions fetch failed: %s", ticker, exc)
            return []

        metadata_list: list[FilingMetadata] = []
        recent = filings_json.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        docs = recent.get("primaryDocument", [])

        cutoff_year = int(time.strftime("%Y")) - years_back

        for form, date, acc, doc in zip(forms, dates, accessions, docs):
            if form not in form_types:
                continue
            filing_year = int(date[:4])
            if filing_year < cutoff_year:
                continue

            acc_clean = acc.replace("-", "")
            doc_url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                f"{acc_clean}/{doc}"
            )
            meta = FilingMetadata(
                ticker=ticker,
                company_name=company_name,
                cik=cik,
                form_type=form,
                filing_date=date,
                fiscal_year=filing_year,
                accession_number=acc,
                primary_document=doc,
                document_url=doc_url,
            )

            downloaded = await self._download_filing(meta)
            if downloaded:
                metadata_list.append(meta)

        return metadata_list

    async def _get_submissions(self, ticker: str) -> tuple[str, str, dict]:
        """Return (cik_padded, company_name, submissions_json) for a ticker."""
        # Look up CIK via company search endpoint
        search_url = (
            f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
            f"&dateRange=custom&startdt=2000-01-01&forms=10-K&hits.hits._source=period_of_report"
        )
        # Use the ticker→CIK mapping endpoint instead (more reliable)
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        async with self._semaphore:
            await asyncio.sleep(settings.sec_request_delay)
            async with self._session.get(tickers_url) as resp:
                resp.raise_for_status()
                all_tickers: dict = await resp.json(content_type=None)

        # Find CIK for ticker (dict values: {cik_str, ticker, title})
        cik_str = None
        company_name = ticker
        for entry in all_tickers.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                cik_str = str(entry["cik_str"]).zfill(10)
                company_name = entry.get("title", ticker)
                break

        if cik_str is None:
            raise ValueError(f"CIK not found for ticker {ticker}")

        submissions_url = self.SUBMISSIONS_URL.format(cik=cik_str)
        async with self._semaphore:
            await asyncio.sleep(settings.sec_request_delay)
            async with self._session.get(submissions_url) as resp:
                resp.raise_for_status()
                submissions = await resp.json(content_type=None)

        return cik_str, company_name, submissions

    async def _download_filing(self, meta: FilingMetadata) -> bool:
        """
        Download the primary filing document.

        Returns True if the file was downloaded (or already exists).
        Saves raw file + metadata.json sidecar.
        """
        # Sanitize form type for filesystem (DEF 14A → DEF_14A)
        form_dir_name = meta.form_type.replace(" ", "_")
        local_dir = (
            settings.raw_data_dir
            / meta.ticker
            / form_dir_name
            / meta.filing_date
        )
        local_dir.mkdir(parents=True, exist_ok=True)

        ext = Path(meta.primary_document).suffix or ".htm"
        local_file = local_dir / f"document{ext}"
        meta_file = local_dir / "metadata.json"

        # Skip if already downloaded
        if local_file.exists() and meta_file.exists():
            meta.local_path = str(local_file)
            return True

        # Download with retry
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=30),
            ):
                with attempt:
                    async with self._semaphore:
                        await asyncio.sleep(settings.sec_request_delay)
                        async with self._session.get(
                            meta.document_url, timeout=aiohttp.ClientTimeout(total=60)
                        ) as resp:
                            if resp.status == 404:
                                logger.debug("404 for %s — skipping", meta.document_url)
                                return False
                            resp.raise_for_status()
                            content = await resp.read()
        except Exception as exc:
            logger.warning("Failed to download %s: %s", meta.document_url, exc)
            return False

        # Save document
        async with aiofiles.open(local_file, "wb") as f:
            await f.write(content)

        meta.local_path = str(local_file)

        # Save metadata sidecar
        async with aiofiles.open(meta_file, "w") as f:
            await f.write(json.dumps(meta.to_dict(), indent=2))

        logger.debug("Downloaded %s %s %s", meta.ticker, meta.form_type, meta.filing_date)
        return True


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

async def scrape_sec_filings(
    tickers: Optional[list[str]] = None,
    form_types: Optional[list[str]] = None,
    years_back: Optional[int] = None,
) -> list[FilingMetadata]:
    """Top-level async entry point for scraping."""
    async with EDGARScraper() as scraper:
        return await scraper.scrape(tickers, form_types, years_back)
