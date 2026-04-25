"""
bot/scrapers/sp_constituents.py
---------------------------------
Scrapes S&P 500, S&P MidCap 400, and S&P SmallCap 600 constituent lists
from Wikipedia. Uses pandas.read_html — no API key required.

Returns a single DataFrame with columns:
    symbol         Ticker symbol
    company        Company name
    sector         GICS sector
    sub_industry   GICS sub-industry (S&P 500 only)
    index          One of: 'sp500', 'sp400', 'sp600'
"""

import logging
import time

import pandas as pd
from io import StringIO
import requests

logger = logging.getLogger(__name__)

WIKI_URLS = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}

# Wikipedia's column names vary between the three pages; we normalise here
COLUMN_ALIASES = {
    "symbol":       ["Symbol", "Ticker symbol", "Ticker"],
    "company":      ["Security", "Company"],
    "sector":       ["GICS Sector", "GICS sector"],
    "sub_industry": ["GICS Sub-Industry", "GICS Sub Industry", "GICS Sub-industry"],
}

USER_AGENT = "trading-bot-1/0.1 (educational project)"


def fetch_all_constituents() -> pd.DataFrame:
    """
    Fetch and merge all three S&P index constituent lists.

    Returns:
        DataFrame with columns [symbol, company, sector, sub_industry, index].
        Symbols deduplicated — if a symbol appears in multiple indexes,
        the largest one wins (sp500 > sp400 > sp600).
    """
    frames = []
    for index_name, url in WIKI_URLS.items():
        try:
            df = _fetch_one_index(url, index_name)
            frames.append(df)
            logger.info("Fetched %d constituents from %s.", len(df), index_name)
            time.sleep(1)   # be polite to Wikipedia
        except Exception as exc:
            logger.error("Failed to fetch %s constituents: %s", index_name, exc, exc_info=True)

    if not frames:
        logger.error("No constituent data fetched from any index.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Deduplicate — keep first occurrence (sp500 > sp400 > sp600 due to dict order)
    combined.drop_duplicates(subset=["symbol"], keep="first", inplace=True)

    # Clean ticker symbols
    combined["symbol"] = combined["symbol"].str.upper().str.strip()
    # Alpaca uses dot notation for class shares (BRK.B not BRK-B), but actually
    # rejects most of these on free tier. Drop dotted/dashed multi-class shares
    # to avoid batch failures - we lose <2% of the universe.
    combined = combined[~combined["symbol"].str.contains(r"[\.\-]", regex=True)]
    combined.reset_index(drop=True, inplace=True)

    logger.info(
        "Total unique constituents: %d (sp500=%d, sp400=%d, sp600=%d)",
        len(combined),
        (combined["index"] == "sp500").sum(),
        (combined["index"] == "sp400").sum(),
        (combined["index"] == "sp600").sum(),
    )
    return combined


def _fetch_one_index(url: str, index_name: str) -> pd.DataFrame:
    """Fetch one Wikipedia constituent table and normalise columns."""
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    if not tables:
        raise ValueError(f"No tables found at {url}")

    # The constituent table is always the first one with a 'Symbol' column
    df = None
    for tbl in tables:
        cols = [str(c) for c in tbl.columns]
        if any(any(alias in col for alias in COLUMN_ALIASES["symbol"]) for col in cols):
            df = tbl
            break

    if df is None:
        raise ValueError(f"No constituent table found at {url}")

    df = _normalise_columns(df)
    df["index"] = index_name
    return df[["symbol", "company", "sector", "sub_industry", "index"]]


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Wikipedia's varying column names to our standard schema."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for col in df.columns:
            if col in aliases or any(alias in col for alias in aliases):
                rename_map[col] = canonical
                break

    df = df.rename(columns=rename_map)

    # Add missing columns as empty strings
    for canonical in ["symbol", "company", "sector", "sub_industry"]:
        if canonical not in df.columns:
            df[canonical] = ""

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    df = fetch_all_constituents()
    print(df.head(20))
    print(f"\nTotal: {len(df)} symbols")
