"""
bot/universe.py
----------------
Builds and refreshes the trading universe.

Pipeline:
  1. Fetch S&P 500/400/600 constituents from Wikipedia
  2. Pull current price + 14-day avg volume from Alpaca for each symbol
  3. Apply eligibility filters:
       - 14-day avg volume > 1,000,000 shares
       - Current price >= $5
       - No OTC (Alpaca only returns NYSE/NASDAQ/AMEX so this is implicit)
  4. Save filtered universe to data/universe.parquet

Run:
    python -m bot.universe
"""

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from bot.scrapers.sp_constituents import fetch_all_constituents

load_dotenv()

logger = logging.getLogger(__name__)

# Eligibility filters
MIN_AVG_VOLUME    = 100_000     # 14-day avg vol (IEX feed only - ~5% of total market)
MIN_PRICE         = 5.00        # USD — no penny stocks
VOLUME_LOOKBACK   = 14          # trading days

# Output
ROOT          = Path(__file__).parent.parent
UNIVERSE_FILE = ROOT / "data" / "universe.parquet"

# Alpaca batch fetch — 200 req/min limit, batch of 100 symbols per request
BATCH_SIZE       = 100
BATCH_DELAY_SECS = 0.5


def build_universe(save: bool = True) -> pd.DataFrame:
    """
    Build the eligible universe from S&P 500/400/600 lists + Alpaca filtering.

    Args:
        save: If True, save result to data/universe.parquet.

    Returns:
        DataFrame with columns:
            symbol, company, sector, sub_industry, index,
            avg_volume_14d, current_price, eligible, reason, last_refreshed
    """
    logger.info("=" * 60)
    logger.info("Building universe — %s", datetime.utcnow().isoformat())
    logger.info("=" * 60)

    # 1. Fetch constituents
    constituents = fetch_all_constituents()
    if constituents.empty:
        logger.error("No constituents fetched — aborting universe build.")
        return pd.DataFrame()

    # 2. Pull market data from Alpaca
    market_data = _fetch_market_data(constituents["symbol"].tolist())

    # 3. Merge and apply filters
    df = constituents.merge(market_data, on="symbol", how="left")
    df = _apply_filters(df)

    # 4. Save
    df["last_refreshed"] = datetime.utcnow()
    if save:
        UNIVERSE_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(UNIVERSE_FILE, compression="snappy")
        logger.info("Saved universe to %s.", UNIVERSE_FILE)

    eligible_count = int(df["eligible"].sum())
    logger.info(
        "Universe complete — %d eligible / %d candidates.",
        eligible_count, len(df),
    )
    return df


def load_universe(eligible_only: bool = True) -> pd.DataFrame:
    """
    Load the saved universe from disk.

    Args:
        eligible_only: If True, return only symbols passing all filters.

    Returns:
        DataFrame with universe data, or empty DataFrame if file missing.
    """
    if not UNIVERSE_FILE.exists():
        logger.warning("Universe file not found: %s", UNIVERSE_FILE)
        return pd.DataFrame()

    df = pd.read_parquet(UNIVERSE_FILE)
    if eligible_only and "eligible" in df.columns:
        df = df[df["eligible"]].copy()
    return df


def get_top_n_by_volume(n: int = 100, eligible_only: bool = True) -> list:
    """
    Return the top N symbols by 14-day average volume.

    Args:
        n: Number of symbols to return.
        eligible_only: If True, filter to eligible symbols only.

    Returns:
        List of ticker symbols, ordered by avg volume descending.
    """
    df = load_universe(eligible_only=eligible_only)
    if df.empty or "avg_volume_14d" not in df.columns:
        return []
    df = df.sort_values("avg_volume_14d", ascending=False, na_position="last")
    return df["symbol"].head(n).tolist()


def _fetch_market_data(symbols: list[str]) -> pd.DataFrame:
    """
    Pull recent bars from Alpaca to compute avg volume and current price.

    Returns:
        DataFrame with columns: symbol, avg_volume_14d, current_price.
    """
    client = StockHistoricalDataClient(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_SECRET_KEY"),
    )

    end   = datetime.utcnow()
    start = end - timedelta(days=VOLUME_LOOKBACK + 10)  # buffer for weekends

    rows = []
    total_batches = (len(symbols) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        logger.info("Fetching market data batch %d/%d (%d symbols)...", batch_num, total_batches, len(batch))

        try:
            request = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=start, end=end,
                feed="iex",
            )
            bars = client.get_stock_bars(request)
            df_all = bars.df if hasattr(bars, "df") else pd.DataFrame()
            rows.extend(_summarise_batch(df_all, batch))
        except Exception as exc:
            logger.warning("Batch %d failed (%s) — retrying symbol by symbol...", batch_num, exc)
            rows.extend(_fetch_individually(client, batch, start, end))

        time.sleep(BATCH_DELAY_SECS)

    return pd.DataFrame(rows)




def _fetch_individually(client, symbols: list, start, end) -> list[dict]:
    """Fall back to one-by-one fetching when a batch fails due to invalid symbols."""
    rows = []
    for sym in symbols:
        try:
            request = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=start, end=end,
                feed="iex",
            )
            bars = client.get_stock_bars(request)
            df_all = bars.df if hasattr(bars, "df") else pd.DataFrame()
            rows.extend(_summarise_batch(df_all, [sym]))
        except Exception:
            rows.append({"symbol": sym, "avg_volume_14d": None, "current_price": None})
        time.sleep(0.05)
    return rows

def _summarise_batch(df_all: pd.DataFrame, expected_symbols: list[str]) -> list[dict]:
    """Compute per-symbol summary stats from a multi-symbol bars DataFrame."""
    rows = []
    if df_all.empty:
        return [{"symbol": s, "avg_volume_14d": None, "current_price": None}
                for s in expected_symbols]

    # Alpaca returns multi-index: (symbol, timestamp)
    for sym in expected_symbols:
        try:
            sym_df = df_all.loc[sym] if sym in df_all.index.get_level_values(0) else None
            if sym_df is None or sym_df.empty:
                rows.append({"symbol": sym, "avg_volume_14d": None, "current_price": None})
                continue

            sym_df = sym_df.tail(VOLUME_LOOKBACK)
            avg_vol  = float(sym_df["volume"].mean()) if "volume" in sym_df.columns else None
            last_px  = float(sym_df["close"].iloc[-1]) if "close" in sym_df.columns else None
            rows.append({
                "symbol":         sym,
                "avg_volume_14d": avg_vol,
                "current_price":  last_px,
            })
        except Exception as exc:
            logger.debug("Summary failed for %s: %s", sym, exc)
            rows.append({"symbol": sym, "avg_volume_14d": None, "current_price": None})

    return rows


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Mark rows as eligible/ineligible with reason."""
    df = df.copy()
    df["eligible"] = True
    df["reason"]   = ""

    # Coerce to numeric upfront so .fillna on object dtype doesn't
    # trigger the pandas Downcasting FutureWarning.
    vol   = pd.to_numeric(df["avg_volume_14d"], errors="coerce")
    price = pd.to_numeric(df["current_price"],  errors="coerce")

    # Filter 1: missing market data
    mask_no_data = vol.isna() | price.isna()
    df.loc[mask_no_data, "eligible"] = False
    df.loc[mask_no_data, "reason"]   = "no market data"

    # Filter 2: volume below threshold
    mask_low_vol = (vol.fillna(0) < MIN_AVG_VOLUME) & ~mask_no_data
    df.loc[mask_low_vol, "eligible"] = False
    df.loc[mask_low_vol, "reason"]   = f"volume below {MIN_AVG_VOLUME:,}"

    # Filter 3: penny stock
    mask_penny = (price.fillna(0) < MIN_PRICE) & ~mask_no_data & ~mask_low_vol
    df.loc[mask_penny, "eligible"] = False
    df.loc[mask_penny, "reason"]   = f"price below ${MIN_PRICE}"

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    df = build_universe()
    if not df.empty:
        eligible = df[df["eligible"]]
        print(f"\nEligible: {len(eligible)} of {len(df)}")
        print("\nTop 10 by volume:")
        print(eligible.nlargest(10, "avg_volume_14d")[["symbol", "company", "sector", "avg_volume_14d", "current_price"]])
