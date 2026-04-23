"""
bot/pipeline.py
---------------
Orchestrates the full data pipeline:
  1. Fetch raw OHLCV bars from Alpaca
  2. Engineer features
  3. Save processed data to disk

Run this directly to refresh your local training data:
    python -m bot.pipeline
"""

import logging
import os
from datetime import datetime

from dotenv import load_dotenv

from bot.data_fetcher import DataFetcher
from bot.data_store import DataStore
from bot.feature_engineer import add_all_features

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

# Symbols to fetch — update this list as your strategy grows
SYMBOLS = ["AAPL", "TSLA", "MSFT", "NVDA", "SPY"]

# Days of history to pull (365 = 1 year of daily bars for training)
LOOKBACK_DAYS = 365


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(symbols: list[str] = SYMBOLS, lookback_days: int = LOOKBACK_DAYS) -> None:
    """
    Run the full data pipeline for all symbols.

    Args:
        symbols: List of ticker symbols to process.
        lookback_days: Number of calendar days of history to fetch.
    """
    logger.info("=" * 60)
    logger.info("Starting data pipeline — %s", datetime.utcnow().isoformat())
    logger.info("Symbols: %s | Lookback: %d days", symbols, lookback_days)
    logger.info("=" * 60)

    fetcher = DataFetcher()
    store = DataStore()

    raw_data = fetcher.fetch_bars(symbols=symbols, lookback_days=lookback_days)

    for symbol, df_raw in raw_data.items():
        if df_raw.empty:
            logger.warning("Skipping %s — no data returned.", symbol)
            continue

        # Save raw bars
        store.save(df_raw, symbol=symbol, tag="raw")

        # Engineer features
        df_features = add_all_features(df_raw)

        if df_features.empty:
            logger.warning("Skipping %s — no rows after feature engineering.", symbol)
            continue

        # Save processed data
        store.save(df_features, symbol=symbol, tag="features")

        logger.info(
            "%s: %d raw bars → %d feature rows | columns: %s",
            symbol,
            len(df_raw),
            len(df_features),
            list(df_features.columns),
        )

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    run_pipeline()
