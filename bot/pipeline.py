"""
bot/pipeline.py
----------------
Data pipeline: fetch OHLCV bars from Alpaca, engineer features, save to disk.

Features:
  - Reads symbol list from universe.parquet (top N by volume)
  - Incremental fetch — only pulls new bars since last stored timestamp
  - Resumable — restart skips symbols already up to date
  - Progress logging — clear "X of Y complete" output
  - Configurable batch size and delay for Alpaca rate limits

Run modes:
    python -m bot.pipeline                  # Fetch top INITIAL_UNIVERSE_SIZE
    python -m bot.pipeline --all            # Fetch all eligible symbols
    python -m bot.pipeline --symbols AAPL,MSFT
    python -m bot.pipeline --batch 100 --offset 100   # Symbols 100-200
"""

import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from bot.config       import (
    DATA_LOOKBACK_DAYS, INITIAL_UNIVERSE_SIZE, ALPACA_BATCH_DELAY_SECS,
    LOG_FORMAT, LOG_LEVEL,
)
from bot.data_fetcher    import DataFetcher
from bot.data_store      import DataStore
from bot.feature_engineer import add_all_features
from bot.universe        import load_universe, get_top_n_by_volume

load_dotenv()

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def run_pipeline(
    symbols:         Optional[list[str]] = None,
    lookback_days:   int  = DATA_LOOKBACK_DAYS,
    incremental:     bool = True,
    skip_features:   bool = False,
) -> dict[str, int]:
    """
    Run the full data pipeline.

    Args:
        symbols:       List of symbols to process. If None, uses top universe symbols.
        lookback_days: Max history depth in days.
        incremental:   If True, only fetch bars newer than what's saved.
        skip_features: If True, skip feature engineering (raw bars only).

    Returns:
        Dict with counts: {processed, skipped, failed, new_rows}
    """
    if symbols is None:
        symbols = get_top_n_by_volume(n=INITIAL_UNIVERSE_SIZE, eligible_only=True)
        if not symbols:
            logger.warning("Universe empty — falling back to default symbols.")
            symbols = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"]

    logger.info("=" * 60)
    logger.info("Pipeline started — %d symbols, %d-day lookback, incremental=%s",
                len(symbols), lookback_days, incremental)
    logger.info("=" * 60)

    fetcher = DataFetcher()
    store   = DataStore()

    counts = {"processed": 0, "skipped": 0, "failed": 0, "new_rows": 0}
    total  = len(symbols)

    for i, symbol in enumerate(symbols, 1):
        try:
            logger.info("[%d/%d] %s — checking...", i, total, symbol)

            existing  = store.load(symbol=symbol, tag="raw")
            start_dt  = _resolve_start_date(existing, lookback_days, incremental)

            if start_dt is None:
                logger.info("[%d/%d] %s — already up to date.", i, total, symbol)
                counts["skipped"] += 1
                continue

            df_new = fetcher.fetch_single(
                symbol=symbol,
                lookback_days=(datetime.utcnow() - start_dt).days,
            )

            if df_new.empty:
                logger.warning("[%d/%d] %s — no new data returned.", i, total, symbol)
                counts["failed"] += 1
                continue

            # Merge with existing if incremental
            if incremental and not existing.empty:
                df_combined = _merge_bars(existing, df_new)
            else:
                df_combined = df_new

            store.save(df_combined, symbol=symbol, tag="raw")
            new_count = len(df_combined) - len(existing)
            counts["new_rows"] += max(0, new_count)

            if not skip_features:
                df_features = add_all_features(df_combined)
                if not df_features.empty:
                    store.save(df_features, symbol=symbol, tag="features")

            counts["processed"] += 1
            logger.info(
                "[%d/%d] %s — done. Total bars: %d (+%d new)",
                i, total, symbol, len(df_combined), max(0, new_count),
            )

            time.sleep(ALPACA_BATCH_DELAY_SECS)

        except Exception as exc:
            logger.error("[%d/%d] %s — failed: %s", i, total, symbol, exc, exc_info=False)
            counts["failed"] += 1

    logger.info("=" * 60)
    logger.info(
        "Pipeline complete — processed=%d skipped=%d failed=%d new_rows=%d",
        counts["processed"], counts["skipped"], counts["failed"], counts["new_rows"],
    )
    logger.info("=" * 60)
    return counts


def _resolve_start_date(
    existing: pd.DataFrame,
    lookback_days: int,
    incremental: bool,
) -> Optional[datetime]:
    """
    Decide where to start fetching from. None means 'no fetch needed'.
    """
    target_start = datetime.utcnow() - timedelta(days=lookback_days)

    if existing.empty or not incremental:
        return target_start

    # Incremental: start from day after last stored bar
    last_ts = pd.to_datetime(existing.index.max())
    if last_ts.tz is not None:
        last_ts = last_ts.tz_convert(None)
    next_start = last_ts + timedelta(days=1)

    # If we're already current (within 1 day of today), skip
    if next_start.date() >= datetime.utcnow().date():
        return None

    return next_start


def _merge_bars(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Merge new bars into existing, deduplicating by index."""
    existing.index = pd.to_datetime(existing.index)
    new.index      = pd.to_datetime(new.index)
    combined = pd.concat([existing, new])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)
    return combined


def main():
    parser = argparse.ArgumentParser(description="Run the trading bot data pipeline.")
    parser.add_argument("--all",      action="store_true", help="Fetch all eligible symbols (~700)")
    parser.add_argument("--symbols",  type=str,            help="Comma-separated symbols")
    parser.add_argument("--batch",    type=int,            help="Batch size — fetch this many")
    parser.add_argument("--offset",   type=int, default=0, help="Skip first N symbols (for batched runs)")
    parser.add_argument("--full",     action="store_true", help="Disable incremental — refetch everything")
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    elif args.all:
        df = load_universe(eligible_only=True)
        df = df.sort_values("avg_volume_14d", ascending=False, na_position="last")
        symbols = df["symbol"].tolist()
        if args.batch:
            symbols = symbols[args.offset:args.offset + args.batch]
    elif args.batch:
        df = load_universe(eligible_only=True)
        df = df.sort_values("avg_volume_14d", ascending=False, na_position="last")
        symbols = df["symbol"].iloc[args.offset:args.offset + args.batch].tolist()
    else:
        symbols = None  # use INITIAL_UNIVERSE_SIZE default

    run_pipeline(symbols=symbols, incremental=not args.full)


if __name__ == "__main__":
    main()
