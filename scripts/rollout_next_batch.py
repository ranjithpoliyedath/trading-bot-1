"""
scripts/rollout_next_batch.py
-------------------------------
Rolls out the next 100 symbols that don't yet have data on disk.

Used as the cron job that gradually fetches the remaining ~600 symbols
after the initial top-100 fetch is complete. Runs nightly at 11 PM.

When all eligible symbols have data, this script becomes a no-op.
"""

import logging
from pathlib import Path

from bot.config   import DATA_DIR, LOG_FORMAT, LOG_LEVEL
from bot.pipeline import run_pipeline
from bot.universe import load_universe

BATCH_SIZE = 100

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def find_missing_symbols(batch_size: int = BATCH_SIZE) -> list[str]:
    """
    Return the next batch of eligible symbols that don't have features saved.

    Symbols are sorted by 14-day average volume so the most-traded names
    are fetched first.
    """
    df = load_universe(eligible_only=True)
    if df.empty:
        logger.warning("Universe empty — run bot.universe first.")
        return []

    df = df.sort_values("avg_volume_14d", ascending=False, na_position="last")
    all_symbols = df["symbol"].tolist()

    existing = {p.stem.replace("_features", "")
                for p in DATA_DIR.glob("*_features.parquet")}

    missing = [s for s in all_symbols if s not in existing]
    return missing[:batch_size]


def main():
    symbols = find_missing_symbols()
    if not symbols:
        logger.info("No missing symbols — universe data is complete.")
        return
    logger.info("Rolling out %d new symbols.", len(symbols))
    run_pipeline(symbols=symbols)


if __name__ == "__main__":
    main()
