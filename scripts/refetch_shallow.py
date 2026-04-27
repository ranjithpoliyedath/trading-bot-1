"""
scripts/refetch_shallow.py
---------------------------
Find symbols whose processed features file has fewer than ``--min-rows``
bars and re-fetch them from scratch with a full lookback so the
walk-forward folds in the dashboard become wider than 30 days.

This catches the original "day-1 megacap fetch" cohort whose lookback
was set short before the 6-year config landed:

    AAPL, NVDA, MSFT, GOOGL, META, AMZN, TSLA,
    SPY, QQQ, DIA, IWM, VTI, …

Run:
    python scripts/refetch_shallow.py                 # dry-run report
    python scripts/refetch_shallow.py --apply         # actually re-fetch
    python scripts/refetch_shallow.py --apply --min-rows 1000

After the run, re-engineer features for the affected symbols:
    python -m bot.pipeline --symbols AAPL,NVDA,…
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from bot.config   import DATA_DIR, DATA_LOOKBACK_DAYS   # noqa: E402
from bot.pipeline import run_pipeline                   # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("refetch_shallow")


def find_shallow_symbols(min_rows: int) -> list[tuple[str, int, str]]:
    rows = []
    for path in sorted(Path(DATA_DIR).glob("*_features.parquet")):
        symbol = path.name.split("_")[0]
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            logger.warning("Could not read %s: %s — skipping.", path.name, exc)
            continue
        if len(df) < min_rows:
            start = str(df.index.min().date()) if len(df) else "—"
            rows.append((symbol, len(df), start))
    rows.sort(key=lambda r: r[1])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-rows", type=int, default=1000,
                    help="Symbols with fewer rows than this get re-fetched (default 1000 ≈ 4yr).")
    ap.add_argument("--apply",    action="store_true",
                    help="Actually run the pipeline.  Without this flag, only reports.")
    ap.add_argument("--lookback-days", type=int, default=DATA_LOOKBACK_DAYS,
                    help=f"Override DATA_LOOKBACK_DAYS (default: {DATA_LOOKBACK_DAYS}).")
    args = ap.parse_args()

    shallow = find_shallow_symbols(args.min_rows)
    if not shallow:
        logger.info("No shallow symbols found — every features file has ≥ %d rows.", args.min_rows)
        return 0

    logger.info("Found %d shallow symbols (< %d rows):", len(shallow), args.min_rows)
    for sym, n, start in shallow:
        logger.info("  %-6s rows=%5d  earliest=%s", sym, n, start)

    if not args.apply:
        logger.info("")
        logger.info("Dry-run only.  Re-run with --apply to fetch %d symbols at lookback=%d days.",
                    len(shallow), args.lookback_days)
        return 0

    syms = [s for s, _, _ in shallow]
    logger.info("Re-fetching %d symbols with %d-day lookback (incremental=False)…",
                len(syms), args.lookback_days)
    run_pipeline(symbols=syms, lookback_days=args.lookback_days, incremental=False)
    logger.info("Done.  Verify with: python scripts/refetch_shallow.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
