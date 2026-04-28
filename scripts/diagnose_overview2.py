"""
scripts/diagnose_overview2.py
-------------------------------
Deeper diagnostic for sentiment and news panels.

Tells us:
  - Whether sentiment values are actually saved in the parquet files
  - Whether the news cache file exists and what's in it
  - What Alpaca News returns for a single symbol
  - Whether the news_fetcher module can be imported successfully

Run:
    python scripts/diagnose_overview2.py
"""

import json
import os
import sys
import traceback
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/processed")
NEWS_CACHE = Path("data/cache/news.json")


def section(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main():
    section("1. Feature files - sentiment column check")
    files = sorted(DATA_DIR.glob("*_features.parquet"))
    print(f"Total feature files: {len(files)}")

    # Check 5 random files for sentiment values
    samples = ["AAPL", "TSLA", "NVDA", "AMD", "META"]
    for sym in samples:
        path = DATA_DIR / f"{sym}_features.parquet"
        if not path.exists():
            print(f"  {sym}: file not found")
            continue
        df = pd.read_parquet(path)
        if "combined_sentiment" not in df.columns:
            print(f"  {sym}: no combined_sentiment column")
            continue
        nonzero = (df["combined_sentiment"].abs() > 0).sum()
        latest_val = df["combined_sentiment"].iloc[-1]
        latest_dt = str(df.index[-1])[:10]
        print(f"  {sym}: nonzero days={nonzero}, latest={latest_val:.3f} on {latest_dt}")

    section("2. features_with_sentiment files (output of sentiment pipeline)")
    sent_files = list(DATA_DIR.glob("*_features_with_sentiment.parquet"))
    print(f"Total features_with_sentiment files: {len(sent_files)}")
    if sent_files:
        sample = sent_files[0]
        df = pd.read_parquet(sample)
        sent_cols = [c for c in df.columns if "sentiment" in c.lower() or "news" in c.lower() or c.startswith("st_")]
        print(f"  Sample: {sample.name}")
        print(f"  Sentiment-related columns: {sent_cols}")
        if "combined_sentiment" in df.columns:
            nz = (df["combined_sentiment"].abs() > 0).sum()
            print(f"  combined_sentiment non-zero days: {nz}")

    section("3. News cache file")
    if NEWS_CACHE.exists():
        try:
            data = json.loads(NEWS_CACHE.read_text())
            entries = data.get("data", [])
            print(f"  Cache exists: {len(entries)} entries")
            if entries:
                print(f"  First entry: {entries[0]}")
        except Exception as exc:
            print(f"  Cache exists but unreadable: {exc}")
    else:
        print("  No cache file at data/cache/news.json")

    section("4. Alpaca News fetcher - live test")
    try:
        sys.path.insert(0, ".")
        from bot.sentiment.news_fetcher import NewsFetcher
        print("  Module import: OK")

        fetcher = NewsFetcher()
        print("  Fetcher init: OK")

        articles = fetcher.fetch(symbols=["AAPL"], lookback_days=2, limit=5)
        print(f"  Fetched {len(articles)} articles for AAPL")
        if articles:
            print(f"  First headline: {articles[0].get('headline', '')[:80]}")
            print(f"  Keys in article: {list(articles[0].keys())}")
    except Exception as exc:
        print(f"  ERROR: {exc}")
        traceback.print_exc()

    section("5. Universe load test")
    try:
        from bot.universe import load_universe
        u = load_universe(eligible_only=True)
        print(f"  Eligible universe: {len(u)} symbols")
        if not u.empty:
            print(f"  First 5: {u['symbol'].head(5).tolist()}")
    except Exception as exc:
        print(f"  ERROR: {exc}")


if __name__ == "__main__":
    main()
