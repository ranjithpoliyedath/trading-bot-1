"""
bot/market_overview.py
-----------------------
Aggregates data for the dashboard Market Overview page.

Six panels:
  1. Market mood       — fear & greed index (CNN)
  2. Index snapshot    — SPY, QQQ, DIA, IWM, VTI daily change
  3. Sector leaders    — top 3 sector ETFs by daily % change
  4. Volume movers     — symbols with volume_ratio > 2.0
  5. Sentiment heatmap — universe symbols colored by combined_sentiment
  6. News headlines    — last N headlines from Alpaca News

All panels read from existing Parquet files except the fear & greed scrape.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from bot.config              import DATA_DIR, NEWS_LOOKBACK_DAYS
from bot.scrapers.fear_greed import get_fear_greed
from bot.universe            import load_universe

logger = logging.getLogger(__name__)

INDEX_ETFS = ["SPY", "QQQ", "DIA", "IWM", "VTI"]

SECTOR_ETFS = {
    "XLK":  "Technology",
    "XLV":  "Health Care",
    "XLF":  "Financials",
    "XLE":  "Energy",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLI":  "Industrials",
    "XLU":  "Utilities",
    "XLB":  "Materials",
    "XLRE": "Real Estate",
    "XLC":  "Communication",
}


def get_market_overview() -> dict:
    """
    Return a dict with data for all six panels of the overview page.

    Returns:
        {
            "fear_greed":  {...},
            "indexes":     [...],
            "sectors":     [...],
            "volume_movers": [...],
            "sentiment_heatmap": [...],
            "news":        [...]
        }
    """
    return {
        "fear_greed":        _get_fear_greed_panel(),
        "indexes":           _get_index_snapshot(),
        "sectors":           _get_sector_leaders(),
        "volume_movers":     _get_volume_movers(),
        "sentiment_heatmap": _get_sentiment_heatmap(),
        "news":              _get_recent_news(),
    }


def _get_fear_greed_panel() -> dict:
    return get_fear_greed()


def _get_index_snapshot() -> list[dict]:
    """Return today's % change for each major index ETF."""
    rows = []
    for symbol in INDEX_ETFS:
        info = _latest_change(symbol)
        if info:
            rows.append({"symbol": symbol, **info})
    return rows


def _get_sector_leaders(top_n: int = 3) -> list[dict]:
    """Return top N sector ETFs by today's % change."""
    rows = []
    for symbol, sector_name in SECTOR_ETFS.items():
        info = _latest_change(symbol)
        if info:
            rows.append({"symbol": symbol, "sector": sector_name, **info})
    rows.sort(key=lambda r: r["change_pct"], reverse=True)
    return rows[:top_n]


def _get_volume_movers(threshold: float = 2.0, limit: int = 10) -> list[dict]:
    """Return symbols with abnormally high volume today."""
    universe = load_universe(eligible_only=True)
    if universe.empty:
        return []

    movers = []
    for symbol in universe["symbol"].head(200):
        df = _load_features(symbol)
        if df.empty or "volume_ratio" not in df.columns:
            continue
        last = df.iloc[-1]
        vr   = last.get("volume_ratio", 0)
        if vr and vr >= threshold:
            movers.append({
                "symbol":       symbol,
                "volume_ratio": round(float(vr), 2),
                "close":        round(float(last.get("close", 0)), 2),
                "change_pct":   round(float(last.get("price_change_1d", 0)) * 100, 2),
            })
    movers.sort(key=lambda r: r["volume_ratio"], reverse=True)
    return movers[:limit]


def _get_sentiment_heatmap(limit: int = 25) -> list[dict]:
    """Return universe symbols with their latest sentiment score for the heatmap."""
    universe = load_universe(eligible_only=True)
    if universe.empty:
        return []

    rows = []
    for symbol in universe["symbol"].head(limit):
        df = _load_features(symbol)
        if df.empty:
            continue
        last       = df.iloc[-1]
        sentiment  = last.get("combined_sentiment", 0)
        sentiment  = 0.0 if pd.isna(sentiment) else float(sentiment)
        rows.append({
            "symbol":     symbol,
            "sentiment":  round(sentiment, 3),
            "close":      round(float(last.get("close", 0)), 2),
            "change_pct": round(float(last.get("price_change_1d", 0)) * 100, 2),
        })
    return rows


def _get_recent_news(limit: int = 10) -> list[dict]:
    """Return the most recent news headlines fetched by the sentiment pipeline."""
    try:
        from bot.sentiment.news_fetcher import NewsFetcher
        fetcher = NewsFetcher()
        universe = load_universe(eligible_only=True)
        symbols  = universe["symbol"].head(20).tolist() if not universe.empty else ["SPY"]
        articles = fetcher.fetch(symbols=symbols, lookback_days=2, limit=5)
        articles.sort(key=lambda a: a.get("published_at") or datetime.min, reverse=True)
        return [
            {
                "symbol":       a["symbol"],
                "headline":     a["headline"][:120],
                "source":       a.get("source", ""),
                "published_at": str(a.get("published_at", "")),
                "url":          a.get("url", ""),
            }
            for a in articles[:limit]
        ]
    except Exception as exc:
        logger.warning("News panel fetch failed: %s", exc)
        return []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_features(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_features.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _latest_change(symbol: str) -> dict | None:
    """Return latest close + 1-day % change for a symbol."""
    df = _load_features(symbol)
    if df.empty:
        df = _load_raw(symbol)
    if df.empty or "close" not in df.columns:
        return None
    if len(df) < 2:
        return None
    last_close = float(df["close"].iloc[-1])
    prev_close = float(df["close"].iloc[-2])
    change_pct = (last_close - prev_close) / prev_close * 100 if prev_close else 0
    return {
        "close":      round(last_close, 2),
        "change_pct": round(change_pct, 2),
        "date":       str(df.index[-1])[:10],
    }


def _load_raw(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_raw.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()
