"""
bot/sentiment/aggregator.py
----------------------------
Aggregates raw scored articles/posts into daily sentiment features per symbol.

Sources:
  - Alpaca News      scored by FinBERT/VADER (sentiment_score field)
  - StockTwits posts pre-labelled Bullish/Bearish (raw_score field)

Features produced:
    news_sentiment_mean   avg FinBERT score from Alpaca news (-1 to +1)
    news_sentiment_std    volatility of news sentiment
    news_count            number of articles that day
    st_sentiment_mean     avg StockTwits sentiment score (-1 to +1)
    st_sentiment_std      volatility of StockTwits sentiment
    st_bullish_ratio      fraction of labelled posts that are bullish
    st_likes_sum          total likes (retail attention proxy)
    combined_sentiment    weighted avg: 60% news + 40% StockTwits
"""

import logging
import numpy as np
import pandas as pd

try:
    from bot.config import SENTIMENT_CUTOFF_HOUR_UTC
except Exception:
    SENTIMENT_CUTOFF_HOUR_UTC = None

logger = logging.getLogger(__name__)

NEWS_WEIGHT = 0.60
ST_WEIGHT   = 0.40


def _bucket_to_trading_day(ts_series: pd.Series) -> pd.Series:
    """
    Bucket each timestamp to a trading-day date string.  Articles
    published after ``SENTIMENT_CUTOFF_HOUR_UTC`` roll forward to the
    next calendar day so they don't leak into the prior day's close.
    Returns a series of dates (00:00:00) ready to groupby.
    """
    ts = pd.to_datetime(ts_series, utc=True, errors="coerce")
    if SENTIMENT_CUTOFF_HOUR_UTC is None:
        return ts.dt.tz_convert(None).dt.normalize()

    # Add 1 day where hour >= cutoff
    rolled = ts.where(
        ts.dt.hour < SENTIMENT_CUTOFF_HOUR_UTC,
        ts + pd.Timedelta(days=1),
    )
    return rolled.dt.tz_convert(None).dt.normalize()


def aggregate_sentiment(
    news_scored,
    stocktwits_scored,
    symbols,
):
    result = {}
    for symbol in symbols:
        news_df = _news_to_daily(news_scored, symbol)
        st_df   = _stocktwits_to_daily(stocktwits_scored, symbol)
        merged  = _merge_sources(news_df, st_df)
        result[symbol] = merged
        logger.info("Aggregated sentiment for %s: %d days.", symbol, len(merged))
    return result


def _news_to_daily(records, symbol):
    rows = [r for r in records if r.get("symbol") == symbol and "sentiment_score" in r]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = _bucket_to_trading_day(df["published_at"])
    daily = df.groupby("date")["sentiment_score"].agg(["mean", "std", "count"])
    daily.columns = ["news_sentiment_mean", "news_sentiment_std", "news_count"]
    return daily.fillna(0)


def _stocktwits_to_daily(records, symbol):
    rows = [r for r in records if r.get("symbol") == symbol]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = _bucket_to_trading_day(df["published_at"])

    if "sentiment_score" in df.columns:
        df["score"] = df["raw_score"].fillna(df["sentiment_score"]).fillna(0.0)
    else:
        df["score"] = df["raw_score"].fillna(0.0)

    df["is_bullish"] = df["sentiment"].apply(
        lambda s: 1 if s == "Bullish" else (0 if s == "Bearish" else np.nan)
    )

    daily = df.groupby("date").agg(
        st_sentiment_mean=("score",      "mean"),
        st_sentiment_std= ("score",      "std"),
        st_bullish_ratio= ("is_bullish", "mean"),
        st_likes_sum=     ("likes",      "sum"),
    ).fillna(0)
    return daily


def _merge_sources(news_df, st_df):
    news_empty = news_df.empty if isinstance(news_df, pd.DataFrame) else True
    st_empty   = st_df.empty  if isinstance(st_df,   pd.DataFrame) else True

    if news_empty and st_empty:
        return pd.DataFrame()

    if news_empty:
        merged = st_df.copy()
        merged["combined_sentiment"] = merged["st_sentiment_mean"]
        return merged

    if st_empty:
        merged = news_df.copy()
        merged["combined_sentiment"] = merged["news_sentiment_mean"]
        return merged

    merged = news_df.join(st_df, how="outer").fillna(0)
    merged["combined_sentiment"] = (
        merged["news_sentiment_mean"] * NEWS_WEIGHT +
        merged["st_sentiment_mean"]   * ST_WEIGHT
    )
    return merged


SENTIMENT_FEATURE_COLUMNS = [
    "news_sentiment_mean",
    "news_sentiment_std",
    "news_count",
    "st_sentiment_mean",
    "st_sentiment_std",
    "st_bullish_ratio",
    "st_likes_sum",
    "combined_sentiment",
]
