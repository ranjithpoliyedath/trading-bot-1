"""
bot/sentiment/sentiment_features.py
-------------------------------------
Merges sentiment features from the sentiment pipeline into the
main feature DataFrame used by the ML model.

This module is called inside feature_engineer.py add_all_features()
when sentiment data is available for a symbol.
"""

import logging
from pathlib import Path

import pandas as pd

from bot.data_store import DataStore

logger = logging.getLogger(__name__)

SENTIMENT_COLUMNS = [
    "news_sentiment_mean",
    "news_sentiment_std",
    "news_count",
    "reddit_sentiment_mean",
    "reddit_sentiment_std",
    "reddit_score_sum",
    "combined_sentiment",
]


def load_and_merge_sentiment(
    df: pd.DataFrame,
    symbol: str,
    store: DataStore = None,
) -> pd.DataFrame:
    """
    Load saved sentiment features for a symbol and merge into the feature DataFrame.

    If no sentiment data exists yet, returns the original df unchanged
    so the pipeline continues to work before sentiment data is collected.

    Args:
        df: Existing feature DataFrame indexed by date.
        symbol: Ticker symbol e.g. 'AAPL'.
        store: DataStore instance. If None, creates one with default path.

    Returns:
        DataFrame with sentiment columns added (filled with 0 where missing).
    """
    if store is None:
        store = DataStore()

    df_sentiment = store.load(symbol=symbol, tag="sentiment")

    if df_sentiment.empty:
        logger.debug("No sentiment data found for %s — skipping merge.", symbol)
        for col in SENTIMENT_COLUMNS:
            df[col] = 0.0
        return df

    df.index         = pd.to_datetime(df.index).normalize()
    df_sentiment.index = pd.to_datetime(df_sentiment.index).normalize()

    available_cols = [c for c in SENTIMENT_COLUMNS if c in df_sentiment.columns]
    df = df.join(df_sentiment[available_cols], how="left")

    missing_cols = [c for c in SENTIMENT_COLUMNS if c not in df.columns]
    for col in missing_cols:
        df[col] = 0.0

    df[SENTIMENT_COLUMNS] = df[SENTIMENT_COLUMNS].fillna(0.0)

    logger.info(
        "%s: merged %d sentiment features (%d rows with real data).",
        symbol,
        len(available_cols),
        df[available_cols[0]].astype(bool).sum() if available_cols else 0,
    )

    return df


def add_sentiment_momentum(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Add rolling sentiment momentum features.

    These capture trend in sentiment — a rising sentiment score over
    3 days is a stronger signal than a single-day spike.

    Args:
        df: DataFrame with SENTIMENT_COLUMNS already present.
        window: Rolling window in trading days.

    Returns:
        DataFrame with additional momentum columns added.
    """
    if "combined_sentiment" not in df.columns:
        return df

    df["sentiment_momentum"] = df["combined_sentiment"].rolling(window).mean()
    df["sentiment_change"]   = df["combined_sentiment"].diff(1)
    df["sentiment_accel"]    = df["sentiment_change"].diff(1)

    # The first few rows are NaN because rolling/diff need lookback.
    # Treat "no prior data" as zero momentum so downstream models and
    # the no-NaN invariant hold.
    for col in ("sentiment_momentum", "sentiment_change", "sentiment_accel"):
        df[col] = df[col].fillna(0.0)

    logger.debug("Added sentiment momentum features (window=%d).", window)
    return df
