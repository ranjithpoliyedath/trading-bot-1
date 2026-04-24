"""
bot/sentiment/sentiment_pipeline.py
-------------------------------------
Orchestrates the full sentiment pipeline:
  1. Fetch news from Alpaca News API
  2. Fetch posts from StockTwits (no API key needed)
  3. Score all text with FinBERT (or VADER fallback)
     StockTwits pre-labelled posts skip NLP — use raw_score directly
  4. Aggregate into daily sentiment features per symbol
  5. Merge with existing OHLCV + technical features
  6. Save enriched features to data/processed/

Run directly to refresh sentiment features:
    python -m bot.sentiment.sentiment_pipeline
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from bot.sentiment.news_fetcher        import NewsFetcher
from bot.sentiment.stocktwits_fetcher  import StockTwitsFetcher
from bot.sentiment.scorer              import SentimentScorer
from bot.sentiment.aggregator          import aggregate_sentiment, SENTIMENT_FEATURE_COLUMNS
from bot.data_store                    import DataStore

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

from bot.config import SYMBOLS, NEWS_LOOKBACK_DAYS as LOOKBACK_DAYS


def run_sentiment_pipeline(
    symbols:       list[str] = SYMBOLS,
    lookback_days: int       = LOOKBACK_DAYS,
) -> dict[str, pd.DataFrame]:
    """
    Run the full sentiment pipeline and merge results with existing features.

    Args:
        symbols: Ticker symbols to process.
        lookback_days: Days of news history to fetch from Alpaca.

    Returns:
        Dict mapping symbol -> enriched DataFrame with sentiment + technical features.
    """
    logger.info("=" * 60)
    logger.info("Sentiment pipeline started — %s", datetime.utcnow().isoformat())
    logger.info("Symbols: %s | Lookback: %d days", symbols, lookback_days)
    logger.info("=" * 60)

    scorer = SentimentScorer(backend="auto")
    store  = DataStore()

    # ── 1. Fetch and score Alpaca news ────────────────────────────────────────
    news_scored = []
    try:
        news_fetcher = NewsFetcher()
        raw_news     = news_fetcher.fetch(symbols=symbols, lookback_days=lookback_days)
        logger.info("Scoring %d news articles with FinBERT/VADER...", len(raw_news))
        for article in raw_news:
            text  = article.get("headline", "") + ". " + article.get("summary", "")
            score = scorer.score(text)
            news_scored.append({**article, "sentiment_score": score})
        logger.info("News scoring complete — %d articles scored.", len(news_scored))
    except Exception as exc:
        logger.error("News pipeline failed: %s", exc, exc_info=True)

    # ── 2. Fetch StockTwits posts (no key needed) ─────────────────────────────
    st_scored = []
    try:
        st_fetcher = StockTwitsFetcher()
        raw_st     = st_fetcher.fetch(symbols=symbols)
        logger.info("Processing %d StockTwits posts...", len(raw_st))

        for post in raw_st:
            if post["raw_score"] is not None:
                # Pre-labelled — use directly, skip NLP
                st_scored.append({**post, "sentiment_score": post["raw_score"]})
            else:
                # Unlabelled — score with FinBERT/VADER
                score = scorer.score(post.get("text", ""))
                st_scored.append({**post, "sentiment_score": score})

        labelled   = sum(1 for p in st_scored if p["raw_score"] is not None)
        unlabelled = len(st_scored) - labelled
        logger.info(
            "StockTwits: %d pre-labelled (no NLP needed), %d scored with NLP.",
            labelled, unlabelled,
        )
    except Exception as exc:
        logger.warning("StockTwits pipeline failed (skipping): %s", exc)

    # ── 3. Aggregate into daily features ──────────────────────────────────────
    sentiment_features = aggregate_sentiment(
        news_scored=news_scored,
        stocktwits_scored=st_scored,
        symbols=symbols,
    )

    # ── 4. Merge with technical features and save ─────────────────────────────
    enriched = {}
    for symbol in symbols:
        df_tech = store.load(symbol=symbol, tag="features")
        df_sent = sentiment_features.get(symbol, pd.DataFrame())

        if df_tech.empty:
            logger.warning("%s: no technical features found — skipping.", symbol)
            continue

        if df_sent.empty:
            logger.warning("%s: no sentiment data — using technical only.", symbol)
            enriched[symbol] = df_tech
            continue

        df_tech.index = pd.to_datetime(df_tech.index).normalize()
        df_sent.index = pd.to_datetime(df_sent.index).normalize()

        available = [c for c in SENTIMENT_FEATURE_COLUMNS if c in df_sent.columns]
        df_merged = df_tech.join(df_sent[available], how="left")
        df_merged[available] = df_merged[available].fillna(0)

        # Fill missing sentiment columns with 0
        for col in SENTIMENT_FEATURE_COLUMNS:
            if col not in df_merged.columns:
                df_merged[col] = 0.0

        store.save(df_merged, symbol=symbol, tag="features_with_sentiment")
        enriched[symbol] = df_merged

        logger.info(
            "%s: %d technical + %d sentiment features → %d rows.",
            symbol, len(df_tech.columns), len(available), len(df_merged),
        )

    logger.info("Sentiment pipeline complete.")
    return enriched


if __name__ == "__main__":
    run_sentiment_pipeline()
