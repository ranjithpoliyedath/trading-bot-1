"""
bot/sentiment/stocktwits_fetcher.py
-------------------------------------
Fetches posts from StockTwits for a list of symbols.
No API key required — uses the public StockTwits REST API.

StockTwits posts already include a user-labelled sentiment tag:
    Bullish  → we map to +1.0
    Bearish  → we map to -1.0
    None     → we score with FinBERT/VADER fallback

Rate limit: ~200 requests/hour on the public endpoint.
Each symbol fetch = 1 request (returns up to 30 most recent messages).
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
REQUEST_DELAY  = 0.5   # seconds between requests to be polite to the API
MAX_RETRIES    = 3


class StockTwitsFetcher:
    """
    Fetches StockTwits messages for stock symbols.

    No authentication required. Uses public StockTwits stream endpoint.

    Usage:
        fetcher = StockTwitsFetcher()
        posts   = fetcher.fetch(["AAPL", "TSLA"])
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "trading-bot-sentiment/1.0 (educational project)",
        })
        logger.info("StockTwitsFetcher initialised (no auth required).")

    def fetch(
        self,
        symbols:   list[str],
        limit:     int = 30,
    ) -> list[dict]:
        """
        Fetch recent StockTwits messages for a list of symbols.

        Args:
            symbols: List of ticker symbols e.g. ['AAPL', 'TSLA'].
            limit: Max messages per symbol (StockTwits caps at 30 for public API).

        Returns:
            List of dicts with keys:
                symbol        — ticker symbol
                text          — message body
                published_at  — datetime (UTC)
                sentiment     — 'Bullish', 'Bearish', or None
                raw_score     — +1.0 (bullish), -1.0 (bearish), or None
                likes         — number of likes (proxy for attention)
                url           — link to original post
        """
        all_posts = []
        for symbol in symbols:
            posts = self._fetch_symbol(symbol, limit)
            all_posts.extend(posts)
            time.sleep(REQUEST_DELAY)

        logger.info(
            "StockTwits: fetched %d posts across %d symbols.",
            len(all_posts), len(symbols),
        )
        return all_posts

    def _fetch_symbol(self, symbol: str, limit: int) -> list[dict]:
        url = STOCKTWITS_URL.format(symbol=symbol.upper())

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.get(url, timeout=10)

                if resp.status_code == 429:
                    wait = 60 * attempt
                    logger.warning("StockTwits rate limited — waiting %ds.", wait)
                    time.sleep(wait)
                    continue

                if resp.status_code == 404:
                    logger.warning("Symbol not found on StockTwits: %s", symbol)
                    return []

                resp.raise_for_status()
                data = resp.json()
                return self._parse_messages(data.get("messages", []), symbol)

            except requests.RequestException as exc:
                logger.error(
                    "StockTwits fetch error for %s (attempt %d/%d): %s",
                    symbol, attempt, MAX_RETRIES, exc,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** attempt)

        return []

    def _parse_messages(self, messages: list, symbol: str) -> list[dict]:
        posts = []
        for msg in messages:
            sentiment_label = None
            raw_score       = None

            entities = msg.get("entities", {})
            sentiment_data = entities.get("sentiment", None)
            if isinstance(sentiment_data, dict):
                sentiment_label = sentiment_data.get("basic")
            if sentiment_label == "Bullish":
                raw_score = 1.0
            elif sentiment_label == "Bearish":
                raw_score = -1.0

            created_str = msg.get("created_at", "")
            try:
                published_at = datetime.strptime(
                    created_str, "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=timezone.utc)
            except Exception:
                published_at = datetime.now(tz=timezone.utc)

            user    = msg.get("user", {})
            post_id = msg.get("id", "")

            posts.append({
                "symbol":       symbol,
                "text":         msg.get("body", "").strip(),
                "published_at": published_at,
                "sentiment":    sentiment_label,
                "raw_score":    raw_score,
                "likes":        msg.get("likes", {}).get("total", 0),
                "url":          f"https://stocktwits.com/{user.get('username','')}/message/{post_id}",
            })

        logger.debug("Parsed %d messages for %s.", len(posts), symbol)
        return posts
