"""
bot/sentiment/news_fetcher.py
------------------------------
Fetches financial news headlines from Alpaca News API.
Returns timestamped articles ready for NLP scoring.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class NewsFetcher:
    """
    Fetches news articles from Alpaca for one or more symbols.

    Args:
        api_key: Alpaca API key. Defaults to env var ALPACA_API_KEY.
        secret_key: Alpaca secret key. Defaults to env var ALPACA_SECRET_KEY.
    """

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        self.api_key    = api_key    or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API key and secret must be set in .env")
        self.client = NewsClient(self.api_key, self.secret_key)
        logger.info("NewsFetcher initialised.")

    def fetch(
        self,
        symbols: list[str],
        lookback_days: int = 30,
        limit: int = 200,
    ) -> list[dict]:
        """
        Fetch recent news articles for a list of symbols.

        Args:
            symbols: List of ticker symbols e.g. ['AAPL', 'TSLA'].
            lookback_days: How many days back to fetch.
            limit: Max number of articles per symbol.

        Returns:
            List of dicts with keys: symbol, headline, summary, published_at, source, url
        """
        end   = datetime.utcnow()
        start = end - timedelta(days=lookback_days)

        articles = []
        for symbol in symbols:
            try:
                request = NewsRequest(
                    symbols=symbol,
                    start=start,
                    end=end,
                    limit=limit,
                    include_content=False,
                )
                response = self.client.get_news(request)
                # Alpaca SDK returns a NewsSet whose `.data` dict is
                # {"news": [News, ...]}.  Iterating the NewsSet yields
                # ("news", [list]) tuples — not individual articles —
                # which is what produced empty headlines previously.
                items = []
                if hasattr(response, "data") and isinstance(response.data, dict):
                    items = response.data.get("news", []) or []
                elif isinstance(response, dict):
                    items = response.get("news", []) or []
                else:
                    items = list(response)

                count = 0
                for item in items:
                    if isinstance(item, dict):
                        headline     = item.get("headline") or item.get("title", "")
                        summary      = item.get("summary", "")
                        published_at = item.get("created_at") or item.get("updated_at")
                        source       = item.get("source", "")
                        url          = item.get("url", "")
                    else:
                        headline     = getattr(item, "headline", "") or ""
                        summary      = getattr(item, "summary",  "") or ""
                        published_at = getattr(item, "created_at", None)
                        source       = getattr(item, "source",   "") or ""
                        url          = getattr(item, "url",      "") or ""
                    articles.append({
                        "symbol":       symbol,
                        "headline":     headline,
                        "summary":      summary,
                        "published_at": published_at,
                        "source":       source,
                        "url":          url,
                    })
                    count += 1
                logger.info("Fetched %d articles for %s.", count, symbol)
            except Exception as exc:
                logger.error("News fetch failed for %s: %s", symbol, exc, exc_info=True)

        return articles
