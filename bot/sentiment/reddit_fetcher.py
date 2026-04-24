"""
bot/sentiment/reddit_fetcher.py
--------------------------------
Fetches posts and comments from finance-related subreddits.
Filters by symbol mentions and returns text ready for NLP scoring.

Required .env variables:
    REDDIT_CLIENT_ID
    REDDIT_CLIENT_SECRET
    REDDIT_USER_AGENT   (e.g. "trading-bot/0.1 by YourUsername")
"""

import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional

import praw
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

SUBREDDITS = ["stocks", "wallstreetbets", "investing", "StockMarket"]


class RedditFetcher:
    """
    Fetches Reddit posts mentioning specific stock symbols.

    Args:
        client_id: Reddit app client ID. Defaults to env var REDDIT_CLIENT_ID.
        client_secret: Reddit app secret. Defaults to env var REDDIT_CLIENT_SECRET.
        user_agent: Reddit API user agent string. Defaults to env var REDDIT_USER_AGENT.
    """

    def __init__(
        self,
        client_id:     Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent:    Optional[str] = None,
    ):
        self.client_id     = client_id     or os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        self.user_agent    = user_agent    or os.getenv("REDDIT_USER_AGENT", "trading-bot/0.1")

        if not self.client_id or not self.client_secret:
            raise ValueError("Reddit credentials must be set in .env: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET")

        self.reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent,
        )
        logger.info("RedditFetcher initialised (read-only).")

    def fetch(
        self,
        symbols: list[str],
        subreddits: list[str] = SUBREDDITS,
        limit_per_sub: int = 50,
    ) -> list[dict]:
        """
        Search subreddits for posts mentioning each symbol.

        Args:
            symbols: List of ticker symbols e.g. ['AAPL', 'TSLA'].
            subreddits: List of subreddit names to search.
            limit_per_sub: Max posts to fetch per subreddit per symbol.

        Returns:
            List of dicts with keys: symbol, text, score, published_at, subreddit, url
        """
        posts = []
        for symbol in symbols:
            for sub_name in subreddits:
                try:
                    sub = self.reddit.subreddit(sub_name)
                    for post in sub.search(f"${symbol} OR {symbol}", limit=limit_per_sub, sort="new"):
                        if not _mentions_symbol(post.title + " " + (post.selftext or ""), symbol):
                            continue
                        text = post.title
                        if post.selftext:
                            text += ". " + post.selftext[:500]
                        posts.append({
                            "symbol":       symbol,
                            "text":         text.strip(),
                            "score":        post.score,
                            "published_at": datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
                            "subreddit":    sub_name,
                            "url":          f"https://reddit.com{post.permalink}",
                        })
                except Exception as exc:
                    logger.error("Reddit fetch failed for %s/%s: %s", sub_name, symbol, exc)

            logger.info("Fetched %d Reddit posts for %s.", sum(1 for p in posts if p["symbol"] == symbol), symbol)

        return posts


def _mentions_symbol(text: str, symbol: str) -> bool:
    """Return True if the text explicitly mentions the ticker symbol."""
    pattern = rf"\b\$?{re.escape(symbol)}\b"
    return bool(re.search(pattern, text, re.IGNORECASE))
