"""
bot/scrapers/fear_greed.py
---------------------------
Scrapes CNN's Fear & Greed Index from their public JSON endpoint.
Caches the result for 1 hour to be polite and avoid hammering the API.

Returns a dict:
    score     0-100 number (0=extreme fear, 100=extreme greed)
    label     'Extreme Fear' | 'Fear' | 'Neutral' | 'Greed' | 'Extreme Greed'
    timestamp ISO timestamp of the data point
    yesterday previous day's score for comparison
    week_ago  one-week-ago score for comparison
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

CNN_URL    = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
CACHE_FILE = Path(__file__).parent.parent.parent / "data" / "cache" / "fear_greed.json"
CACHE_TTL  = 3600   # 1 hour

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def get_fear_greed() -> dict:
    """
    Return the current Fear & Greed Index, using cached value if fresh.

    Returns:
        Dict with keys score, label, timestamp, yesterday, week_ago.
        On error returns a sensible default with score=50 (neutral).
    """
    cached = _read_cache()
    if cached is not None:
        return cached

    try:
        data = _fetch_live()
        _write_cache(data)
        return data
    except Exception as exc:
        logger.warning("Fear & Greed fetch failed (%s) — using stale cache or fallback.", exc)
        stale = _read_cache(ignore_ttl=True)
        if stale:
            return stale
        return _fallback()


def _fetch_live() -> dict:
    """Hit CNN's JSON endpoint and parse the response."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept":     "application/json",
    }
    response = requests.get(CNN_URL, headers=headers, timeout=10)
    response.raise_for_status()
    payload = response.json()

    fg     = payload.get("fear_and_greed", {})
    score  = float(fg.get("score", 50))
    label  = str(fg.get("rating", "Neutral")).title()
    ts     = fg.get("timestamp", datetime.utcnow().isoformat())

    return {
        "score":     round(score, 1),
        "label":     label,
        "timestamp": ts,
        "yesterday": round(float(fg.get("previous_close", score)), 1),
        "week_ago":  round(float(fg.get("previous_1_week",  score)), 1),
    }


def _read_cache(ignore_ttl: bool = False) -> dict | None:
    if not CACHE_FILE.exists():
        return None
    try:
        with open(CACHE_FILE) as f:
            entry = json.load(f)
        if not ignore_ttl and time.time() - entry["fetched_at"] > CACHE_TTL:
            return None
        return entry["data"]
    except Exception:
        return None


def _write_cache(data: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump({"fetched_at": time.time(), "data": data}, f)


def _fallback() -> dict:
    return {
        "score":     50.0,
        "label":     "Neutral",
        "timestamp": datetime.utcnow().isoformat(),
        "yesterday": 50.0,
        "week_ago":  50.0,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(get_fear_greed(), indent=2))
