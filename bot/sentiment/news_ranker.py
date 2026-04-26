"""
bot/sentiment/news_ranker.py
-----------------------------
Ranks news articles 1-5 with conflict-aware insight.

Confidence is composed from:
  * sentiment magnitude (FinBERT/VADER score on title + summary)
  * source weight (Reuters/Bloomberg > generic blogs)
  * recency decay (24-hour half-life)
  * keyword boost (earnings, guidance, FDA, M&A, downgrades, …)

Each article is then placed in context:
  * stock-level sentiment (rolling combined_sentiment for the symbol)
  * sector-level sentiment (sector ETF rolling combined_sentiment)
  * market-level sentiment (SPY rolling combined_sentiment)

Articles that *contradict* the broader picture (stock + while sector −,
or stock − while market +) are flagged as high-signal — they often
carry the alpha.  Final 1-5 stars combines confidence with the
contradiction boost.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from bot.config import DATA_DIR

logger = logging.getLogger(__name__)


# ── Tunables ────────────────────────────────────────────────────────────────

SOURCE_WEIGHTS = {
    "reuters":         1.00,
    "bloomberg":       1.00,
    "wall street journal": 0.95,
    "wsj":             0.95,
    "financial times": 0.95,
    "ft":              0.95,
    "cnbc":            0.85,
    "marketwatch":     0.80,
    "barron's":        0.80,
    "barrons":         0.80,
    "seeking alpha":   0.65,
    "benzinga":        0.55,
    "zacks":           0.55,
    "motley fool":     0.45,
    "the fly":         0.50,
}
DEFAULT_SOURCE_WEIGHT = 0.60

# (regex-free) keyword → boost.  Boost stacks (capped at 1.4×).
KEYWORD_BOOSTS = {
    # Earnings / guidance
    "earnings":     0.20, "beat":       0.15, "miss":      0.18, "guidance":   0.20,
    "raises forecast": 0.25, "lowers forecast": 0.25, "warns": 0.20,
    # Ratings
    "upgrade":      0.18, "downgrade":  0.20, "price target": 0.10, "buy rating": 0.10,
    "sell rating":  0.12,
    # Corporate action
    "acquires":     0.22, "acquired":   0.22, "merger":    0.22, "spinoff":    0.18,
    "buyback":      0.15, "dividend":   0.12, "split":     0.10,
    # Regulatory / catalyst
    "fda":          0.25, "approval":   0.20, "lawsuit":   0.18, "investigation": 0.20,
    "recall":       0.22, "subpoena":   0.20, "ceo":       0.10, "resigns":     0.18,
    # Market-moving macro tags
    "fed":          0.10, "rate cut":   0.15, "rate hike": 0.15, "inflation":   0.10,
}

# Sector ETF reference (matches market_overview.SECTOR_ETFS)
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

MARKET_PROXY = "SPY"


@dataclass
class RankedArticle:
    symbol:        str
    headline:      str
    summary:       str
    source:        str
    url:           str
    published_at:  datetime
    sentiment:     float          # raw -1..1
    confidence:    float          # 0..1 composite
    stars:         int            # 1..5
    direction:     str            # "bullish" | "bearish" | "neutral"
    insight:       str            # human-readable conflict / context line
    flags:         list[str]      = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol":       self.symbol,
            "headline":     self.headline,
            "summary":      self.summary,
            "source":       self.source,
            "url":          self.url,
            "published_at": self.published_at.isoformat() if self.published_at else "",
            "sentiment":    round(self.sentiment, 3),
            "confidence":   round(self.confidence, 3),
            "stars":        self.stars,
            "direction":    self.direction,
            "insight":      self.insight,
            "flags":        self.flags,
        }


# ── Per-article scoring ─────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _to_aware(dt) -> Optional[datetime]:
    if dt is None or dt == "":
        return None
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except ValueError:
            return None
    if isinstance(dt, datetime) and dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _recency_weight(published_at: Optional[datetime], half_life_hours: float = 24.0) -> float:
    """Exponential decay; 1.0 at t=0, 0.5 at t=half_life."""
    if published_at is None:
        return 0.5
    delta = _now() - published_at
    hours = max(delta.total_seconds() / 3600.0, 0.0)
    return float(math.pow(0.5, hours / half_life_hours))


def _source_weight(source: str) -> float:
    s = (source or "").strip().lower()
    return SOURCE_WEIGHTS.get(s, DEFAULT_SOURCE_WEIGHT)


def _keyword_boost(text: str) -> tuple[float, list[str]]:
    t = (text or "").lower()
    boost = 1.0
    hits: list[str] = []
    for kw, b in KEYWORD_BOOSTS.items():
        if kw in t:
            boost += b
            hits.append(kw)
    return min(boost, 1.4), hits


def _confidence(article: dict) -> tuple[float, list[str]]:
    """Compose 0..1 confidence + return matched keyword flags."""
    magnitude = abs(float(article.get("sentiment_score", 0.0)))
    src       = _source_weight(article.get("source", ""))
    recency   = _recency_weight(_to_aware(article.get("published_at")))
    text      = f"{article.get('headline','')} {article.get('summary','')}"
    boost, hits = _keyword_boost(text)
    raw = magnitude * src * recency * boost
    # Soft squash so scores spread out reasonably for 1-5 mapping
    conf = 1.0 - math.exp(-2.5 * raw)
    return float(min(max(conf, 0.0), 1.0)), hits


# ── Context (stock / sector / market sentiment) ─────────────────────────────

def _load_recent_sentiment(symbol: str, lookback: int = 5) -> Optional[float]:
    """Mean of last N days' combined_sentiment for symbol, or None."""
    for tag in ("features_with_sentiment", "features"):
        path = DATA_DIR / f"{symbol}_{tag}.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
            if "combined_sentiment" not in df.columns or df.empty:
                continue
            tail = df["combined_sentiment"].tail(lookback)
            tail = tail[tail != 0]
            if tail.empty:
                continue
            return float(tail.mean())
        except Exception as exc:
            logger.debug("sentiment load failed for %s: %s", symbol, exc)
    return None


def _classify(value: Optional[float], pos: float = 0.05, neg: float = -0.05) -> str:
    if value is None: return "unknown"
    if value >  pos:  return "+"
    if value <  neg:  return "-"
    return "0"


def _sector_for(symbol: str) -> Optional[str]:
    """Map a stock to a sector ETF.  Best-effort using universe parquet."""
    try:
        from bot.universe import load_universe
        universe = load_universe(eligible_only=False)
        if universe.empty or "sector" not in universe.columns:
            return None
        row = universe.loc[universe["symbol"] == symbol]
        if row.empty:
            return None
        sector = row.iloc[0]["sector"]
        # Map sector → ETF
        sector_to_etf = {
            "Information Technology":   "XLK",
            "Technology":               "XLK",
            "Health Care":              "XLV",
            "Healthcare":               "XLV",
            "Financials":               "XLF",
            "Energy":                   "XLE",
            "Consumer Discretionary":   "XLY",
            "Consumer Staples":         "XLP",
            "Industrials":              "XLI",
            "Utilities":                "XLU",
            "Materials":                "XLB",
            "Real Estate":              "XLRE",
            "Communication Services":   "XLC",
            "Communications":           "XLC",
        }
        return sector_to_etf.get(str(sector))
    except Exception:
        return None


@dataclass
class _Context:
    market: Optional[float]
    sector: dict[str, float]   # sector ETF -> sentiment
    cache:  dict[str, float]   # per-symbol sentiment


def _build_context(symbols: Iterable[str]) -> _Context:
    market = _load_recent_sentiment(MARKET_PROXY)
    sector = {etf: (_load_recent_sentiment(etf) or 0.0) for etf in SECTOR_ETFS}
    cache: dict[str, float] = {}
    for s in symbols:
        v = _load_recent_sentiment(s)
        if v is not None:
            cache[s] = v
    return _Context(market=market, sector=sector, cache=cache)


# ── Insight composer ────────────────────────────────────────────────────────

def _insight(
    article_dir: str,
    stock_dir:   str,
    sector_dir:  str,
    market_dir:  str,
    sector_name: Optional[str],
) -> tuple[str, list[str], float]:
    """
    Return (insight_text, flags, contradiction_boost).
    Boost is multiplicative on confidence — bigger when the article
    contradicts the broader picture.
    """
    flags: list[str] = []
    parts: list[str] = []
    boost = 1.0

    # The article itself
    ad = "+ve" if article_dir == "+" else "-ve" if article_dir == "-" else "neutral"
    parts.append(f"Article {ad}")

    # Vs sector
    sector_label = sector_name or "sector"
    if sector_dir != "unknown":
        parts.append(f"{sector_label} {sector_dir}")
        if article_dir in ("+", "-") and sector_dir in ("+", "-") and article_dir != sector_dir:
            flags.append("vs-sector")
            boost *= 1.25

    # Vs market
    if market_dir != "unknown":
        parts.append(f"market {market_dir}")
        if article_dir in ("+", "-") and market_dir in ("+", "-") and article_dir != market_dir:
            flags.append("vs-market")
            boost *= 1.20

    # Vs the stock's own rolling sentiment (does article reverse the trend?)
    if stock_dir != "unknown":
        if article_dir in ("+", "-") and stock_dir in ("+", "-") and article_dir != stock_dir:
            flags.append("trend-reversal")
            boost *= 1.15

    if "vs-sector" in flags and "vs-market" in flags:
        flags.append("strong-divergence")

    return ", ".join(parts), flags, min(boost, 1.6)


# ── Public API ──────────────────────────────────────────────────────────────

def rank_articles(
    articles_with_scores: list[dict],
    min_confidence: float = 0.0,
) -> list[RankedArticle]:
    """
    Rank a list of scored articles.

    Args:
        articles_with_scores: dicts with keys symbol, headline, summary,
            source, url, published_at, sentiment_score (already produced
            by the sentiment pipeline / scorer).
        min_confidence: drop anything below this composite confidence.

    Returns:
        Articles sorted by stars desc, confidence desc.
    """
    if not articles_with_scores:
        return []

    symbols = sorted({a.get("symbol", "") for a in articles_with_scores if a.get("symbol")})
    ctx = _build_context(symbols)

    ranked: list[RankedArticle] = []
    for a in articles_with_scores:
        symbol = a.get("symbol", "")
        sent = float(a.get("sentiment_score", 0.0))
        article_dir = "+" if sent > 0.05 else "-" if sent < -0.05 else "0"

        stock_v  = ctx.cache.get(symbol)
        sector_etf = _sector_for(symbol)
        sector_v = ctx.sector.get(sector_etf) if sector_etf else None
        market_v = ctx.market

        stock_dir  = _classify(stock_v)
        sector_dir = _classify(sector_v)
        market_dir = _classify(market_v)

        conf, kw_hits = _confidence(a)
        text, flags, boost = _insight(
            article_dir, stock_dir, sector_dir, market_dir,
            SECTOR_ETFS.get(sector_etf or ""),
        )
        if kw_hits:
            flags = [*flags, *[f"kw:{k}" for k in kw_hits]]
        conf = min(conf * boost, 1.0)
        if conf < min_confidence:
            continue

        # 1-5 mapping with thresholds tuned so 5★ is genuinely rare.
        if   conf >= 0.80: stars = 5
        elif conf >= 0.60: stars = 4
        elif conf >= 0.40: stars = 3
        elif conf >= 0.20: stars = 2
        else:              stars = 1

        ranked.append(RankedArticle(
            symbol       = symbol,
            headline     = a.get("headline", "") or "",
            summary      = a.get("summary",  "") or "",
            source       = a.get("source",   "") or "",
            url          = a.get("url",      "") or "",
            published_at = _to_aware(a.get("published_at")) or _now(),
            sentiment    = sent,
            confidence   = conf,
            stars        = stars,
            direction    = "bullish" if article_dir == "+" else "bearish" if article_dir == "-" else "neutral",
            insight      = text,
            flags        = flags,
        ))

    ranked.sort(key=lambda r: (r.stars, r.confidence), reverse=True)
    return ranked


def get_top_news(
    symbols:        Optional[list[str]] = None,
    lookback_days:  int                 = 3,
    limit:          int                 = 15,
    min_stars:      int                 = 4,
    include_conflicts: bool             = True,
) -> list[dict]:
    """
    Convenience wrapper used by the dashboard.  Fetches via NewsFetcher,
    scores with SentimentScorer, ranks and filters.

    Returns list[dict] (RankedArticle.to_dict() rows).
    """
    from bot.sentiment.news_fetcher import NewsFetcher
    from bot.sentiment.scorer       import SentimentScorer
    from bot.universe               import load_universe

    if symbols is None:
        u = load_universe(eligible_only=True)
        symbols = u["symbol"].head(40).tolist() if not u.empty else [MARKET_PROXY]

    try:
        fetcher = NewsFetcher()
        raw = fetcher.fetch(symbols=symbols, lookback_days=lookback_days, limit=20)
    except Exception as exc:
        logger.warning("News fetch failed: %s", exc)
        return []
    if not raw:
        return []

    scorer = SentimentScorer(backend="auto")
    scored = []
    for art in raw:
        text = (art.get("headline") or "") + ". " + (art.get("summary") or "")
        scored.append({**art, "sentiment_score": scorer.score(text)})

    ranked = rank_articles(scored)

    out = []
    for r in ranked:
        if r.stars >= min_stars:
            out.append(r.to_dict())
        elif include_conflicts and any(f in r.flags for f in ("vs-sector", "vs-market", "strong-divergence")):
            out.append(r.to_dict())
        if len(out) >= limit:
            break
    return out
