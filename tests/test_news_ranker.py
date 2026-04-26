"""
tests/test_news_ranker.py
--------------------------
Unit tests for the conflict-aware news ranker.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from bot.sentiment.news_ranker import (
    rank_articles, _confidence, _recency_weight, _source_weight,
    _keyword_boost, _classify, _insight,
)


def _art(symbol="AAPL", headline="Test", summary="", source="Reuters",
         sent=0.5, hours_ago=1.0):
    return {
        "symbol":         symbol,
        "headline":       headline,
        "summary":        summary,
        "source":         source,
        "url":            "http://x",
        "published_at":   datetime.now(timezone.utc) - timedelta(hours=hours_ago),
        "sentiment_score": sent,
    }


class TestPrimitives:

    def test_recency_decays(self):
        now  = _recency_weight(datetime.now(timezone.utc))
        old  = _recency_weight(datetime.now(timezone.utc) - timedelta(hours=48))
        assert now > old
        assert 0.0 <= old <= 1.0

    def test_source_weight_default(self):
        assert _source_weight("UnknownBlog") < _source_weight("Reuters")

    def test_keyword_boost_stacks(self):
        single, _ = _keyword_boost("Apple beats earnings")
        none,   _ = _keyword_boost("nothing here")
        assert single > none
        assert single <= 1.4

    def test_classify_thresholds(self):
        assert _classify(0.10)  == "+"
        assert _classify(-0.10) == "-"
        assert _classify(0.0)   == "0"
        assert _classify(None)  == "unknown"


class TestRankArticles:

    def _patch_ctx(self, market=None, sector=None, stock=None):
        from bot.sentiment import news_ranker
        ctx = news_ranker._Context(
            market=market,
            sector=sector or {etf: 0.0 for etf in news_ranker.SECTOR_ETFS},
            cache=stock or {},
        )
        return [
            patch("bot.sentiment.news_ranker._build_context", return_value=ctx),
            patch("bot.sentiment.news_ranker._sector_for", return_value=None),
        ]

    def _ranked(self, articles, **ctx_kwargs):
        ps = self._patch_ctx(**ctx_kwargs)
        for p in ps: p.start()
        try:
            return rank_articles(articles)
        finally:
            for p in ps: p.stop()

    def test_empty_returns_empty(self):
        assert rank_articles([]) == []

    def test_strong_article_gets_5_stars(self):
        a = _art(headline="Apple beats earnings, raises guidance",
                 sent=0.85, source="Reuters", hours_ago=0.5)
        out = self._ranked([a])
        assert out[0].stars == 5

    def test_weak_article_low_stars(self):
        a = _art(headline="Apple holds investor day",
                 sent=0.05, source="UnknownBlog", hours_ago=72)
        out = self._ranked([a])
        assert out[0].stars <= 2

    def test_direction_classification(self):
        bull = self._ranked([_art(sent=0.7)])[0]
        bear = self._ranked([_art(sent=-0.7)])[0]
        neut = self._ranked([_art(sent=0.0)])[0]
        assert bull.direction == "bullish"
        assert bear.direction == "bearish"
        assert neut.direction == "neutral"

    def test_keyword_flags_recorded(self):
        a = _art(headline="FDA approval triggers buyout", sent=0.6)
        out = self._ranked([a])
        kw = [f for f in out[0].flags if f.startswith("kw:")]
        assert any("fda" in f for f in kw)

    def test_sorted_by_stars_desc(self):
        strong = _art(headline="Apple beats earnings", sent=0.9, source="Reuters")
        weak   = _art(headline="minor headline", sent=0.05, source="Blog", hours_ago=80)
        out = self._ranked([weak, strong])
        assert out[0].stars >= out[1].stars


class TestInsight:

    def test_vs_sector_flag(self):
        text, flags, boost = _insight("+", "0", "-", "0", "Tech")
        assert "vs-sector" in flags
        assert boost > 1.0

    def test_strong_divergence(self):
        text, flags, boost = _insight("+", "0", "-", "-", "Tech")
        assert "strong-divergence" in flags
        assert boost > 1.3

    def test_no_flag_when_aligned(self):
        text, flags, boost = _insight("+", "+", "+", "+", "Tech")
        assert "vs-sector" not in flags
        assert "vs-market" not in flags
        assert boost == 1.0
