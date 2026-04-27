"""
tests/test_sentiment.py
------------------------
Unit tests for sentiment pipeline — scorer, StockTwits fetcher, aggregator.
No real API calls made — all external services are mocked.
"""

import pandas as pd
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from bot.sentiment.scorer              import SentimentScorer
from bot.sentiment.stocktwits_fetcher  import StockTwitsFetcher
from bot.sentiment.aggregator          import aggregate_sentiment, SENTIMENT_FEATURE_COLUMNS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dt(date_str):
    return datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)

def _news(symbol, date_str, score):
    return {"symbol": symbol, "published_at": _dt(date_str),
            "sentiment_score": score, "headline": "Test"}

def _st(symbol, date_str, raw_score=None, sentiment=None, likes=10):
    return {"symbol": symbol, "published_at": _dt(date_str),
            "raw_score": raw_score, "sentiment": sentiment,
            "likes": likes, "text": "Test post"}


# ── SentimentScorer tests ─────────────────────────────────────────────────────

class TestSentimentScorer:

    def _make_scorer(self):
        """Return a scorer using whichever backend is available."""
        return SentimentScorer(backend="auto")

    def test_bullish_headline_positive(self):
        score = self._make_scorer().score("Apple smashes earnings with record profits")
        assert isinstance(score, float)
        assert score >= -1.0 and score <= 1.0

    def test_bearish_headline_negative(self):
        score = self._make_scorer().score("Stock crashes after catastrophic earnings miss")
        assert isinstance(score, float)
        assert score >= -1.0 and score <= 1.0

    def test_empty_string_returns_zero(self):
        assert self._make_scorer().score("") == 0.0

    def test_none_returns_zero(self):
        assert self._make_scorer().score(None) == 0.0

    def test_score_within_range(self):
        scorer = self._make_scorer()
        for text in ["Great earnings", "Neutral day", "Huge losses"]:
            assert -1.0 <= scorer.score(text) <= 1.0

    def test_batch_length_matches(self):
        scores = self._make_scorer().score_batch(["good", "bad", "neutral"])
        assert len(scores) == 3

    def test_finbert_parsed_correctly(self):
        mock_pipe = MagicMock(return_value=[[
            {"label": "positive", "score": 0.8},
            {"label": "negative", "score": 0.1},
            {"label": "neutral",  "score": 0.1},
        ]])
        scorer = SentimentScorer.__new__(SentimentScorer)
        scorer.backend   = "finbert"
        scorer._pipeline = mock_pipe
        scorer._vader    = None
        assert abs(scorer.score("Earnings beat") - 0.7) < 0.01


# ── StockTwitsFetcher tests ───────────────────────────────────────────────────

class TestStockTwitsFetcher:

    def _mock_response(self, messages):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"messages": messages}
        resp.raise_for_status = MagicMock()
        return resp

    def _sample_message(self, body="AAPL looks great", sentiment="Bullish", likes=5):
        return {
            "id": 123,
            "body": body,
            "created_at": "2024-01-15T10:00:00Z",
            "likes": {"total": likes},
            "entities": {"sentiment": {"basic": sentiment}},
            "user": {"username": "trader1"},
        }

    def test_bullish_post_has_positive_raw_score(self):
        fetcher = StockTwitsFetcher()
        fetcher.session.get = MagicMock(
            return_value=self._mock_response([self._sample_message(sentiment="Bullish")])
        )
        posts = fetcher.fetch(["AAPL"])
        assert posts[0]["raw_score"] == 1.0

    def test_bearish_post_has_negative_raw_score(self):
        fetcher = StockTwitsFetcher()
        fetcher.session.get = MagicMock(
            return_value=self._mock_response([self._sample_message(sentiment="Bearish")])
        )
        posts = fetcher.fetch(["AAPL"])
        assert posts[0]["raw_score"] == -1.0

    def test_unlabelled_post_has_none_raw_score(self):
        msg = self._sample_message()
        msg["entities"] = {}
        fetcher = StockTwitsFetcher()
        fetcher.session.get = MagicMock(
            return_value=self._mock_response([msg])
        )
        posts = fetcher.fetch(["AAPL"])
        assert posts[0]["raw_score"] is None

    def test_symbol_assigned_correctly(self):
        fetcher = StockTwitsFetcher()
        fetcher.session.get = MagicMock(
            return_value=self._mock_response([self._sample_message()])
        )
        posts = fetcher.fetch(["TSLA"])
        assert posts[0]["symbol"] == "TSLA"

    def test_404_returns_empty(self):
        resp = MagicMock()
        resp.status_code = 404
        fetcher = StockTwitsFetcher()
        fetcher.session.get = MagicMock(return_value=resp)
        posts = fetcher.fetch(["UNKNOWN"])
        assert posts == []

    def test_likes_parsed(self):
        fetcher = StockTwitsFetcher()
        fetcher.session.get = MagicMock(
            return_value=self._mock_response([self._sample_message(likes=42)])
        )
        posts = fetcher.fetch(["AAPL"])
        assert posts[0]["likes"] == 42


# ── Aggregator tests ──────────────────────────────────────────────────────────

class TestAggregator:

    def test_combined_sentiment_in_range(self):
        news = [_news("AAPL", "2024-01-15", 0.8)]
        st   = [_st("AAPL", "2024-01-15", raw_score=1.0, sentiment="Bullish")]
        df   = aggregate_sentiment(news, st, ["AAPL"])["AAPL"]
        assert (df["combined_sentiment"].between(-1, 1)).all()

    def test_news_only_works(self):
        news = [_news("AAPL", "2024-01-15", 0.5)]
        df   = aggregate_sentiment(news, [], ["AAPL"])["AAPL"]
        assert not df.empty
        assert "combined_sentiment" in df.columns

    def test_stocktwits_only_works(self):
        st = [_st("TSLA", "2024-01-15", raw_score=-1.0, sentiment="Bearish")]
        df = aggregate_sentiment([], st, ["TSLA"])["TSLA"]
        assert not df.empty
        assert df["st_sentiment_mean"].iloc[0] == -1.0

    def test_empty_inputs_returns_empty(self):
        assert aggregate_sentiment([], [], ["AAPL"])["AAPL"].empty

    def test_bullish_ratio_correct(self):
        st = [
            _st("AAPL", "2024-01-15", raw_score=1.0,  sentiment="Bullish"),
            _st("AAPL", "2024-01-15", raw_score=-1.0, sentiment="Bearish"),
            _st("AAPL", "2024-01-15", raw_score=1.0,  sentiment="Bullish"),
        ]
        df = aggregate_sentiment([], st, ["AAPL"])["AAPL"]
        assert abs(df["st_bullish_ratio"].iloc[0] - 2/3) < 0.01

    def test_likes_summed(self):
        st = [
            _st("AAPL", "2024-01-15", raw_score=1.0, likes=10),
            _st("AAPL", "2024-01-15", raw_score=1.0, likes=25),
        ]
        df = aggregate_sentiment([], st, ["AAPL"])["AAPL"]
        assert df["st_likes_sum"].iloc[0] == 35

    def test_news_weighted_60pct(self):
        news = [_news("AAPL", "2024-01-15", 1.0)]
        st   = [_st("AAPL",  "2024-01-15", raw_score=0.0, sentiment=None)]
        df   = aggregate_sentiment(news, st, ["AAPL"])["AAPL"]
        # combined = 1.0 * 0.60 + 0.0 * 0.40 = 0.60
        assert abs(df["combined_sentiment"].iloc[0] - 0.60) < 0.01

    def test_symbol_isolation(self):
        news = [_news("AAPL", "2024-01-15", 0.9), _news("TSLA", "2024-01-15", -0.7)]
        result = aggregate_sentiment(news, [], ["AAPL", "TSLA"])
        assert result["AAPL"]["news_sentiment_mean"].iloc[0] > 0
        assert result["TSLA"]["news_sentiment_mean"].iloc[0] < 0

    def test_multi_day_aggregation(self):
        news = [_news("AAPL", "2024-01-14", 0.5), _news("AAPL", "2024-01-15", -0.3)]
        df   = aggregate_sentiment(news, [], ["AAPL"])["AAPL"]
        assert len(df) == 2


# ── Look-ahead guard (article published after the close rolls forward) ─────

class TestLookaheadGuard:

    def _at(self, dt_str: str, score: float):
        return {
            "symbol":          "AAPL",
            "published_at":    datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc),
            "sentiment_score": score,
            "headline":        "test",
        }

    def test_pre_cutoff_stays_on_same_day(self):
        # 19:00 UTC = 3pm ET — before close, lands on Jan 15.
        news = [self._at("2024-01-15T19:00:00", 0.5)]
        df = aggregate_sentiment(news, [], ["AAPL"])["AAPL"]
        assert pd.Timestamp("2024-01-15") in df.index

    def test_post_cutoff_rolls_to_next_day(self):
        # 22:00 UTC = 6pm ET — after close, should roll to Jan 16.
        news = [self._at("2024-01-15T22:00:00", 0.5)]
        df = aggregate_sentiment(news, [], ["AAPL"])["AAPL"]
        assert pd.Timestamp("2024-01-16") in df.index
        assert pd.Timestamp("2024-01-15") not in df.index

    def test_mixed_pre_post_cutoff_buckets_correctly(self):
        news = [
            self._at("2024-01-15T15:00:00",  0.8),    # pre-cutoff → Jan 15
            self._at("2024-01-15T23:00:00", -0.6),    # post-cutoff → Jan 16
        ]
        df = aggregate_sentiment(news, [], ["AAPL"])["AAPL"]
        assert df.loc["2024-01-15", "news_sentiment_mean"] >  0
        assert df.loc["2024-01-16", "news_sentiment_mean"] <  0
