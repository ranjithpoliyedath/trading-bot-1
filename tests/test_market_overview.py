"""
tests/test_market_overview.py
-------------------------------
Unit tests for fear & greed scraper and market overview data aggregator.
All HTTP calls and file reads are mocked.
"""

import json
import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bot.scrapers.fear_greed import (
    get_fear_greed, _fetch_live, _fallback,
)


# ── Fear & Greed scraper ──────────────────────────────────────────────────────

class TestFearGreed:

    def test_fallback_returns_neutral_score(self):
        result = _fallback()
        assert result["score"] == 50.0
        assert result["label"] == "Neutral"

    def test_fetch_live_parses_cnn_payload(self):
        cnn_payload = {
            "fear_and_greed": {
                "score":            72.3,
                "rating":           "greed",
                "timestamp":        "2026-04-25T12:00:00",
                "previous_close":   68.0,
                "previous_1_week":  55.5,
            }
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = cnn_payload
        mock_resp.raise_for_status = MagicMock()

        with patch("bot.scrapers.fear_greed.requests.get", return_value=mock_resp):
            result = _fetch_live()

        assert result["score"]     == 72.3
        assert result["label"]     == "Greed"
        assert result["yesterday"] == 68.0
        assert result["week_ago"]  == 55.5

    def test_fetch_live_handles_missing_fields(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"fear_and_greed": {}}
        mock_resp.raise_for_status = MagicMock()

        with patch("bot.scrapers.fear_greed.requests.get", return_value=mock_resp):
            result = _fetch_live()

        assert result["score"] == 50
        assert result["label"] == "Neutral"

    def test_get_fear_greed_uses_cached_when_fresh(self, tmp_path):
        cache_file = tmp_path / "fg_cache.json"
        cached = {
            "fetched_at": time.time(),
            "data": {"score": 65.0, "label": "Greed", "timestamp": "x", "yesterday": 60, "week_ago": 50},
        }
        cache_file.write_text(json.dumps(cached))

        with patch("bot.scrapers.fear_greed.CACHE_FILE", cache_file):
            with patch("bot.scrapers.fear_greed._fetch_live") as mock_fetch:
                result = get_fear_greed()
                mock_fetch.assert_not_called()

        assert result["score"] == 65.0

    def test_get_fear_greed_falls_back_on_network_error(self, tmp_path):
        cache_file = tmp_path / "missing.json"

        with patch("bot.scrapers.fear_greed.CACHE_FILE", cache_file):
            with patch("bot.scrapers.fear_greed.requests.get", side_effect=Exception("network")):
                result = get_fear_greed()

        assert result["score"] == 50.0


# ── Market overview aggregator ────────────────────────────────────────────────

class TestMarketOverview:

    @patch("bot.market_overview.get_fear_greed")
    @patch("bot.market_overview._latest_change")
    @patch("bot.market_overview.load_universe")
    @patch("bot.market_overview._load_features")
    def test_overview_has_all_keys(self, mock_features, mock_universe, mock_latest, mock_fg):
        from bot.market_overview import get_market_overview

        mock_fg.return_value = {"score": 60, "label": "Greed", "timestamp": "x", "yesterday": 55, "week_ago": 50}
        mock_latest.return_value = {"close": 100, "change_pct": 1.5, "date": "2026-04-25"}
        mock_universe.return_value = pd.DataFrame({"symbol": ["AAPL"]})
        mock_features.return_value = pd.DataFrame({
            "close":              [100.0],
            "volume_ratio":       [1.0],
            "combined_sentiment": [0.2],
            "price_change_1d":    [0.01],
        })

        result = get_market_overview()

        for key in ("fear_greed", "indexes", "sectors", "volume_movers", "sentiment_heatmap", "news"):
            assert key in result

    @patch("bot.market_overview._load_features")
    @patch("bot.market_overview.load_universe")
    def test_volume_movers_filter_threshold(self, mock_universe, mock_features):
        from bot.market_overview import _get_volume_movers

        mock_universe.return_value = pd.DataFrame({"symbol": ["LOW", "HIGH"]})

        def fake_features(sym):
            vr = 1.0 if sym == "LOW" else 3.5
            return pd.DataFrame({
                "close":              [100.0],
                "volume_ratio":       [vr],
                "price_change_1d":    [0.02],
            })
        mock_features.side_effect = fake_features

        result = _get_volume_movers(threshold=2.0)
        symbols = [r["symbol"] for r in result]
        assert "HIGH" in symbols
        assert "LOW"  not in symbols

    @patch("bot.market_overview._latest_change")
    def test_sector_leaders_sorted_by_change(self, mock_latest):
        from bot.market_overview import _get_sector_leaders

        # Make XLE the leader, XLU second, XLF third
        def fake_change(sym):
            mapping = {"XLE": 3.5, "XLU": 1.2, "XLF": 0.5}
            return {"close": 100, "change_pct": mapping.get(sym, -1.0), "date": "x"}
        mock_latest.side_effect = fake_change

        result = _get_sector_leaders(top_n=3)
        symbols = [r["symbol"] for r in result]
        assert symbols[0] == "XLE"
