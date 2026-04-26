"""
tests/test_screener.py
-----------------------
Unit tests for the stock screener.
No I/O — _load_latest is patched to return synthetic Series.
"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from bot.screener import Filter, run_screener, list_fields, SCREENER_FIELDS


def _row(**kwargs) -> pd.Series:
    base = {
        "close":              100.0,
        "rsi_14":             50.0,
        "macd_hist":          0.0,
        "combined_sentiment": 0.0,
        "volume_ratio":       1.0,
        "price_change_1d":    0.0,
        "news_count":         0,
    }
    base.update(kwargs)
    return pd.Series(base)


class TestFilter:

    def test_gt_matches(self):
        assert Filter("rsi_14", ">", 40).matches(_row(rsi_14=50))

    def test_gt_excludes(self):
        assert not Filter("rsi_14", ">", 60).matches(_row(rsi_14=50))

    def test_missing_field_excluded(self):
        assert not Filter("nope", ">", 0).matches(_row())

    def test_nan_excluded(self):
        assert not Filter("rsi_14", ">", 0).matches(_row(rsi_14=float("nan")))

    def test_invalid_op_raises(self):
        with pytest.raises(ValueError):
            Filter("rsi_14", "??", 0).matches(_row())


class TestRunScreener:

    def _patch_loader(self, fixtures):
        """Replace _load_latest with a dict-based stub and clamp candidates."""
        return [
            patch("bot.screener._candidate_symbols",
                  return_value=list(fixtures.keys())),
            patch("bot.screener._load_latest",
                  side_effect=lambda s: fixtures.get(s)),
        ]

    def _run_with(self, fixtures, **kwargs):
        ps = self._patch_loader(fixtures)
        for p in ps:
            p.start()
        try:
            return run_screener(**kwargs)
        finally:
            for p in ps:
                p.stop()

    def test_basic_filter_ranks_results(self):
        fixtures = {
            "AAA": _row(rsi_14=20, combined_sentiment=0.5),
            "BBB": _row(rsi_14=80, combined_sentiment=0.5),
            "CCC": _row(rsi_14=15, combined_sentiment=0.5),
        }
        rows = self._run_with(
            fixtures,
            filters=[Filter("rsi_14", "<", 30)],
            descending=False,
        )
        assert [r["symbol"] for r in rows] == ["CCC", "AAA"]

    def test_and_combination(self):
        fixtures = {
            "AAA": _row(rsi_14=25, combined_sentiment=0.6),
            "BBB": _row(rsi_14=25, combined_sentiment=-0.1),
            "CCC": _row(rsi_14=80, combined_sentiment=0.6),
        }
        rows = self._run_with(
            fixtures,
            filters=[
                Filter("rsi_14", "<", 30),
                Filter("combined_sentiment", ">", 0.0),
            ],
        )
        assert [r["symbol"] for r in rows] == ["AAA"]

    def test_limit_respected(self):
        fixtures = {
            f"S{i}": _row(rsi_14=i, volume_ratio=1.5) for i in range(10)
        }
        rows = self._run_with(
            fixtures,
            filters=[Filter("rsi_14", "<", 100)],
            limit=3,
        )
        assert len(rows) == 3

    def test_no_matches_returns_empty(self):
        fixtures = {"AAA": _row(rsi_14=80)}
        rows = self._run_with(
            fixtures,
            filters=[Filter("rsi_14", "<", 30)],
        )
        assert rows == []

    def test_result_payload_shape(self):
        fixtures = {"AAA": _row(close=42.5, rsi_14=20, combined_sentiment=0.4)}
        rows = self._run_with(
            fixtures,
            filters=[Filter("rsi_14", "<", 30)],
        )
        assert rows[0]["symbol"] == "AAA"
        assert rows[0]["close"]  == 42.5
        assert rows[0]["matched"] == {"rsi_14": 20.0}
        assert "combined_sentiment" in rows[0]["extras"]


class TestFieldCatalogue:

    def test_list_fields_non_empty(self):
        opts = list_fields()
        assert len(opts) == len(SCREENER_FIELDS)
        for o in opts:
            assert {"value", "label", "group"} <= set(o.keys())
