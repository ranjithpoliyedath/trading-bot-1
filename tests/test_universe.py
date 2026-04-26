"""
tests/test_universe.py
-----------------------
Unit tests for universe building and filtering.
All external calls (Wikipedia, Alpaca) are mocked.
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from bot.universe                    import _apply_filters, build_universe, MIN_AVG_VOLUME, MIN_PRICE
from bot.scrapers.sp_constituents    import _normalise_columns


# Test fixtures key off the live thresholds so the suite stays green if
# we re-tune MIN_AVG_VOLUME / MIN_PRICE in production.
LOW_VOL  = MIN_AVG_VOLUME  // 2     # below threshold
HIGH_VOL = MIN_AVG_VOLUME  * 100    # well above
LOW_PRICE  = MIN_PRICE - 1.0        # penny
HIGH_PRICE = MIN_PRICE * 36         # comfortably above


# ── Filter tests ──────────────────────────────────────────────────────────────

class TestApplyFilters:

    def test_eligible_when_volume_and_price_pass(self):
        df = pd.DataFrame({
            "symbol":         ["AAPL"],
            "avg_volume_14d": [HIGH_VOL],
            "current_price":  [HIGH_PRICE],
        })
        result = _apply_filters(df)
        assert result["eligible"].iloc[0] == True
        assert result["reason"].iloc[0] == ""

    def test_ineligible_when_volume_below_threshold(self):
        df = pd.DataFrame({
            "symbol":         ["LOWVOL"],
            "avg_volume_14d": [LOW_VOL],
            "current_price":  [HIGH_PRICE],
        })
        result = _apply_filters(df)
        assert result["eligible"].iloc[0] == False
        assert "volume" in result["reason"].iloc[0]

    def test_ineligible_when_penny_stock(self):
        df = pd.DataFrame({
            "symbol":         ["PENNY"],
            "avg_volume_14d": [HIGH_VOL],
            "current_price":  [LOW_PRICE],
        })
        result = _apply_filters(df)
        assert result["eligible"].iloc[0] == False
        assert "price" in result["reason"].iloc[0]

    def test_ineligible_when_no_market_data(self):
        df = pd.DataFrame({
            "symbol":         ["UNKNOWN"],
            "avg_volume_14d": [None],
            "current_price":  [None],
        })
        result = _apply_filters(df)
        assert result["eligible"].iloc[0] == False
        assert "no market data" in result["reason"].iloc[0]

    def test_volume_filter_takes_precedence_over_price(self):
        # If both fail, volume reason should be set first
        df = pd.DataFrame({
            "symbol":         ["BOTH"],
            "avg_volume_14d": [LOW_VOL],
            "current_price":  [LOW_PRICE],
        })
        result = _apply_filters(df)
        assert result["eligible"].iloc[0] == False
        assert "volume" in result["reason"].iloc[0]

    def test_multiple_symbols(self):
        df = pd.DataFrame({
            "symbol":         ["AAPL",   "PENNY",    "LOWVOL", "UNKNOWN"],
            "avg_volume_14d": [HIGH_VOL, HIGH_VOL,   LOW_VOL,  None],
            "current_price":  [HIGH_PRICE, LOW_PRICE, HIGH_PRICE, None],
        })
        result = _apply_filters(df)
        assert result["eligible"].tolist() == [True, False, False, False]


# ── Column normalisation tests ────────────────────────────────────────────────

class TestNormaliseColumns:

    def test_sp500_columns_renamed(self):
        df = pd.DataFrame({
            "Symbol":            ["AAPL"],
            "Security":          ["Apple Inc."],
            "GICS Sector":       ["Information Technology"],
            "GICS Sub-Industry": ["Technology Hardware"],
        })
        result = _normalise_columns(df)
        assert "symbol"       in result.columns
        assert "company"      in result.columns
        assert "sector"       in result.columns
        assert "sub_industry" in result.columns

    def test_missing_columns_added_as_empty(self):
        df = pd.DataFrame({"Symbol": ["AAPL"]})
        result = _normalise_columns(df)
        assert (result["company"]      == "").all()
        assert (result["sector"]       == "").all()
        assert (result["sub_industry"] == "").all()


# ── Build pipeline tests ──────────────────────────────────────────────────────

class TestBuildUniverse:

    @patch("bot.universe._fetch_market_data")
    @patch("bot.universe.fetch_all_constituents")
    def test_build_merges_constituents_and_market_data(
        self, mock_const, mock_market
    ):
        mock_const.return_value = pd.DataFrame({
            "symbol":       ["AAPL", "PENNY"],
            "company":      ["Apple", "Penny Co"],
            "sector":       ["Tech", "Other"],
            "sub_industry": ["", ""],
            "index":        ["sp500", "sp600"],
        })
        mock_market.return_value = pd.DataFrame({
            "symbol":         ["AAPL",   "PENNY"],
            "avg_volume_14d": [HIGH_VOL, HIGH_VOL],
            "current_price":  [HIGH_PRICE, LOW_PRICE],
        })

        result = build_universe(save=False)

        assert len(result) == 2
        assert result.loc[result["symbol"] == "AAPL",  "eligible"].iloc[0]
        assert not result.loc[result["symbol"] == "PENNY", "eligible"].iloc[0]

    @patch("bot.universe.fetch_all_constituents")
    def test_returns_empty_when_no_constituents(self, mock_const):
        mock_const.return_value = pd.DataFrame()
        result = build_universe(save=False)
        assert result.empty
