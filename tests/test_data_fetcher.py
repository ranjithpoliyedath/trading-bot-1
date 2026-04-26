"""
tests/test_data_fetcher.py
--------------------------
Unit tests for DataFetcher.
All Alpaca API calls are mocked — no real API is ever called in tests.
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from bot.data_fetcher import DataFetcher


@pytest.fixture
def mock_fetcher():
    """Return a DataFetcher with fake credentials."""
    with patch("bot.data_fetcher.StockHistoricalDataClient"):
        fetcher = DataFetcher(api_key="fake_key", secret_key="fake_secret")
    return fetcher


def _make_bar_df(n: int = 10, symbol: str = "AAPL") -> pd.DataFrame:
    """
    Helper: create a multi-indexed OHLCV DataFrame matching the shape
    Alpaca's BarSet.df returns (MultiIndex of (symbol, timestamp)).
    """
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    idx = pd.MultiIndex.from_product([[symbol], dates],
                                     names=["symbol", "timestamp"])
    return pd.DataFrame(
        {
            "open":   [100.0 + i for i in range(n)],
            "high":   [105.0 + i for i in range(n)],
            "low":    [98.0  + i for i in range(n)],
            "close":  [102.0 + i for i in range(n)],
            "volume": [1_000_000 + i * 1000 for i in range(n)],
            "vwap":   [101.5 + i for i in range(n)],
        },
        index=idx,
    )


def _mock_bars(df: pd.DataFrame):
    """Wrap a DataFrame in a BarSet-shaped mock (exposes .df)."""
    bars = MagicMock()
    bars.df = df
    return bars


def test_fetch_bars_returns_dataframe(mock_fetcher):
    """fetch_bars should return a dict mapping symbol to DataFrame."""
    fake_df = _make_bar_df(10, "AAPL")
    mock_fetcher.client.get_stock_bars = MagicMock(return_value=_mock_bars(fake_df))

    result = mock_fetcher.fetch_bars(symbols=["AAPL"])

    assert "AAPL" in result
    assert isinstance(result["AAPL"], pd.DataFrame)
    assert len(result["AAPL"]) == 10


def test_fetch_bars_missing_symbol_returns_empty(mock_fetcher):
    """fetch_bars should return an empty DataFrame for symbols with no data."""
    empty = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume", "vwap"],
        index=pd.MultiIndex.from_arrays([[], []], names=["symbol", "timestamp"]),
    )
    mock_fetcher.client.get_stock_bars = MagicMock(return_value=_mock_bars(empty))

    result = mock_fetcher.fetch_bars(symbols=["UNKNOWN"])

    assert "UNKNOWN" in result
    assert result["UNKNOWN"].empty


def test_fetch_single_returns_dataframe(mock_fetcher):
    """fetch_single should return a single DataFrame for one symbol."""
    fake_df = _make_bar_df(20, "TSLA")
    mock_fetcher.client.get_stock_bars = MagicMock(return_value=_mock_bars(fake_df))

    result = mock_fetcher.fetch_single("TSLA")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 20


def test_missing_api_key_raises():
    """DataFetcher should raise if API credentials are missing."""
    with patch("bot.data_fetcher.StockHistoricalDataClient"):
        with patch.dict("os.environ", {"ALPACA_API_KEY": "", "ALPACA_SECRET_KEY": ""}):
            with pytest.raises(ValueError, match="Alpaca API key"):
                DataFetcher(api_key=None, secret_key=None)


def test_fetch_bars_raises_on_api_error(mock_fetcher):
    """fetch_bars should raise when Alpaca client throws an exception."""
    mock_fetcher.client.get_stock_bars = MagicMock(
        side_effect=Exception("API error")
    )

    with pytest.raises(Exception, match="API error"):
        mock_fetcher.fetch_bars(symbols=["AAPL"])
