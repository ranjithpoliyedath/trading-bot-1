"""
tests/test_feature_engineer.py
-------------------------------
Unit tests for feature_engineer.py.
Uses synthetic OHLCV data — no external API calls.
"""

import numpy as np
import pandas as pd
import pytest

from bot.feature_engineer import (
    FEATURE_COLUMNS,
    add_all_features,
    _add_rsi,
    _add_macd,
    _add_bollinger_bands,
    _add_atr,
    _add_volume_ratio,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """300 rows of synthetic OHLCV data — enough for every warm-up,
    including the 252-bar perf_1y / pct_from_52w_low windows added
    in the 2026-05-01 'Best Winners' feature pack."""
    n = 300
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "open":   close * 0.998,
            "high":   close * 1.005,
            "low":    close * 0.995,
            "close":  close,
            "volume": np.random.randint(500_000, 2_000_000, n).astype(float),
            "vwap":   close * 1.001,
        },
        index=pd.date_range("2023-01-01", periods=n, freq="D"),
    )


def test_all_feature_columns_present(sample_df):
    """add_all_features should produce every column in FEATURE_COLUMNS."""
    result = add_all_features(sample_df)
    for col in FEATURE_COLUMNS:
        assert col in result.columns, f"Missing feature column: {col}"


def test_no_nan_in_features_after_dropna(sample_df):
    """After warm-up rows are dropped, no NaN should remain in feature columns."""
    result = add_all_features(sample_df)
    assert result[FEATURE_COLUMNS].isna().sum().sum() == 0


def test_row_count_reduced_by_warmup(sample_df):
    """Warm-up rows should be dropped — output must be shorter than input."""
    result = add_all_features(sample_df)
    assert len(result) < len(sample_df)


def test_rsi_bounds(sample_df):
    """RSI must always be between 0 and 100."""
    result = _add_rsi(sample_df.copy())
    rsi = result["rsi_14"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_macd_histogram_equals_diff(sample_df):
    """MACD histogram should equal MACD line minus signal line."""
    result = _add_macd(sample_df.copy())
    diff = (result["macd"] - result["macd_signal"]).round(8)
    hist = result["macd_hist"].round(8)
    pd.testing.assert_series_equal(diff, hist, check_names=False)


def test_bollinger_upper_above_lower(sample_df):
    """Bollinger upper band must always be above the lower band."""
    result = _add_bollinger_bands(sample_df.copy())
    valid = result[["bb_upper", "bb_lower"]].dropna()
    assert (valid["bb_upper"] > valid["bb_lower"]).all()


def test_atr_positive(sample_df):
    """ATR must always be positive."""
    result = _add_atr(sample_df.copy())
    atr = result["atr_14"].dropna()
    assert (atr > 0).all()


def test_volume_ratio_mean_near_one(sample_df):
    """Volume ratio relative to its own rolling window should average near 1."""
    result = _add_volume_ratio(sample_df.copy())
    mean_ratio = result["volume_ratio"].dropna().mean()
    assert 0.8 < mean_ratio < 1.2


def test_empty_dataframe_returns_empty():
    """Empty input should return empty output without raising."""
    result = add_all_features(pd.DataFrame())
    assert result.empty
