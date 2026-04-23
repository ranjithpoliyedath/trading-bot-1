"""
bot/feature_engineer.py
-----------------------
Computes technical indicators and ML-ready features from raw OHLCV data.
All features are added as new columns to the input DataFrame.
No external TA library required — all indicators computed with pandas/numpy.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature column names — update here if you add/remove features
FEATURE_COLUMNS = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "bb_pct",
    "ema_9",
    "ema_21",
    "ema_cross",
    "atr_14",
    "volume_ratio",
    "price_change_1d",
    "price_change_5d",
    "high_low_ratio",
    "close_to_vwap",
]


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicator features to a raw OHLCV DataFrame.

    Args:
        df: DataFrame with columns [open, high, low, close, volume, vwap].
            Must be sorted by date ascending with no gaps.

    Returns:
        DataFrame with all FEATURE_COLUMNS added. Rows with NaN features
        (warm-up period) are dropped.
    """
    if df.empty:
        logger.warning("Empty DataFrame passed to feature engineer.")
        return df

    df = df.copy()

    df = _add_rsi(df, period=14)
    df = _add_macd(df)
    df = _add_bollinger_bands(df, period=20)
    df = _add_ema(df, spans=[9, 21])
    df = _add_atr(df, period=14)
    df = _add_volume_ratio(df, period=20)
    df = _add_price_changes(df)
    df = _add_ratio_features(df)

    before = len(df)
    df.dropna(subset=FEATURE_COLUMNS, inplace=True)
    dropped = before - len(df)

    if dropped:
        logger.debug("Dropped %d warm-up rows after feature engineering.", dropped)

    logger.info("Feature engineering complete. %d rows, %d features.", len(df), len(FEATURE_COLUMNS))
    return df


# ── Individual indicator functions ────────────────────────────────────────────

def _add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    return df


def _add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def _add_bollinger_bands(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Bollinger Bands: upper, lower, width, and %B."""
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    df["bb_upper"] = sma + 2 * std
    df["bb_lower"] = sma - 2 * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"]
    ).replace(0, np.nan)
    return df


def _add_ema(df: pd.DataFrame, spans: list[int]) -> pd.DataFrame:
    """Exponential Moving Averages and EMA crossover signal."""
    for span in spans:
        df[f"ema_{span}"] = df["close"].ewm(span=span, adjust=False).mean()
    if 9 in spans and 21 in spans:
        df["ema_cross"] = (df["ema_9"] - df["ema_21"]) / df["close"]
    return df


def _add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range — measures volatility."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f"atr_{period}"] = true_range.ewm(com=period - 1, min_periods=period).mean()
    return df


def _add_volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Volume relative to its rolling average — detects unusual activity."""
    avg_volume = df["volume"].rolling(period).mean()
    df["volume_ratio"] = df["volume"] / avg_volume.replace(0, np.nan)
    return df


def _add_price_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Percentage price changes over 1 and 5 days."""
    df["price_change_1d"] = df["close"].pct_change(1)
    df["price_change_5d"] = df["close"].pct_change(5)
    return df


def _add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """High/low range ratio and close-to-VWAP distance."""
    df["high_low_ratio"] = (df["high"] - df["low"]) / df["close"]
    if "vwap" in df.columns:
        df["close_to_vwap"] = (df["close"] - df["vwap"]) / df["vwap"]
    else:
        df["close_to_vwap"] = np.nan
    return df
