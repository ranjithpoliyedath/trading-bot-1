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
    # Extended EMAs — used by the screener's "indicator preset"
    # filter packs (e.g. "Above EMA10/20/50", "Bear stack").
    "ema_10",
    "ema_20",
    "ema_50",
    "ema_200",
    # Pre-computed boolean/numeric helpers so the screener can
    # filter without manually combining columns.  All are 0/1.
    "above_ema_10",
    "above_ema_20",
    "above_ema_50",
    "above_ema_200",
    "above_ema_10_20",       # above both 10 and 20
    "above_ema_10_20_50",    # above 10, 20, AND 50
    "above_all_emas",        # above 10, 20, 50, 200
    "below_all_emas",        # below 10, 20, 50, 200
    "ema_bull_stack",        # close > ema10 > ema20 > ema50 > ema200
    "ema_bear_stack",        # close < ema10 < ema20 < ema50 < ema200
    # ── 2026-05-01: "Best Winners" momentum/quality filter set ──
    # Short-warmup columns are kept in the canonical FEATURE_COLUMNS
    # list (tested for "no NaN after dropna").  The long-warmup ones
    # (perf_3m: 63 bars, perf_6m: 126, perf_1y: 252, pct_from_52w_low:
    # 20–252) are emitted by the helper but NOT listed here — adding
    # them would force the warm-up dropna to discard 252 rows from
    # every parquet, killing recent IPOs.  They're still surfaced
    # through SCREENER_FIELDS so the screener / "Best Winners" preset
    # finds them on disk.
    "ema_60",                # extra EMA used by some momentum filters
    "dollar_volume_30d",     # close × 30-day mean volume (USD turnover)
    "dollar_volume_today",   # close × today's volume
    "adr_pct",               # Average Daily Range over 14d as % of close
    # Sentiment features — added by sentiment pipeline
    "news_sentiment_mean",
    "news_sentiment_std",
    "news_count",
    "reddit_sentiment_mean",
    "reddit_sentiment_std",
    "reddit_score_sum",
    "combined_sentiment",
    "sentiment_momentum",
    "sentiment_change",
    "sentiment_accel",
]

# Core technical columns only — used for dropna (sentiment cols may be 0 legitimately)
TECHNICAL_COLUMNS = FEATURE_COLUMNS[:17]


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
    df = _add_ema(df, spans=[9, 10, 20, 21, 50, 60, 200])
    df = _add_ema_relations(df)
    df = _add_atr(df, period=14)
    df = _add_volume_ratio(df, period=20)
    df = _add_price_changes(df)
    df = _add_ratio_features(df)
    df = _add_winners_features(df)

    before = len(df)
    df.dropna(subset=TECHNICAL_COLUMNS, inplace=True)
    dropped = before - len(df)

    if dropped:
        logger.debug("Dropped %d warm-up rows after feature engineering.", dropped)

    # Fill sentiment columns with 0 if not yet collected
    try:
        from bot.sentiment.sentiment_features import SENTIMENT_COLUMNS, add_sentiment_momentum
        for col in SENTIMENT_COLUMNS + ["sentiment_momentum", "sentiment_change", "sentiment_accel"]:
            if col not in df.columns:
                df[col] = 0.0
        df = add_sentiment_momentum(df)
    except Exception as exc:
        logger.debug("Sentiment features skipped: %s", exc)

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


def _add_ema_relations(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean / numeric helper columns describing the relationship
    between price and the EMA stack.  These let the screener filter
    on common patterns ("above 10, 20 AND 50", "bull stack", etc.)
    without forcing the user to chain multiple raw filter rows.

    All outputs are 0/1 (so screener `>0` and `==1` work out of the
    box).  When an EMA hasn't warmed up yet (NaN), the corresponding
    relation is NaN — filters skip those rows naturally.
    """
    c   = df["close"]
    e10  = df.get("ema_10")
    e20  = df.get("ema_20")
    e50  = df.get("ema_50")
    e200 = df.get("ema_200")

    if e10 is not None:
        df["above_ema_10"]  = (c > e10).astype("float64").where(e10.notna())
    if e20 is not None:
        df["above_ema_20"]  = (c > e20).astype("float64").where(e20.notna())
    if e50 is not None:
        df["above_ema_50"]  = (c > e50).astype("float64").where(e50.notna())
    if e200 is not None:
        df["above_ema_200"] = (c > e200).astype("float64").where(e200.notna())

    if e10 is not None and e20 is not None:
        df["above_ema_10_20"] = (
            (c > e10) & (c > e20)
        ).astype("float64").where(e10.notna() & e20.notna())

    if e10 is not None and e20 is not None and e50 is not None:
        df["above_ema_10_20_50"] = (
            (c > e10) & (c > e20) & (c > e50)
        ).astype("float64").where(e10.notna() & e20.notna() & e50.notna())

    if all(s is not None for s in (e10, e20, e50, e200)):
        all_present = e10.notna() & e20.notna() & e50.notna() & e200.notna()
        df["above_all_emas"] = (
            (c > e10) & (c > e20) & (c > e50) & (c > e200)
        ).astype("float64").where(all_present)
        df["below_all_emas"] = (
            (c < e10) & (c < e20) & (c < e50) & (c < e200)
        ).astype("float64").where(all_present)
        df["ema_bull_stack"] = (
            (c > e10) & (e10 > e20) & (e20 > e50) & (e50 > e200)
        ).astype("float64").where(all_present)
        df["ema_bear_stack"] = (
            (c < e10) & (e10 < e20) & (e20 < e50) & (e50 < e200)
        ).astype("float64").where(all_present)

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


def _add_winners_features(df: pd.DataFrame) -> pd.DataFrame:
    """Multi-period returns + dollar-volume + 52-week-low distance +
    ADR%.  These power the "Best Winners" momentum filter preset
    (US momentum-screener heuristic).

    Trading-day approximations:
      • 3 months  ≈ 63 bars
      • 6 months  ≈ 126 bars
      • 1 year    ≈ 252 bars
      • 52 weeks  = 252 bars

    All outputs are dimensionless ratios (or USD turnover for the
    dollar-volume columns) so screener thresholds work across
    symbols of different price levels.
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # Multi-period total return (used for Perf 3M/6M/1Y filters)
    if "perf_3m" not in df.columns:
        df["perf_3m"] = close.pct_change(63)
    if "perf_6m" not in df.columns:
        df["perf_6m"] = close.pct_change(126)
    if "perf_1y" not in df.columns:
        df["perf_1y"] = close.pct_change(252)

    # Distance above the 52-week low (>=0.70 means "70%+ off the low").
    # Uses min_periods=1 so newly-listed symbols still get a sensible
    # value once they have any history; the 52w window simply hasn't
    # filled yet.
    if "pct_from_52w_low" not in df.columns:
        low_52w = low.rolling(252, min_periods=20).min()
        df["pct_from_52w_low"] = (close - low_52w) / low_52w

    # Dollar volume — USD turnover.  Used to filter for liquidity
    # ("Price * Avg Vol 30D > 15M USD" type rules).
    if "dollar_volume_30d" not in df.columns:
        avg_vol_30d = volume.rolling(30, min_periods=10).mean()
        df["dollar_volume_30d"] = close * avg_vol_30d
    if "dollar_volume_today" not in df.columns:
        df["dollar_volume_today"] = close * volume

    # ADR% — Average Daily Range as % of close.  A volatility/edge
    # measure that some momentum screeners use as a "moves enough to
    # be worth trading" gate.  4-5% is the typical floor for
    # day-trading strategies on US large/mid-caps.
    if "adr_pct" not in df.columns:
        daily_range_pct = (high - low) / close
        df["adr_pct"]  = daily_range_pct.rolling(14, min_periods=7).mean() * 100.0

    return df
