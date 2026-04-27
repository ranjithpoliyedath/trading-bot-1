"""
bot/indicators.py
------------------
Lazy technical-indicator helpers shared across the strategy library.

Each ``add_*`` function takes a feature DataFrame, appends one or more
indicator columns, and returns it.  All are vectorised over pandas /
numpy — no third-party TA libraries needed.

Strategies in ``bot/models/builtin/`` call these in their
``predict_batch`` so the feature pipeline doesn't have to be re-run
when a new indicator is needed.  The composite ``add_all_indicators``
is the one-call entry point most models use.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Moving averages ─────────────────────────────────────────────────────────

def add_sma(df: pd.DataFrame, periods=(50, 200)) -> pd.DataFrame:
    """Append simple moving averages (sma_50, sma_200, …)."""
    out = df.copy()
    for n in periods:
        col = f"sma_{n}"
        if col not in out.columns:
            out[col] = out["close"].rolling(n, min_periods=max(2, n // 3)).mean()
    return out


# ── ADX (Wilder) ────────────────────────────────────────────────────────────

def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Append ADX, +DI, -DI columns (Wilder's smoothing).
    """
    if {"adx_14", "plus_di_14", "minus_di_14"}.issubset(df.columns):
        return df.copy()

    out = df.copy()
    high = out["high"]; low = out["low"]; close = out["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    up   = high.diff()
    down = -low.diff()
    plus_dm  = np.where((up > down)   & (up > 0),   up,   0.0)
    minus_dm = np.where((down > up)   & (down > 0), down, 0.0)

    # Wilder's smoothing == EMA with alpha = 1/period
    atr      = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm, index=out.index).ewm(
                  alpha=1/period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=out.index).ewm(
                  alpha=1/period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    out[f"adx_{period}"]      = adx
    out[f"plus_di_{period}"]  = plus_di
    out[f"minus_di_{period}"] = minus_di
    return out


# ── Donchian channel ────────────────────────────────────────────────────────

def add_donchian(df: pd.DataFrame, periods=(20, 10)) -> pd.DataFrame:
    """
    Append donchian_high_<n> and donchian_low_<n> for each n in periods.
    Use shifted highs/lows so today's close crossing is a fresh signal.
    """
    out = df.copy()
    for n in periods:
        out[f"donchian_high_{n}"] = out["high"].rolling(n, min_periods=2).max().shift(1)
        out[f"donchian_low_{n}"]  = out["low"].rolling(n,  min_periods=2).min().shift(1)
    return out


# ── Keltner channel (ATR-based) ─────────────────────────────────────────────

def add_keltner(
    df: pd.DataFrame,
    period:     int   = 20,
    multiplier: float = 2.0,
    atr_period: int   = 14,
) -> pd.DataFrame:
    """
    Append keltner_upper / _middle / _lower bands.

    Centerline is an EMA of close over ``period`` bars; bands are
    ``multiplier × ATR(atr_period)``.  Both are configurable so the
    caller can match Chester Keltner's original (period=10, mult=1.5)
    or Linda Raschke's later refinement (period=20, mult=2.0).
    """
    atr_col = f"atr_{atr_period}"
    if atr_col not in df.columns:
        df = add_atr(df, period=atr_period)

    out = df.copy()
    middle = out["close"].ewm(span=period, min_periods=2, adjust=False).mean()
    band   = multiplier * out[atr_col]
    out["keltner_upper"]  = middle + band
    out["keltner_middle"] = middle
    out["keltner_lower"]  = middle - band
    return out


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Defensive: re-compute atr_14 if it's missing."""
    if f"atr_{period}" in df.columns:
        return df.copy()
    out = df.copy()
    high = out["high"]; low = out["low"]; close = out["close"]
    prev = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev).abs(),
        (low  - prev).abs(),
    ], axis=1).max(axis=1)
    out[f"atr_{period}"] = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return out


# ── On-Balance Volume ───────────────────────────────────────────────────────

def add_obv(df: pd.DataFrame, slope_window: int = 20) -> pd.DataFrame:
    """
    Append ``obv`` (cumulative on-balance volume) and
    ``obv_slope_<window>`` (rolling linear-regression slope of OBV
    over the last ``slope_window`` bars).

    For a daily universe of ~500 symbols this is the slowest indicator
    helper but still completes in under 2 s — fine for nightly cron.
    """
    out = df.copy()
    direction = np.sign(out["close"].diff().fillna(0))
    out["obv"] = (direction * out["volume"]).cumsum()

    n_x = np.arange(int(slope_window), dtype=float)
    def _slope(arr: np.ndarray) -> float:
        if len(arr) < 2 or not np.isfinite(arr).all():
            return np.nan
        # Use the slice's actual length (n_x is sized for full windows;
        # rolling.apply hands shorter arrays at the head).
        x = n_x[: len(arr)]
        x_mean, y_mean = x.mean(), arr.mean()
        var_x = float(((x - x_mean) ** 2).sum())
        if var_x == 0:
            return np.nan
        return float(((x - x_mean) * (arr - y_mean)).sum() / var_x)

    out[f"obv_slope_{slope_window}"] = (
        out["obv"].rolling(int(slope_window), min_periods=max(2, slope_window // 3))
                  .apply(_slope, raw=True)
    )
    return out


# ── Internal Bar Strength ───────────────────────────────────────────────────

def add_ibs(df: pd.DataFrame) -> pd.DataFrame:
    """
    IBS = (close - low) / (high - low).  Range [0, 1].  Low IBS = closed
    near today's low → mean-reversion candidate.
    """
    out = df.copy()
    rng = (out["high"] - out["low"]).replace(0, np.nan)
    out["ibs"] = (out["close"] - out["low"]) / rng
    out["ibs"] = out["ibs"].clip(0.0, 1.0)
    return out


# ── RSI(2) (Connors short-period RSI) ──────────────────────────────────────

def add_rsi(df: pd.DataFrame, period: int = 2) -> pd.DataFrame:
    """Standard Wilder RSI but with a configurable period (typically 2 or 14)."""
    col = f"rsi_{period}"
    if col in df.columns:
        return df.copy()
    out = df.copy()
    delta = out["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    out[col] = 100 - (100 / (1 + rs))
    return out


# ── Z-score (mean reversion) ────────────────────────────────────────────────

def add_zscore(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Append zscore_close_<period> = (close - rolling_mean) / rolling_std."""
    out = df.copy()
    mean = out["close"].rolling(period, min_periods=max(2, period // 3)).mean()
    std  = out["close"].rolling(period, min_periods=max(2, period // 3)).std()
    out[f"zscore_close_{period}"] = (out["close"] - mean) / std.replace(0, np.nan)
    return out


# ── Composite ───────────────────────────────────────────────────────────────

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply every indicator in this module — convenient one-call entry
    point for ``predict_batch`` paths that need several at once.
    Existing columns are left untouched (each helper is idempotent).
    """
    df = add_sma(df, periods=(20, 50, 200))
    df = add_adx(df, period=14)
    df = add_donchian(df, periods=(20, 10))
    df = add_keltner(df, period=20, multiplier=2.0)
    df = add_obv(df, slope_window=20)
    df = add_ibs(df)
    df = add_rsi(df, period=2)
    df = add_zscore(df, period=20)
    return df
