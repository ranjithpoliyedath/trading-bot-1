"""
bot/patterns.py
----------------
Breakout pattern detectors for Qullamaggie and VCP-style setups.

These functions take a daily OHLCV+features DataFrame and return the
same DataFrame with extra columns appended:

  prior_runup_pct        — % gain over the last RUNUP_LOOKBACK days
  consolidation_range    — (high-low)/close over the last CONSOL_LEN days
  consolidation_vol_drop — recent / longer-term volume ratio (<1 = drying up)
  contraction_count      — number of progressively smaller pullbacks in
                           the last VCP_LOOKBACK bars
  pivot_high             — recent N-day high (consolidation top)
  breakout_today         — bool, today's close > pivot on >1.5× avg volume
  qullamaggie_setup      — bool, runup + tight consol + breakout
  vcp_setup              — bool, contraction count >= 2 + tight base + breakout

These columns are then read by the rule-based models in
bot/models/builtin/qullamaggie_v1.py and vcp_v1.py.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Tunables ────────────────────────────────────────────────────────────────

RUNUP_LOOKBACK = 60
RUNUP_MIN_PCT  = 0.30      # +30% prior move

CONSOL_LEN     = 20
CONSOL_MAX_RNG = 0.15      # high-low range / close <= 15%
CONSOL_VOL_DROP_MAX = 0.85 # recent volume <= 85% of longer-term avg

VCP_LOOKBACK   = 90
VCP_MIN_CONTRACTIONS = 2

BREAKOUT_VOL_MULTIPLE = 1.5


# ── Internals ───────────────────────────────────────────────────────────────

def _rolling_max(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(2, n // 3)).max()


def _rolling_min(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(2, n // 3)).min()


def _count_contractions(highs: np.ndarray, lows: np.ndarray) -> int:
    """
    Walk backwards through pivots and count progressively smaller pullbacks.

    A "pullback" is a swing from a local high to a local low.  We compare
    consecutive pullback magnitudes; each one must be strictly smaller
    than the previous to count as a contraction.
    """
    if len(highs) < 4:
        return 0

    # Identify simple pivot highs/lows: local max/min on a 5-bar window.
    pivots: list[tuple[int, float, str]] = []
    n = len(highs)
    for i in range(2, n - 2):
        if highs[i] >= highs[i-1] and highs[i] >= highs[i-2] \
                and highs[i] >= highs[i+1] and highs[i] >= highs[i+2]:
            pivots.append((i, float(highs[i]), "H"))
        elif lows[i] <= lows[i-1] and lows[i] <= lows[i-2] \
                and lows[i] <= lows[i+1] and lows[i] <= lows[i+2]:
            pivots.append((i, float(lows[i]), "L"))

    # Reduce to alternating H/L sequence
    cleaned: list[tuple[int, float, str]] = []
    for p in pivots:
        if not cleaned or cleaned[-1][2] != p[2]:
            cleaned.append(p)
        else:
            # Keep the more extreme pivot
            if p[2] == "H" and p[1] > cleaned[-1][1]:
                cleaned[-1] = p
            elif p[2] == "L" and p[1] < cleaned[-1][1]:
                cleaned[-1] = p

    # Compute swing magnitudes (H→L pullbacks)
    swings: list[float] = []
    for i in range(1, len(cleaned)):
        prev, cur = cleaned[i-1], cleaned[i]
        if prev[2] == "H" and cur[2] == "L":
            swings.append(abs(prev[1] - cur[1]) / prev[1])

    # Walk swings from latest backwards, counting strictly decreasing ones
    count = 0
    for i in range(len(swings) - 1, 0, -1):
        if swings[i] < swings[i - 1]:
            count += 1
        else:
            break
    return count


# ── Public API ──────────────────────────────────────────────────────────────

def add_breakout_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append breakout / VCP columns to a feature DataFrame.

    Required input columns: high, low, close, volume.  The function is
    a no-op (returns input unchanged) if any are missing.
    """
    needed = {"high", "low", "close", "volume"}
    if not needed.issubset(df.columns):
        logger.debug("breakout features skipped — missing %s",
                     needed - set(df.columns))
        return df

    out = df.copy()

    close = out["close"]
    high  = out["high"]
    low   = out["low"]
    vol   = out["volume"]

    # Prior run-up
    out["prior_runup_pct"] = close / close.shift(RUNUP_LOOKBACK) - 1.0

    # Consolidation tightness
    consol_high = _rolling_max(high, CONSOL_LEN)
    consol_low  = _rolling_min(low,  CONSOL_LEN)
    out["consolidation_range"] = (consol_high - consol_low) / close

    # Volume dry-up
    recent_vol = vol.rolling(CONSOL_LEN, min_periods=5).mean()
    long_vol   = vol.rolling(CONSOL_LEN * 3, min_periods=15).mean()
    out["consolidation_vol_drop"] = recent_vol / long_vol

    # Pivot high used for breakouts (yesterday's rolling max so today
    # crossing it is a fresh signal)
    out["pivot_high"] = _rolling_max(high, CONSOL_LEN).shift(1)
    avg_vol = vol.rolling(CONSOL_LEN, min_periods=5).mean()
    out["breakout_today"] = (
        (close > out["pivot_high"])
        & (vol > avg_vol * BREAKOUT_VOL_MULTIPLE)
    ).fillna(False)

    # Contraction count over VCP_LOOKBACK
    contractions = np.zeros(len(out), dtype=int)
    h_arr = high.to_numpy()
    l_arr = low.to_numpy()
    for i in range(len(out)):
        start = max(0, i - VCP_LOOKBACK)
        contractions[i] = _count_contractions(h_arr[start:i+1], l_arr[start:i+1])
    out["contraction_count"] = contractions

    # Composite setups
    out["qullamaggie_setup"] = (
        (out["prior_runup_pct"] >= RUNUP_MIN_PCT)
        & (out["consolidation_range"] <= CONSOL_MAX_RNG)
        & (out["consolidation_vol_drop"] <= CONSOL_VOL_DROP_MAX)
        & out["breakout_today"]
    ).fillna(False)

    out["vcp_setup"] = (
        (out["contraction_count"] >= VCP_MIN_CONTRACTIONS)
        & (out["consolidation_range"] <= CONSOL_MAX_RNG)
        & out["breakout_today"]
    ).fillna(False)

    return out
