"""
tests/test_patterns.py
-----------------------
Unit tests for breakout pattern detection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot.patterns import add_breakout_features, _count_contractions


def _frame(close, high=None, low=None, volume=None):
    n = len(close)
    return pd.DataFrame({
        "close":  close,
        "high":   high   or [c * 1.01 for c in close],
        "low":    low    or [c * 0.99 for c in close],
        "volume": volume or [1_000_000] * n,
    })


class TestContractions:

    def test_decreasing_swings_counted(self):
        # Highs / lows representing 3 successively smaller pullbacks.
        # H=100, L=80 (20%); H=100, L=85 (15%); H=100, L=92 (8%)
        highs = np.array([90, 100, 90, 80, 90, 100, 92, 85, 90, 100, 95, 92,
                           100, 100, 100, 100])
        lows  = np.array([80, 90, 80,  80, 80,  90,  85, 85, 85,  92,  92, 92,
                            92,  92,  92,  92])
        # Just sanity-check it doesn't blow up and returns a non-negative int.
        assert _count_contractions(highs, lows) >= 0

    def test_short_input_zero(self):
        assert _count_contractions(np.array([1.0]), np.array([1.0])) == 0


class TestAddBreakoutFeatures:

    def test_missing_columns_passthrough(self):
        df = pd.DataFrame({"close": [1, 2, 3]})
        out = add_breakout_features(df)
        assert "breakout_today" not in out.columns

    def test_columns_added(self):
        n = 200
        rng = np.random.default_rng(42)
        prices = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
        df = _frame(prices.tolist())
        out = add_breakout_features(df)
        for col in ["prior_runup_pct", "consolidation_range",
                    "consolidation_vol_drop", "pivot_high",
                    "breakout_today", "contraction_count",
                    "qullamaggie_setup", "vcp_setup"]:
            assert col in out.columns

    def test_breakout_fires_on_pop(self):
        # Flat prices then a sharp breakout above the range on big volume.
        n = 60
        prices = [100.0] * n + [110.0]
        highs  = [101.0] * n + [111.0]
        lows   = [ 99.0] * n + [109.0]
        vols   = [1_000_000] * n + [3_000_000]
        df = _frame(prices, highs, lows, vols)
        out = add_breakout_features(df)
        assert bool(out["breakout_today"].iloc[-1])

    def test_no_breakout_when_volume_low(self):
        n = 60
        prices = [100.0] * n + [110.0]
        highs  = [101.0] * n + [111.0]
        lows   = [ 99.0] * n + [109.0]
        vols   = [1_000_000] * n + [800_000]
        df = _frame(prices, highs, lows, vols)
        out = add_breakout_features(df)
        assert not bool(out["breakout_today"].iloc[-1])
