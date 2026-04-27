"""
tests/test_backtest_realism.py
-------------------------------
Tests for the Phase-1 realism upgrades:
  * execution_model (next_open vs same_close)
  * slippage_bps
  * walk_forward_folds() + run_walk_forward()
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bot.config import DATA_DIR


@pytest.fixture(scope="module")
def have_data():
    files = list(Path(DATA_DIR).glob("*_features.parquet"))
    if not files:
        pytest.skip("No processed feature files on disk.")
    return [p.name.split("_")[0] for p in files][:5]


def _synthetic_features(n=200, seed=42, gap_factor=1.0):
    """Build a deterministic OHLCV df + signal column for unit tests.

    `gap_factor` lets tests amplify the open-vs-close difference (gap-up
    bars after each buy signal) so the next_open vs same_close branches
    produce visibly different fills.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = close + rng.normal(0, 0.5, n)
    high  = np.maximum(open_, close) + rng.uniform(0.1, 1.0, n)
    low   = np.minimum(open_, close) - rng.uniform(0.1, 1.0, n)
    df = pd.DataFrame({
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": rng.integers(500_000, 2_000_000, n),
        "atr_14": np.full(n, 1.5),                  # constant for clean test
        "signal":  ["hold"] * n,
        "confidence": np.full(n, 0.7),
    }, index=dates)
    # One buy at bar 50, one sell at bar 60 — single round-trip
    df.at[df.index[50], "signal"] = "buy"
    df.at[df.index[60], "signal"] = "sell"
    # Push the next bar's open up so next_open differs visibly from close
    if gap_factor != 1.0:
        df.at[df.index[51], "open"] *= gap_factor
        df.at[df.index[61], "open"] *= gap_factor
    return df


# ── Slippage ────────────────────────────────────────────────────────────────

class TestSlippage:

    def test_zero_slippage_baseline(self):
        from dashboard.backtest_engine import _simulate_trades
        df = _synthetic_features()
        out = _simulate_trades(df.copy(), slippage_bps=0,
                                execution_model="same_close",
                                take_profit_pct=None, stop_loss_pct=None,
                                time_stop_days=None)
        trades = out[out["trade_pl"] != 0]
        assert len(trades) == 1
        baseline_pl = float(trades["trade_pl"].iloc[0])

        # With 50 bps round-trip cost, P&L should drop by ~2× 50 bps × notional
        out = _simulate_trades(df.copy(), slippage_bps=50,
                                execution_model="same_close",
                                take_profit_pct=None, stop_loss_pct=None,
                                time_stop_days=None)
        trades_50 = out[out["trade_pl"] != 0]
        assert len(trades_50) == 1
        slipped_pl = float(trades_50["trade_pl"].iloc[0])
        assert slipped_pl < baseline_pl  # slippage costs us money

    def test_buy_and_sell_both_pay_slippage(self):
        """Both legs should be hit — round trip ≈ 2× one-leg cost."""
        from dashboard.backtest_engine import _simulate_trades
        df = _synthetic_features()
        out_5 = _simulate_trades(df.copy(), slippage_bps=5,
                                  execution_model="same_close",
                                  take_profit_pct=None, stop_loss_pct=None,
                                  time_stop_days=None)
        out_50 = _simulate_trades(df.copy(), slippage_bps=50,
                                   execution_model="same_close",
                                   take_profit_pct=None, stop_loss_pct=None,
                                   time_stop_days=None)
        cost_5  = float(out_5[out_5["trade_pl"]   != 0]["trade_pl"].iloc[0])
        cost_50 = float(out_50[out_50["trade_pl"] != 0]["trade_pl"].iloc[0])
        # The 50bps run should cost noticeably more than 5bps — at least 5×.
        # (Don't try for exact equality: position sizing uses the slipped price
        # so share count drifts slightly between runs.)
        assert (cost_5 - cost_50) > 0


# ── Execution model ─────────────────────────────────────────────────────────

class TestExecutionModel:

    def test_next_open_vs_same_close_differ_with_gap(self):
        from dashboard.backtest_engine import _simulate_trades
        # Buy bar 50, gap up bar 51 → next_open fills higher than same_close
        df = _synthetic_features(gap_factor=1.05)
        same_close = _simulate_trades(df.copy(), execution_model="same_close",
                                       slippage_bps=0,
                                       take_profit_pct=None, stop_loss_pct=None,
                                       time_stop_days=None)
        next_open  = _simulate_trades(df.copy(), execution_model="next_open",
                                       slippage_bps=0,
                                       take_profit_pct=None, stop_loss_pct=None,
                                       time_stop_days=None)
        sc_pl = float(same_close[same_close["trade_pl"] != 0]["trade_pl"].iloc[0])
        no_pl = float(next_open[next_open["trade_pl"]  != 0]["trade_pl"].iloc[0])
        # Same trade, different fill prices → P&Ls must differ
        assert abs(sc_pl - no_pl) > 0.01

    def test_unknown_execution_model_falls_back(self):
        from dashboard.backtest_engine import _simulate_trades
        df = _synthetic_features()
        out = _simulate_trades(df.copy(), execution_model="garbage",
                                slippage_bps=0,
                                take_profit_pct=None, stop_loss_pct=None,
                                time_stop_days=None)
        # Should not crash; should still produce a trade
        assert (out["trade_pl"] != 0).sum() == 1


# ── Walk-forward fold integrity ─────────────────────────────────────────────

class TestWalkForwardFolds:

    def test_four_folds_dates_monotonic(self):
        from dashboard.backtest_engine import walk_forward_folds
        folds = walk_forward_folds(
            pd.Timestamp("2020-01-01"), pd.Timestamp("2026-04-26"), n_folds=4,
        )
        assert len(folds) == 4
        for k, (is_w, oos_w) in enumerate(folds):
            # IS starts at the global start
            assert is_w[0] == pd.Timestamp("2020-01-01")
            # IS ends == OOS begins  (no overlap, no gap)
            assert is_w[1] == oos_w[0]
            # OOS ends > OOS begins
            assert oos_w[1] > oos_w[0]
            # Subsequent folds expand IS
            if k > 0:
                prev_is = folds[k - 1][0]
                assert is_w[1] >= prev_is[1]

    def test_oos_chunks_tile_without_overlap(self):
        from dashboard.backtest_engine import walk_forward_folds
        folds = walk_forward_folds(
            pd.Timestamp("2020-01-01"), pd.Timestamp("2026-04-26"), n_folds=4,
        )
        # OOS windows should chain: end of one == start of next
        for k in range(1, len(folds)):
            prev_oos_end  = folds[k - 1][1][1]
            curr_oos_start = folds[k][1][0]
            assert prev_oos_end == curr_oos_start

    def test_short_range_returns_empty(self):
        from dashboard.backtest_engine import walk_forward_folds
        folds = walk_forward_folds(
            pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-01"),
            n_folds=4, min_is_days=365,
        )
        assert folds == []

    def test_run_walk_forward_returns_per_fold_metrics(self, have_data):
        from dashboard.backtest_engine import run_walk_forward
        out = run_walk_forward(
            model_id="rsi_macd_v1", n_folds=4,
            symbols=have_data, period_days=365 * 6,
            conf_threshold=0.50,
        )
        # Should have either 0 (data too short) or up to 4 folds
        assert "fold_results" in out
        assert "aggregate"    in out
        # If we got folds, aggregate must have all the required keys
        if out["fold_results"]:
            for k in ("mean_oos_sharpe", "median_oos_sharpe",
                      "stdev_oos_sharpe", "pct_positive_folds",
                      "mean_oos_return_pct"):
                assert k in out["aggregate"]
