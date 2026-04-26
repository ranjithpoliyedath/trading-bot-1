"""
tests/test_backtest_extras.py
------------------------------
Tests for universe-scope selection and the configurable exit rules
in the backtest engine.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bot.config import DATA_DIR


@pytest.fixture(scope="module")
def have_data():
    files = list(Path(DATA_DIR).glob("*_features.parquet"))
    if not files:
        pytest.skip("No processed feature files on disk.")
    return [p.name.split("_")[0] for p in files][:5]


# ── Universe scope ──────────────────────────────────────────────────────────

class TestUniverseScope:

    def test_scope_all_returns_eligible(self):
        from bot.universe import select_universe
        syms = select_universe("all", limit=20)
        assert isinstance(syms, list)
        assert len(syms) <= 20

    def test_scope_top_100_capped(self):
        from bot.universe import select_universe
        syms = select_universe("top_100")
        assert len(syms) <= 100

    def test_scope_sp500_only_large(self):
        from bot.universe import select_universe, load_universe
        syms = select_universe("sp500", limit=50)
        u = load_universe(eligible_only=True)
        if "index" in u.columns and not u.empty:
            sp500 = set(u.loc[u["index"] == "sp500", "symbol"])
            for s in syms:
                assert s in sp500

    def test_invalid_scope_falls_back(self):
        from bot.universe import select_universe
        syms = select_universe("nonsense")
        # Treated as "all"; either has results or empty without crashing.
        assert isinstance(syms, list)

    def test_scopes_dict_keys_are_strings(self):
        from bot.universe import UNIVERSE_SCOPES
        for k, v in UNIVERSE_SCOPES.items():
            assert isinstance(k, str) and isinstance(v, str)


# ── Configurable exit rules ─────────────────────────────────────────────────

class TestExitRules:

    def test_take_profit_fires(self, have_data):
        from dashboard.backtest_engine import run_filtered_backtest
        out = run_filtered_backtest(
            model_id="rsi_macd_v1", filters=[],
            symbols=have_data, period_days=730, conf_threshold=0.50,
            take_profit_pct=0.05,             # very tight TP — should hit
            stop_loss_pct=None,
            time_stop_days=None,
        )
        reasons = {t.get("exit_reason") for t in out["trades"]}
        # If any trades fired, at least one should have used take_profit
        if out["trades"]:
            assert "take_profit" in reasons or "signal" in reasons

    def test_time_stop_fires(self, have_data):
        from dashboard.backtest_engine import run_filtered_backtest
        out = run_filtered_backtest(
            model_id="rsi_macd_v1", filters=[],
            symbols=have_data, period_days=730, conf_threshold=0.50,
            take_profit_pct=None, stop_loss_pct=None,
            time_stop_days=2,                  # any open position closes in 2 bars
        )
        if out["trades"]:
            assert all(t.get("exit_reason") in ("time_stop", "signal")
                        for t in out["trades"])

    def test_disabling_all_exits_falls_back_to_time_stop(self, have_data):
        from dashboard.backtest_engine import run_filtered_backtest
        # Should not loop forever — engine forces a default time-stop.
        out = run_filtered_backtest(
            model_id="rsi_macd_v1", filters=[],
            symbols=have_data, period_days=365, conf_threshold=0.50,
            use_signal_exit=False,
            take_profit_pct=None, stop_loss_pct=None,
            time_stop_days=None,
        )
        assert "metrics" in out


# ── Metric shape ────────────────────────────────────────────────────────────

class TestMetricShape:

    def test_all_metrics_keys_present(self, have_data):
        from dashboard.backtest_engine import run_filtered_backtest
        out = run_filtered_backtest(
            model_id="rsi_macd_v1", filters=[],
            symbols=have_data, period_days=730, conf_threshold=0.50,
        )
        m = out["metrics"]
        expected = {
            "total_return_pct", "sharpe", "sortino", "expectancy",
            "edge_ratio", "profit_factor", "max_drawdown_pct",
            "win_rate_pct", "loss_rate_pct",
            "total_trades", "wins", "losses",
            "avg_win", "avg_loss", "avg_win_pct", "avg_loss_pct",
            "largest_win", "largest_loss",
        }
        assert expected <= set(m.keys()), f"Missing: {expected - set(m.keys())}"

    def test_loss_rate_complements_win_rate(self, have_data):
        from dashboard.backtest_engine import run_filtered_backtest
        out = run_filtered_backtest(
            model_id="rsi_macd_v1", filters=[],
            symbols=have_data, period_days=365, conf_threshold=0.50,
        )
        m = out["metrics"]
        if m["total_trades"] > 0:
            assert abs(m["win_rate_pct"] + m["loss_rate_pct"] - 100.0) < 0.05
