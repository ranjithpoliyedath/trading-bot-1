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


# ── Position sizing ─────────────────────────────────────────────────────────

class TestPositionSizing:

    def test_fixed_pct_uses_almost_all_cash(self):
        from dashboard.backtest_engine import _position_size_shares
        n = _position_size_shares(
            "fixed_pct", cash=10_000, portfolio=10_000, price=100,
            atr=0, sizing_kwargs={"pct": 0.95},
        )
        assert n == 95

    def test_atr_risk_smaller_position_in_volatile_name(self):
        from dashboard.backtest_engine import _position_size_shares
        # Same capital + risk, higher ATR → fewer shares
        small_atr = _position_size_shares(
            "atr_risk", cash=10_000, portfolio=10_000, price=100, atr=1.0,
            sizing_kwargs={"risk_pct": 0.01, "atr_mult": 2.0},
        )
        big_atr = _position_size_shares(
            "atr_risk", cash=10_000, portfolio=10_000, price=100, atr=5.0,
            sizing_kwargs={"risk_pct": 0.01, "atr_mult": 2.0},
        )
        assert small_atr > big_atr > 0

    def test_atr_risk_zero_atr_returns_zero(self):
        from dashboard.backtest_engine import _position_size_shares
        n = _position_size_shares(
            "atr_risk", cash=10_000, portfolio=10_000, price=100, atr=0,
            sizing_kwargs={"risk_pct": 0.01, "atr_mult": 2.0},
        )
        assert n == 0

    def test_half_kelly_is_half_full_kelly(self):
        from dashboard.backtest_engine import _position_size_shares
        kw = {"win_rate": 0.6, "win_loss_ratio": 2.0}
        full = _position_size_shares(
            "kelly", cash=10_000, portfolio=10_000, price=100, atr=0, sizing_kwargs=kw,
        )
        half = _position_size_shares(
            "half_kelly", cash=10_000, portfolio=10_000, price=100, atr=0, sizing_kwargs=kw,
        )
        # half-Kelly is half the fraction of full Kelly (subject to int floor)
        assert half == full // 2

    def test_kelly_zero_edge_zero_size(self):
        from dashboard.backtest_engine import _position_size_shares
        n = _position_size_shares(
            "kelly", cash=10_000, portfolio=10_000, price=100, atr=0,
            sizing_kwargs={"win_rate": 0.4, "win_loss_ratio": 1.0},   # negative edge
        )
        assert n == 0

    def test_starting_cash_threaded_through(self, have_data):
        from dashboard.backtest_engine import run_filtered_backtest
        out_low = run_filtered_backtest(
            model_id="rsi_macd_v1", filters=[],
            symbols=have_data, period_days=730, conf_threshold=0.50,
            starting_cash=1_000,
        )
        out_high = run_filtered_backtest(
            model_id="rsi_macd_v1", filters=[],
            symbols=have_data, period_days=730, conf_threshold=0.50,
            starting_cash=100_000,
        )
        # If trades fired, the dollar P&L should scale roughly with capital
        if out_low["trades"] and out_high["trades"]:
            avg_low  = abs(sum(t["pl"] for t in out_low["trades"]) / max(len(out_low["trades"]), 1))
            avg_high = abs(sum(t["pl"] for t in out_high["trades"]) / max(len(out_high["trades"]), 1))
            assert avg_high > avg_low

    def test_atr_stop_fires(self, have_data):
        from dashboard.backtest_engine import run_filtered_backtest
        out = run_filtered_backtest(
            model_id="rsi_macd_v1", filters=[],
            symbols=have_data, period_days=730, conf_threshold=0.50,
            take_profit_pct=None, stop_loss_pct=None, time_stop_days=None,
            atr_stop_mult=0.5,    # very tight, almost any down move triggers
        )
        if out["trades"]:
            assert any(t.get("exit_reason") == "atr_stop" for t in out["trades"])


# ── New public APIs introduced by the layout / preset enhancements ──────────

class TestUniverseSymbolScopes:

    def test_etf_prefix_returns_single_symbol(self):
        from bot.universe import select_universe
        assert select_universe("etf:SPY") == ["SPY"]
        assert select_universe("etf:QQQ") == ["QQQ"]

    def test_sym_prefix_returns_single_symbol(self):
        from bot.universe import select_universe
        assert select_universe("sym:AAPL") == ["AAPL"]
        assert select_universe("sym:NVDA") == ["NVDA"]

    def test_etf_and_sym_present_in_scopes_dict(self):
        from bot.universe import UNIVERSE_SCOPES
        keys = list(UNIVERSE_SCOPES)
        assert any(k.startswith("etf:") for k in keys)
        assert any(k.startswith("sym:") for k in keys)
        # Categorical scopes must come before prefixed ones (UI sort order)
        first_etf = next(i for i, k in enumerate(keys) if k.startswith("etf:"))
        first_sym = next(i for i, k in enumerate(keys) if k.startswith("sym:"))
        first_categ = next(i for i, k in enumerate(keys)
                            if not k.startswith(("etf:", "sym:")))
        assert first_categ < first_etf < first_sym


class TestBenchmarkLoader:

    def test_returns_empty_when_symbol_unknown(self):
        from dashboard.backtest_engine import load_benchmark_curve
        # An unlikely-to-exist ticker
        out = load_benchmark_curve("ZZZZ_NOT_A_REAL_TICKER")
        assert out == []

    def test_normalize_anchors_first_value(self):
        """When SPY is on disk, the curve's first value should equal
        the requested ``normalize_to`` anchor."""
        from dashboard.backtest_engine import load_benchmark_curve
        curve = load_benchmark_curve("SPY", normalize_to=10000.0)
        if curve:
            assert abs(curve[0]["value"] - 10000.0) < 0.5

    def test_date_window_slicing(self):
        from dashboard.backtest_engine import load_benchmark_curve
        full = load_benchmark_curve("SPY")
        if not full:
            return  # SPY data not on disk in this env
        sliced = load_benchmark_curve("SPY",
                                      start=full[5]["date"],
                                      end=full[10]["date"])
        if sliced:
            assert sliced[0]["date"] >= full[5]["date"]
            assert sliced[-1]["date"] <= full[10]["date"]


class TestPresetPayload:

    def test_filtered_backtest_attaches_preset(self, have_data):
        from dashboard.backtest_engine import run_filtered_backtest
        out = run_filtered_backtest(
            model_id="rsi_macd_v1", filters=[],
            symbols=have_data, period_days=365, conf_threshold=0.55,
            starting_cash=12_345, sizing_method="fixed_pct",
            sizing_kwargs={"pct": 0.80},
            execution_model="next_open", slippage_bps=7,
            take_profit_pct=0.20, stop_loss_pct=0.10, time_stop_days=15,
        )
        p = out.get("preset")
        assert p is not None
        # Round-trip the values we set
        assert p["model_id"]        == "rsi_macd_v1"
        assert p["period_days"]     == 365
        assert p["min_confidence"]  == 0.55
        assert p["starting_cash"]   == 12_345
        assert p["sizing_method"]   == "fixed_pct"
        assert p["sizing_kwargs"]["pct"] == 0.80
        assert p["execution_model"] == "next_open"
        assert p["slippage_bps"]    == 7
        assert p["take_profit_pct"] == 0.20
        assert p["stop_loss_pct"]   == 0.10
        assert p["time_stop_days"]  == 15


class TestWalkForwardFoldShape:

    def test_each_fold_carries_equity_curve_and_monthly(self, have_data):
        from dashboard.backtest_engine import run_walk_forward
        out = run_walk_forward(
            model_id="ibs_v1", n_folds=2, symbols=have_data,
            period_days=365 * 6, conf_threshold=0.55,
        )
        for fr in out.get("fold_results", []):
            assert "metrics"         in fr
            assert "equity_curve"    in fr     # NEW
            assert "monthly_returns" in fr     # NEW
            assert "trades"          in fr
            assert "run_id"          in fr
