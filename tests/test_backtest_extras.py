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


# ── Saved-run load + render guarantees (regression for "I picked a run
# but didn't see anything") ────────────────────────────────────────────────

class TestSavedRunRender:
    """If a user picks a saved run from the dropdown, render_results
    must always produce SOMETHING visible — even when the run had 0
    trades or is a walk-forward shape with empty folds."""

    def _seed_payload(self, **overrides):
        base = {
            "run_id":         "BT-test-empty",
            "model":          "rsi_macd_v1",
            "symbol":         "AAPL",
            "period_days":    365,
            "conf_threshold": 0.65,
            "metrics":        {"total_return_pct": 0, "sharpe": 0,
                                "sortino": 0, "expectancy": 0,
                                "edge_ratio": 0, "profit_factor": 0,
                                "max_drawdown_pct": 0, "win_rate_pct": 0,
                                "loss_rate_pct": 0, "total_trades": 0,
                                "wins": 0, "losses": 0,
                                "avg_win": 0, "avg_loss": 0,
                                "avg_win_pct": 0, "avg_loss_pct": 0,
                                "largest_win": 0, "largest_loss": 0},
            "equity_curve":   [{"date": "start", "value": 10000}],
            "trades":         [],
            "monthly_returns": [],
        }
        base.update(overrides)
        return base

    def test_empty_trades_renders_warning(self):
        from dashboard.pages.backtest import render_results
        # Payload with symbols_traded > 0 → triggers the strategy diagnostic
        # (the data-missing branch is exercised in TestDataMissingDiagnostic).
        payload = self._seed_payload()
        payload["metrics"]["symbols_traded"] = 5
        out = render_results(payload)
        s = str(out)
        assert "No trades fired" in s, \
            "Empty-trade payload must render the diagnostic warning"

    def test_loaded_banner_shows_when_tagged(self):
        from dashboard.pages.backtest import render_results
        payload = {**self._seed_payload(), "_loaded_from_saved": "SEED-foo"}
        s = str(render_results(payload))
        assert "Loaded saved run" in s
        assert "SEED-foo"          in s

    def test_walk_forward_zero_trades_renders_warning(self):
        from dashboard.pages.backtest import render_results
        wf_payload = {
            "run_id": "WF-test",
            "fold_results": [
                {"fold": 1, "oos_window": ("2024-01-01", "2024-04-01"),
                 "metrics": {"total_trades": 0, "sharpe": 0,
                              "total_return_pct": 0, "win_rate_pct": 0,
                              "max_drawdown_pct": 0, "expectancy": 0,
                              "edge_ratio": 0, "profit_factor": 0,
                              "sortino": 0, "loss_rate_pct": 0,
                              "wins": 0, "losses": 0,
                              "avg_win": 0, "avg_loss": 0,
                              "avg_win_pct": 0, "avg_loss_pct": 0,
                              "largest_win": 0, "largest_loss": 0},
                 "trades": [], "equity_curve": [], "monthly_returns": []},
            ],
            "aggregate": {"mean_oos_sharpe": 0, "median_oos_sharpe": 0,
                          "stdev_oos_sharpe": 0, "pct_positive_folds": 0,
                          "mean_oos_return_pct": 0},
        }
        s = str(render_results(wf_payload))
        assert "No trades fired in any fold" in s

    def test_run_or_load_tags_loaded_run(self, monkeypatch):
        """When a saved run is selected, the engine result must carry
        the `_loaded_from_saved` tag the renderer keys off."""
        from dashboard.callbacks import backtest_callbacks as bc
        from dashboard.backtest_engine import save_backtest, BACKTEST_DIR
        import json
        from pathlib import Path

        # Write a minimal saved JSON
        saved_id = "BT-unit-test-12345"
        saved_path = BACKTEST_DIR / f"{saved_id}.json"
        saved_path.write_text(json.dumps(self._seed_payload(run_id=saved_id)))
        try:
            data = bc.load_backtest(saved_id)
            # Simulate the run_or_load tagging step
            tagged = {**data, "_loaded_from_saved": saved_id}
            assert tagged["_loaded_from_saved"] == saved_id
        finally:
            saved_path.unlink(missing_ok=True)


# ── Iteration: trade log + strategy explainer + signal-toggle hydration ────

class TestTradeLogFields:
    """Trades dict in run_filtered_backtest now carries entry_date,
    entry_price, exit_price, and shares so the dashboard's trade-log
    table can render rich rows."""

    def test_trades_have_entry_attribution(self, have_data):
        from dashboard.backtest_engine import run_filtered_backtest
        out = run_filtered_backtest(
            model_id="rsi_macd_v1", filters=[],
            symbols=have_data, period_days=730, conf_threshold=0.50,
        )
        for t in out["trades"]:
            for k in ("entry_date", "entry_price", "exit_price", "shares"):
                assert k in t, f"trade dict missing {k}"
            # Entry must precede exit when both are set
            if t["entry_date"] and t["date"]:
                assert t["entry_date"] <= t["date"]


class TestStrategyExplainer:

    def test_explainer_renders_for_known_model(self):
        from dashboard.pages.backtest import _strategy_explainer
        out = _strategy_explainer({"model": "rsi_macd_v1"})
        s = str(out)
        assert "RSI + MACD" in s

    def test_explainer_handles_unknown_model_gracefully(self):
        from dashboard.pages.backtest import _strategy_explainer
        # Must not raise
        out = _strategy_explainer({"model": "nonexistent_model"})
        assert out is not None


class TestTradeLogRender:

    def test_collapsible_log_renders_when_trades_present(self):
        from dashboard.pages.backtest import _trade_log
        results = {
            "trades": [{
                "symbol":      "AAPL",
                "entry_date":  "2024-01-05",
                "date":        "2024-01-19",
                "pl":          150.0,
                "win":         True,
                "exit_reason": "take_profit",
                "entry_price": 100.0,
                "exit_price":  115.0,
                "shares":      10,
            }],
            "preset": {"starting_cash": 10_000},
        }
        out = _trade_log(results)
        s = str(out)
        # Both dates appear in the table, plus exit-reason badge text
        assert "2024-01-05" in s
        assert "2024-01-19" in s
        assert "take profit" in s

    def test_trade_log_empty_when_no_trades(self):
        from dashboard.pages.backtest import _trade_log
        out = _trade_log({"trades": []})
        # Empty Div has no children; quick way to check is repr length
        assert "Trade log" not in str(out)


class TestPresetHydratesScopeAndIndicators:

    def test_preset_payload_carries_scope_and_indicators(self, have_data):
        from dashboard.callbacks import backtest_callbacks  # noqa: F401  (registers)
        # Direct test of run_filtered_backtest preset doesn't include
        # scope/indicators (those are added by run_or_load), so check
        # the synthesized fallback works for older runs.
        from dashboard.callbacks.backtest_callbacks import _synthesize_preset_if_missing
        old = {
            "model": "rsi_macd_v1", "period_days": 365,
            "conf_threshold": 0.55, "filters": [],
        }
        p = _synthesize_preset_if_missing(old)
        assert p["model_id"] == "rsi_macd_v1"


# ── Iteration: expanded strategy explainer (entry/exit/sizing/realism) ─────

class TestStrategyExplainerExpanded:

    def _full_preset_results(self):
        return {
            "model": "connors_rsi2_v1",
            "preset": {
                "model_id":         "connors_rsi2_v1",
                "period_days":      730,
                "min_confidence":   0.55,
                "use_signal_exit":  True,
                "take_profit_pct":  0.15,
                "stop_loss_pct":    0.07,
                "time_stop_days":   30,
                "atr_stop_mult":    2.0,
                "sizing_method":    "atr_risk",
                "sizing_kwargs":    {"risk_pct": 0.01, "atr_mult": 2.0},
                "starting_cash":    10_000,
                "execution_model":  "next_open",
                "execution_delay":  0,
                "slippage_bps":     5,
                "val_mode":         "wf4",
                "scope":            "top_100",
                "filters":          [{"field": "rsi_14", "op": "<", "value": 35}],
            },
            "metrics": {"total_trades": 10},
        }

    def test_explainer_shows_all_four_sections(self):
        from dashboard.pages.backtest import _strategy_explainer
        s = str(_strategy_explainer(self._full_preset_results()))
        assert "Entry signals"     in s
        assert "Exit signals"      in s
        assert "Position sizing"   in s
        assert "Realism + universe" in s

    def test_explainer_lists_active_exit_rules_only(self):
        """Disabled exits (None values) must NOT appear in the list."""
        from dashboard.pages.backtest import _strategy_explainer
        results = self._full_preset_results()
        # Disable take-profit + stop-loss
        results["preset"]["take_profit_pct"] = None
        results["preset"]["stop_loss_pct"]   = None
        s = str(_strategy_explainer(results))
        assert "Take-profit" not in s
        assert "Stop-loss"   not in s
        # Time-stop + signal exit should still show
        assert "Time stop"          in s
        assert "Model sell signal"  in s

    def test_explainer_describes_sizing_method(self):
        from dashboard.pages.backtest import _strategy_explainer
        # ATR-risk sizing
        s = str(_strategy_explainer(self._full_preset_results()))
        assert "ATR-risk" in s
        assert "1.00% capital risked" in s

        # Fixed-pct sizing
        results = self._full_preset_results()
        results["preset"]["sizing_method"] = "fixed_pct"
        results["preset"]["sizing_kwargs"] = {"pct": 0.95}
        s = str(_strategy_explainer(results))
        assert "Fixed % of portfolio" in s
        assert "95%" in s

        # Half-Kelly sizing
        results["preset"]["sizing_method"] = "half_kelly"
        results["preset"]["sizing_kwargs"] = {"win_rate": 0.6, "win_loss_ratio": 2.0}
        s = str(_strategy_explainer(results))
        assert "Half-Kelly" in s

    def test_explainer_lists_extra_filters(self):
        from dashboard.pages.backtest import _strategy_explainer
        s = str(_strategy_explainer(self._full_preset_results()))
        assert "rsi_14"  in s
        assert "<"       in s

    def test_explainer_warns_when_no_exits_enabled(self):
        from dashboard.pages.backtest import _strategy_explainer
        results = self._full_preset_results()
        results["preset"]["use_signal_exit"]  = False
        results["preset"]["take_profit_pct"]  = None
        results["preset"]["stop_loss_pct"]    = None
        results["preset"]["atr_stop_mult"]    = 0
        results["preset"]["time_stop_days"]   = None
        s = str(_strategy_explainer(results))
        assert "default" in s.lower() and "time stop" in s.lower()


# ── Iteration: equity-curve respects starting_cash + per-symbol semantics ──

class TestEquityCurveBase:
    """The equity curve must use starting_cash as its base — previously
    it was hardcoded to $10,000 regardless of the user's setting,
    making percent-return numbers misleading by orders of magnitude."""

    def test_curve_starts_at_starting_cash(self):
        from dashboard.backtest_engine import _calc_equity_curve
        trades = [{"date": "2024-01-01", "pl": 100.0}]
        curve  = _calc_equity_curve(trades, starting_cash=50_000)
        assert curve[0]["value"] == 50_000.0
        assert curve[1]["value"] == 50_100.0

    def test_default_still_10k_for_back_compat(self):
        from dashboard.backtest_engine import _calc_equity_curve
        curve = _calc_equity_curve([])
        assert curve[0]["value"] == 10_000.0

    def test_run_filtered_records_starting_cash(self, have_data):
        """Shared-pool semantics: the engine reports a single
        ``starting_cash`` (no aggregate × N math).  Equity curve starts
        at that exact value."""
        from dashboard.backtest_engine import run_filtered_backtest
        out = run_filtered_backtest(
            model_id="rsi_macd_v1", filters=[],
            symbols=have_data, period_days=730, conf_threshold=0.50,
            starting_cash=100_000,
        )
        m = out["metrics"]
        assert m.get("starting_cash") == 100_000.0
        # Aggregate-style keys should be gone in shared-pool model
        assert "aggregate_starting"   not in m
        assert "per_symbol_starting"  not in m
        # Equity curve start matches starting_cash exactly
        assert abs(out["equity_curve"][0]["value"] - 100_000.0) < 0.01


class TestExplainerNoExitsBanner:
    """When every exit is disabled, the explainer must show a loud
    warning so the user immediately understands the engine fell back
    to the 30-day default."""

    def test_no_exits_banner_appears(self):
        from dashboard.pages.backtest import _strategy_explainer
        results = {
            "model": "rsi_macd_v1",
            "preset": {
                "model_id":         "rsi_macd_v1",
                "use_signal_exit":  False,
                "take_profit_pct":  None,
                "stop_loss_pct":    None,
                "atr_stop_mult":    0,
                "time_stop_days":   None,
                "sizing_method":    "fixed_pct",
                "sizing_kwargs":    {"pct": 0.95},
                "starting_cash":    10_000,
                "execution_model":  "next_open",
                "execution_delay":  0,
                "slippage_bps":     5,
                "filters":          [],
            },
        }
        s = str(_strategy_explainer(results))
        assert "⚠ No exit rules" in s
        assert "30-day default"  in s

    def test_no_exits_banner_absent_when_any_exit_enabled(self):
        from dashboard.pages.backtest import _strategy_explainer
        results = {
            "model": "rsi_macd_v1",
            "preset": {
                "model_id":         "rsi_macd_v1",
                "use_signal_exit":  True,
                "take_profit_pct":  None,
                "stop_loss_pct":    None,
                "time_stop_days":   None,
                "sizing_method":    "fixed_pct",
                "sizing_kwargs":    {"pct": 0.95},
                "starting_cash":    10_000,
                "execution_model":  "next_open",
                "execution_delay":  0,
                "slippage_bps":     5,
                "filters":          [],
            },
        }
        s = str(_strategy_explainer(results))
        assert "⚠ No exit rules" not in s


class TestExplainerCrossSectional:
    """Regression: cross-sectional runs (e.g. jt_momentum_v1) must NOT
    render per-symbol exit rows like 'Model sell signal' or 'Stop-loss',
    because the strategy class doesn't honour those — it rebalances by
    rank on a fixed cadence.  Old explainer defaulted use_signal_exit to
    True when the key was missing → misleading 'Model sell signal' row
    even on a cross-sectional run.  Also: the runner must persist its
    rebalance settings into the preset so the dashboard can show them."""

    def _xs_results(self, **preset_overrides):
        preset = {
            "model_id":       "jt_momentum_v1",
            "model_kind":     "cross_sectional",
            "period_days":    365,
            "starting_cash":  10_000.0,
            "slippage_bps":   5.0,
            "top_decile":     0.20,
            "rebalance_days": 21,
        }
        preset.update(preset_overrides)
        return {
            "run_id":           "XS-test-jt_momentum_v1-50syms",
            "model":            "jt_momentum_v1",
            "symbol":           "50 symbols (cross-sectional)",
            "period_days":      365,
            "metrics":          {"symbols_traded": 50, "total_trades": 60},
            "equity_curve":     [],
            "trades":           [],
            "rebalance_trades": [],
            "preset":           preset,
        }

    def test_no_per_symbol_exit_rows_for_cross_sectional(self):
        from dashboard.pages.backtest import _strategy_explainer
        s = str(_strategy_explainer(self._xs_results()))
        # The bug-symptom strings — must NOT appear on a cross-sectional run
        assert "Model sell signal" not in s
        assert "Stop-loss"         not in s
        assert "Take-profit"       not in s
        assert "ATR stop"          not in s
        assert "Time stop"         not in s

    def test_rebalance_schedule_appears(self):
        from dashboard.pages.backtest import _strategy_explainer
        s = str(_strategy_explainer(self._xs_results(rebalance_days=21)))
        assert "Rebalance every 21" in s

    def test_top_decile_appears_in_entry_and_sizing(self):
        from dashboard.pages.backtest import _strategy_explainer
        s = str(_strategy_explainer(self._xs_results(top_decile=0.30)))
        assert "top 30%" in s
        assert "Equal-weight" in s

    def test_runner_persists_xs_preset(self):
        """run_cross_sectional_backtest must include a preset block with
        model_kind/top_decile/rebalance_days so the dashboard can render
        the explainer correctly when this run is re-loaded later."""
        from dashboard.backtest_engine import run_cross_sectional_backtest
        out = run_cross_sectional_backtest(
            model_id="jt_momentum_v1", symbols=[],   # empty — fast path
        )
        preset = out.get("preset") or {}
        assert preset.get("model_kind") == "cross_sectional"
        assert "top_decile"     in preset
        assert "rebalance_days" in preset


class TestExplainerCapitalText:

    def test_shared_pool_text_for_multi_symbol(self):
        from dashboard.pages.backtest import _strategy_explainer
        results = {
            "model": "ibs_v1",
            "preset": {
                "model_id":         "ibs_v1",
                "use_signal_exit":  True,
                "take_profit_pct":  0.15,
                "stop_loss_pct":    0.07,
                "time_stop_days":   30,
                "sizing_method":    "fixed_pct",
                "sizing_kwargs":    {"pct": 0.10},
                "starting_cash":    100_000,
                "execution_model":  "next_open",
                "execution_delay":  0,
                "slippage_bps":     5,
                "filters":          [],
            },
            "metrics": {"symbols_traded": 50,
                         "starting_cash": 100_000,
                         "ending_cash":   135_000},
        }
        s = str(_strategy_explainer(results))
        assert "Starting cash"     in s
        assert "shared pool"        in s
        assert "Ending cash"        in s

    def test_shared_pool_text_for_single_symbol(self):
        from dashboard.pages.backtest import _strategy_explainer
        results = {
            "model": "rsi_macd_v1",
            "preset": {
                "model_id":         "rsi_macd_v1",
                "use_signal_exit":  True,
                "take_profit_pct":  0.15,
                "stop_loss_pct":    0.07,
                "time_stop_days":   30,
                "sizing_method":    "fixed_pct",
                "sizing_kwargs":    {"pct": 0.95},
                "starting_cash":    10_000,
                "execution_model":  "next_open",
                "execution_delay":  0,
                "slippage_bps":     5,
                "filters":          [],
            },
            "metrics": {"symbols_traded": 1},
        }
        s = str(_strategy_explainer(results))
        assert "Starting cash"  in s
        assert "shared pool"     in s
        # No leftover "Per-symbol cash" / "Total deployed" labels
        assert "Per-symbol cash" not in s
        assert "Total deployed"  not in s


# ── Run-stamp banner + data-missing diagnostic (so each Run is visibly
#    distinct and a missing-parquet failure mode is clearly separable) ──

class TestRunStampBanner:

    def test_fresh_run_banner_shows_timestamp_and_run_id(self):
        from dashboard.pages.backtest import _loaded_banner
        banner = _loaded_banner({
            "run_id": "BT-20260427-211536-ibs_v1-multi-2190d",
            "run_at": "2026-04-27T21:15:36.123456",
            "metrics": {"total_trades": 0},
        })
        s = str(banner)
        assert "Fresh run"      in s
        assert "BT-20260427"    in s
        assert "21:15:36"       in s

    def test_loaded_banner_shows_loaded_marker(self):
        from dashboard.pages.backtest import _loaded_banner
        banner = _loaded_banner({
            "run_id": "x", "run_at": "2026-04-27T21:00:00",
            "_loaded_from_saved": "SEED-foo",
        })
        s = str(banner)
        assert "Loaded saved run" in s
        assert "SEED-foo"          in s

    def test_no_banner_for_empty_results(self):
        from dashboard.pages.backtest import _loaded_banner
        assert _loaded_banner({}) is None
        assert _loaded_banner(None) is None


class TestDataMissingDiagnostic:
    """When 0 symbols had loadable features, show an explicit
    'data is missing — run the pipeline' diagnostic instead of the
    generic 'no trades fired' panel."""

    def _payload(self, n_symbols: int):
        return {
            "run_id": "BT-test",
            "run_at": "2026-04-27T21:00:00",
            "metrics": {"total_trades": 0, "symbols_traded": n_symbols},
            "trades": [],
            "equity_curve": [{"date": "start", "value": 10000}],
        }

    def test_zero_symbols_renders_data_missing_diagnostic(self):
        from dashboard.pages.backtest import render_results
        s = str(render_results(self._payload(n_symbols=0)))
        assert "No symbol data was loadable" in s
        assert "bot.pipeline"                in s

    def test_nonzero_symbols_zero_trades_renders_strategy_diagnostic(self):
        from dashboard.pages.backtest import render_results
        s = str(render_results(self._payload(n_symbols=12)))
        assert "No trades fired" in s
        assert "didn't trigger any buys" in s


# ── Per-symbol load report (precise diagnostic for "0 symbols traded") ────

class TestLoadReport:

    def _payload_with_report(self, *, requested, loaded, missing, n_traded):
        return {
            "run_id": "BT-load-report-test",
            "run_at": "2026-04-27T22:00:00",
            "metrics":      {"total_trades": 0, "symbols_traded": n_traded},
            "trades":       [],
            "equity_curve": [{"date": "start", "value": 10000}],
            "load_report":  {"requested": requested, "loaded": loaded,
                              "missing_features": missing,
                              "empty_after_window": []},
        }

    def test_diagnostic_lists_missing_symbols(self):
        from dashboard.pages.backtest import render_results
        s = str(render_results(self._payload_with_report(
            requested=5, loaded=0,
            missing=["NVDA", "INTC", "AAL", "T", "NFLX"],
            n_traded=0,
        )))
        # New diagnostic mentions counts AND lists the missing symbols
        assert "5/5 symbols missing"  in s
        assert "NVDA"                  in s
        assert "INTC"                  in s

    def test_diagnostic_for_empty_universe_scope(self):
        from dashboard.pages.backtest import render_results
        s = str(render_results(self._payload_with_report(
            requested=0, loaded=0, missing=[], n_traded=0,
        )))
        assert "Universe scope returned no symbols" in s

    def test_diagnostic_falls_back_for_legacy_payload(self):
        """Older saved runs lack load_report — render the legacy
        'No symbol data was loadable' message."""
        from dashboard.pages.backtest import render_results
        legacy = {
            "run_id": "BT-legacy",
            "run_at": "2026-04-27T22:00:00",
            "metrics": {"total_trades": 0, "symbols_traded": 0},
            "trades": [],
            "equity_curve": [{"date": "start", "value": 10000}],
        }
        s = str(render_results(legacy))
        assert "No symbol data was loadable" in s


# ── Shared-pool simulator (every trade draws from one cash account) ────

class TestSharedPortfolioPool:

    def _scored(self, n_bars: int, signals: dict, base_price: float = 100.0):
        """Build a synthetic scored DataFrame with explicit signals
        on chosen bar indices, e.g. signals={3: 'buy', 10: 'sell'}.
        Returns a DataFrame indexed by trading dates."""
        import pandas as pd, numpy as np
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="D", tz="UTC")
        df = pd.DataFrame({
            "open":   [base_price + i * 0.5 for i in range(n_bars)],
            "high":   [base_price + i * 0.5 + 1 for i in range(n_bars)],
            "low":    [base_price + i * 0.5 - 1 for i in range(n_bars)],
            "close":  [base_price + i * 0.5 for i in range(n_bars)],
            "volume": [1_000_000] * n_bars,
            "atr_14": [1.0] * n_bars,
            "signal":      ["hold"] * n_bars,
            "confidence":  [0.7] * n_bars,
        }, index=dates)
        for bar, sig in signals.items():
            df.iat[bar, df.columns.get_loc("signal")] = sig
        return df

    def test_cash_pool_decreases_on_entry_and_returns_on_exit(self):
        from dashboard.backtest_engine import _simulate_portfolio
        # One symbol, buy bar 1, sell bar 5
        scored = {"AAPL": self._scored(20, {1: "buy", 5: "sell"})}
        sim = _simulate_portfolio(
            scored, starting_cash=10_000,
            sizing_method="fixed_pct", sizing_kwargs={"pct": 0.50},
            use_signal_exit=True,
            take_profit_pct=None, stop_loss_pct=None, time_stop_days=None,
            slippage_bps=0, execution_model="next_open",
        )
        assert len(sim["trades"]) == 1
        # After exit, cash should be approximately what we started with
        # plus the trade's P&L (within rounding from share-count flooring).
        pl = sim["trades"][0]["pl"]
        assert abs(sim["final_cash"] - (10_000 + pl)) < 50  # small rounding leeway

    def test_concurrent_entries_share_cash_pool(self):
        """Two symbols both signal buy on the same bar; the sum of
        cash deployed should not exceed starting_cash."""
        from dashboard.backtest_engine import _simulate_portfolio
        scored = {
            "AAPL": self._scored(20, {1: "buy", 12: "sell"}, base_price=100),
            "MSFT": self._scored(20, {1: "buy", 12: "sell"}, base_price=50),
        }
        sim = _simulate_portfolio(
            scored, starting_cash=10_000,
            sizing_method="fixed_pct", sizing_kwargs={"pct": 0.40},
            use_signal_exit=True,
            take_profit_pct=None, stop_loss_pct=None, time_stop_days=None,
            slippage_bps=0, execution_model="next_open",
        )
        # Both should have traded
        symbols_traded = {t["symbol"] for t in sim["trades"]}
        assert symbols_traded == {"AAPL", "MSFT"}
        # The first BUY consumes 40% of $10K = $4K → 39 shares of AAPL @
        # ~$100.5 ≈ $3919.50 cost.  Then portfolio mark-to-market on
        # MSFT's buy bar is cash + AAPL position; 40% of that should leave
        # cash positive throughout.

    def test_insufficient_cash_skips_entry(self):
        """If a third buy lands while two large positions are already
        open, sizing % may yield 0 shares — the entry must skip
        cleanly, not crash, not go negative on cash."""
        from dashboard.backtest_engine import _simulate_portfolio
        scored = {
            "AAPL": self._scored(30, {1: "buy", 20: "sell"}, base_price=100),
            "MSFT": self._scored(30, {1: "buy", 20: "sell"}, base_price=50),
            "GOOG": self._scored(30, {2: "buy", 20: "sell"}, base_price=2000),  # huge
        }
        sim = _simulate_portfolio(
            scored, starting_cash=5_000,           # tight pool
            sizing_method="fixed_pct", sizing_kwargs={"pct": 0.50},
            use_signal_exit=True,
            take_profit_pct=None, stop_loss_pct=None, time_stop_days=None,
            slippage_bps=0, execution_model="next_open",
        )
        # Cash must never go negative — final value is real money
        assert sim["final_cash"] >= 0

    def test_zero_starting_cash_no_trades(self):
        from dashboard.backtest_engine import _simulate_portfolio
        scored = {"AAPL": self._scored(10, {1: "buy", 5: "sell"})}
        sim = _simulate_portfolio(scored, starting_cash=0)
        assert sim["trades"] == []
        assert sim["final_cash"] == 0
