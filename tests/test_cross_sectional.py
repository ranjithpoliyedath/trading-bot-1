"""
tests/test_cross_sectional.py
------------------------------
Tests for the CrossSectionalModel base class, the JT 12-1 momentum
implementation, and the run_cross_sectional_backtest runner.
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
    return [p.name.split("_")[0] for p in files][:8]


# ── Base abstraction ────────────────────────────────────────────────────────

class TestBaseAbstraction:

    def test_cross_sectional_class_imports(self):
        from bot.models.base import CrossSectionalModel
        assert CrossSectionalModel.__name__ == "CrossSectionalModel"

    def test_jt_momentum_registered(self):
        from bot.models.registry import list_models
        ids = {m.id: m for m in list_models()}
        assert "jt_momentum_v1" in ids
        assert ids["jt_momentum_v1"].type == "cross_sectional"

    def test_get_model_returns_cross_sectional_instance(self):
        from bot.models.registry import get_model
        from bot.models.base     import CrossSectionalModel
        m = get_model("jt_momentum_v1")
        assert isinstance(m, CrossSectionalModel)


# ── JT momentum logic ─────────────────────────────────────────────────────

class TestJtMomentum:

    def _panel(self, n=400, n_syms=10, seed=0):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        cols = {f"S{i:02d}": 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
                for i in range(n_syms)}
        return pd.DataFrame(cols, index=dates)

    def test_rank_universe_in_unit_interval(self):
        from bot.models.registry import get_model
        m = get_model("jt_momentum_v1")
        ranks = m.rank_universe(self._panel())
        valid = ranks.dropna()
        if not valid.empty:
            assert (valid.values >= 0).all() and (valid.values <= 1).all()

    def test_top_ranked_is_best_performer(self):
        from bot.models.registry import get_model
        m = get_model("jt_momentum_v1")

        # Build a panel where one symbol is the obvious winner over the
        # 12-1 formation period.  273 bars covers 12 months × 21 + skip.
        n = 320
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        flat = pd.DataFrame(
            {f"S{i:02d}": np.linspace(100, 100, n) for i in range(5)},
            index=dates,
        )
        # S00 ramps from 100 → 200, others stay flat
        flat["S00"] = np.linspace(100, 200, n)

        ranks = m.rank_universe(flat)
        last  = ranks.iloc[-1].dropna()
        if not last.empty:
            assert last.idxmax() == "S00"


# ── Backtest runner ────────────────────────────────────────────────────────

class TestRunner:

    def test_runner_returns_results_envelope(self, have_data):
        from dashboard.backtest_engine import run_cross_sectional_backtest
        out = run_cross_sectional_backtest(
            model_id="jt_momentum_v1", symbols=have_data,
            period_days=365 * 6, top_decile=0.30, rebalance_days=21,
        )
        for k in ("run_id", "metrics", "equity_curve", "trades"):
            assert k in out

    def test_runner_handles_empty_universe(self):
        from dashboard.backtest_engine import run_cross_sectional_backtest
        out = run_cross_sectional_backtest(
            model_id="jt_momentum_v1", symbols=[],
        )
        # Empty results envelope rather than crash
        assert out.get("metrics", {}).get("total_trades", 0) == 0

    def test_runner_pl_per_trade_matches_equity_change(self, have_data):
        """
        Regression test for the piggyback-storage bug: trade P&L summed
        across the run should track the equity curve's net change.  An
        old version of the runner stuffed entry-prices into the holdings
        dict alongside share counts, so liquidations multiplied entry-
        price floats as if they were shares — corrupting both cash and
        the trade log silently.
        """
        from dashboard.backtest_engine import run_cross_sectional_backtest
        out = run_cross_sectional_backtest(
            model_id="jt_momentum_v1", symbols=have_data,
            period_days=365 * 6, top_decile=0.30, rebalance_days=21,
        )
        trades = out.get("trades", [])
        if not trades:
            return                                  # nothing to assert on
        total_pl = sum(float(t.get("pl", 0) or 0) for t in trades)
        # At least one trade should have non-zero P&L if there were trades
        assert any(float(t.get("pl", 0) or 0) != 0 for t in trades), \
            "every trade has pl=0 — likely the piggyback-storage bug returned"
        # And the summed P&L should be in the same ballpark as the equity
        # curve's net change (not necessarily exactly equal — buy/sell
        # slippage on each leg, partial-share rounding).
        ec = out.get("equity_curve", [])
        if len(ec) >= 2:
            net = float(ec[-1]["value"]) - float(ec[0]["value"])
            assert abs(total_pl - net) <= max(50.0, abs(net) * 0.05), \
                f"trade P&L sum ({total_pl:.2f}) far from equity change ({net:.2f})"

    def test_runner_trade_win_flag_matches_pl_sign(self, have_data):
        from dashboard.backtest_engine import run_cross_sectional_backtest
        out = run_cross_sectional_backtest(
            model_id="jt_momentum_v1", symbols=have_data,
            period_days=365 * 6, top_decile=0.30, rebalance_days=21,
        )
        for t in out.get("trades", []):
            pl = float(t.get("pl", 0) or 0)
            assert bool(t.get("win")) == (pl > 0), (
                f"trade win flag {t.get('win')} doesn't match pl sign "
                f"({pl}) for {t}"
            )


# ── UI guards: cross-sectional models hidden from per-symbol dropdowns ──────

class TestUiGuards:

    def test_topbar_dropdown_excludes_cross_sectional(self):
        from dashboard.components.global_controls import _load_models
        opts = _load_models()
        assert all("jt_momentum" not in o["value"] for o in opts)

    def test_backtest_page_dropdown_excludes_cross_sectional(self):
        from dashboard.pages.backtest import _model_options
        opts = _model_options()
        assert all("jt_momentum" not in o["value"] for o in opts)

    def test_finder_dropdown_excludes_cross_sectional(self):
        from dashboard.pages.strategy_finder import _strategy_options
        opts = _strategy_options()
        assert all("jt_momentum" not in o["value"] for o in opts)
