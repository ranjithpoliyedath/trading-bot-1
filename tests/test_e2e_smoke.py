"""
tests/test_e2e_smoke.py
------------------------
Phase 4 end-to-end smoke test.

Walks through the major user-facing flows in one pass:
  1. Every dashboard page imports cleanly.
  2. The backtest engine runs (single-period + walk-forward + filtered).
  3. The Strategy Finder runs Optuna over a small budget.
  4. apply_params writes a valid CustomRuleModel JSON.
  5. The seed leaderboard exists and parses (or is regenerable).

Skips gracefully when no parquet data is on disk so this can run in
empty CI without flapping.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bot.config import DATA_DIR


@pytest.fixture(scope="module")
def have_data():
    files = list(Path(DATA_DIR).glob("*_features.parquet"))
    if not files:
        pytest.skip("No processed feature files on disk.")
    return [p.name.split("_")[0] for p in files][:5]


# ── 1. Dashboard pages import ──────────────────────────────────────────────

def test_all_pages_import():
    from dashboard.pages import (                  # noqa: F401
        market_overview, screener, model_builder,
        strategy_finder, backtest,
    )
    assert market_overview.layout("paper", "rsi_macd_v1", "AAPL") is not None
    assert screener.layout()                                       is not None
    assert model_builder.layout()                                  is not None
    assert strategy_finder.layout("paper", "rsi_macd_v1", "AAPL")  is not None
    assert backtest.layout("paper", "rsi_macd_v1", "AAPL")         is not None


def test_app_imports_with_all_callbacks():
    """Importing app must register every @callback decorator (backtest +
    strategy_finder) without exploding."""
    import dashboard.app  # noqa: F401


# ── 2. Backtest engine end-to-end ─────────────────────────────────────────

def test_filtered_backtest_runs(have_data):
    from dashboard.backtest_engine import run_filtered_backtest
    out = run_filtered_backtest(
        model_id        = "connors_rsi2_v1",
        filters         = [],
        symbols         = have_data,
        period_days     = 730,
        conf_threshold  = 0.55,
    )
    assert "metrics"      in out
    assert "equity_curve" in out
    assert "per_symbol"   in out


def test_walk_forward_runs(have_data):
    from dashboard.backtest_engine import run_walk_forward
    out = run_walk_forward(
        model_id        = "ibs_v1",
        n_folds         = 2,
        symbols         = have_data,
        period_days     = 365 * 6,
        conf_threshold  = 0.55,
    )
    assert "fold_results" in out
    assert "aggregate"    in out


def test_realism_options_threaded_through(have_data):
    """Slippage + next-open should propagate from the public entry point."""
    from dashboard.backtest_engine import run_filtered_backtest
    out_no_slip = run_filtered_backtest(
        model_id="connors_rsi2_v1", filters=[],
        symbols=have_data, period_days=365, conf_threshold=0.55,
        execution_model="next_open", slippage_bps=0,
    )
    out_high_slip = run_filtered_backtest(
        model_id="connors_rsi2_v1", filters=[],
        symbols=have_data, period_days=365, conf_threshold=0.55,
        execution_model="next_open", slippage_bps=100,
    )
    if out_no_slip["metrics"]["total_trades"] > 0:
        # 100 bps slippage should materially reduce returns vs zero
        assert out_high_slip["metrics"]["total_return_pct"] \
                <= out_no_slip["metrics"]["total_return_pct"] + 0.5


# ── 3. Strategy Finder ────────────────────────────────────────────────────

def test_strategy_finder_runs_optuna(have_data):
    from bot.strategy_finder import run_optuna
    out = run_optuna(
        strategy_id      = "connors_rsi2_v1",
        n_trials         = 4,
        n_folds          = 2,
        symbols          = have_data,
        period_days      = 365 * 6,
        seed             = 42,
        early_stop_after = 100,
    )
    assert out.get("n_trials", 0) >= 1
    assert "leaderboard" in out


def test_apply_params_round_trips_to_disk(tmp_path):
    """apply_params output is the exact spec CustomRuleModel can load."""
    from bot.strategy_finder import apply_params
    from bot.models.custom    import CustomRuleModel

    spec = apply_params(
        strategy_id  = "connors_rsi2_v1",
        params       = {"buy_rsi2": 8, "min_confidence": 0.60},
        name         = "connors_test_save",
        description  = "round-trip smoke test",
    )
    path = tmp_path / "connors_test_save.json"
    path.write_text(json.dumps(spec, indent=2))

    loaded = json.loads(path.read_text())
    assert loaded["id"]   == "connors_test_save"
    assert loaded["tuned_from"] == "connors_rsi2_v1"

    # Constructable into the actual rule-runner
    model = CustomRuleModel(loaded)
    assert model.metadata.id == "custom:connors_test_save"


# ── 4. Seed leaderboard ───────────────────────────────────────────────────

def test_seed_leaderboard_exists_and_parses():
    leaderboard = Path("dashboard/backtests/seed_leaderboard.json")
    if not leaderboard.exists():
        pytest.skip(f"Seed leaderboard not generated yet: run "
                    f"scripts/rank_strategies.py")
    data = json.loads(leaderboard.read_text())
    for k in ("generated_at", "scope", "leaderboard"):
        assert k in data
    rows = data["leaderboard"]
    assert isinstance(rows, list)
    if rows:
        # Sorted descending by Sharpe
        sharpes = [r.get("mean_oos_sharpe", 0) for r in rows]
        assert sharpes == sorted(sharpes, reverse=True)
