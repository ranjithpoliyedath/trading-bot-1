"""
tests/test_strategy_finder.py
------------------------------
Tests for the Strategy Finder backend:
  * param_space + params_to_filters + apply_params
  * run_optuna determinism + leaderboard shape
  * suggest_with_claude opt-in path (Anthropic SDK mocked)
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from bot.config import DATA_DIR


@pytest.fixture(scope="module")
def have_data():
    files = list(Path(DATA_DIR).glob("*_features.parquet"))
    if not files:
        pytest.skip("No processed feature files on disk.")
    return [p.name.split("_")[0] for p in files][:5]


# ── Param space + spec building ─────────────────────────────────────────────

class TestParamSpace:

    def test_known_strategies_have_space(self):
        from bot.strategy_finder import param_space
        for sid in ("rsi_macd_v1", "bollinger_v1", "sentiment_v1",
                    "qullamaggie_v1", "vcp_v1"):
            assert param_space(sid), f"No space for {sid}"

    def test_unknown_strategy_returns_empty(self):
        from bot.strategy_finder import param_space
        assert param_space("nonsense") == []

    def test_params_to_filters_shape(self):
        from bot.strategy_finder import params_to_filters
        out = params_to_filters("rsi_macd_v1",
                                  {"buy_rsi": 25, "sell_rsi": 70,
                                   "min_confidence": 0.6})
        assert all({"field", "op", "value"} <= set(f) for f in out)

    def test_apply_params_produces_valid_spec(self):
        from bot.strategy_finder import apply_params
        spec = apply_params("rsi_macd_v1",
                             {"buy_rsi": 25, "min_confidence": 0.6,
                              "sell_rsi": 70},
                             name="rsi_tuned",
                             description="test")
        for k in ("id", "name", "buy_when", "sell_when",
                  "min_confidence", "tuned_from", "tuned_params"):
            assert k in spec
        assert spec["id"] == "rsi_tuned"
        assert spec["tuned_from"] == "rsi_macd_v1"


# ── Optuna driver ───────────────────────────────────────────────────────────

class TestRunOptuna:

    def test_returns_leaderboard_for_known_strategy(self, have_data):
        from bot.strategy_finder import run_optuna
        out = run_optuna(
            strategy_id      = "rsi_macd_v1",
            n_trials         = 5,
            n_folds          = 2,
            symbols          = have_data,
            period_days      = 365 * 6,
            seed             = 42,
            early_stop_after = 100,
        )
        assert "leaderboard" in out
        assert out["n_trials"] >= 1
        assert "study_state" in out
        for row in out["leaderboard"]:
            for k in ("trial", "mean_oos_sharpe", "mean_oos_return_pct",
                      "pct_positive_folds", "total_trades", "params"):
                assert k in row

    def test_unknown_strategy_returns_error(self):
        from bot.strategy_finder import run_optuna
        out = run_optuna(strategy_id="nope", n_trials=2)
        assert out.get("error")
        assert out.get("leaderboard") == []

    def test_deterministic_with_same_seed(self, have_data):
        from bot.strategy_finder import run_optuna
        out_a = run_optuna(
            strategy_id="rsi_macd_v1", n_trials=4, n_folds=2,
            symbols=have_data, period_days=365*6, seed=42,
            early_stop_after=100,
        )
        out_b = run_optuna(
            strategy_id="rsi_macd_v1", n_trials=4, n_folds=2,
            symbols=have_data, period_days=365*6, seed=42,
            early_stop_after=100,
        )
        # Same seed → same trial sequence → same params per trial
        assert [r["params"] for r in out_a["leaderboard"]] == \
               [r["params"] for r in out_b["leaderboard"]]


# ── Anthropic opt-in path (mocked) ─────────────────────────────────────────

class TestSuggestWithClaude:

    def test_no_api_key_returns_empty(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from bot.strategy_finder import suggest_with_claude
        assert suggest_with_claude("rsi_macd_v1", []) == []

    def test_unknown_strategy_returns_empty(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
        from bot.strategy_finder import suggest_with_claude
        assert suggest_with_claude("nonsense", []) == []

    def test_well_formed_proposals_clipped_to_range(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

        # Fake tool_use response
        tool_block = MagicMock()
        tool_block.type  = "tool_use"
        tool_block.input = {"proposals": [
            {"params": {"buy_rsi": 999, "sell_rsi": -10,
                         "min_confidence": 0.99},
             "rationale": "out-of-range — should be clipped"},
        ]}
        response = MagicMock()
        response.content = [tool_block]

        client = MagicMock()
        client.messages.create.return_value = response

        with patch("anthropic.Anthropic", return_value=client):
            from bot.strategy_finder import suggest_with_claude
            out = suggest_with_claude("rsi_macd_v1", [], n_proposals=1)

        assert len(out) == 1
        params = out[0].params
        # Clipped to declared ranges
        assert 15 <= params["buy_rsi"]  <= 40
        assert 60 <= params["sell_rsi"] <= 85
        assert 0.50 <= params["min_confidence"] <= 0.85

    def test_no_tool_use_block_returns_empty(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

        text_block = MagicMock()
        text_block.type = "text"
        response = MagicMock()
        response.content = [text_block]

        client = MagicMock()
        client.messages.create.return_value = response

        with patch("anthropic.Anthropic", return_value=client):
            from bot.strategy_finder import suggest_with_claude
            assert suggest_with_claude("rsi_macd_v1", []) == []


# ── Holdout confirmation ───────────────────────────────────────────────────

class TestConfirmHoldout:

    def test_returns_one_row_per_config(self, have_data):
        from bot.strategy_finder import confirm_holdout
        rows = confirm_holdout(
            strategy_id  = "rsi_macd_v1",
            top_k_params = [
                {"params": {"buy_rsi": 30, "sell_rsi": 70, "min_confidence": 0.6}},
                {"params": {"buy_rsi": 25, "sell_rsi": 75, "min_confidence": 0.55}},
            ],
            holdout_start = pd.Timestamp("2025-01-01"),
            holdout_end   = pd.Timestamp("2026-01-01"),
            symbols       = have_data,
            period_days   = 365 * 6,
        )
        assert len(rows) == 2
        for r in rows:
            for k in ("params", "holdout_sharpe", "holdout_return", "holdout_trades"):
                assert k in r
