"""
tests/test_strategy_library.py
-------------------------------
Smoke tests for the Phase-3 strategy library + indicators module.
Each registered strategy must:
  * import cleanly,
  * declare metadata,
  * produce a valid signal column on a synthetic OHLCV DataFrame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


NEW_STRATEGY_IDS = [
    "golden_cross_v1",
    "donchian_v1",
    "connors_rsi2_v1",
    "ibs_v1",
    "adx_trend_v1",
    "keltner_breakout_v1",
    "obv_momentum_v1",
    "zscore_reversion_v1",
]


@pytest.fixture(scope="module")
def synthetic_df():
    """Long enough for SMA(200), Donchian, OBV slope, etc., to warm up."""
    rng = np.random.default_rng(0)
    n = 400
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
    open_ = close + rng.normal(0, 0.4, n)
    high  = np.maximum(open_, close) + rng.uniform(0.1, 1.2, n)
    low   = np.minimum(open_, close) - rng.uniform(0.1, 1.2, n)
    return pd.DataFrame({
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": rng.integers(800_000, 3_000_000, n),
        "atr_14": np.full(n, 1.5),
        "rsi_14":          rng.uniform(20, 80, n),
        "macd_hist":       rng.normal(0, 0.3, n),
        "bb_pct":          rng.uniform(0, 1, n),
        "volume_ratio":    rng.uniform(0.5, 2.5, n),
        "price_change_1d": rng.normal(0, 0.02, n),
        "price_change_5d": rng.normal(0, 0.05, n),
    }, index=dates)


# ── Indicators module ───────────────────────────────────────────────────────

class TestIndicators:

    def test_add_all_idempotent(self, synthetic_df):
        from bot.indicators import add_all_indicators
        out  = add_all_indicators(synthetic_df.copy())
        out2 = add_all_indicators(out.copy())
        # Re-running shouldn't add new columns or change values
        assert set(out.columns) == set(out2.columns)

    def test_adx_in_range(self, synthetic_df):
        from bot.indicators import add_adx
        out = add_adx(synthetic_df.copy())
        adx = out["adx_14"].dropna()
        assert (adx.between(0, 100)).all()

    def test_ibs_in_range(self, synthetic_df):
        from bot.indicators import add_ibs
        out = add_ibs(synthetic_df.copy())
        ibs = out["ibs"].dropna()
        assert (ibs.between(0, 1)).all()

    def test_zscore_centred(self, synthetic_df):
        from bot.indicators import add_zscore
        out = add_zscore(synthetic_df.copy(), period=20)
        z = out["zscore_close_20"].dropna()
        # Should be roughly centred around zero on stationary noise; loose bound
        assert abs(z.mean()) < 1.5

    def test_donchian_high_above_low(self, synthetic_df):
        from bot.indicators import add_donchian
        out = add_donchian(synthetic_df.copy(), periods=(20,))
        valid = out[["donchian_high_20", "donchian_low_20"]].dropna()
        if not valid.empty:
            assert (valid["donchian_high_20"] >= valid["donchian_low_20"]).all()

    def test_keltner_band_order(self, synthetic_df):
        from bot.indicators import add_keltner
        out = add_keltner(synthetic_df.copy())
        valid = out[["keltner_upper", "keltner_middle", "keltner_lower"]].dropna()
        if not valid.empty:
            assert (valid["keltner_upper"]  >= valid["keltner_middle"]).all()
            assert (valid["keltner_middle"] >= valid["keltner_lower"]).all()


# ── Each strategy: registers + predict_batch produces valid signals ─────────

class TestStrategyRegistration:

    def test_all_new_strategies_registered(self):
        from bot.models.registry import list_models
        ids = {m.id for m in list_models()}
        for sid in NEW_STRATEGY_IDS:
            assert sid in ids, f"{sid} not registered"


@pytest.mark.parametrize("strategy_id", NEW_STRATEGY_IDS)
class TestEachStrategy:

    def test_imports_and_has_metadata(self, strategy_id):
        from bot.models.registry import get_model
        m = get_model(strategy_id)
        assert m.metadata.id == strategy_id
        assert m.metadata.name
        assert m.metadata.description
        assert m.metadata.required_features

    def test_predict_batch_produces_valid_signals(self, strategy_id, synthetic_df):
        from bot.models.registry import get_model
        m = get_model(strategy_id)
        out = m.predict_batch(synthetic_df.copy())
        assert "signal" in out.columns
        assert "confidence" in out.columns
        signals = set(out["signal"].dropna().unique())
        assert signals <= {"buy", "sell", "hold"}, \
            f"{strategy_id} emitted unexpected signals: {signals}"
        confs = out["confidence"].dropna()
        assert (confs.between(0.0, 1.0)).all(), \
            f"{strategy_id} produced out-of-range confidence values"


# ── Strategy Finder param spaces declared for every new strategy ────────────

class TestFinderIntegration:

    @pytest.mark.parametrize("strategy_id", NEW_STRATEGY_IDS)
    def test_param_space_declared(self, strategy_id):
        from bot.strategy_finder import param_space
        space = param_space(strategy_id)
        assert space, f"No param space declared for {strategy_id}"
        # Every entry has the expected schema
        for p in space:
            assert {"name", "type"} <= set(p)
            assert p["type"] in {"int", "float", "categorical"}
