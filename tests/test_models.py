"""
tests/test_models.py
---------------------
Unit tests for the model registry and built-in models.
"""

import pandas as pd
import pytest

from bot.models.registry import get_model, list_models
from bot.models.custom   import CustomRuleModel


# ── Registry tests ────────────────────────────────────────────────────────────

class TestRegistry:

    def test_list_models_returns_builtin_models(self):
        models = list_models()
        ids = [m.id for m in models]
        assert "rsi_macd_v1"  in ids
        assert "bollinger_v1" in ids
        assert "sentiment_v1" in ids

    def test_get_model_returns_predicting_instance(self):
        model = get_model("rsi_macd_v1")
        assert hasattr(model, "predict")
        row = pd.Series({"rsi_14": 25.0, "macd_hist": 0.5})
        signal, conf = model.predict(row)
        assert signal in ("buy", "sell", "hold")
        assert 0.0 <= conf <= 1.0

    def test_unknown_model_raises_keyerror(self):
        with pytest.raises(KeyError):
            get_model("nonexistent_model_xyz")


# ── RSI + MACD model ──────────────────────────────────────────────────────────

class TestRsiMacd:

    def setup_method(self):
        self.model = get_model("rsi_macd_v1")

    def test_buy_when_oversold_and_macd_positive(self):
        row = pd.Series({"rsi_14": 25.0, "macd_hist": 0.3})
        signal, conf = self.model.predict(row)
        assert signal == "buy"
        assert conf > 0.5

    def test_sell_when_overbought_and_macd_negative(self):
        row = pd.Series({"rsi_14": 75.0, "macd_hist": -0.3})
        signal, conf = self.model.predict(row)
        assert signal == "sell"
        assert conf > 0.5

    def test_hold_when_oversold_but_macd_negative(self):
        row = pd.Series({"rsi_14": 25.0, "macd_hist": -0.3})
        signal, _ = self.model.predict(row)
        assert signal == "hold"

    def test_hold_when_overbought_but_macd_positive(self):
        row = pd.Series({"rsi_14": 75.0, "macd_hist": 0.3})
        signal, _ = self.model.predict(row)
        assert signal == "hold"

    def test_hold_when_neutral(self):
        row = pd.Series({"rsi_14": 50.0, "macd_hist": 0.0})
        signal, _ = self.model.predict(row)
        assert signal == "hold"

    def test_handles_nan_gracefully(self):
        row = pd.Series({"rsi_14": float("nan"), "macd_hist": 0.3})
        signal, _ = self.model.predict(row)
        assert signal == "hold"

    def test_more_extreme_rsi_higher_confidence(self):
        sig_extreme, c_extreme = self.model.predict(pd.Series({"rsi_14": 15.0, "macd_hist": 0.3}))
        sig_mild,    c_mild    = self.model.predict(pd.Series({"rsi_14": 28.0, "macd_hist": 0.3}))
        assert c_extreme > c_mild


# ── Bollinger model ───────────────────────────────────────────────────────────

class TestBollinger:

    def setup_method(self):
        self.model = get_model("bollinger_v1")

    def test_buy_at_lower_band_with_positive_sentiment(self):
        row = pd.Series({"bb_pct": 0.05, "combined_sentiment": 0.3})
        signal, _ = self.model.predict(row)
        assert signal == "buy"

    def test_sell_at_upper_band_with_negative_sentiment(self):
        row = pd.Series({"bb_pct": 0.95, "combined_sentiment": -0.3})
        signal, _ = self.model.predict(row)
        assert signal == "sell"

    def test_hold_at_lower_band_with_negative_sentiment(self):
        row = pd.Series({"bb_pct": 0.05, "combined_sentiment": -0.5})
        signal, _ = self.model.predict(row)
        assert signal == "hold"


# ── Sentiment model ───────────────────────────────────────────────────────────

class TestSentiment:

    def setup_method(self):
        self.model = get_model("sentiment_v1")

    def test_buy_when_strongly_bullish_with_news(self):
        row = pd.Series({"combined_sentiment": 0.7, "news_count": 10})
        signal, _ = self.model.predict(row)
        assert signal == "buy"

    def test_sell_when_strongly_bearish_with_news(self):
        row = pd.Series({"combined_sentiment": -0.7, "news_count": 10})
        signal, _ = self.model.predict(row)
        assert signal == "sell"

    def test_hold_when_news_volume_too_low(self):
        row = pd.Series({"combined_sentiment": 0.8, "news_count": 2})
        signal, _ = self.model.predict(row)
        assert signal == "hold"

    def test_hold_when_sentiment_too_weak(self):
        row = pd.Series({"combined_sentiment": 0.3, "news_count": 10})
        signal, _ = self.model.predict(row)
        assert signal == "hold"


# ── Custom model from JSON spec ───────────────────────────────────────────────

class TestCustomModel:

    def test_custom_buy_when_all_conditions_met(self):
        spec = {
            "id": "test_strategy",
            "name": "Test",
            "buy_when":  [{"field": "rsi_14", "op": "<", "value": 30}],
            "sell_when": [{"field": "rsi_14", "op": ">", "value": 70}],
            "min_confidence": 0.7,
        }
        model = CustomRuleModel(spec)
        row = pd.Series({"rsi_14": 25.0})
        signal, conf = model.predict(row)
        assert signal == "buy"
        assert conf == 0.7

    def test_custom_multiple_and_conditions(self):
        spec = {
            "id": "multi",
            "buy_when": [
                {"field": "rsi_14",       "op": "<", "value": 30},
                {"field": "volume_ratio", "op": ">", "value": 1.5},
            ],
            "sell_when": [],
        }
        model = CustomRuleModel(spec)
        # Both true -> buy
        signal, _ = model.predict(pd.Series({"rsi_14": 25.0, "volume_ratio": 2.0}))
        assert signal == "buy"
        # Only one true -> hold
        signal, _ = model.predict(pd.Series({"rsi_14": 25.0, "volume_ratio": 1.0}))
        assert signal == "hold"

    def test_custom_metadata_includes_required_features(self):
        spec = {
            "id": "test",
            "buy_when":  [{"field": "rsi_14",    "op": "<", "value": 30}],
            "sell_when": [{"field": "macd_hist", "op": "<", "value": 0}],
        }
        model = CustomRuleModel(spec)
        assert "rsi_14"    in model.metadata.required_features
        assert "macd_hist" in model.metadata.required_features

    def test_custom_handles_nan_gracefully(self):
        spec = {
            "id": "nan_test",
            "buy_when":  [{"field": "rsi_14", "op": "<", "value": 30}],
            "sell_when": [],
        }
        model = CustomRuleModel(spec)
        signal, _ = model.predict(pd.Series({"rsi_14": float("nan")}))
        assert signal == "hold"
