"""
bot/models/builtin/keltner_breakout_v1.py
------------------------------------------
Keltner channel breakout (volatility-adjusted).

Source: Chester Keltner (1960) *How to Make Money in Commodities*,
modernised by Linda Bradford Raschke and others.  The channels are
EMA(close, 20) ± 2 × ATR(14) — width adapts to each symbol's
volatility, so the breakout level scales with the recent regime.

Buy when:  close > Keltner upper band
Sell when: close < Keltner middle (the EMA)
"""
from __future__ import annotations

import pandas as pd

from bot.indicators       import add_keltner
from bot.models.base     import BaseModel, ModelMetadata, Signal
from bot.models.registry import register_model


@register_model
class KeltnerBreakoutModel(BaseModel):
    metadata = ModelMetadata(
        id          = "keltner_breakout_v1",
        name        = "Keltner channel breakout",
        description = "Volatility-adjusted breakout: buy outside upper Keltner band.",
        type        = "rule",
        required_features = ["keltner_upper", "keltner_middle", "close"],
    )

    def predict(self, row: pd.Series) -> tuple[Signal, float]:
        close = row.get("close")
        upper = row.get("keltner_upper")
        mid   = row.get("keltner_middle")
        if any(pd.isna(x) for x in (close, upper, mid)):
            return ("hold", 0.50)

        if close > upper:
            spread = (close - upper) / upper if upper else 0
            conf   = 0.60 + min(0.30, max(0, spread) * 8)
            return ("buy", round(min(conf, 0.90), 3))
        if close < mid:
            return ("sell", 0.60)
        return ("hold", 0.55)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if "keltner_upper" not in df.columns:
            df = add_keltner(df, period=20, multiplier=2.0)
        return super().predict_batch(df)
