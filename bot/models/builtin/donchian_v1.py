"""
bot/models/builtin/donchian_v1.py
----------------------------------
Donchian channel breakout — Turtle-style.

Source: Curtis Faith *Way of the Turtle* (2007), describing the
Dennis-Eckhardt program.  Original Turtle System 1: buy on 20-day
high breakout, exit on 10-day low.  Trend-following with strong
historical edge in commodities; mixed in equities, but useful as
a benchmark.

Buy when:  close > rolling 20-day high (shifted yesterday)
Sell when: close < rolling 10-day low (shifted yesterday)

Confidence rises with how far above the channel we broke.
"""
from __future__ import annotations

import pandas as pd

from bot.indicators       import add_donchian
from bot.models.base     import BaseModel, ModelMetadata, Signal
from bot.models.registry import register_model


@register_model
class DonchianModel(BaseModel):
    metadata = ModelMetadata(
        id          = "donchian_v1",
        name        = "Donchian breakout (Turtle)",
        description = "Buy on 20-day high breakout, exit on 10-day low.",
        type        = "rule",
        required_features = ["donchian_high_20", "donchian_low_10", "close"],
    )

    def predict(self, row: pd.Series) -> tuple[Signal, float]:
        close   = row.get("close")
        hi20    = row.get("donchian_high_20")
        lo10    = row.get("donchian_low_10")
        if any(pd.isna(x) for x in (close, hi20, lo10)):
            return ("hold", 0.50)

        if close > hi20:
            spread = (close - hi20) / hi20 if hi20 else 0
            conf   = 0.60 + min(0.30, max(0, spread) * 8)
            return ("buy", round(min(conf, 0.92), 3))
        if close < lo10:
            return ("sell", 0.65)
        return ("hold", 0.55)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if "donchian_high_20" not in df.columns:
            df = add_donchian(df, periods=(20, 10))
        return super().predict_batch(df)
