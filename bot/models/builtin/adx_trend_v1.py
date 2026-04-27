"""
bot/models/builtin/adx_trend_v1.py
-----------------------------------
ADX-gated trend follower.

Source: J. Welles Wilder Jr., *New Concepts in Technical Trading
Systems* (1978).  ADX measures trend strength regardless of direction.
Wilder argued readings above 25 indicate a trending market suitable
for momentum systems; below 20 the market is ranging and trend
strategies should stand aside.

Buy when:  ADX(14) > 25  AND  +DI > -DI  AND  close > SMA(50)
Sell when: ADX(14) < 20  OR  close < SMA(50)
"""
from __future__ import annotations

import pandas as pd

from bot.indicators       import add_adx, add_sma
from bot.models.base     import BaseModel, ModelMetadata, Signal
from bot.models.registry import register_model


@register_model
class AdxTrendModel(BaseModel):
    metadata = ModelMetadata(
        id          = "adx_trend_v1",
        name        = "ADX trend filter",
        description = "Trend follower gated by Wilder's ADX > 25 and a 50-SMA filter.",
        type        = "rule",
        required_features = ["adx_14", "plus_di_14", "minus_di_14", "sma_50", "close"],
    )

    BUY_ADX  = 25
    EXIT_ADX = 20

    def predict(self, row: pd.Series) -> tuple[Signal, float]:
        adx     = row.get("adx_14")
        plus    = row.get("plus_di_14")
        minus   = row.get("minus_di_14")
        sma50   = row.get("sma_50")
        close   = row.get("close")
        if any(pd.isna(x) for x in (adx, plus, minus, sma50, close)):
            return ("hold", 0.50)

        if adx > self.BUY_ADX and plus > minus and close > sma50:
            # Confidence rises with stronger trend (capped at ADX 50)
            conf = 0.55 + min(0.35, (adx - self.BUY_ADX) / 50)
            return ("buy", round(min(conf, 0.92), 3))
        if adx < self.EXIT_ADX or close < sma50:
            return ("sell", 0.60)
        return ("hold", 0.55)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if "adx_14" not in df.columns:
            df = add_adx(df, period=14)
        if "sma_50" not in df.columns:
            df = add_sma(df, periods=(50,))
        return super().predict_batch(df)
