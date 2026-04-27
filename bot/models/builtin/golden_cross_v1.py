"""
bot/models/builtin/golden_cross_v1.py
--------------------------------------
SMA 50 / 200 crossover ("Golden Cross / Death Cross").

Source: classical TA, popularised in John Murphy's *Technical Analysis
of the Financial Markets* (1999).  One of the most-watched long-term
trend signals on Wall Street.

Buy when:  SMA(50) > SMA(200)  AND  close > SMA(50)
Sell when: SMA(50) < SMA(200)

Confidence scales with how far above/below the 50-day MA the price is.
"""
from __future__ import annotations

import pandas as pd

from bot.indicators       import add_sma
from bot.models.base     import BaseModel, ModelMetadata, Signal
from bot.models.registry import register_model


@register_model
class GoldenCrossModel(BaseModel):
    metadata = ModelMetadata(
        id          = "golden_cross_v1",
        name        = "Golden Cross 50/200",
        description = "Long-term trend follower: buy when 50-SMA above 200-SMA.",
        type        = "rule",
        required_features = ["sma_50", "sma_200", "close"],
    )

    def predict(self, row: pd.Series) -> tuple[Signal, float]:
        sma50  = row.get("sma_50")
        sma200 = row.get("sma_200")
        close  = row.get("close")
        if any(pd.isna(x) for x in (sma50, sma200, close)):
            return ("hold", 0.50)

        if sma50 > sma200 and close > sma50:
            spread = (close - sma50) / sma50
            conf   = 0.55 + min(0.30, max(0, spread) * 5)
            return ("buy", round(min(conf, 0.90), 3))
        if sma50 < sma200:
            return ("sell", 0.65)
        return ("hold", 0.55)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if "sma_200" not in df.columns:
            df = add_sma(df, periods=(50, 200))
        return super().predict_batch(df)
