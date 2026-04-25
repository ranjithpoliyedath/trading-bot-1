"""
bot/models/builtin/rsi_macd_v1.py
----------------------------------
Classic mean-reversion + momentum confirmation strategy.

Buy when:  RSI < 30 (oversold) AND MACD histogram > 0 (momentum turning up)
Sell when: RSI > 70 (overbought) AND MACD histogram < 0 (momentum turning down)
Hold otherwise.

Confidence scales with how extreme the RSI reading is — RSI of 20 is more
confident than RSI of 29.
"""

import pandas as pd

from bot.models.base     import BaseModel, ModelMetadata, Signal
from bot.models.registry import register_model


@register_model
class RsiMacdModel(BaseModel):
    metadata = ModelMetadata(
        id          = "rsi_macd_v1",
        name        = "RSI + MACD",
        description = "Mean reversion: buy oversold with positive momentum, sell overbought with negative momentum.",
        type        = "rule",
        required_features = ["rsi_14", "macd_hist"],
    )

    BUY_RSI_THRESHOLD  = 30
    SELL_RSI_THRESHOLD = 70

    def predict(self, row: pd.Series) -> tuple[Signal, float]:
        rsi  = row.get("rsi_14")
        macd = row.get("macd_hist")

        if pd.isna(rsi) or pd.isna(macd):
            return ("hold", 0.50)

        if rsi < self.BUY_RSI_THRESHOLD and macd > 0:
            confidence = 0.50 + min(0.40, (self.BUY_RSI_THRESHOLD - rsi) / 60)
            return ("buy", round(confidence, 3))

        if rsi > self.SELL_RSI_THRESHOLD and macd < 0:
            confidence = 0.50 + min(0.40, (rsi - self.SELL_RSI_THRESHOLD) / 60)
            return ("sell", round(confidence, 3))

        return ("hold", 0.55)
