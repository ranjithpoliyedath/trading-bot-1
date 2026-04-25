"""
bot/models/builtin/bollinger_v1.py
-----------------------------------
Bollinger Band squeeze strategy with sentiment confirmation.

Buy when:  Price touches lower band (bb_pct < 0.1) AND sentiment is positive
Sell when: Price touches upper band (bb_pct > 0.9) AND sentiment is negative
Hold otherwise.

Using bb_pct (where price sits in the band, 0=lower, 1=upper) makes the
threshold symbol-agnostic. Sentiment confirmation reduces false signals
during sustained trends.
"""

import pandas as pd

from bot.models.base     import BaseModel, ModelMetadata, Signal
from bot.models.registry import register_model


@register_model
class BollingerModel(BaseModel):
    metadata = ModelMetadata(
        id          = "bollinger_v1",
        name        = "Bollinger + Sentiment",
        description = "Mean reversion at band extremes, confirmed by sentiment direction.",
        type        = "rule",
        required_features = ["bb_pct", "combined_sentiment"],
    )

    BUY_BB_THRESHOLD  = 0.10
    SELL_BB_THRESHOLD = 0.90

    def predict(self, row: pd.Series) -> tuple[Signal, float]:
        bb_pct    = row.get("bb_pct")
        sentiment = row.get("combined_sentiment", 0.0)

        if pd.isna(bb_pct):
            return ("hold", 0.50)

        sentiment = 0.0 if pd.isna(sentiment) else sentiment

        if bb_pct < self.BUY_BB_THRESHOLD and sentiment >= 0:
            confidence = 0.55 + min(0.30, (self.BUY_BB_THRESHOLD - bb_pct) * 2 + sentiment * 0.2)
            return ("buy", round(confidence, 3))

        if bb_pct > self.SELL_BB_THRESHOLD and sentiment <= 0:
            confidence = 0.55 + min(0.30, (bb_pct - self.SELL_BB_THRESHOLD) * 2 + abs(sentiment) * 0.2)
            return ("sell", round(confidence, 3))

        return ("hold", 0.55)
