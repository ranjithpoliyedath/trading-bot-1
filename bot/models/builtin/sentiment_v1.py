"""
bot/models/builtin/sentiment_v1.py
-----------------------------------
Pure sentiment-driven strategy.

Buy when:  combined_sentiment > 0.5 AND news_count >= 5 (strong + well-covered)
Sell when: combined_sentiment < -0.5 AND news_count >= 5
Hold otherwise.

The news_count gate prevents acting on a single noisy headline.
Confidence scales with the magnitude of the sentiment score.
"""

import pandas as pd

from bot.models.base     import BaseModel, ModelMetadata, Signal
from bot.models.registry import register_model


@register_model
class SentimentModel(BaseModel):
    metadata = ModelMetadata(
        id          = "sentiment_v1",
        name        = "Sentiment-driven",
        description = "Buy on strongly positive news+social sentiment, sell on strongly negative — gated by news volume.",
        type        = "rule",
        required_features = ["combined_sentiment", "news_count"],
    )

    BUY_THRESHOLD     = 0.50
    SELL_THRESHOLD    = -0.50
    MIN_NEWS_VOLUME   = 5

    def predict(self, row: pd.Series) -> tuple[Signal, float]:
        sentiment  = row.get("combined_sentiment")
        news_count = row.get("news_count", 0)

        if pd.isna(sentiment):
            return ("hold", 0.50)

        news_count = 0 if pd.isna(news_count) else news_count

        if news_count < self.MIN_NEWS_VOLUME:
            return ("hold", 0.55)

        if sentiment > self.BUY_THRESHOLD:
            confidence = 0.55 + min(0.40, (sentiment - self.BUY_THRESHOLD) * 0.8)
            return ("buy", round(confidence, 3))

        if sentiment < self.SELL_THRESHOLD:
            confidence = 0.55 + min(0.40, (self.SELL_THRESHOLD - sentiment) * 0.8)
            return ("sell", round(confidence, 3))

        return ("hold", 0.55)
