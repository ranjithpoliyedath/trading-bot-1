"""
bot/models/builtin/pct52w_high_v1.py
-------------------------------------
52-Week High Momentum.

Source: George & Hwang — "The 52-Week High and Momentum Investing"
(Journal of Finance, 2004).  Documents that stocks trading near
their 52-week high outperform.  The intuition: anchoring bias —
investors hesitate to buy stocks that look "too high" relative to
their highs, even when fundamentals justify it.

Rule: buy when close >= 95% of trailing 252-bar high
      AND volume > recent average (confirmation).
Sell when close drops below 85% of the 52-week high.

Confidence scales with proximity to the high (1.00 = at the high).
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from bot.models.base     import BaseModel, ModelMetadata, Signal
from bot.models.registry import register_model


@register_model
class Pct52WeekHighModel(BaseModel):
    metadata = ModelMetadata(
        id          = "pct52w_high_v1",
        name        = "52-Week High Momentum (George-Hwang)",
        description = ("Buy when close is within 5% of the 52-week high "
                       "with above-average volume. Sell on 15% retracement "
                       "from the high."),
        type        = "rule",
        required_features = ["close", "high", "volume"],
    )

    LOOKBACK_BARS = 252
    BUY_THRESHOLD = 0.95   # within 5% of the high
    SELL_THRESHOLD = 0.85  # 15% below the high → exit

    def predict(self, row: pd.Series) -> tuple[Signal, float]:
        pct  = row.get("pct_52w_high")
        vol_ratio = row.get("volume_ratio", 1.0)
        if pd.isna(pct):
            return ("hold", 0.50)
        if pct >= self.BUY_THRESHOLD and (vol_ratio is not None and vol_ratio >= 1.0):
            # The closer to 1.00, the higher the confidence
            conf = 0.55 + min(0.35, (pct - self.BUY_THRESHOLD) * 7)
            return ("buy", round(conf, 3))
        if pct < self.SELL_THRESHOLD:
            return ("sell", 0.65)
        return ("hold", 0.55)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if "pct_52w_high" not in df.columns:
            df = df.copy()
            high_52w = df["high"].rolling(self.LOOKBACK_BARS,
                                            min_periods=self.LOOKBACK_BARS).max()
            df["pct_52w_high"] = df["close"] / high_52w
        if "volume_ratio" not in df.columns:
            df["volume_ratio"] = (df["volume"] /
                                  df["volume"].rolling(20, min_periods=10).mean())
        return super().predict_batch(df)
