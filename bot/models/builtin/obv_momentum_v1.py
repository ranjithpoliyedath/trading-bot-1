"""
bot/models/builtin/obv_momentum_v1.py
--------------------------------------
On-Balance Volume momentum confirmation.

Source: Joseph Granville (1963) *Granville's New Key to Stock Market
Profits*.  OBV adds volume on up days, subtracts on down days — the
cumulative line is a rough proxy for accumulation/distribution.
Long signals are most reliable when *both* price and OBV are trending
up; OBV diverging from price is the classic warning.

Buy when:  20-day price return > 0  AND  OBV slope (20d) > 0  AND
           close > SMA(50)
Sell when: OBV slope < 0  OR  close < SMA(50)
"""
from __future__ import annotations

import pandas as pd

from bot.indicators       import add_obv, add_sma
from bot.models.base     import BaseModel, ModelMetadata, Signal
from bot.models.registry import register_model


@register_model
class ObvMomentumModel(BaseModel):
    metadata = ModelMetadata(
        id          = "obv_momentum_v1",
        name        = "OBV momentum",
        description = "Volume-confirmed trend: price up + OBV slope up + above 50-SMA.",
        type        = "rule",
        required_features = ["obv", "obv_slope_20", "sma_50", "close", "price_change_5d"],
    )

    def predict(self, row: pd.Series) -> tuple[Signal, float]:
        slope  = row.get("obv_slope_20")
        sma50  = row.get("sma_50")
        close  = row.get("close")
        ret_5  = row.get("price_change_5d", 0)
        if any(pd.isna(x) for x in (slope, sma50, close)):
            return ("hold", 0.50)

        if slope > 0 and close > sma50 and ret_5 > 0:
            # Magnitude term: stronger 5d return = more confidence
            conf = 0.55 + min(0.30, max(0, ret_5) * 4)
            return ("buy", round(min(conf, 0.88), 3))
        if slope < 0 or close < sma50:
            return ("sell", 0.60)
        return ("hold", 0.55)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if "obv_slope_20" not in df.columns:
            df = add_obv(df, slope_window=20)
        if "sma_50" not in df.columns:
            df = add_sma(df, periods=(50,))
        return super().predict_batch(df)
