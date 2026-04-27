"""
bot/models/builtin/ibs_v1.py
-----------------------------
Internal Bar Strength (IBS) mean reversion.

Source: Liew & Roberts (2013) and the QuantPedia "IBS effect" body
of research.  IBS = (close − low) / (high − low).  Values near 0
mean the bar closed at its low (oversold intraday); values near 1
mean closed at the high.  Buying low-IBS bars in an uptrend is one
of the simplest profitable daily systems on US equities.

Buy when:  IBS < 0.20  AND  close > SMA(200)
Sell when: IBS > 0.80
"""
from __future__ import annotations

import pandas as pd

from bot.indicators       import add_ibs, add_sma
from bot.models.base     import BaseModel, ModelMetadata, Signal
from bot.models.registry import register_model


@register_model
class IbsModel(BaseModel):
    metadata = ModelMetadata(
        id          = "ibs_v1",
        name        = "Internal Bar Strength",
        description = "Buy bars closed near low in an uptrend; exit when bar closes near high.",
        type        = "rule",
        required_features = ["ibs", "sma_200", "close"],
    )

    BUY_IBS_MAX  = 0.20
    SELL_IBS_MIN = 0.80

    def predict(self, row: pd.Series) -> tuple[Signal, float]:
        ibs    = row.get("ibs")
        sma200 = row.get("sma_200")
        close  = row.get("close")
        if any(pd.isna(x) for x in (ibs, sma200, close)):
            return ("hold", 0.50)

        if ibs < self.BUY_IBS_MAX and close > sma200:
            conf = 0.55 + min(0.30, (self.BUY_IBS_MAX - ibs) * 1.5)
            return ("buy", round(min(conf, 0.88), 3))
        if ibs > self.SELL_IBS_MIN:
            return ("sell", 0.60)
        return ("hold", 0.55)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if "ibs" not in df.columns:
            df = add_ibs(df)
        if "sma_200" not in df.columns:
            df = add_sma(df, periods=(200,))
        return super().predict_batch(df)
