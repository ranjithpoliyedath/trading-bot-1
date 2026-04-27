"""
bot/models/builtin/zscore_reversion_v1.py
------------------------------------------
Z-score statistical mean reversion.

Source: classic stat-arb literature (Avellaneda-Lee 2010, Pole 2007).
Computes the z-score of close vs a rolling 20-day mean / stdev.
Extreme negative z = stretched below the local mean → mean-reversion
buy.  Pairs naturally with a 200-SMA trend filter to avoid catching
falling knives.

Buy when:  z-score(20) < -1.5  AND  close > SMA(200)
Sell when: z-score(20) > 0.0   (snap-back to mean is enough)
"""
from __future__ import annotations

import pandas as pd

from bot.indicators       import add_zscore, add_sma
from bot.models.base     import BaseModel, ModelMetadata, Signal
from bot.models.registry import register_model


@register_model
class ZscoreReversionModel(BaseModel):
    metadata = ModelMetadata(
        id          = "zscore_reversion_v1",
        name        = "Z-score reversion",
        description = "Buy stretched-down stocks (z<-1.5) above the 200-SMA; exit at the mean.",
        type        = "rule",
        required_features = ["zscore_close_20", "sma_200", "close"],
    )

    BUY_Z   = -1.5
    EXIT_Z  =  0.0

    def predict(self, row: pd.Series) -> tuple[Signal, float]:
        z      = row.get("zscore_close_20")
        sma200 = row.get("sma_200")
        close  = row.get("close")
        if any(pd.isna(x) for x in (z, sma200, close)):
            return ("hold", 0.50)

        if z < self.BUY_Z and close > sma200:
            # Confidence rises with how stretched we got (capped at z=-3)
            conf = 0.55 + min(0.30, max(0, (self.BUY_Z - z)) / 3)
            return ("buy", round(min(conf, 0.88), 3))
        if z > self.EXIT_Z:
            return ("sell", 0.60)
        return ("hold", 0.55)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if "zscore_close_20" not in df.columns:
            df = add_zscore(df, period=20)
        if "sma_200" not in df.columns:
            df = add_sma(df, periods=(200,))
        return super().predict_batch(df)
