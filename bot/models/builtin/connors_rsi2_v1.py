"""
bot/models/builtin/connors_rsi2_v1.py
--------------------------------------
Connors' RSI(2) short-term mean reversion.

Source: Larry Connors & Cesar Alvarez, *Short-Term Trading Strategies
That Work* (2009).  One of the most cited daily-bar mean-reversion
systems.  The 2-period RSI is so jumpy that levels below 10 indicate
short-term over-sold conditions; combined with a 200-SMA trend filter
it captures pullbacks inside up-trends.

Buy when:  RSI(2) < 10  AND  close > SMA(200)
Sell when: close > SMA(5)  (let the bounce run to the short MA)
"""
from __future__ import annotations

import pandas as pd

from bot.indicators       import add_rsi, add_sma
from bot.models.base     import BaseModel, ModelMetadata, Signal
from bot.models.registry import register_model


@register_model
class ConnorsRsi2Model(BaseModel):
    metadata = ModelMetadata(
        id          = "connors_rsi2_v1",
        name        = "Connors RSI(2)",
        description = "Daily mean-reversion: RSI(2) < 10 above the 200-SMA, exit at the 5-SMA.",
        type        = "rule",
        required_features = ["rsi_2", "sma_200", "sma_5", "close"],
    )

    def predict(self, row: pd.Series) -> tuple[Signal, float]:
        rsi2   = row.get("rsi_2")
        sma200 = row.get("sma_200")
        sma5   = row.get("sma_5")
        close  = row.get("close")
        if any(pd.isna(x) for x in (rsi2, sma200, close)):
            return ("hold", 0.50)

        if rsi2 < 10 and close > sma200:
            # Confidence higher the deeper RSI(2) sank
            conf = 0.55 + min(0.30, (10 - rsi2) / 30)
            return ("buy", round(min(conf, 0.90), 3))
        if pd.notna(sma5) and close > sma5:
            return ("sell", 0.60)
        return ("hold", 0.55)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if "rsi_2" not in df.columns:
            df = add_rsi(df, period=2)
        if "sma_200" not in df.columns:
            df = add_sma(df, periods=(5, 200))
        elif "sma_5" not in df.columns:
            df = add_sma(df, periods=(5,))
        return super().predict_batch(df)
