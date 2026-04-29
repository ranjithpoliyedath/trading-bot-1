"""
bot/models/builtin/tsmom_v1.py
-------------------------------
Time-Series Momentum (TSMOM).

Source: Moskowitz, Ooi, Pedersen — "Time Series Momentum" (Journal of
Financial Economics, 2012).  One of the most-cited momentum papers
in finance literature.

Rule: buy when the trailing 12-month excess return is positive,
hold for 1 month.  Unlike Jegadeesh-Titman (which is *cross-
sectional* — buying winners relative to peers), TSMOM is
*time-series* — each symbol fights its own past.  Works
across asset classes; here we apply to single-name equities.

Buy when:  trailing 252-bar return > 0
           AND close above 50-SMA (additional trend filter)
Sell when: trailing 252-bar return < 0

The 50-SMA filter isn't in the original paper — added here to
suppress flippy entries on choppy stocks where the 12-month
window is barely positive.

Confidence scales with the magnitude of the 12-month return.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from bot.indicators       import add_sma
from bot.models.base     import BaseModel, ModelMetadata, Signal
from bot.models.registry import register_model


@register_model
class TimeSeriesMomentumModel(BaseModel):
    metadata = ModelMetadata(
        id          = "tsmom_v1",
        name        = "Time-Series Momentum (Moskowitz-Pedersen)",
        description = ("Buy when trailing 12-month return is positive AND "
                       "close > 50-SMA. Academic momentum factor that "
                       "works across asset classes."),
        type        = "rule",
        required_features = ["close", "sma_50"],
    )

    LOOKBACK_BARS = 252  # ~12 months

    def predict(self, row: pd.Series) -> tuple[Signal, float]:
        # Implemented in predict_batch since we need a rolling lookback.
        # Single-row predict is best-effort: use the precomputed mom_12m
        # column if predict_batch already ran.
        mom = row.get("mom_12m")
        sma50 = row.get("sma_50")
        close = row.get("close")
        if any(pd.isna(x) for x in (mom, sma50, close)):
            return ("hold", 0.50)
        if mom > 0 and close > sma50:
            conf = 0.55 + min(0.35, mom * 0.5)   # cap at 0.90
            return ("buy", round(conf, 3))
        if mom < 0:
            return ("sell", round(0.55 + min(0.30, -mom * 0.5), 3))
        return ("hold", 0.55)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if "sma_50" not in df.columns:
            df = add_sma(df, periods=(50,))
        if "mom_12m" not in df.columns:
            # 12-month total return — close / close 252 bars ago - 1
            df = df.copy()
            df["mom_12m"] = df["close"].pct_change(self.LOOKBACK_BARS)
        return super().predict_batch(df)
