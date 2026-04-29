"""
bot/models/builtin/weinstein_v1.py
-----------------------------------
Stan Weinstein Stage Analysis — Stage 2 entries.

Source: Stan Weinstein — *Secrets for Profiting in Bull and Bear
Markets* (1988).  Heavily referenced in trading subreddits and
financial blogs as the "stage analysis" method.

Weinstein divides every chart into 4 stages:

  Stage 1 — Basing area (sideways below flat 30-week MA)
  Stage 2 — Advancing (price above 30-week MA, MA sloping up)   ← BUY
  Stage 3 — Top area (price above MA but momentum stalling)
  Stage 4 — Declining (price below MA, MA sloping down)         ← AVOID/SHORT

This model trades **Stage 2 only**: buy on a confirmed Stage-2
breakout (close > 30-week SMA, SMA slope positive, breakout above
prior consolidation), and exit when the stock enters Stage 3
(MA flattens) or Stage 4 (price breaks below MA).

30 weeks ≈ 150 trading days.  We use sma_150 for the regime MA.
The slope filter is a 5-day change in sma_150 — positive = "MA
rising", negative or near-zero = "MA flat or falling".

Buy when: close > sma_150 AND sma_150 trending up
          AND volume confirms (volume_ratio > 1.0)
Sell when: close < sma_150 OR sma_150 sloping down
"""
from __future__ import annotations

import pandas as pd

from bot.models.base     import BaseModel, ModelMetadata, Signal
from bot.models.registry import register_model


@register_model
class WeinsteinStageModel(BaseModel):
    metadata = ModelMetadata(
        id          = "weinstein_v1",
        name        = "Weinstein Stage 2 (30-week MA)",
        description = ("Stan Weinstein's stage analysis: buy in Stage 2 "
                       "(above rising 30-week SMA with volume), exit "
                       "into Stage 3 (MA flattens) or Stage 4 (close "
                       "below MA)."),
        type        = "rule",
        required_features = ["close", "sma_150", "volume"],
    )

    SLOPE_WINDOW = 5  # bars to measure SMA slope direction

    def predict(self, row: pd.Series) -> tuple[Signal, float]:
        close       = row.get("close")
        sma150      = row.get("sma_150")
        # Use the price-normalised slope/extension so the rules are
        # comparable across symbols (a $10 stock and a $1000 stock
        # see the same threshold semantics).
        slope_pct   = row.get("sma_150_slope_pct")
        extension   = row.get("sma_150_extension")
        vol_r       = row.get("volume_ratio", 1.0)

        if any(pd.isna(x) for x in (close, sma150, slope_pct, extension)):
            return ("hold", 0.50)

        # Stage 4: below 30-wk MA — avoid / sell
        if close < sma150:
            return ("sell", 0.65)

        # Stage 1: above MA but MA flat/falling — wait
        if slope_pct <= 0:
            return ("hold", 0.50)

        # Late Stage 2 / Stage 3 risk: price too extended above MA
        if extension > 0.30:
            return ("hold", 0.55)

        # Stage 2: above MA, MA rising — BUY.  Confidence = strength of
        # all three signals.  All inputs are dimensionless ratios, so
        # confidence is comparable across symbols.
        vol_kicker = min(0.10, max(0, (vol_r or 1.0) - 1.0) * 0.10)
        slope_kick = min(0.15, slope_pct * 1.5)   # 0.10%/bar slope → +0.15
        ext_kick   = min(0.15, extension * 3.0)   # 5% above MA → +0.15
        conf = 0.55 + ext_kick + slope_kick + vol_kicker
        return ("buy", round(min(conf, 0.90), 3))

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "sma_150" not in df.columns:
            df["sma_150"] = df["close"].rolling(150, min_periods=150).mean()
        # Normalise slope to a percentage of the SMA so thresholds
        # are dimensionless and tunable across symbols.
        if "sma_150_slope_pct" not in df.columns:
            raw_slope = df["sma_150"].diff(self.SLOPE_WINDOW)
            df["sma_150_slope_pct"] = (raw_slope / df["sma_150"]) * 100.0
        # Distance of close above (or below, signed) the SMA.
        if "sma_150_extension" not in df.columns:
            df["sma_150_extension"] = (df["close"] - df["sma_150"]) / df["sma_150"]
        if "volume_ratio" not in df.columns:
            df["volume_ratio"] = (df["volume"] /
                                  df["volume"].rolling(20, min_periods=10).mean())
        return super().predict_batch(df)
