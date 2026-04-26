"""
bot/models/builtin/qullamaggie_v1.py
-------------------------------------
Qullamaggie-style breakout setup (Kristjan Kullamägi).

Buy when ALL of:
  * Prior 60-day run-up of >= 30%   (stage-2 strength)
  * 20-day consolidation range <= 15%  (tight base)
  * 20-day average volume <= 85% of 60-day average volume  (volume dry-up)
  * Today's close > 20-day pivot high  AND  volume > 1.5× 20-day avg volume

Sell:
  * Today's close < pivot high - 7%   (loss of structure)

These conditions are pre-computed by ``bot.patterns.add_breakout_features``.
The model overrides ``predict_batch`` so it works straight from a feature
DataFrame even if the underlying parquet hasn't been re-engineered yet.
"""
from __future__ import annotations

import pandas as pd

from bot.models.base     import BaseModel, ModelMetadata, Signal
from bot.models.registry import register_model
from bot.patterns        import add_breakout_features


@register_model
class QullamaggieModel(BaseModel):
    metadata = ModelMetadata(
        id          = "qullamaggie_v1",
        name        = "Qullamaggie breakout",
        description = "Stage-2 strength + tight consolidation + breakout on volume.",
        type        = "rule",
        required_features = [
            "prior_runup_pct",
            "consolidation_range",
            "consolidation_vol_drop",
            "breakout_today",
            "pivot_high",
            "close",
        ],
    )

    PULLBACK_STOP_PCT = 0.07

    def predict(self, row: pd.Series) -> tuple[Signal, float]:
        breakout = bool(row.get("breakout_today", False))
        runup    = row.get("prior_runup_pct")
        rng      = row.get("consolidation_range")
        vol_drop = row.get("consolidation_vol_drop")
        close    = row.get("close")
        pivot    = row.get("pivot_high")

        if any(pd.isna(x) for x in (runup, rng, vol_drop, close, pivot)):
            return ("hold", 0.50)

        # Buy: all conditions
        if (breakout and runup >= 0.30 and rng <= 0.15 and vol_drop <= 0.85):
            # Confidence rises with stronger runup and tighter base
            tightness = max(0.0, (0.15 - rng) / 0.15)
            strength  = min(1.0, runup / 1.0)
            conf = 0.55 + 0.20 * tightness + 0.15 * strength
            return ("buy", round(min(conf, 0.95), 3))

        # Sell: structure breaks
        if pivot and close < pivot * (1 - self.PULLBACK_STOP_PCT):
            return ("sell", 0.65)

        return ("hold", 0.55)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if "breakout_today" not in df.columns:
            df = add_breakout_features(df)
        return super().predict_batch(df)
