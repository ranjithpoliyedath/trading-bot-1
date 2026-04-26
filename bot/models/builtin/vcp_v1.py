"""
bot/models/builtin/vcp_v1.py
-----------------------------
Mark Minervini-style Volatility Contraction Pattern.

Buy when:
  * At least 2 successively smaller pullbacks in the last 90 bars
  * Final base is tight (20-day range <= 15%)
  * Today's close breaks above the 20-day pivot high on >1.5× avg volume

Sell when:
  * Close < pivot - 5% (failed breakout)

Required pattern columns are computed by
``bot.patterns.add_breakout_features``.
"""
from __future__ import annotations

import pandas as pd

from bot.models.base     import BaseModel, ModelMetadata, Signal
from bot.models.registry import register_model
from bot.patterns        import add_breakout_features


@register_model
class VcpModel(BaseModel):
    metadata = ModelMetadata(
        id          = "vcp_v1",
        name        = "VCP breakout",
        description = "Volatility Contraction Pattern: shrinking pullbacks then breakout on volume.",
        type        = "rule",
        required_features = [
            "contraction_count",
            "consolidation_range",
            "breakout_today",
            "pivot_high",
            "close",
        ],
    )

    FAILURE_STOP_PCT = 0.05

    def predict(self, row: pd.Series) -> tuple[Signal, float]:
        contractions = row.get("contraction_count", 0) or 0
        rng          = row.get("consolidation_range")
        close        = row.get("close")
        pivot        = row.get("pivot_high")
        breakout     = bool(row.get("breakout_today", False))

        if any(pd.isna(x) for x in (rng, close, pivot)):
            return ("hold", 0.50)

        if breakout and contractions >= 2 and rng <= 0.15:
            tightness = max(0.0, (0.15 - rng) / 0.15)
            depth     = min(1.0, contractions / 4)
            conf = 0.55 + 0.20 * tightness + 0.20 * depth
            return ("buy", round(min(conf, 0.95), 3))

        if pivot and close < pivot * (1 - self.FAILURE_STOP_PCT):
            return ("sell", 0.60)

        return ("hold", 0.55)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if "contraction_count" not in df.columns:
            df = add_breakout_features(df)
        return super().predict_batch(df)
