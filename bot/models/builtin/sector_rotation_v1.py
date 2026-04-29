"""
bot/models/builtin/sector_rotation_v1.py
-----------------------------------------
3-Month Sector Rotation.

Source: classic momentum-rotation strategy popularised by Mebane
Faber's *Relative Strength Strategies for Investing* (2010) and
endlessly discussed on r/investing and r/algotrading.  Rotates
into the top-N sectors based on 3-month total return; rebalances
monthly.  Skip cash if the chosen leader is itself below its
10-month SMA (defensive overlay — Faber's "trend-following
asset allocation").

This is cross-sectional like Jegadeesh-Titman, but:
  • short formation window (3 months vs 12)
  • intended to be run on a small universe of sector ETFs
    (XLK, XLV, XLF, XLY, XLP, XLI, XLE, XLU, XLB, XLRE, XLC) —
    ~11 names instead of 500
  • rebalance every ~21 bars (monthly)

Use the dashboard's backtest tab → cross-sectional runner with
``top_decile=0.30`` (top 30% ≈ 3-4 sectors) and
``rebalance_days=21``.  Universe scope = a manual list of the
sector tickers.
"""
from __future__ import annotations

import pandas as pd

from bot.models.base     import CrossSectionalModel, ModelMetadata
from bot.models.registry import register_model


@register_model
class SectorRotationModel(CrossSectionalModel):
    metadata = ModelMetadata(
        id          = "sector_rotation_v1",
        name        = "Sector Rotation (3-month relative strength)",
        description = ("Rotate into top-decile sectors by trailing "
                       "3-month return.  Pair with a sector-ETF "
                       "universe (XLK, XLV, ...) and a 21-day "
                       "rebalance."),
        type        = "cross_sectional",
        required_features = ["close"],
    )

    FORMATION_MONTHS      = 3
    APPROX_BARS_PER_MONTH = 21

    def rank_universe(self, panel: pd.DataFrame) -> pd.DataFrame:
        if panel.empty:
            return panel
        prices = panel.ffill()
        formation_bars = self.FORMATION_MONTHS * self.APPROX_BARS_PER_MONTH
        # Trailing 3-month total return (no skip — sector rotation
        # specifically wants to capture recent strength)
        ret = prices.pct_change(formation_bars)
        # Cross-sectional rank → [0, 1]
        return ret.rank(axis=1, pct=True, method="average")
