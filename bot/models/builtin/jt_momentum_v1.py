"""
bot/models/builtin/jt_momentum_v1.py
-------------------------------------
Jegadeesh-Titman 12-1 cross-sectional momentum.

Source: Jegadeesh & Titman (1993), *Returns to Buying Winners and
Selling Losers: Implications for Stock Market Efficiency*, Journal
of Finance.  One of the foundational anomaly papers.  Each month we
rank the universe by total return over the past 12 months
*excluding the most recent month* (the "12 minus 1" formation
period — the recent month is dropped to avoid short-term reversal
contamination).  Long the top decile, short the bottom decile, hold
for 1 month, rebalance.

This is a **cross-sectional** strategy — the signal for any one
symbol depends on every other symbol's return — so it can't be
expressed via the per-symbol ``BaseModel`` interface.  See
``CrossSectionalModel`` in ``bot.models.base`` and the runner
``run_cross_sectional_backtest`` in ``dashboard.backtest_engine``.
"""
from __future__ import annotations

import pandas as pd

from bot.models.base     import CrossSectionalModel, ModelMetadata
from bot.models.registry import register_model


@register_model
class JegadeeshTitmanModel(CrossSectionalModel):
    metadata = ModelMetadata(
        id          = "jt_momentum_v1",
        name        = "Jegadeesh-Titman 12-1 momentum",
        description = "Cross-sectional momentum: long top-decile 12-1 return.",
        type        = "cross_sectional",
        required_features = ["close"],
    )

    FORMATION_MONTHS = 12   # months of return to use
    SKIP_MONTHS      = 1    # most recent month is dropped
    APPROX_BARS_PER_MONTH = 21

    def rank_universe(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        ``panel``: wide df, index=date, columns=symbol, values=close.

        Compute each symbol's trailing 12-1 return on every bar and
        rank cross-sectionally to [0, 1] (1 = best performer).
        """
        if panel.empty:
            return panel

        # Forward-fill so a holiday doesn't NaN out the ranks
        prices = panel.ffill()

        formation_bars = self.FORMATION_MONTHS * self.APPROX_BARS_PER_MONTH
        skip_bars      = self.SKIP_MONTHS * self.APPROX_BARS_PER_MONTH

        # 12-1 return = price[t-skip] / price[t-formation] - 1
        anchor = prices.shift(skip_bars)
        base   = prices.shift(formation_bars)
        ret    = anchor / base - 1.0

        # Cross-sectional rank along columns, scaled to [0,1]
        ranks = ret.rank(axis=1, pct=True, method="average")
        return ranks
