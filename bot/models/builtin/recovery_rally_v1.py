"""
bot/models/builtin/recovery_rally_v1.py
----------------------------------------
"Sit out the downtrend, ride the recovery."

User-requested defensive strategy: stay in cash while the broad
market is falling (close below both 50-SMA and 200-SMA), and re-
enter only when the recovery has confirmed itself with TWO signals:

  1. Price reclaims the 50-SMA (early-stage recovery)
  2. MACD histogram turns positive (momentum confirms)
  3. Optional: combined sentiment > 0 (news mood agrees)

This is a *meta-strategy* — it ignores the specific stock setup and
only trades when the macro regime is friendly.  Designed to skip
bear markets and sideways chop.  In flat / uptrending markets it's
essentially long-only with a trend filter.

Buy when: close > sma_50 AND macd_hist > 0 AND price > sma_200
Sell when: close < sma_50  (early defensive exit)
Hold when: in cash and conditions aren't fully bullish

Confidence scales with how clearly all three signals agree.

NOTE: This is a per-symbol model.  For *portfolio-level* macro
timing (gate every trade on SPY's 200-SMA), use the existing
``Market regime exit`` toggle in the Realism panel.
"""
from __future__ import annotations

import pandas as pd

from bot.indicators       import add_sma
from bot.models.base     import BaseModel, ModelMetadata, Signal
from bot.models.registry import register_model


@register_model
class RecoveryRallyModel(BaseModel):
    metadata = ModelMetadata(
        id          = "recovery_rally_v1",
        name        = "Recovery Rally (defensive trend follower)",
        description = ("Sits in cash when below 50/200 SMAs.  Re-enters "
                       "when price reclaims 50-SMA AND MACD histogram "
                       "turns positive.  Designed to skip drawdowns."),
        type        = "rule",
        required_features = ["close", "sma_50", "sma_200", "macd_hist"],
    )

    def predict(self, row: pd.Series) -> tuple[Signal, float]:
        close   = row.get("close")
        sma50   = row.get("sma_50")
        sma200  = row.get("sma_200")
        mhist   = row.get("macd_hist")
        sentiment = row.get("combined_sentiment", 0)

        if any(pd.isna(x) for x in (close, sma50, sma200, mhist)):
            return ("hold", 0.50)

        # Defensive exit: drop below 50-SMA = trend broken, sit out
        if close < sma50:
            return ("sell", 0.65)

        # Bullish entry: above BOTH SMAs AND momentum confirming
        if close > sma50 and close > sma200 and mhist > 0:
            # Three components for confidence — all dimensionless
            # ratios so confidence values are comparable across
            # symbols (a $10 vs $1000 stock see the same scale):
            #   1. spread above sma_50 (trend strength)
            #   2. macd_hist as % of price (momentum strength)
            #   3. sentiment tailwind (news agreement, optional)
            spread   = (close - sma50) / sma50
            mom_pct  = abs(mhist) / max(close, 1e-6)   # dimensionless ratio
            sent     = float(sentiment) if not pd.isna(sentiment) else 0.0
            conf = (
                0.55
                + min(0.20, spread * 4)               # trend
                + min(0.10, mom_pct * 50)             # momentum (1% mhist → +0.10)
                + min(0.05, max(0, sent) * 0.10)      # sentiment kicker
            )
            return ("buy", round(min(conf, 0.90), 3))

        # Mid-state: between 50 and 200, trend repairing — wait
        return ("hold", 0.55)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if "sma_50" not in df.columns or "sma_200" not in df.columns:
            df = add_sma(df, periods=(50, 200))
        return super().predict_batch(df)
