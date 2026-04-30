"""
bot/models/builtin/leaders_breakout_v1.py
------------------------------------------
"Leaders Breakout" — user-designed momentum strategy.

Entry — ALL must be true:
  1. Bull stack: close > sma_10, sma_20, sma_50, sma_200
  2. Volume spike: max(volume_ratio over last 5 bars) >= 2.5
  3. Big move:     price_change_5d >= 3%
  4. Breakout:     close > 20-day rolling high (excluding today)

Exit — either fires:
  • close < any of sma_10, sma_20, sma_50  (model sell signal)
  • 15% stop-loss from entry              (engine stop_loss_pct, set
                                            in the run config)

When several candidates fire on the same bar and cash is limited,
the simulator picks the highest-confidence ones first.  We scale
confidence by 5-day return so the user's "sort by % returns in
last 5 days" rule is honoured naturally — strongest leaders
consume cash first; laggards are skipped if the pool is empty.

Sector-cap rule (max 2 per sector) — DOCUMENTED LIMITATION.
The engine doesn't currently group entries by sector at simulation
time.  In practice the natural diversification effect of fixed_pct
sizing (e.g. 10% per position → ~10 positions max) plus this
strategy's selectivity (it only fires on a tiny fraction of bars)
gives you concentration similar to a manual cap.  A proper engine-
side ``max_per_sector`` flag is a separate follow-up.

Recommended run config:
  • Universe       : top_500 or sp500 (deep + diverse)
  • Sizing         : fixed_pct 5–10%
  • Stop-loss      : 0.15  (matches the user's 15% rule)
  • Take-profit    : off  (let the SMA-break or stop close it)
  • Use signal exit: True  (let the strategy emit its own SMA-break sell)
  • Time stop      : off
"""
from __future__ import annotations

import pandas as pd

from bot.models.base     import BaseModel, ModelMetadata, Signal
from bot.models.registry import register_model


@register_model
class LeadersBreakoutModel(BaseModel):
    metadata = ModelMetadata(
        id          = "leaders_breakout_v1",
        name        = "Leaders Breakout (volume + breakout + bull stack)",
        description = ("Buy on a 20-day breakout in symbols above all "
                       "four SMAs (10/20/50/200), confirmed by a "
                       "≥2.5× volume spike and ≥3% 5-day move.  "
                       "Sort by 5d return.  Exit below 10/20/50 SMA "
                       "or 15% stop."),
        type        = "rule",
        required_features = ["close", "high", "volume",
                              "sma_10", "sma_20", "sma_50", "sma_200"],
    )

    # Entry thresholds — exposed so Optuna can search around them
    BREAKOUT_LOOKBACK         = 20
    VOLUME_SPIKE_THRESHOLD    = 2.5
    VOLUME_SPIKE_LOOKBACK     = 5
    MIN_5D_RETURN             = 0.03    # 3% in 5 days = "big price move"

    def predict(self, row: pd.Series) -> tuple[Signal, float]:
        close   = row.get("close")
        sma10   = row.get("sma_10")
        sma20   = row.get("sma_20")
        sma50   = row.get("sma_50")
        sma200  = row.get("sma_200")
        ret5d   = row.get("price_change_5d")
        volspk  = row.get("volume_spike_5d_max")
        is_brk  = row.get("breakout_today_20d")

        # If any of the core trend SMAs are NaN (still in warm-up), hold
        if any(pd.isna(x) for x in (close, sma10, sma20, sma50, sma200)):
            return ("hold", 0.50)

        # ── EXIT: drop below any of 10/20/50 SMA ─────────────────
        # The 200-SMA is intentionally excluded from the exit — many
        # leaders dip below sma_200 briefly during pullbacks; we only
        # want to bail when the shorter-term trend cracks.
        if close < sma10 or close < sma20 or close < sma50:
            return ("sell", 0.65)

        # ── ENTRY: ALL four conditions ───────────────────────────
        bull_stack = close > sma200      # already ensured > 10/20/50 by exit-check above

        ret5d_v   = float(ret5d)  if not pd.isna(ret5d)  else 0.0
        volspk_v  = float(volspk) if not pd.isna(volspk) else 0.0
        is_brk_v  = bool(is_brk) if not pd.isna(is_brk) else False

        big_volume = volspk_v >= self.VOLUME_SPIKE_THRESHOLD
        big_move   = ret5d_v  >= self.MIN_5D_RETURN

        if bull_stack and big_volume and big_move and is_brk_v:
            # Confidence scales with 5-day return so cash-limited bars
            # let the strongest leaders win the entry race.  The
            # confidence delta from 0 → 0.60 maps to a 5d return of
            # 0% → 20%.  Capped at 0.92.
            conf = 0.60 + min(0.30, ret5d_v * 1.5)
            # Volume kicker: a 4× spike beats a 2.5× spike
            conf += min(0.05, max(0, volspk_v - 2.5) * 0.025)
            return ("buy", round(min(conf, 0.92), 3))

        return ("hold", 0.55)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ── SMAs (ensure all 4 are present) ────────────────────
        for span in (10, 20, 50, 200):
            col = f"sma_{span}"
            if col not in df.columns:
                df[col] = df["close"].rolling(span, min_periods=span).mean()

        # ── Volume ratio + 5-day rolling max ────────────────────
        if "volume_ratio" not in df.columns:
            avg = df["volume"].rolling(20, min_periods=10).mean()
            df["volume_ratio"] = df["volume"] / avg
        if "volume_spike_5d_max" not in df.columns:
            df["volume_spike_5d_max"] = (
                df["volume_ratio"]
                .rolling(self.VOLUME_SPIKE_LOOKBACK, min_periods=1)
                .max()
            )

        # ── 5-day return (price change) ─────────────────────────
        if "price_change_5d" not in df.columns:
            df["price_change_5d"] = df["close"].pct_change(5)

        # ── 20-day breakout flag ────────────────────────────────
        # Today's close > rolling-20-day high through *yesterday*.
        # We shift(1) so today's high doesn't make every bar look
        # like a breakout (look-ahead-safe).
        if "breakout_today_20d" not in df.columns:
            prior_max = (df["high"].shift(1)
                          .rolling(self.BREAKOUT_LOOKBACK,
                                    min_periods=self.BREAKOUT_LOOKBACK).max())
            df["breakout_today_20d"] = (df["close"] > prior_max).astype("float64")

        return super().predict_batch(df)
