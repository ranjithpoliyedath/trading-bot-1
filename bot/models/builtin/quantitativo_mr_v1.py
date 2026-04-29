"""
bot/models/builtin/quantitativo_mr_v1.py
-----------------------------------------
Mean-reversion strategy from quantitativo.com — original article:
"A Mean Reversion Strategy with 2.11 Sharpe" (the published Sharpe
on QQQ over 1993–2024, hence the "211" in the article URL).

The author tested several variants and settled on the one below as
the preferred ("dynamic stop loss") version.  We implement that
final version here.

Entry (long-only):
  • IBS = (close − low) / (high − low) < 0.3
  • close < lower_band, where:
        rolling_high  = high.rolling(10).max()
        rolling_range = (high − low).rolling(25).mean()
        lower_band    = rolling_high − 2.5 * rolling_range

Exit (either fires — "dynamic stop loss"):
  • close > previous day's high   (mean-reversion target hit)
  • close < 300-day SMA           (regime defensive)

The asset of choice in the article was QQQ.  Nothing in the rules
locks the model to QQQ — it'll run on any symbol — but QQQ is what
the published 2.11 Sharpe was achieved on.

Reported (1993–2024 on QQQ):
  Sharpe              2.11
  Annualised return   13.0%   (Buy & Hold: 9.2%)
  Max drawdown       −20.3%
  Max DD duration    < 1 year
  Time in market      ~20%

A long-and-short variant (QQQ longs in bull markets, PSQ in bear)
got Sharpe 2.02 / DD −13.3%.  Not implemented here; the platform
doesn't yet support seamless long/short symbol-swap.  Add as a
follow-up if needed.
"""
from __future__ import annotations

import pandas as pd

from bot.models.base     import BaseModel, ModelMetadata, Signal
from bot.models.registry import register_model


@register_model
class QuantitativoMeanReversionModel(BaseModel):
    metadata = ModelMetadata(
        id          = "quantitativo_mr_v1",
        name        = "Quantitativo Mean Reversion (Sharpe 2.11)",
        description = ("Mean reversion: enter long when price closes "
                       "below a 10-day rolling-high minus 2.5x the "
                       "25-day mean(H−L), AND IBS < 0.3.  Exit when "
                       "price clears yesterday's high, OR when below "
                       "the 300-day SMA.  Designed for QQQ."),
        type        = "rule",
        required_features = ["open", "high", "low", "close", "volume"],
    )

    # Strategy constants — exactly as published in the article.
    HIGH_LOOKBACK    = 10    # rolling-high window for the lower band
    RANGE_LOOKBACK   = 25    # rolling mean(H−L) window
    BAND_MULTIPLIER  = 2.5   # band width in mean-range units
    IBS_BUY_MAX      = 0.30  # IBS must be below this to buy
    REGIME_SMA       = 300   # defensive exit below this

    def predict(self, row: pd.Series) -> tuple[Signal, float]:
        """Single-row predict — works once predict_batch has populated
        the derived columns.  Falls back to "hold, 0.5" on missing data.

        Note: the "exit when close > yesterday's high" rule needs a
        lag column (``prev_high``).  predict_batch precomputes it.
        """
        close       = row.get("close")
        prev_high   = row.get("prev_high")
        sma_300     = row.get("sma_300")
        ibs         = row.get("ibs")
        lower_band  = row.get("qmr_lower_band")
        in_position = row.get("_qmr_in_position", False)

        if any(pd.isna(x) for x in (close, sma_300, lower_band, ibs)):
            return ("hold", 0.50)

        # ── EXITS ─────────────────────────────────────────────────
        # Two triggers, either fires.  We emit the sell signal even
        # when we're not "in position" — the simulator interprets
        # signal=sell as "no entry on this bar", consistent with the
        # rest of the strategy zoo.
        if pd.notna(prev_high) and close > prev_high:
            return ("sell", 0.70)
        if close < sma_300:
            return ("sell", 0.65)

        # ── ENTRY ─────────────────────────────────────────────────
        if close < lower_band and ibs < self.IBS_BUY_MAX:
            # Confidence: how deeply oversold are we?  IBS near 0 +
            # close well below the band = stronger buy.
            band_dive = (lower_band - close) / max(close, 1e-6)
            conf = (
                0.60
                + min(0.20, max(0.0, band_dive) * 5.0)   # how far below band
                + min(0.10, (self.IBS_BUY_MAX - ibs) * 0.5)  # how oversold IBS
            )
            return ("buy", round(min(conf, 0.92), 3))

        return ("hold", 0.55)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Precompute the derived columns once for the whole bar
        history, then defer to BaseModel.predict_batch which iterates
        rows through ``predict``.

        We compute and attach as columns:
          • ibs              — Internal Bar Strength (0..1)
          • qmr_range_mean   — 25-day rolling mean(H−L)
          • qmr_high_n       — 10-day rolling high
          • qmr_lower_band   — high_n − 2.5 × range_mean
          • prev_high        — shifted high for the "yesterday's high" rule
          • sma_300          — defensive regime filter
        """
        df = df.copy()

        if "ibs" not in df.columns:
            denom = (df["high"] - df["low"]).replace(0, pd.NA)
            df["ibs"] = (df["close"] - df["low"]) / denom

        rng = df["high"] - df["low"]
        if "qmr_range_mean" not in df.columns:
            df["qmr_range_mean"] = rng.rolling(
                self.RANGE_LOOKBACK, min_periods=self.RANGE_LOOKBACK
            ).mean()
        if "qmr_high_n" not in df.columns:
            df["qmr_high_n"] = df["high"].rolling(
                self.HIGH_LOOKBACK, min_periods=self.HIGH_LOOKBACK
            ).max()
        if "qmr_lower_band" not in df.columns:
            df["qmr_lower_band"] = (
                df["qmr_high_n"] - self.BAND_MULTIPLIER * df["qmr_range_mean"]
            )
        if "qmr_band_dive" not in df.columns:
            # How far BELOW the band the close is, as a positive
            # ratio.  Negative or zero means the close is at/above
            # the band (no oversold setup).  Useful for Optuna
            # tuning ("only buy when at least X% below band").
            df["qmr_band_dive"] = (
                (df["qmr_lower_band"] - df["close"]) / df["close"].abs()
            ).clip(lower=0)
        if "prev_high" not in df.columns:
            df["prev_high"] = df["high"].shift(1)
        if "sma_300" not in df.columns:
            df["sma_300"] = df["close"].rolling(
                self.REGIME_SMA, min_periods=self.REGIME_SMA
            ).mean()

        return super().predict_batch(df)
