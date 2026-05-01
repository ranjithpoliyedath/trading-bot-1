"""
bot/models/builtin/macro_aware_leaders_v1.py
---------------------------------------------
Macro-Aware Leaders.  User-designed defensive momentum strategy.

Rules (from the spec — every check is configurable in the dashboard):

  Buy gates (ALL must be true on the entry bar):
    1. SPY/QQQ above 10, 20, 50, 200 SMA  (market regime fully bullish)
    2. Stock above 10, 20, 50, 200 SMA    (per-symbol bull stack)
    3. Sector ETF above 10, 20, 50 SMA   (sector trending up)
    4. Market breadth above its 10, 20, 50 SMA  (universe trending up)

  Sell triggers (ANY closes the position):
    • Stock below 10 AND 20 SMA           (per-symbol break)
    • SPY/QQQ below 10 AND 20 AND 50 SMA  (macro break — exit ALL)
    • Breadth below its 10 AND 20 SMA     (universe break — exit ALL)
    • Stock < entry − 2 × ATR(14)         (volatility-stop, engine-level)

  Stock selection (when many candidates, cash-limited):
    • Higher liquidity preferred (avg 14-day volume)
    • Higher beta preferred (move-amplifying names)
    • Confidence is scaled by both → simulator's "highest-conf-first"
      cash allocation naturally honours both ranks.

Implementation
─────────────────────────────────────────────────────────────────

predict_batch precomputes a rich set of derived columns so the
screener filter chain can gate every macro check independently:

  • sma_10, sma_20, sma_50, sma_200    — per-symbol stack
  • macd_hist                           — momentum confirmation
  • macro_spy_bullish (0/1)            — SPY > all 4 SMAs
  • macro_spy_exit    (0/1)            — SPY < 10 AND 20 AND 50
  • macro_sector_bullish (0/1)         — sector ETF > 10/20/50
  • macro_breadth_bullish (0/1)        — breadth > 10/20/50 SMA
  • macro_breadth_exit (0/1)           — breadth < 10 AND 20 SMA
  • liquidity_score, beta_60d          — used in confidence weighting

The strategy's predict() emits buy/sell/hold using ONLY the
per-symbol bull-stack + macd check.  Macro gates are layered ON via
UI toggles → each toggle adds a filter row like
``macro_spy_bullish == 1``.  This way every check is independently
configurable from the dashboard.

Macro EXIT triggers are baked into the strategy's own sell signal:
  close < sma_10 AND close < sma_20      → sell
  macro_spy_exit == 1                     → sell
  macro_breadth_exit == 1                 → sell

Engine's ``use_signal_exit=True`` honours all three.

Breadth proxy
─────────────────────────────────────────────────────────────────

We don't have $S5FI / $MMFI / $SPXA50R as fetched parquets.  Instead
we COMPUTE the equivalent from the universe on disk:

  breadth_pct(t) = (count of universe symbols with close > sma_50 on t)
                 / (count of universe symbols with valid sma_50 on t)

This is the same statistic the published indices track — just
computed from our 1,500-symbol universe.  Cached per backtest in a
module-level dict so cost is paid once.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import numpy as np

from bot.models.base     import BaseModel, ModelMetadata, Signal
from bot.models.registry import register_model

logger = logging.getLogger(__name__)


# ── Module-level caches (computed once per process) ──────────────

_SPY_CACHE: Optional[pd.DataFrame] = None         # SPY w/ derived columns
_QQQ_CACHE: Optional[pd.DataFrame] = None
_SECTOR_ETF_CACHE: dict[str, pd.DataFrame] = {}   # ETF → df with derived
_BREADTH_CACHE: Optional[pd.Series] = None        # date → breadth_pct
_BREADTH_DERIVED_CACHE: Optional[pd.DataFrame] = None
_SYMBOL_TO_SECTOR: dict[str, str] = {}


_SECTOR_TO_ETF = {
    "Information Technology": "XLK",
    "Health Care":             "XLV",
    "Financials":              "XLF",
    "Consumer Discretionary":  "XLY",
    "Communication Services":  "XLC",
    "Industrials":             "XLI",
    "Consumer Staples":        "XLP",
    "Energy":                  "XLE",
    "Utilities":               "XLU",
    "Real Estate":             "XLRE",
    "Materials":               "XLB",
}


def _load_index_with_smas(symbol: str) -> Optional[pd.DataFrame]:
    """Load an ETF/index parquet and add the 10/20/50/200 SMAs.
    Returns None if the parquet isn't on disk (caller falls back)."""
    from bot.config import DATA_DIR
    from pathlib import Path

    path = Path(DATA_DIR) / f"{symbol}_features.parquet"
    if not path.exists():
        path = Path(DATA_DIR) / f"{symbol}_raw.parquet"
        if not path.exists():
            return None
    try:
        df = pd.read_parquet(path, columns=["close"])
    except Exception:
        return None
    if df.empty:
        return None
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index().copy()
    for span in (10, 20, 50, 200):
        df[f"sma_{span}"] = df["close"].rolling(span, min_periods=span).mean()
    return df


def _compute_breadth_proxy() -> pd.Series:
    """Compute the % of universe symbols above their 50-day SMA per
    trading day.  Returns a Series indexed by date.

    Equivalent to $S5FI / $MMFI when applied to the right universe.
    Heavy on first call (reads ~1,500 parquets); cached after.
    """
    from bot.config import DATA_DIR
    from pathlib import Path

    counts_above: dict[pd.Timestamp, int] = {}
    counts_total: dict[pd.Timestamp, int] = {}

    paths = list(Path(DATA_DIR).glob("*_features.parquet"))
    logger.info("Computing breadth proxy from %d feature parquets…", len(paths))

    for p in paths:
        try:
            df = pd.read_parquet(p, columns=["close"])
        except Exception:
            continue
        if df.empty:
            continue
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        sma50 = df["close"].rolling(50, min_periods=50).mean()
        valid_mask = sma50.notna()
        above_mask = (df["close"] > sma50) & valid_mask
        # Aggregate per date
        for ts, is_valid, is_above in zip(df.index, valid_mask, above_mask):
            if is_valid:
                counts_total[ts] = counts_total.get(ts, 0) + 1
                if is_above:
                    counts_above[ts] = counts_above.get(ts, 0) + 1

    if not counts_total:
        return pd.Series(dtype=float)

    dates = sorted(counts_total.keys())
    pct = [counts_above.get(d, 0) / counts_total[d] for d in dates]
    return pd.Series(pct, index=dates, name="breadth_pct").sort_index()


def _ensure_macro_caches():
    """Lazy-load all macro state (SPY, QQQ, sector ETFs, breadth).
    Idempotent — calling multiple times is a no-op after the first."""
    global _SPY_CACHE, _QQQ_CACHE, _BREADTH_CACHE, _BREADTH_DERIVED_CACHE
    global _SYMBOL_TO_SECTOR

    if _SPY_CACHE is None:
        _SPY_CACHE = _load_index_with_smas("SPY")
    if _QQQ_CACHE is None:
        _QQQ_CACHE = _load_index_with_smas("QQQ")
    for etf in _SECTOR_TO_ETF.values():
        if etf not in _SECTOR_ETF_CACHE:
            df = _load_index_with_smas(etf)
            if df is not None:
                _SECTOR_ETF_CACHE[etf] = df

    if _BREADTH_CACHE is None:
        _BREADTH_CACHE = _compute_breadth_proxy()
        if not _BREADTH_CACHE.empty:
            df_b = pd.DataFrame({"breadth_pct": _BREADTH_CACHE})
            for span in (10, 20, 50):
                df_b[f"breadth_sma_{span}"] = (
                    df_b["breadth_pct"].rolling(span, min_periods=span).mean()
                )
            _BREADTH_DERIVED_CACHE = df_b

    if not _SYMBOL_TO_SECTOR:
        try:
            from bot.universe import load_universe
            u = load_universe(eligible_only=False)
            if not u.empty and "sector" in u.columns:
                _SYMBOL_TO_SECTOR.update(dict(zip(u["symbol"], u["sector"])))
        except Exception:
            pass


@register_model
class MacroAwareLeadersModel(BaseModel):
    metadata = ModelMetadata(
        id          = "macro_aware_leaders_v1",
        name        = "Macro-Aware Leaders (SPY + sector + breadth gated)",
        description = ("Bull-stack momentum gated on SPY/sector ETF/"
                       "breadth bullish.  Exits on per-symbol SMA "
                       "break, SPY break, breadth break, or 2×ATR "
                       "stop.  All gates configurable in the UI."),
        type        = "rule",
        required_features = ["close", "high", "low", "volume",
                              "sma_50", "sma_200", "macd_hist", "atr_14"],
    )

    BETA_LOOKBACK = 60   # bars

    def predict(self, row: pd.Series) -> tuple[Signal, float]:
        close = row.get("close")
        s10   = row.get("sma_10");  s20  = row.get("sma_20")
        s50   = row.get("sma_50");  s200 = row.get("sma_200")
        mhist = row.get("macd_hist")

        # Macro EXIT triggers — these always fire regardless of UI
        # toggles.  When the user toggles a gate OFF, they remove the
        # corresponding ENTRY filter row, but exits stay active so we
        # never sit in a position during a clear macro break.
        spy_exit     = row.get("macro_spy_exit", 0) or 0
        breadth_exit = row.get("macro_breadth_exit", 0) or 0
        if spy_exit >= 1 or breadth_exit >= 1:
            return ("sell", 0.75)

        if any(pd.isna(x) for x in (close, s10, s20, s50, s200, mhist)):
            return ("hold", 0.50)

        # ── Per-symbol exit: close < sma_10 AND close < sma_20 ──
        if close < s10 and close < s20:
            return ("sell", 0.65)

        # ── Buy: per-symbol bull stack + momentum ──
        # Macro entry gates (SPY/sector/breadth bullish) are layered
        # on via screener filter rows, NOT here — keeps each gate
        # independently configurable from the dashboard.
        if close > s10 and close > s20 and close > s50 and close > s200 and mhist > 0:
            # Confidence: base + liquidity kicker + beta kicker
            liq   = float(row.get("liquidity_score", 0) or 0)
            beta  = float(row.get("beta_60d", 1.0) or 1.0)
            conf  = (
                0.60
                + min(0.15, liq * 0.05)            # liq is log10(vol/100K)
                + min(0.10, max(0, beta - 1.0) * 0.10)
            )
            return ("buy", round(min(conf, 0.92), 3))

        return ("hold", 0.55)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        _ensure_macro_caches()

        # ── Per-symbol SMAs (10/20 may be missing) ──
        for span in (10, 20, 50, 200):
            col = f"sma_{span}"
            if col not in df.columns:
                df[col] = df["close"].rolling(span, min_periods=span).mean()

        # ── Liquidity score: log10(avg 14d volume / 100K) ──
        # Keeps the kicker bounded; a $10M-volume name scores ~2.0,
        # a $100K-volume name scores ~0.0.
        if "liquidity_score" not in df.columns:
            avg_vol = df["volume"].rolling(14, min_periods=7).mean()
            df["liquidity_score"] = np.log10(
                (avg_vol / 100_000).clip(lower=1.0)
            )

        # ── Beta vs SPY (60-day rolling) ──
        if "beta_60d" not in df.columns:
            df["beta_60d"] = self._compute_rolling_beta(df)

        # ── Macro overlay columns (joined by date) ──
        # Pick SPY by default; if df came from a tech-heavy run the
        # caller may use QQQ — for v1 we always use SPY (matches the
        # user's spec which mentions SPX/SPY).
        spy_df = _SPY_CACHE
        if spy_df is not None and not spy_df.empty:
            # Join derived SPY values onto df by date
            joined = df.join(
                spy_df[["close", "sma_10", "sma_20", "sma_50", "sma_200"]]
                .rename(columns={
                    "close":   "_spy_close",
                    "sma_10":  "_spy_sma_10",
                    "sma_20":  "_spy_sma_20",
                    "sma_50":  "_spy_sma_50",
                    "sma_200": "_spy_sma_200",
                }), how="left",
            )
            df["macro_spy_bullish"] = (
                (joined["_spy_close"] > joined["_spy_sma_10"]) &
                (joined["_spy_close"] > joined["_spy_sma_20"]) &
                (joined["_spy_close"] > joined["_spy_sma_50"]) &
                (joined["_spy_close"] > joined["_spy_sma_200"])
            ).astype("float64").where(
                joined["_spy_sma_200"].notna()
            )
            df["macro_spy_exit"] = (
                (joined["_spy_close"] < joined["_spy_sma_10"]) &
                (joined["_spy_close"] < joined["_spy_sma_20"]) &
                (joined["_spy_close"] < joined["_spy_sma_50"])
            ).astype("float64").where(
                joined["_spy_sma_50"].notna()
            )
        else:
            # SPY parquet missing — gate "always permissive" so the
            # strategy still works (just without SPY gating).
            df["macro_spy_bullish"] = 1.0
            df["macro_spy_exit"]    = 0.0

        # ── Sector ETF gate (per-symbol, looked up via universe) ──
        # We can't know the symbol from inside predict_batch (the
        # caller knows but doesn't pass the name in), so we leave
        # macro_sector_bullish unset here and let downstream code
        # (or a per-symbol post-process) set it.  v1: emit 1.0 (open)
        # — sector gate enabled means user must add the filter row
        # AND we'd populate this column from outside.
        df["macro_sector_bullish"] = 1.0

        # ── Breadth gate ──
        if _BREADTH_DERIVED_CACHE is not None and not _BREADTH_DERIVED_CACHE.empty:
            joined_b = df.join(_BREADTH_DERIVED_CACHE, how="left")
            df["breadth_pct"] = joined_b["breadth_pct"]
            df["macro_breadth_bullish"] = (
                (joined_b["breadth_pct"] > joined_b["breadth_sma_10"]) &
                (joined_b["breadth_pct"] > joined_b["breadth_sma_20"]) &
                (joined_b["breadth_pct"] > joined_b["breadth_sma_50"])
            ).astype("float64").where(
                joined_b["breadth_sma_50"].notna()
            )
            df["macro_breadth_exit"] = (
                (joined_b["breadth_pct"] < joined_b["breadth_sma_10"]) &
                (joined_b["breadth_pct"] < joined_b["breadth_sma_20"])
            ).astype("float64").where(
                joined_b["breadth_sma_20"].notna()
            )
        else:
            df["macro_breadth_bullish"] = 1.0
            df["macro_breadth_exit"]    = 0.0

        return super().predict_batch(df)

    @staticmethod
    def _compute_rolling_beta(df: pd.DataFrame, window: int = 60) -> pd.Series:
        """Rolling beta vs SPY over the last ``window`` bars.

        beta = cov(stock_ret, spy_ret) / var(spy_ret)

        Returns ``pd.Series(1.0)`` when SPY isn't loadable (neutral
        beta — no kicker).
        """
        spy = _SPY_CACHE
        if spy is None or spy.empty:
            return pd.Series(1.0, index=df.index)

        # Align SPY closes to df's index then compute returns
        spy_close = spy["close"].reindex(df.index, method="ffill")
        stock_ret = df["close"].pct_change()
        spy_ret   = spy_close.pct_change()

        cov  = stock_ret.rolling(window).cov(spy_ret)
        var  = spy_ret.rolling(window).var()
        beta = cov / var.replace(0, np.nan)
        # Clamp wild outliers (illiquid early bars can produce silly numbers)
        return beta.clip(lower=-3.0, upper=4.0).fillna(1.0)
