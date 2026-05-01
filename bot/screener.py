"""
bot/screener.py
----------------
Stock discovery / screener.

Loads the latest row of every symbol in the universe (preferring the
sentiment-enriched feature file when present) and applies a list of
filter conditions.  Returns a ranked list of matching symbols.

This module is data-only; the dashboard page in
``dashboard/pages/screener.py`` builds a UI on top of it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from bot.config   import DATA_DIR
from bot.universe import load_universe

logger = logging.getLogger(__name__)


# ── Field catalogue ──────────────────────────────────────────────────────────
#
# The screener exposes a curated subset of the columns produced by
# feature_engineer + sentiment pipeline.  Keep this in sync with the
# UI dropdowns; keys are the raw column names, labels are display strings.

SCREENER_FIELDS: dict[str, dict[str, Any]] = {
    # Technical
    "rsi_14":             {"label": "RSI (14)",               "group": "Technical"},
    "macd_hist":          {"label": "MACD histogram",         "group": "Technical"},
    "ema_cross":          {"label": "EMA 9/21 cross",         "group": "Technical"},
    "bb_pct":             {"label": "Bollinger %B",           "group": "Technical"},
    "atr_14":             {"label": "ATR (14)",               "group": "Technical"},
    # EMA stack relations — boolean (use op `==` 1 or `>` 0)
    "above_ema_10":         {"label": "Price > EMA 10",         "group": "EMA"},
    "above_ema_20":         {"label": "Price > EMA 20",         "group": "EMA"},
    "above_ema_50":         {"label": "Price > EMA 50",         "group": "EMA"},
    "above_ema_200":        {"label": "Price > EMA 200",        "group": "EMA"},
    "above_ema_10_20":      {"label": "Price > EMA 10 AND 20",  "group": "EMA"},
    "above_ema_10_20_50":   {"label": "Price > EMA 10, 20, 50", "group": "EMA"},
    "above_all_emas":       {"label": "Price > all EMAs",       "group": "EMA"},
    "below_all_emas":       {"label": "Price < all EMAs",       "group": "EMA"},
    "ema_bull_stack":       {"label": "Bull stack (10>20>50>200)", "group": "EMA"},
    "ema_bear_stack":       {"label": "Bear stack (10<20<50<200)", "group": "EMA"},
    # Raw EMA values — useful for distance/ratio comparisons
    "ema_10":               {"label": "EMA 10",                  "group": "EMA"},
    "ema_20":               {"label": "EMA 20",                  "group": "EMA"},
    "ema_50":               {"label": "EMA 50",                  "group": "EMA"},
    "ema_200":              {"label": "EMA 200",                 "group": "EMA"},
    # Sentiment
    "combined_sentiment": {"label": "Combined sentiment",     "group": "Sentiment"},
    "news_count":         {"label": "News count",             "group": "Sentiment"},
    "st_bullish_ratio":   {"label": "StockTwits bullish %",   "group": "Sentiment"},
    # Price action
    "price_change_1d":    {"label": "Price change 1d",        "group": "Price"},
    "price_change_5d":    {"label": "Price change 5d",        "group": "Price"},
    "volume_ratio":       {"label": "Volume vs avg",          "group": "Price"},
    # ── Multi-period returns + liquidity + range (Best Winners preset) ──
    "close":                {"label": "Close price (USD)",         "group": "Price"},
    "perf_3m":              {"label": "Performance 3-month",       "group": "Momentum"},
    "perf_6m":              {"label": "Performance 6-month",       "group": "Momentum"},
    "perf_1y":              {"label": "Performance 1-year",        "group": "Momentum"},
    "pct_from_52w_low":     {"label": "% from 52-week low",        "group": "Momentum"},
    "dollar_volume_30d":    {"label": "Dollar volume (30d avg)",   "group": "Liquidity"},
    "dollar_volume_today":  {"label": "Dollar volume (today)",     "group": "Liquidity"},
    "adr_pct":              {"label": "Average Daily Range %",     "group": "Volatility"},
    "ema_60":               {"label": "EMA 60",                    "group": "EMA"},
    # Breakout patterns (computed lazily by add_breakout_features)
    "prior_runup_pct":      {"label": "Prior 60d runup %",     "group": "Breakout"},
    "consolidation_range":  {"label": "Consolidation range",   "group": "Breakout"},
    "consolidation_vol_drop": {"label": "Volume dry-up ratio", "group": "Breakout"},
    "contraction_count":    {"label": "VCP contractions",       "group": "Breakout"},
    "breakout_today":       {"label": "Breakout today (1=yes)", "group": "Breakout"},
}


# Indicator presets — friendly names mapped to ready-to-go filter rows.
# The dashboard's "Indicator preset" dropdown applies one of these in
# one click instead of forcing the user to chain raw filter rows.
INDICATOR_PRESETS: dict[str, dict] = {
    "ema_bull_stack": {
        "label":   "EMA bull stack (price > 10 > 20 > 50 > 200)",
        "group":   "EMA",
        "filters": [{"field": "ema_bull_stack", "op": "==", "value": 1}],
    },
    "ema_bear_stack": {
        "label":   "EMA bear stack (price < 10 < 20 < 50 < 200)",
        "group":   "EMA",
        "filters": [{"field": "ema_bear_stack", "op": "==", "value": 1}],
    },
    "above_short_emas": {
        "label":   "Above short EMAs (10 AND 20)",
        "group":   "EMA",
        "filters": [{"field": "above_ema_10_20", "op": "==", "value": 1}],
    },
    "above_short_mid_emas": {
        "label":   "Above 10, 20, AND 50 EMA",
        "group":   "EMA",
        "filters": [{"field": "above_ema_10_20_50", "op": "==", "value": 1}],
    },
    "above_long_ema": {
        "label":   "Above 200 EMA only (long-term uptrend)",
        "group":   "EMA",
        "filters": [{"field": "above_ema_200", "op": "==", "value": 1}],
    },
    "above_50_below_200": {
        "label":   "Above 50 EMA but below 200 (early reversal)",
        "group":   "EMA",
        "filters": [
            {"field": "above_ema_50",  "op": "==", "value": 1},
            {"field": "above_ema_200", "op": "==", "value": 0},
        ],
    },
    "above_all_emas": {
        "label":   "Above all EMAs (10, 20, 50, 200)",
        "group":   "EMA",
        "filters": [{"field": "above_all_emas", "op": "==", "value": 1}],
    },
    "below_all_emas": {
        "label":   "Below all EMAs (full breakdown)",
        "group":   "EMA",
        "filters": [{"field": "below_all_emas", "op": "==", "value": 1}],
    },
    "rsi_oversold": {
        "label":   "RSI oversold (<30)",
        "group":   "Momentum",
        "filters": [{"field": "rsi_14", "op": "<", "value": 30}],
    },
    "rsi_overbought": {
        "label":   "RSI overbought (>70)",
        "group":   "Momentum",
        "filters": [{"field": "rsi_14", "op": ">", "value": 70}],
    },
    "macd_bullish_above_50": {
        "label":   "MACD bullish + above EMA 50",
        "group":   "Trend",
        "filters": [
            {"field": "macd_hist",     "op": ">",  "value": 0},
            {"field": "above_ema_50",  "op": "==", "value": 1},
        ],
    },
    "high_volume_breakout": {
        "label":   "High volume + breakout today",
        "group":   "Breakout",
        "filters": [
            {"field": "volume_ratio",  "op": ">",  "value": 1.5},
            {"field": "breakout_today", "op": "==", "value": 1},
        ],
    },
    "bullish_sentiment_uptrend": {
        "label":   "Bullish sentiment + uptrend (above 50 EMA)",
        "group":   "Sentiment",
        "filters": [
            {"field": "combined_sentiment", "op": ">",  "value": 0.2},
            {"field": "above_ema_50",       "op": "==", "value": 1},
        ],
    },
    "best_winners": {
        # User-supplied "best winners" momentum/quality screen — the
        # filter set seen in popular US momentum screeners.  Looks
        # for stocks that are well off their 52-week low, have
        # positive 3M/6M/1Y returns, decent dollar-volume turnover,
        # high enough ADR to be tradable, and sit above the 60-EMA.
        "label":   "Best Winners (momentum + quality + liquidity)",
        "group":   "Winners",
        "filters": [
            # Quality price-floor (ignore penny stocks)
            {"field": "close",                "op": ">",  "value": 1.0},
            # 70%+ off 52-week low — meaningful run-up already
            {"field": "pct_from_52w_low",     "op": ">=", "value": 0.70},
            # Positive trailing returns at three horizons
            {"field": "perf_3m",              "op": ">",  "value": 0.0},
            {"field": "perf_6m",              "op": ">",  "value": 0.0},
            {"field": "perf_1y",              "op": ">",  "value": 0.0},
            # Liquidity: ≥$15M average daily $ volume + ≥$5M today
            {"field": "dollar_volume_30d",    "op": ">=", "value": 15_000_000},
            {"field": "dollar_volume_today",  "op": ">=", "value": 5_000_000},
            # Trend filter: above the 60-day EMA
            {"field": "above_ema_50",         "op": "==", "value": 1},
            # Volatility: enough daily range to be worth trading
            {"field": "adr_pct",              "op": ">=", "value": 4.5},
        ],
    },
}

OPS = {
    ">":  lambda a, b: a >  b,
    ">=": lambda a, b: a >= b,
    "<":  lambda a, b: a <  b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


@dataclass
class Filter:
    """A single screener condition: ``<field> <op> <value>``."""
    field: str
    op:    str
    value: float

    def matches(self, row: pd.Series) -> bool:
        if self.field not in row.index:
            return False
        cell = row[self.field]
        if pd.isna(cell):
            return False
        op = OPS.get(self.op)
        if op is None:
            raise ValueError(f"Unsupported op: {self.op}")
        try:
            return bool(op(float(cell), float(self.value)))
        except (TypeError, ValueError):
            return False


@dataclass
class ScreenerResult:
    symbol:    str
    close:     float
    matched:   dict[str, float]      = field(default_factory=dict)
    extras:    dict[str, float]      = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol":  self.symbol,
            "close":   self.close,
            "matched": self.matched,
            "extras":  self.extras,
        }


# ── Loaders ──────────────────────────────────────────────────────────────────

def _load_latest(symbol: str) -> Optional[pd.Series]:
    """
    Return the last row of ``symbol`` features (sentiment-enriched if
    present), with breakout columns appended on the fly.
    """
    from bot.patterns import add_breakout_features

    for tag in ("features_with_sentiment", "features"):
        path = DATA_DIR / f"{symbol}_{tag}.parquet"
        if path.exists():
            try:
                df = pd.read_parquet(path)
                if df.empty:
                    return None
                df = add_breakout_features(df)
                return df.iloc[-1]
            except Exception as exc:
                logger.warning("Failed to load %s for %s: %s", tag, symbol, exc)
    return None


def _candidate_symbols(symbols: Optional[Iterable[str]] = None) -> list[str]:
    """
    Return the symbol scan list, ordered most-tradeable first.

    Behaviour:
      * If ``symbols`` is given explicitly, use that list verbatim.
      * Otherwise read the eligible universe and sort by 14-day average
        volume descending so the most liquid names get scanned first.
      * Filter out symbols that don't have a processed features parquet
        on disk — those would just `continue` in the scan loop and
        silently shrink the effective universe size, which makes
        ``max_symbols`` accidentally drop the names you actually want.
      * Fall back to whatever has been processed on disk if the
        universe parquet is unavailable.
    """
    if symbols is not None:
        return [s.upper() for s in symbols]

    on_disk = {
        p.name.split("_")[0]
        for p in Path(DATA_DIR).glob("*_features*.parquet")
    }

    universe = load_universe(eligible_only=True)
    if universe.empty:
        return sorted(on_disk)

    df = universe.copy()
    if "avg_volume_14d" in df.columns:
        df = df.sort_values("avg_volume_14d", ascending=False)
    ordered = df["symbol"].tolist()
    return [s for s in ordered if s in on_disk] if on_disk else ordered


# ── Public API ───────────────────────────────────────────────────────────────

def run_screener(
    filters:  list[Filter],
    symbols:  Optional[Iterable[str]] = None,
    sort_by:  Optional[str]           = None,
    descending: bool                  = True,
    limit:    int                     = 50,
) -> list[dict[str, Any]]:
    """
    Run all filter conditions across the universe and return matches.

    Args:
        filters:     List of ``Filter`` conditions, all combined with AND.
        symbols:     Restrict scan to these symbols.  Defaults to universe.
        sort_by:     Field to sort matched results by.  Defaults to the
                     first filter's field; falls back to ``volume_ratio``.
        descending:  Sort direction.
        limit:       Maximum number of rows to return.

    Returns:
        List of result dicts ready for table rendering.
    """
    candidates = _candidate_symbols(symbols)
    logger.info("Screening %d symbols against %d filters.",
                len(candidates), len(filters))

    matches: list[ScreenerResult] = []
    for symbol in candidates:
        row = _load_latest(symbol)
        if row is None:
            continue
        if not all(f.matches(row) for f in filters):
            continue

        matched = {}
        for f in filters:
            try:
                matched[f.field] = round(float(row[f.field]), 4)
            except (TypeError, ValueError):
                matched[f.field] = None
        extras = {}
        for col in ("rsi_14", "combined_sentiment", "volume_ratio",
                    "price_change_1d"):
            if col in row.index and not pd.isna(row[col]):
                try:
                    extras[col] = round(float(row[col]), 4)
                except (TypeError, ValueError):
                    pass

        close = float(row.get("close", 0)) if "close" in row.index else 0.0
        matches.append(ScreenerResult(
            symbol=symbol, close=round(close, 2),
            matched=matched, extras=extras,
        ))

    sort_key = sort_by or (filters[0].field if filters else "volume_ratio")

    def _key(r: ScreenerResult) -> float:
        return r.matched.get(sort_key, r.extras.get(sort_key, 0.0)) or 0.0

    matches.sort(key=_key, reverse=descending)
    return [r.to_dict() for r in matches[:limit]]


def list_fields() -> list[dict[str, str]]:
    """Return the field catalogue as dropdown options."""
    return [
        {"value": key, "label": meta["label"], "group": meta["group"]}
        for key, meta in SCREENER_FIELDS.items()
    ]
