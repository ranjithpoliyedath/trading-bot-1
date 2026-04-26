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
    # Sentiment
    "combined_sentiment": {"label": "Combined sentiment",     "group": "Sentiment"},
    "news_count":         {"label": "News count",             "group": "Sentiment"},
    "st_bullish_ratio":   {"label": "StockTwits bullish %",   "group": "Sentiment"},
    # Price action
    "price_change_1d":    {"label": "Price change 1d",        "group": "Price"},
    "price_change_5d":    {"label": "Price change 5d",        "group": "Price"},
    "volume_ratio":       {"label": "Volume vs avg",          "group": "Price"},
    # Breakout patterns (computed lazily by add_breakout_features)
    "prior_runup_pct":      {"label": "Prior 60d runup %",     "group": "Breakout"},
    "consolidation_range":  {"label": "Consolidation range",   "group": "Breakout"},
    "consolidation_vol_drop": {"label": "Volume dry-up ratio", "group": "Breakout"},
    "contraction_count":    {"label": "VCP contractions",       "group": "Breakout"},
    "breakout_today":       {"label": "Breakout today (1=yes)", "group": "Breakout"},
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
    if symbols is not None:
        return [s.upper() for s in symbols]
    universe = load_universe(eligible_only=True)
    if universe.empty:
        # Fall back to whatever has been processed on disk
        return sorted({
            p.name.split("_")[0]
            for p in Path(DATA_DIR).glob("*_features*.parquet")
        })
    return universe["symbol"].tolist()


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
