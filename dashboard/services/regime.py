"""
dashboard/services/regime.py
-----------------------------
Market-regime + sector-regime detection for the backtest engine's
optional "regime exit" feature.

A symbol is in a DOWN-trend when its close is below the **200-day
simple moving average** (SMA-200) of the relevant index/ETF.  This
is the classic "investor-grade" trend-filter rule: above 200-day SMA
= bull, below = bear.  We use SMA (not EMA) because it's the
industry-standard trend filter referenced everywhere from O'Neil's
CAN-SLIM to managed-futures literature.

When the user enables either regime exit, the backtest simulator
force-sells held positions on bars where the relevant regime is
down — overriding the strategy's own buy/sell logic for that bar.

Two separate regimes:

  • **Market regime** — the broad index.  Defaults to SPY for
    diversified universes; switches to QQQ when the universe is
    >50% Information-Technology constituents (so a tech-heavy
    bot's "market" is the Nasdaq-100, not the S&P).

  • **Sector regime** — each held symbol's GICS sector mapped to
    the corresponding SPDR sector ETF (XLK, XLF, XLV, ...).  Looks
    up the sector via the universe parquet's ``sector`` column.

Pre-loads the relevant ETF feature parquets ONCE up-front, indexes
their down-trend Series by date, then offers cheap per-(symbol, date)
lookups during the simulator hot loop.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from bot.config import DATA_DIR

logger = logging.getLogger(__name__)


# Sector → SPDR sector ETF.  Sourced from the eligible-universe parquet's
# ``sector`` column.  GICS short names with the conventional ETF mapping.
_SECTOR_ETF: dict[str, str] = {
    "Information Technology":   "XLK",
    "Health Care":              "XLV",
    "Financials":               "XLF",
    "Consumer Discretionary":   "XLY",
    "Communication Services":   "XLC",
    "Industrials":              "XLI",
    "Consumer Staples":         "XLP",
    "Energy":                   "XLE",
    "Utilities":                "XLU",
    "Real Estate":              "XLRE",
    "Materials":                "XLB",
}


def _load_close_for_regime(symbol: str) -> pd.DataFrame:
    """Read just the close column for an ETF/index parquet.  We
    compute the 200-day SMA on the fly rather than relying on a
    pre-engineered column, so this works even on legacy parquets
    that pre-date the SMA-200 feature.

    Returns an empty DataFrame if the parquet isn't on disk."""
    path = DATA_DIR / f"{symbol}_features.parquet"
    if not path.exists():
        # Fall back to the raw bars
        path = DATA_DIR / f"{symbol}_raw.parquet"
        if not path.exists():
            return pd.DataFrame()
    try:
        df = pd.read_parquet(path, columns=["close"])
        if df.empty:
            return df
        # Strip tz so downstream comparisons with strategy data work
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df.sort_index()
    except Exception as exc:
        logger.warning("Failed to load %s feature parquet: %s", symbol, exc)
        return pd.DataFrame()


def _downtrend_series(df: pd.DataFrame, sma_window: int = 200) -> pd.Series:
    """Boolean Series — True when close < SMA_200.

    Uses simple moving average (not exponential) to match the
    investor-grade "200-day SMA" trend filter convention.

    Args:
        df: DataFrame with a ``close`` column.
        sma_window: Window in bars (200 = the classic trend filter).

    Returns:
        Boolean Series aligned with ``df.index``.  NaN bars (during
        the 200-day warm-up) are treated as False so we don't force-
        sell on the very first bar of a new symbol.
    """
    if df.empty or "close" not in df.columns:
        return pd.Series(dtype=bool)
    sma = df["close"].rolling(window=sma_window, min_periods=sma_window).mean()
    return (df["close"] < sma).fillna(False)


def _pick_market_etf(symbols: list[str]) -> str:
    """Choose the market index for the universe.  >50% tech → QQQ;
    otherwise SPY."""
    try:
        from bot.universe import load_universe
        u = load_universe(eligible_only=False)
        if u.empty or "sector" not in u.columns:
            return "SPY"
        sub = u[u["symbol"].isin(symbols)]
        if sub.empty:
            return "SPY"
        n_tech = (sub["sector"] == "Information Technology").sum()
        if n_tech / len(sub) > 0.50:
            return "QQQ"
        return "SPY"
    except Exception:
        return "SPY"


class RegimeChecker:
    """Pre-loaded regime lookup.  Build once per backtest run, then
    query cheaply per-(symbol, timestamp).

    Args:
        symbols:     The full universe being backtested.  Used to
                     pick SPY vs QQQ as the market index.
        use_market:  Enable market-regime exit?
        use_sector:  Enable sector-regime exit?
    """

    def __init__(
        self,
        symbols:    list[str],
        use_market: bool = False,
        use_sector: bool = False,
    ):
        self.use_market = bool(use_market)
        self.use_sector = bool(use_sector)
        self.market_etf  = ""
        self._market_down: pd.Series = pd.Series(dtype=bool)
        self._sector_down: dict[str, pd.Series] = {}
        self._sym_to_sector: dict[str, str]   = {}
        self._loaded_etfs: list[str]           = []
        self._missing_etfs: list[str]          = []

        if not (use_market or use_sector):
            return

        # Resolve sector lookups for the universe up front
        if use_sector:
            self._build_sector_map(symbols)

        if use_market:
            self.market_etf = _pick_market_etf(symbols)
            df = _load_close_for_regime(self.market_etf)
            if df.empty:
                self._missing_etfs.append(self.market_etf)
                self.use_market = False
            else:
                self._market_down = _downtrend_series(df)
                self._loaded_etfs.append(self.market_etf)

        if use_sector:
            needed = sorted({etf for etf in self._sym_to_sector.values()
                                  if etf})
            for etf in needed:
                df = _load_close_for_regime(etf)
                if df.empty:
                    self._missing_etfs.append(etf)
                    continue
                self._sector_down[etf] = _downtrend_series(df)
                self._loaded_etfs.append(etf)

    def _build_sector_map(self, symbols: list[str]) -> None:
        try:
            from bot.universe import load_universe
            u = load_universe(eligible_only=False)
            if u.empty or "sector" not in u.columns:
                return
            sub = u[u["symbol"].isin(symbols)][["symbol", "sector"]]
            for _, row in sub.iterrows():
                etf = _SECTOR_ETF.get(row["sector"], "")
                if etf:
                    self._sym_to_sector[row["symbol"]] = etf
        except Exception as exc:
            logger.warning("Sector map build failed: %s", exc)

    # ── Public API ────────────────────────────────────────────────

    def is_market_down(self, ts: pd.Timestamp) -> bool:
        if not self.use_market or self._market_down.empty:
            return False
        try:
            v = self._market_down.asof(ts)
            return bool(v) if pd.notna(v) else False
        except (KeyError, TypeError):
            return False

    def is_sector_down(self, symbol: str, ts: pd.Timestamp) -> bool:
        if not self.use_sector:
            return False
        etf = self._sym_to_sector.get(symbol, "")
        if not etf:
            return False
        series = self._sector_down.get(etf)
        if series is None or series.empty:
            return False
        try:
            v = series.asof(ts)
            return bool(v) if pd.notna(v) else False
        except (KeyError, TypeError):
            return False

    def regime_exit_reason(
        self, symbol: str, ts: pd.Timestamp,
    ) -> Optional[str]:
        """Convenience: return the exit_reason string when ANY enabled
        regime is down at this bar, else None."""
        if self.is_market_down(ts):
            return f"market_regime ({self.market_etf})"
        if self.is_sector_down(symbol, ts):
            etf = self._sym_to_sector.get(symbol, "?")
            return f"sector_regime ({etf})"
        return None

    def status_summary(self) -> dict:
        """Diagnostic summary the dashboard surfaces near the result."""
        return {
            "use_market":      self.use_market,
            "use_sector":      self.use_sector,
            "market_etf":      self.market_etf,
            "loaded_etfs":     sorted(set(self._loaded_etfs)),
            "missing_etfs":    sorted(set(self._missing_etfs)),
            "n_symbols_with_sector": sum(
                1 for s in self._sym_to_sector.values() if s
            ),
        }
