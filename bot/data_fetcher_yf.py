"""
bot/data_fetcher_yf.py
----------------------
Yahoo Finance OHLCV fetcher.  Free, no API key, daily history back to
~2010 for most US equities (longer for SPY-class indices).

Why this exists alongside the Alpaca fetcher:
  • Alpaca's free IEX feed has a hard floor at ~2020-07-27 (~5.8 years
    of history).  Asking for 10/15/20 years all return the same set.
  • The paid SIP feed unlocks deeper history but costs ~$99/mo.
  • yfinance gives 16+ years of daily bars for free.

Trade-offs:
  • Unofficial API — Yahoo can rate-limit / break it without warning.
  • Slightly different bar volumes vs Alpaca (different aggregation).
  • Best for *backtesting depth*; live trading should still use Alpaca
    via bot.data_fetcher.DataFetcher.

This class mirrors the public surface of ``DataFetcher`` so the
pipeline can switch sources by changing one import / arg.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class YFinanceDataFetcher:
    """
    Fetches OHLCV bars from Yahoo Finance (free, no auth).

    Public surface intentionally mirrors ``DataFetcher`` so callers can
    swap implementations:

        fetcher = YFinanceDataFetcher()
        bars    = fetcher.fetch_bars(symbols=["AAPL", "MSFT"], lookback_days=365*10)

    All bars are auto-adjusted by default (split + dividend) — that's
    what you want for backtesting so corporate actions don't fake-fire
    signals.  Pass ``auto_adjust=False`` to get raw bars instead.
    """

    def __init__(self, auto_adjust: bool = True):
        # Lazy import so the ``import bot.data_fetcher_yf`` call doesn't
        # fail when yfinance isn't installed (e.g., live-trading-only
        # deployments).  Surface the install hint when fetch_bars is
        # actually called.
        try:
            import yfinance as yf
            self._yf = yf
        except ImportError as exc:
            self._yf = None
            self._import_error = exc

        self.auto_adjust = auto_adjust

        if self._yf is not None:
            logger.info("YFinanceDataFetcher initialised "
                         "(auto_adjust=%s).", auto_adjust)

    # ── Public API ─────────────────────────────────────────────────

    def fetch_bars(
        self,
        symbols:        list[str],
        start:          Optional[datetime] = None,
        end:            Optional[datetime] = None,
        lookback_days:  int = 365,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch daily bars for one or more symbols.

        Args:
            symbols:       List of ticker symbols.
            start, end:    Explicit date range — overrides lookback_days.
            lookback_days: How far back to fetch from today.  Default 1y.

        Returns:
            ``{symbol: DataFrame}`` mapping.  Each DataFrame has
            ``open / high / low / close / volume`` columns indexed by a
            naive timestamp (UTC midnight).  Symbols that returned no
            data map to an empty DataFrame.
        """
        if self._yf is None:
            raise ImportError(
                "yfinance is not installed.  Run: pip install yfinance"
            ) from self._import_error

        if not symbols:
            return {}

        if start is None:
            start = datetime.utcnow() - timedelta(days=lookback_days)
        if end is None:
            end = datetime.utcnow()

        # Normalise dates to naive (yfinance accepts strings or datetimes)
        start_str = start.strftime("%Y-%m-%d") if hasattr(start, "strftime") else str(start)
        end_str   = end.strftime("%Y-%m-%d")   if hasattr(end,   "strftime") else str(end)

        logger.info("yfinance: fetching %d symbols, %s → %s",
                    len(symbols), start_str, end_str)

        # Single batch download — yfinance threads internally with
        # threads=True.  Returns a multi-column DataFrame indexed by
        # date when len(symbols) > 1.
        try:
            df = self._yf.download(
                tickers      = symbols,
                start        = start_str,
                end          = end_str,
                interval     = "1d",
                auto_adjust  = self.auto_adjust,
                progress     = False,
                group_by     = "ticker",
                threads      = True,
                actions      = False,
            )
        except Exception as exc:
            logger.error("yfinance bulk download failed: %s", exc)
            return {sym: pd.DataFrame() for sym in symbols}

        out: dict[str, pd.DataFrame] = {}

        if len(symbols) == 1:
            # Single-symbol request: yfinance returns a flat DataFrame
            # with simple column names (Open/High/Low/Close/Volume).
            out[symbols[0]] = self._normalise_single(df)
        else:
            # Multi-symbol: returns a MultiIndex on columns
            # ((symbol, "Open"), (symbol, "High"), ...).  Slice per sym.
            for sym in symbols:
                try:
                    if isinstance(df.columns, pd.MultiIndex):
                        sub = df[sym].copy() if sym in df.columns.get_level_values(0) else pd.DataFrame()
                    else:
                        sub = pd.DataFrame()  # unexpected shape
                    out[sym] = self._normalise_single(sub)
                except (KeyError, ValueError) as exc:
                    logger.warning("yfinance: no data for %s (%s)", sym, exc)
                    out[sym] = pd.DataFrame()

        loaded = sum(1 for v in out.values() if not v.empty)
        logger.info("yfinance: returned data for %d/%d symbols.",
                    loaded, len(symbols))
        return out

    def fetch_single(
        self,
        symbol:        str,
        lookback_days: int = 365,
    ) -> pd.DataFrame:
        """Convenience wrapper for one symbol."""
        return self.fetch_bars(symbols=[symbol],
                                lookback_days=lookback_days).get(symbol, pd.DataFrame())

    # ── Internals ──────────────────────────────────────────────────

    @staticmethod
    def _normalise_single(df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame in the canonical ``open/high/low/close/volume``
        layout the rest of the pipeline expects.

        yfinance gives capitalised column names; the feature engineer
        and downstream code work in lowercase.  Drop the ``Adj Close``
        column when present (auto_adjust=True already folded that into
        ``Close``)."""
        if df is None or df.empty:
            return pd.DataFrame()

        out = df.copy()
        out.columns = [c.lower().replace(" ", "_") for c in out.columns]

        # When auto_adjust=False yfinance returns 'adj_close' too —
        # the rest of the codebase doesn't use it, drop to keep
        # parity with Alpaca's output shape.
        out = out.drop(columns=[c for c in ("adj_close", "dividends",
                                             "stock_splits", "capital_gains")
                                  if c in out.columns],
                        errors="ignore")

        # Required columns + types
        required = ["open", "high", "low", "close", "volume"]
        missing  = [c for c in required if c not in out.columns]
        if missing:
            logger.debug("yfinance: missing columns %s — discarding bar set", missing)
            return pd.DataFrame()

        out = out[required].copy()
        # Drop rows where every OHLC field is NaN (yfinance fills these
        # for days the symbol wasn't actively trading)
        out = out.dropna(subset=["open", "close"], how="any")

        # ── Daily VWAP proxy ──────────────────────────────────────
        # Alpaca's bars include a ``vwap`` column from intraday tick
        # weights.  yfinance daily bars don't expose VWAP, but the
        # rest of the feature engineer depends on it (the
        # ``close_to_vwap`` feature is in TECHNICAL_COLUMNS, so a
        # missing column → all rows dropped during the warm-up
        # ``dropna``).  Use the typical-price proxy HLC/3 — a
        # standard substitute for daily VWAP when no intraday
        # weights are available.
        if not out.empty:
            out["vwap"] = (out["high"] + out["low"] + out["close"]) / 3.0

        # Index: tz-naive timestamps (so pandas equality with the
        # Alpaca path's tz-naive index works).
        if out.index.tz is not None:
            out.index = out.index.tz_localize(None)
        out.index.name = "timestamp"
        out.sort_index(inplace=True)
        return out
