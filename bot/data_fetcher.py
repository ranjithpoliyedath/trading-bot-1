"""
bot/data_fetcher.py
-------------------
Fetches historical and live OHLCV market data from Alpaca Markets API.
All data is returned as pandas DataFrames ready for feature engineering.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Fetches OHLCV bar data from Alpaca for one or more symbols.

    Args:
        api_key (str): Alpaca API key. Defaults to env var ALPACA_API_KEY.
        secret_key (str): Alpaca secret key. Defaults to env var ALPACA_SECRET_KEY.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API key and secret must be set in .env or passed as arguments."
            )

        self.client = StockHistoricalDataClient(self.api_key, self.secret_key)
        logger.info("DataFetcher initialised with Alpaca client.")

    def fetch_bars(
        self,
        symbols: list[str],
        timeframe: TimeFrame = TimeFrame(1, TimeFrameUnit.Day),
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        lookback_days: int = 365,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV bars for a list of symbols.

        Args:
            symbols: List of ticker symbols e.g. ['AAPL', 'TSLA'].
            timeframe: Alpaca TimeFrame object. Default is 1 day.
            start: Start datetime. If None, uses lookback_days from today.
            end: End datetime. If None, uses today.
            lookback_days: Days of history to fetch when start is not provided.

        Returns:
            Dict mapping each symbol to a DataFrame with columns:
            [open, high, low, close, volume, trade_count, vwap]
        """
        end = end or datetime.utcnow()
        start = start or (end - timedelta(days=lookback_days))

        logger.info(
            "Fetching bars for %s from %s to %s",
            symbols,
            start.date(),
            end.date(),
        )

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
            feed='iex',
        )

        try:
            bars = self.client.get_stock_bars(request)
        except Exception as exc:
            logger.error("Failed to fetch bars from Alpaca: %s", exc, exc_info=True)
            raise

        # Convert to DataFrame — handles both BarSet .df accessor and list-of-bars
        try:
            df_all = bars.df
        except AttributeError:
            # Older/different SDK version returns dict of lists
            rows = []
            for sym, bar_list in bars.items() if hasattr(bars, 'items') else []:
                items = bar_list if isinstance(bar_list, list) else [bar_list]
                for b in items:
                    row = {"symbol": sym}
                    if isinstance(b, dict):
                        row.update(b)
                    else:
                        for f in ["open","high","low","close","volume","trade_count","vwap"]:
                            row[f] = getattr(b, f, None)
                        row["timestamp"] = getattr(b, "timestamp", None)
                    rows.append(row)
            if rows:
                df_all = pd.DataFrame(rows)
                df_all.set_index("timestamp", inplace=True)
            else:
                df_all = pd.DataFrame()

        result = {}
        for symbol in symbols:
            try:
                if "symbol" in df_all.columns:
                    df = df_all[df_all["symbol"] == symbol].drop(columns=["symbol"]).copy()
                else:
                    df = df_all.loc[symbol].copy() if symbol in df_all.index.get_level_values(0) else pd.DataFrame()
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
                result[symbol] = df
                logger.info("Fetched %d bars for %s.", len(df), symbol)
            except (KeyError, Exception) as exc:
                logger.warning("No data for %s: %s", symbol, exc)
                result[symbol] = pd.DataFrame()

        return result

    def fetch_single(
        self,
        symbol: str,
        timeframe: TimeFrame = TimeFrame(1, TimeFrameUnit.Day),
        lookback_days: int = 365,
    ) -> pd.DataFrame:
        """
        Convenience wrapper to fetch bars for a single symbol.

        Args:
            symbol: Ticker symbol e.g. 'AAPL'.
            timeframe: Alpaca TimeFrame object. Default is 1 day.
            lookback_days: Number of calendar days of history.

        Returns:
            DataFrame with OHLCV columns, indexed by timestamp.
        """
        result = self.fetch_bars(
            symbols=[symbol],
            timeframe=timeframe,
            lookback_days=lookback_days,
        )
        return result.get(symbol, pd.DataFrame())
