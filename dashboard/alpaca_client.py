"""
dashboard/alpaca_client.py
--------------------------
Thin wrapper around Alpaca REST API.
Switches between paper and live endpoints based on account argument.
All methods return plain dicts/lists safe to pass to Dash callbacks.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Literal

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL  = "https://api.alpaca.markets"


def _trading_client(account: str) -> TradingClient:
    key    = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    return TradingClient(key, secret, paper=(account == "paper"))


def _data_client() -> StockHistoricalDataClient:
    return StockHistoricalDataClient(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_SECRET_KEY"),
    )


def get_account_summary(account: str = "paper") -> dict:
    """
    Return portfolio value, cash, buying power, and daily P&L.

    Args:
        account: 'paper' or 'live'

    Returns:
        Dict with keys: portfolio_value, cash, buying_power, daily_pl, daily_pl_pct
    """
    try:
        client  = _trading_client(account)
        acct    = client.get_account()
        equity  = float(acct.equity)
        last_eq = float(acct.last_equity)
        daily_pl = equity - last_eq
        daily_pl_pct = (daily_pl / last_eq * 100) if last_eq else 0.0
        return {
            "portfolio_value": equity,
            "cash":            float(acct.cash),
            "buying_power":    float(acct.buying_power),
            "daily_pl":        daily_pl,
            "daily_pl_pct":    daily_pl_pct,
        }
    except Exception as exc:
        logger.error("get_account_summary failed: %s", exc)
        return {"portfolio_value": 0, "cash": 0, "buying_power": 0, "daily_pl": 0, "daily_pl_pct": 0}


def get_positions(account: str = "paper") -> list[dict]:
    """
    Return all open positions.

    Returns:
        List of dicts with keys: symbol, qty, side, market_value, unrealized_pl, unrealized_plpc
    """
    try:
        client = _trading_client(account)
        positions = client.get_all_positions()
        return [
            {
                "symbol":         p.symbol,
                "qty":            float(p.qty),
                "side":           p.side.value,
                "market_value":   float(p.market_value),
                "unrealized_pl":  float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc) * 100,
            }
            for p in positions
        ]
    except Exception as exc:
        logger.error("get_positions failed: %s", exc)
        return []


def get_recent_orders(account: str = "paper", limit: int = 20) -> list[dict]:
    """
    Return the most recent filled orders.

    Returns:
        List of dicts with keys: symbol, side, qty, filled_avg_price, filled_at, pl_est
    """
    try:
        client = _trading_client(account)
        req    = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=limit)
        orders = client.get_orders(filter=req)
        result = []
        for o in orders:
            if o.filled_avg_price:
                result.append({
                    "symbol":           o.symbol,
                    "side":             o.side.value,
                    "qty":              float(o.qty or 0),
                    "filled_avg_price": float(o.filled_avg_price),
                    "filled_at":        o.filled_at.strftime("%H:%M") if o.filled_at else "--",
                    "pl_est":           0.0,
                })
        return result
    except Exception as exc:
        logger.error("get_recent_orders failed: %s", exc)
        return []


def get_bars(symbol: str, days: int = 30) -> list[dict]:
    """
    Return daily OHLCV bars for a symbol.

    Returns:
        List of dicts with keys: date, open, high, low, close, volume
    """
    try:
        client = _data_client()
        end   = datetime.utcnow()
        start = end - timedelta(days=days)
        req   = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=start, end=end,
            feed='iex',
        )
        bars = client.get_stock_bars(req)[symbol]
        return [
            {
                "date":   b.timestamp.strftime("%Y-%m-%d"),
                "open":   float(b.open),
                "high":   float(b.high),
                "low":    float(b.low),
                "close":  float(b.close),
                "volume": int(b.volume),
            }
            for b in bars
        ]
    except Exception as exc:
        logger.error("get_bars failed for %s: %s", symbol, exc)
        return []
