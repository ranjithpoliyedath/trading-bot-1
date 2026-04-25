"""
dashboard/backtest_engine.py
-----------------------------
Runs a full backtest of an ML strategy on historical data.
Returns a results dict with all metrics, equity curve, and trade log.
Saves/loads results as JSON in dashboard/backtests/.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BACKTEST_DIR = Path(__file__).parent / "backtests"
BACKTEST_DIR.mkdir(exist_ok=True)

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def run_backtest(
    model_name: str = "model_v1",
    symbol: str = "AAPL",
    timeframe: str = "1d",
    period_days: int = 365,
    conf_threshold: float = 0.65,
    active_indicators: list = None,
) -> dict:
    """
    Simulate the ML strategy on historical data and return full metrics.

    Args:
        model_name: Name of model file in models/saved/
        symbol: Ticker to backtest
        timeframe: '1d', '4h', '1h', '15m'
        period_days: Number of days of history to use
        conf_threshold: Min confidence to act on signal (0-1)
        active_indicators: List of indicator names to include as features

    Returns:
        Dict with keys: metrics, equity_curve, monthly_returns, trades, run_id
    """
    logger.info("Running backtest: %s %s %sd conf=%.2f", model_name, symbol, period_days, conf_threshold)

    df = _load_features(symbol, period_days)

    if df.empty:
        logger.warning("No data for %s — returning empty results.", symbol)
        return _empty_results()

    df = _filter_indicators(df, active_indicators)
    df = _generate_signals(df, model_name, conf_threshold)
    df = _simulate_trades(df)

    trades       = _extract_trades(df)
    equity_curve = _calc_equity_curve(trades)
    monthly      = _calc_monthly_returns(equity_curve)
    metrics      = _calc_metrics(trades, equity_curve)

    run_id = f"BT-{datetime.now().strftime('%Y%m%d-%H%M')}-{model_name}-{symbol}-{period_days}d"

    return {
        "run_id":         run_id,
        "model":          model_name,
        "symbol":         symbol,
        "period_days":    period_days,
        "conf_threshold": conf_threshold,
        "metrics":        metrics,
        "equity_curve":   equity_curve,
        "monthly_returns": monthly,
        "trades":         trades,
        "run_at":         datetime.now().isoformat(),
    }


def save_backtest(results: dict) -> str:
    """Save backtest results to disk. Returns the run_id."""
    run_id = results["run_id"]
    path   = BACKTEST_DIR / f"{run_id}.json"

    serialisable = {
        k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v)
        for k, v in results.items()
    }
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)

    logger.info("Saved backtest to %s", path)
    return run_id


def load_backtest(run_id: str) -> dict:
    """Load a saved backtest by run_id."""
    path = BACKTEST_DIR / f"{run_id}.json"
    if not path.exists():
        logger.warning("Backtest not found: %s", run_id)
        return {}
    with open(path) as f:
        return json.load(f)


def list_saved_backtests() -> list[dict]:
    """Return list of saved backtest summaries for the dropdown."""
    results = []
    for path in sorted(BACKTEST_DIR.glob("*.json"), reverse=True):
        try:
            with open(path) as f:
                data = json.load(f)
            results.append({
                "label": f"{data.get('run_id', path.stem)} | {data.get('metrics', {}).get('total_return_pct', 0):.1f}%",
                "value": data.get("run_id", path.stem),
            })
        except Exception:
            continue
    return results


def _load_features(symbol: str, period_days: int) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol.upper()}_features.parquet"
    if not path.exists():
        logger.warning("Features file not found: %s", path)
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.sort_index(inplace=True)
    cutoff = df.index.max() - pd.Timedelta(days=period_days)
    return df[df.index >= cutoff].copy()


def _filter_indicators(df: pd.DataFrame, active: list) -> pd.DataFrame:
    if not active:
        return df
    keep = ["open", "high", "low", "close", "volume"] + [
        c for c in df.columns
        if any(ind.lower().replace(" ", "_") in c.lower() for ind in active)
    ]
    return df[[c for c in keep if c in df.columns]]


def _generate_signals(df: pd.DataFrame, model_name: str, conf_threshold: float) -> pd.DataFrame:
    """Load trained model and generate signals, or fall back to rule-based sim."""
    import pickle
    model_path = Path(__file__).parent.parent / "models" / "saved" / f"{model_name}.pkl"

    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        feature_cols = [c for c in df.columns if c not in ["open", "high", "low", "close", "volume", "vwap"]]
        X = df[feature_cols].dropna()
        proba = model.predict_proba(X)
        classes = list(model.classes_)
        df.loc[X.index, "signal"]     = model.predict(X)
        df.loc[X.index, "confidence"] = proba.max(axis=1)
    else:
        logger.info("Model file not found — using rule-based simulation for backtest.")
        df = _rule_based_signals(df)

    df["signal"]     = df.get("signal", "hold").fillna("hold")
    df["confidence"] = df.get("confidence", 0.7).fillna(0.7)
    df.loc[df["confidence"] < conf_threshold, "signal"] = "hold"
    return df


def _rule_based_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Simple RSI+MACD rule for simulation when no trained model exists."""
    df["signal"]     = "hold"
    df["confidence"] = 0.65
    if "rsi_14" in df.columns and "macd_hist" in df.columns:
        df.loc[(df["rsi_14"] < 40) & (df["macd_hist"] > 0), "signal"] = "buy"
        df.loc[(df["rsi_14"] > 60) & (df["macd_hist"] < 0), "signal"] = "sell"
        df["confidence"] = np.where(df["signal"] != "hold",
                                    0.65 + np.abs(df["macd_hist"]).clip(0, 0.15), 0.55)
    return df


def _simulate_trades(df: pd.DataFrame, initial_cash: float = 10_000.0) -> pd.DataFrame:
    """Walk through signals and simulate buy/sell trades."""
    df = df.copy()
    df["position"]    = 0.0
    df["cash"]        = initial_cash
    df["portfolio"]   = initial_cash
    df["trade_pl"]    = 0.0
    df["in_trade"]    = False

    position  = 0.0
    cash      = initial_cash
    entry_price = 0.0

    for i, (idx, row) in enumerate(df.iterrows()):
        price = row["close"]
        sig   = row["signal"]

        if sig == "buy" and position == 0 and cash > price:
            shares      = int(cash * 0.95 / price)
            position    = shares
            cash       -= shares * price
            entry_price = price
            df.at[idx, "in_trade"] = True

        elif sig == "sell" and position > 0:
            pl        = (price - entry_price) * position
            cash     += position * price
            df.at[idx, "trade_pl"] = pl
            position  = 0
            entry_price = 0.0

        df.at[idx, "position"]  = position
        df.at[idx, "cash"]      = cash
        df.at[idx, "portfolio"] = cash + position * price

    return df


def _extract_trades(df: pd.DataFrame) -> list[dict]:
    trades = []
    for idx, row in df.iterrows():
        if row["trade_pl"] != 0:
            trades.append({
                "date":     str(idx.date()) if hasattr(idx, "date") else str(idx),
                "symbol":   "—",
                "pl":       round(float(row["trade_pl"]), 2),
                "win":      row["trade_pl"] > 0,
            })
    return trades


def _calc_equity_curve(trades: list) -> list[dict]:
    equity = 10_000.0
    curve  = [{"date": "start", "value": equity}]
    for t in trades:
        equity += t["pl"]
        curve.append({"date": t["date"], "value": round(equity, 2)})
    return curve


def _calc_monthly_returns(equity_curve: list) -> list[dict]:
    if len(equity_curve) < 2:
        return []
    df = pd.DataFrame(equity_curve[1:])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df.set_index("date", inplace=True)
    monthly = df["value"].resample("ME").last().pct_change().dropna()
    return [{"month": str(m)[:7], "return": round(v * 100, 2)} for m, v in monthly.items()]


def _calc_metrics(trades: list, equity_curve: list) -> dict:
    if not trades:
        return _empty_metrics()

    pls      = [t["pl"] for t in trades]
    wins     = [p for p in pls if p > 0]
    losses   = [p for p in pls if p < 0]

    total_trades = len(pls)
    win_rate     = len(wins) / total_trades if total_trades else 0
    avg_win      = np.mean(wins)  if wins   else 0
    avg_loss     = np.mean(losses) if losses else 0
    expectancy   = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    edge_ratio   = abs(avg_win / avg_loss) if avg_loss else 0
    profit_factor = abs(sum(wins) / sum(losses)) if losses else 0

    values = [e["value"] for e in equity_curve]
    total_return = (values[-1] - values[0]) / values[0] * 100 if values else 0

    returns = pd.Series(pls)
    sharpe  = (returns.mean() / returns.std() * np.sqrt(252)).round(2) if returns.std() > 0 else 0
    sortino_neg = returns[returns < 0].std()
    sortino = (returns.mean() / sortino_neg * np.sqrt(252)).round(2) if sortino_neg > 0 else 0

    peak     = pd.Series(values).cummax()
    drawdown = ((pd.Series(values) - peak) / peak * 100)
    max_dd   = drawdown.min()

    return {
        "total_return_pct":  round(total_return, 2),
        "sharpe":            round(float(sharpe), 2),
        "sortino":           round(float(sortino), 2),
        "expectancy":        round(expectancy, 2),
        "edge_ratio":        round(edge_ratio, 2),
        "profit_factor":     round(profit_factor, 2),
        "max_drawdown_pct":  round(float(max_dd), 2),
        "win_rate_pct":      round(win_rate * 100, 1),
        "total_trades":      total_trades,
        "avg_win":           round(avg_win, 2),
        "avg_loss":          round(avg_loss, 2),
        "largest_win":       round(max(pls), 2),
        "largest_loss":      round(min(pls), 2),
    }


def _empty_metrics() -> dict:
    keys = ["total_return_pct", "sharpe", "sortino", "expectancy", "edge_ratio",
            "profit_factor", "max_drawdown_pct", "win_rate_pct", "total_trades",
            "avg_win", "avg_loss", "largest_win", "largest_loss"]
    return {k: 0 for k in keys}


def _empty_results() -> dict:
    return {
        "run_id": "", "model": "", "symbol": "", "period_days": 0,
        "conf_threshold": 0, "metrics": _empty_metrics(),
        "equity_curve": [], "monthly_returns": [], "trades": [], "run_at": "",
    }
