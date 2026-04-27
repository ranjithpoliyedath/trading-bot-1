"""
dashboard/backtest_engine.py
-----------------------------
Runs a full backtest of an ML strategy on historical data.
Returns a results dict with all metrics, equity curve, and trade log.
Saves/loads results as JSON in dashboard/backtests/.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BACKTEST_DIR = Path(__file__).parent / "backtests"
BACKTEST_DIR.mkdir(exist_ok=True)

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def run_backtest(
    model_name:        str   = "model_v1",
    symbol:            str   = "AAPL",
    timeframe:         str   = "1d",
    period_days:       int   = 365,
    conf_threshold:    float = 0.65,
    active_indicators: list  = None,
    use_signal_exit:   bool             = True,
    take_profit_pct:   Optional[float]  = 0.15,
    stop_loss_pct:     Optional[float]  = 0.07,
    time_stop_days:    Optional[int]    = 30,
    atr_stop_mult:     Optional[float]  = None,
    starting_cash:     float            = 10_000.0,
    sizing_method:     str              = "fixed_pct",
    sizing_kwargs:     Optional[dict]   = None,
    execution_model:   str              = "next_open",
    execution_delay_bars: int           = 0,
    slippage_bps:      float            = 5.0,
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
    df = _simulate_trades(
        df,
        initial_cash         = starting_cash,
        use_signal_exit      = use_signal_exit,
        take_profit_pct      = take_profit_pct,
        stop_loss_pct        = stop_loss_pct,
        time_stop_days       = time_stop_days,
        atr_stop_mult        = atr_stop_mult,
        sizing_method        = sizing_method,
        sizing_kwargs        = sizing_kwargs,
        execution_model      = execution_model,
        execution_delay_bars = execution_delay_bars,
        slippage_bps         = slippage_bps,
    )

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
    """Load features (sentiment-enriched if present) + breakout columns."""
    from bot.patterns import add_breakout_features

    for tag in ("features_with_sentiment", "features"):
        path = DATA_DIR / f"{symbol.upper()}_{tag}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df.sort_index(inplace=True)
            cutoff = df.index.max() - pd.Timedelta(days=period_days)
            df = df[df.index >= cutoff].copy()
            return add_breakout_features(df)
    logger.warning("No features file for %s.", symbol)
    return pd.DataFrame()


# ── Multi-symbol / screener-aware backtest ───────────────────────────────────

def run_filtered_backtest(
    model_id:        str,
    filters:         list[dict],
    symbols:         list[str] | None = None,
    period_days:     int              = 365,
    conf_threshold:  float            = 0.65,
    max_symbols:     int              = 50,
    use_signal_exit: bool             = True,
    take_profit_pct: Optional[float]  = 0.15,
    stop_loss_pct:   Optional[float]  = 0.07,
    time_stop_days:  Optional[int]    = 30,
    atr_stop_mult:   Optional[float]  = None,
    starting_cash:   float            = 10_000.0,
    sizing_method:   str              = "fixed_pct",
    sizing_kwargs:   Optional[dict]   = None,
    execution_model: str              = "next_open",
    execution_delay_bars: int         = 0,
    slippage_bps:    float            = 5.0,
    date_window:     Optional[tuple]  = None,   # (start, end) timestamps for sample slicing
) -> dict:
    """
    Run a backtest across many symbols, only entering on bars that
    satisfy ``filters``.  This is what the NL query and screener
    "Send to backtest" feed into.

    Returns the same shape as ``run_backtest`` but aggregated.
    """
    from bot.screener import _candidate_symbols, OPS
    from bot.models.registry import get_model

    if symbols is None:
        symbols = _candidate_symbols()
    symbols = symbols[:max_symbols]

    try:
        model = get_model(model_id)
    except Exception as exc:
        logger.error("Cannot load model %r: %s", model_id, exc)
        return _empty_results()

    all_trades: list[dict] = []
    metric_per_symbol: list[dict] = []
    initial_cash = float(starting_cash)

    for symbol in symbols:
        df = _load_features(symbol, period_days)
        if df.empty:
            continue

        if date_window is not None:
            start, end = date_window
            try:
                df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
            except Exception:
                pass
            if df.empty:
                continue

        scored = model.predict_batch(df.copy())
        scored["signal"]     = scored.get("signal", "hold").fillna("hold")
        scored["confidence"] = scored.get("confidence", 0.5).fillna(0.5)
        scored.loc[scored["confidence"] < conf_threshold, "signal"] = "hold"

        # Apply screener filters per-bar — only allow buys when all match
        if filters:
            mask = pd.Series(True, index=scored.index)
            for f in filters:
                fld = f["field"]; op = f["op"]; val = float(f["value"])
                if fld not in scored.columns:
                    mask &= False
                    continue
                col = pd.to_numeric(scored[fld], errors="coerce")
                opfn = OPS.get(op)
                if opfn is None:
                    continue
                mask &= opfn(col, val).fillna(False)
            scored.loc[~mask & (scored["signal"] == "buy"), "signal"] = "hold"

        scored = _simulate_trades(
            scored,
            initial_cash         = initial_cash,
            use_signal_exit      = use_signal_exit,
            take_profit_pct      = take_profit_pct,
            stop_loss_pct        = stop_loss_pct,
            time_stop_days       = time_stop_days,
            atr_stop_mult        = atr_stop_mult,
            sizing_method        = sizing_method,
            sizing_kwargs        = sizing_kwargs,
            execution_model      = execution_model,
            execution_delay_bars = execution_delay_bars,
            slippage_bps         = slippage_bps,
        )
        sym_trades = []
        for idx, row in scored.iterrows():
            if row["trade_pl"] != 0:
                sym_trades.append({
                    "date":        str(idx.date()) if hasattr(idx, "date") else str(idx),
                    "symbol":      symbol,
                    "pl":          round(float(row["trade_pl"]), 2),
                    "win":         bool(row["trade_pl"] > 0),
                    "exit_reason": str(row.get("exit_reason", "")),
                })
        all_trades.extend(sym_trades)

        if sym_trades:
            metric_per_symbol.append({
                "symbol":  symbol,
                "trades":  len(sym_trades),
                "pl":      round(sum(t["pl"] for t in sym_trades), 2),
                "win_rate": round(
                    sum(1 for t in sym_trades if t["win"]) / len(sym_trades) * 100, 1),
            })

    all_trades.sort(key=lambda t: t["date"])
    equity_curve = _calc_equity_curve(all_trades)
    monthly      = _calc_monthly_returns(equity_curve)
    metrics      = _calc_metrics(all_trades, equity_curve)
    metrics["symbols_traded"] = len(metric_per_symbol)

    run_id = f"BT-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{model_id}-multi-{period_days}d"

    return {
        "run_id":          run_id,
        "model":           model_id,
        "symbol":          f"{len(metric_per_symbol)} symbols",
        "period_days":     period_days,
        "conf_threshold":  conf_threshold,
        "filters":         filters,
        "metrics":         metrics,
        "equity_curve":    equity_curve,
        "monthly_returns": monthly,
        "trades":          all_trades,
        "per_symbol":      metric_per_symbol,
        "run_at":          datetime.now().isoformat(),
    }


def _filter_indicators(df: pd.DataFrame, active: list) -> pd.DataFrame:
    if not active:
        return df
    keep = ["open", "high", "low", "close", "volume"] + [
        c for c in df.columns
        if any(ind.lower().replace(" ", "_") in c.lower() for ind in active)
    ]
    return df[[c for c in keep if c in df.columns]]


def _generate_signals(df: pd.DataFrame, model_name: str, conf_threshold: float) -> pd.DataFrame:
    """Use the model registry to produce signals; fall back to rule-based sim."""
    try:
        from bot.models.registry import get_model
        model  = get_model(model_name)
        scored = model.predict_batch(df.copy())
        df["signal"]     = scored["signal"].fillna("hold")
        df["confidence"] = scored["confidence"].fillna(0.5)
    except Exception as exc:
        logger.warning("Model %s failed (%s) — using rule fallback.", model_name, exc)
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


def _position_size_shares(
    method:        str,
    cash:          float,
    portfolio:     float,
    price:         float,
    atr:           float,
    sizing_kwargs: dict,
) -> int:
    """
    Compute number of shares to buy.

    Methods:
      * ``fixed_pct`` — use ``pct`` of the *portfolio* (cash + open positions)
        as position notional, then floor to whole shares.  Default 95%.
      * ``kelly`` / ``half_kelly`` — Kelly fraction = W − (1−W)/R, where W is
        the assumed win rate and R the win/loss ratio.  Both supplied via
        ``sizing_kwargs`` so the user can plug in numbers from the prior
        backtest.  half-Kelly halves the result to reduce variance.
      * ``atr_risk`` — size such that (entry − stop) × shares = capital ×
        ``risk_pct``.  Stop is entry − ``atr_mult`` × ATR.  This is the
        "fixed-fractional risk" sizing professional traders favour because
        it normalises position size to the symbol's volatility.

    Returns 0 if the inputs don't make sense (no ATR for atr_risk, etc.).
    """
    if price <= 0 or cash < price:
        return 0

    method = (method or "fixed_pct").lower()

    if method == "fixed_pct":
        pct = float(sizing_kwargs.get("pct", 0.95))
        return max(int(portfolio * pct / price), 0)

    if method in ("kelly", "half_kelly"):
        win_rate    = float(sizing_kwargs.get("win_rate", 0.5))
        wl_ratio    = float(sizing_kwargs.get("win_loss_ratio", 1.5))
        f = win_rate - (1 - win_rate) / max(wl_ratio, 1e-6)
        f = max(0.0, min(f, 1.0))
        if method == "half_kelly":
            f /= 2
        return max(int(portfolio * f / price), 0)

    if method == "atr_risk":
        if atr is None or atr <= 0 or pd.isna(atr):
            return 0
        risk_pct  = float(sizing_kwargs.get("risk_pct", 0.01))
        atr_mult  = float(sizing_kwargs.get("atr_mult", 2.0))
        risk_per_share = atr_mult * atr
        if risk_per_share <= 0:
            return 0
        risk_dollars = portfolio * risk_pct
        shares = int(risk_dollars / risk_per_share)
        # Guardrail: never spend more than 100% of available cash.
        max_by_cash = int(cash / price)
        return max(min(shares, max_by_cash), 0)

    # Unknown method — fall back to fixed 95%.
    return max(int(portfolio * 0.95 / price), 0)


def _simulate_trades(
    df:               pd.DataFrame,
    initial_cash:     float            = 10_000.0,
    use_signal_exit:  bool             = True,
    take_profit_pct:  Optional[float]  = 0.15,
    stop_loss_pct:    Optional[float]  = 0.07,
    time_stop_days:   Optional[int]    = 30,
    atr_stop_mult:    Optional[float]  = None,    # close when price <= entry - N*ATR
    sizing_method:    str              = "fixed_pct",
    sizing_kwargs:    Optional[dict]   = None,
    execution_model:  str              = "next_open",   # "next_open" or "same_close"
    execution_delay_bars: int          = 0,
    slippage_bps:     float            = 5.0,
) -> pd.DataFrame:
    """
    Walk through signals and simulate buy / sell trades with configurable
    sizing and exits.  Pass ``None`` (or False) for any rule to disable it.

    Exit rules (whichever fires first wins):
      1. Model sell signal      (gated by ``use_signal_exit``)
      2. Take-profit:  price >= entry * (1 + take_profit_pct)
      3. Stop-loss:    price <= entry * (1 - stop_loss_pct)
      4. ATR stop:     price <= entry - atr_stop_mult * entry-day ATR
      5. Time-stop:    bars_held >= time_stop_days

    Each exit tags the trade with ``exit_reason`` so the log can show
    why we left.  At least one exit rule must be enabled — otherwise
    a winning position would never close.
    """
    sizing_kwargs = sizing_kwargs or {}
    if (not use_signal_exit and take_profit_pct is None
            and stop_loss_pct is None and time_stop_days is None
            and atr_stop_mult is None):
        # Defensive: nothing would ever close.  Force a default.
        time_stop_days = 30

    execution_model = (execution_model or "next_open").lower()
    if execution_model not in ("next_open", "same_close"):
        execution_model = "next_open"

    # next_open requires at least a 1-bar wait by definition; same_close
    # can fill on the very same bar as the signal.
    required_wait = (1 if execution_model == "next_open" else 0) + max(0, int(execution_delay_bars or 0))
    fill_col       = "open" if execution_model == "next_open" else "close"
    if fill_col not in df.columns:
        # Symbol features may be missing the open column (rare).  Fall back
        # to close + log so the user sees it.
        logger.warning("execution_model=%s but no '%s' column — falling back to close.",
                       execution_model, fill_col)
        fill_col = "close"

    bps  = float(slippage_bps or 0) / 10_000.0
    df = df.copy()
    df["position"]    = 0.0
    df["cash"]        = initial_cash
    df["portfolio"]   = initial_cash
    df["trade_pl"]    = 0.0
    df["exit_reason"] = ""
    df["in_trade"]    = False

    position      = 0.0
    cash          = initial_cash
    entry_price   = 0.0
    entry_idx     = None
    entry_atr     = 0.0

    # Queue a single pending action.  Each is a dict:
    #   {"type": "buy",  "queued_bar": i}
    #   {"type": "exit", "queued_bar": i, "reason": str}
    pending = None

    for i, (idx, row) in enumerate(df.iterrows()):
        price = float(row["close"])      # decisions are always made on close
        sig   = row["signal"]

        # ── 1. Detect new signal / exit and queue it (if nothing pending) ──
        if pending is None:
            if sig == "buy" and position == 0:
                pending = {"type": "buy", "queued_bar": i}
            elif position > 0:
                pl_per_share = price - entry_price
                ret_pct      = pl_per_share / entry_price if entry_price else 0.0
                held_days    = i - (entry_idx or i)

                atr_floor = (entry_price - atr_stop_mult * entry_atr) \
                    if (atr_stop_mult is not None and entry_atr > 0) else None
                exit_reason = ""
                if   use_signal_exit and sig == "sell":                                exit_reason = "signal"
                elif take_profit_pct is not None and ret_pct >=  take_profit_pct:      exit_reason = "take_profit"
                elif stop_loss_pct  is not None and ret_pct <= -stop_loss_pct:         exit_reason = "stop_loss"
                elif atr_floor is not None and price <= atr_floor:                     exit_reason = "atr_stop"
                elif time_stop_days is not None and held_days >= time_stop_days:       exit_reason = "time_stop"
                if exit_reason:
                    pending = {"type": "exit", "queued_bar": i, "reason": exit_reason}

        # ── 2. Try to execute pending action ──────────────────────────────
        if pending is not None and (i - pending["queued_bar"]) >= required_wait:
            raw_fill = float(row[fill_col])

            if pending["type"] == "buy" and position == 0 and cash > raw_fill:
                # Buy-side slippage: pay a bit more than the printed fill.
                fill = raw_fill * (1 + bps)
                atr_now = float(row.get("atr_14", 0) or 0)
                shares = _position_size_shares(
                    method        = sizing_method,
                    cash          = cash,
                    portfolio     = cash,
                    price         = fill,
                    atr           = atr_now,
                    sizing_kwargs = sizing_kwargs,
                )
                if shares > 0:
                    position    = shares
                    cash       -= shares * fill
                    entry_price = fill
                    entry_idx   = i
                    entry_atr   = atr_now
                    df.at[idx, "in_trade"] = True

            elif pending["type"] == "exit" and position > 0:
                # Sell-side slippage: receive a bit less than the print.
                fill = raw_fill * (1 - bps)
                pl   = (fill - entry_price) * position
                cash += position * fill
                df.at[idx, "trade_pl"]    = pl
                df.at[idx, "exit_reason"] = pending["reason"]
                position    = 0
                entry_price = 0.0
                entry_idx   = None
                entry_atr   = 0.0

            pending = None

        df.at[idx, "position"]  = position
        df.at[idx, "cash"]      = cash
        df.at[idx, "portfolio"] = cash + position * price

    return df


def _extract_trades(df: pd.DataFrame) -> list[dict]:
    trades = []
    for idx, row in df.iterrows():
        if row["trade_pl"] != 0:
            trades.append({
                "date":        str(idx.date()) if hasattr(idx, "date") else str(idx),
                "symbol":      "—",
                "pl":          round(float(row["trade_pl"]), 2),
                "win":         bool(row["trade_pl"] > 0),
                "exit_reason": str(row.get("exit_reason", "")),
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
    monthly = df["value"].resample("ME").last().pct_change(fill_method=None).dropna()
    return [{"month": str(m)[:7], "return": round(v * 100, 2)} for m, v in monthly.items()]


def _calc_metrics(trades: list, equity_curve: list) -> dict:
    if not trades:
        return _empty_metrics()

    pls    = [t["pl"] for t in trades]
    wins   = [p for p in pls if p > 0]
    losses = [p for p in pls if p < 0]

    total_trades = len(pls)
    n_wins       = len(wins)
    n_losses     = len(losses)
    win_rate     = n_wins / total_trades if total_trades else 0.0
    avg_win      = float(np.mean(wins))   if wins   else 0.0
    avg_loss     = float(np.mean(losses)) if losses else 0.0   # negative

    # Per-trade % returns (relative to a notional $10k position).  Used
    # for the avg_win_pct / avg_loss_pct fields requested in the UI.
    notional       = 10_000.0
    pct_per_trade  = [p / notional * 100 for p in pls]
    win_pcts       = [p for p in pct_per_trade if p > 0]
    loss_pcts      = [p for p in pct_per_trade if p < 0]
    avg_win_pct    = float(np.mean(win_pcts))  if win_pcts  else 0.0
    avg_loss_pct   = float(np.mean(loss_pcts)) if loss_pcts else 0.0

    expectancy    = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    edge_ratio    = abs(avg_win / avg_loss) if avg_loss else 0.0
    profit_factor = abs(sum(wins) / sum(losses)) if losses else 0.0

    # ── Equity-curve metrics ──────────────────────────────────────────
    # Sharpe / Sortino were previously computed off per-trade P&L which
    # is wrong (and explodes to 0 when there's only one trade).  Use
    # the equity curve resampled to a daily series instead.
    values = [e["value"] for e in equity_curve]
    total_return = (values[-1] - values[0]) / values[0] * 100 if values else 0.0

    sharpe = sortino = 0.0
    try:
        ec_dates = [e["date"] for e in equity_curve]
        ec_df = pd.DataFrame(
            {"value": values},
            index=pd.to_datetime(ec_dates, errors="coerce", format="ISO8601"),
        )
        ec_df = ec_df.dropna()
        if len(ec_df) >= 2:
            daily = ec_df["value"].resample("D").last().ffill().pct_change().dropna()
            if daily.std() > 0:
                sharpe = float(daily.mean() / daily.std() * np.sqrt(252))
            downside = daily[daily < 0].std()
            if downside and downside > 0:
                sortino = float(daily.mean() / downside * np.sqrt(252))
    except Exception:
        pass

    peak     = pd.Series(values).cummax()
    drawdown = ((pd.Series(values) - peak) / peak * 100)
    max_dd   = float(drawdown.min())

    return {
        "total_return_pct":  round(total_return, 2),
        "sharpe":            round(sharpe, 2),
        "sortino":           round(sortino, 2),
        "expectancy":        round(expectancy, 2),
        "edge_ratio":        round(edge_ratio, 2),
        "profit_factor":     round(profit_factor, 2),
        "max_drawdown_pct":  round(max_dd, 2),
        "win_rate_pct":      round(win_rate * 100, 1),
        "loss_rate_pct":     round((1 - win_rate) * 100, 1),
        "total_trades":      total_trades,
        "wins":              n_wins,
        "losses":            n_losses,
        "avg_win":           round(avg_win, 2),
        "avg_loss":          round(avg_loss, 2),
        "avg_win_pct":       round(avg_win_pct, 2),
        "avg_loss_pct":      round(avg_loss_pct, 2),
        "largest_win":       round(max(pls), 2),
        "largest_loss":      round(min(pls), 2),
    }


def _empty_metrics() -> dict:
    keys = ["total_return_pct", "sharpe", "sortino", "expectancy", "edge_ratio",
            "profit_factor", "max_drawdown_pct", "win_rate_pct", "loss_rate_pct",
            "total_trades", "wins", "losses",
            "avg_win", "avg_loss", "avg_win_pct", "avg_loss_pct",
            "largest_win", "largest_loss"]
    return {k: 0 for k in keys}


def _empty_results() -> dict:
    return {
        "run_id": "", "model": "", "symbol": "", "period_days": 0,
        "conf_threshold": 0, "metrics": _empty_metrics(),
        "equity_curve": [], "monthly_returns": [], "trades": [], "run_at": "",
    }


# ── Cross-sectional backtest ──────────────────────────────────────────────

def run_cross_sectional_backtest(
    model_id:       str,
    symbols:        Optional[list]  = None,
    period_days:    int             = 365 * 6,
    top_decile:     float           = 0.10,
    rebalance_days: int             = 21,
    starting_cash:  float           = 10_000.0,
    slippage_bps:   float           = 5.0,
) -> dict:
    """
    Run a cross-sectional strategy.  The model's ``rank_universe``
    produces a wide panel of ranks; on each rebalance day we equal-
    weight the top ``top_decile`` of symbols by rank, hold them for
    ``rebalance_days``, then re-form.

    Long-only — short positions deliberately deferred (proper
    short-side modelling needs borrow / margin assumptions).
    """
    from bot.models.registry import get_model
    from bot.universe        import select_universe

    # Distinguish None ("use default") from [] ("explicitly nothing")
    if symbols is None:
        syms = select_universe("top_100", limit=50)
    else:
        syms = list(symbols)
    if not syms:
        return _empty_results()

    try:
        model = get_model(model_id)
    except Exception as exc:
        logger.error("Cannot load cross-sectional model %r: %s", model_id, exc)
        return _empty_results()

    # Build the wide-format close panel from disk
    closes = {}
    for s in syms:
        df = _load_features(s, period_days)
        if df.empty or "close" not in df.columns:
            continue
        closes[s] = df["close"]
    if not closes:
        logger.warning("No symbols had features data for cross-sectional run.")
        return _empty_results()

    panel = pd.concat(closes, axis=1).sort_index()

    # Get cross-sectional ranks per bar
    try:
        ranks = model.rank_universe(panel.copy())
    except Exception as exc:
        logger.error("rank_universe failed: %s", exc, exc_info=True)
        return _empty_results()

    # Walk the rebalance schedule
    bps    = float(slippage_bps or 0) / 10_000.0
    cash   = float(starting_cash)
    equity = cash
    holdings = {}                    # symbol -> shares
    trades = []
    equity_curve = [{"date": "start", "value": equity}]

    rebal_dates = panel.index[::rebalance_days]

    for i, dt in enumerate(rebal_dates):
        if dt not in ranks.index:
            continue
        row_ranks = ranks.loc[dt].dropna()
        if row_ranks.empty:
            continue

        # Top-decile picks
        threshold = row_ranks.quantile(1 - top_decile)
        picks     = row_ranks[row_ranks >= threshold].index.tolist()
        if not picks:
            continue

        # ── Liquidate everything we currently hold (sell at current close × (1-bps)) ──
        for sym, shares in list(holdings.items()):
            price = panel.at[dt, sym] if sym in panel.columns else None
            if price is None or pd.isna(price):
                continue
            fill = float(price) * (1 - bps)
            cash += shares * fill
            trades.append({
                "date":        str(dt.date()) if hasattr(dt, "date") else str(dt),
                "symbol":      sym,
                "pl":          round(shares * fill - shares * holdings[sym + "_entry_price"], 2)
                                if (sym + "_entry_price") in holdings else 0.0,
                "win":         False,
                "exit_reason": "rebalance",
            })
        holdings = {}

        # ── Allocate equally among picks (buy at close × (1+bps)) ──
        per_position = cash / len(picks)
        for sym in picks:
            price = panel.at[dt, sym]
            if pd.isna(price):
                continue
            fill = float(price) * (1 + bps)
            shares = int(per_position / fill)
            if shares <= 0:
                continue
            cost = shares * fill
            if cost > cash:
                continue
            cash -= cost
            holdings[sym] = shares
            holdings[sym + "_entry_price"] = fill   # piggyback storage

        # mark-to-market equity
        m2m = sum(
            holdings[s] * float(panel.at[dt, s])
            for s in holdings if not s.endswith("_entry_price")
            and s in panel.columns and not pd.isna(panel.at[dt, s])
        )
        equity = cash + m2m
        equity_curve.append({
            "date":  str(dt.date()) if hasattr(dt, "date") else str(dt),
            "value": round(float(equity), 2),
        })

    # Final liquidation at panel's last close so the equity curve is closed
    last_dt = panel.index[-1]
    for sym, shares in list(holdings.items()):
        if sym.endswith("_entry_price") or sym not in panel.columns:
            continue
        price = panel.at[last_dt, sym]
        if pd.isna(price):
            continue
        fill = float(price) * (1 - bps)
        cash += shares * fill
    equity = cash
    equity_curve.append({"date": str(last_dt.date()), "value": round(equity, 2)})

    # Reduce trade dicts to just real trades (the messy storage trick above
    # leaves entry-price-piggyback strings around — strip them)
    trades = [t for t in trades if not t["symbol"].endswith("_entry_price")]
    monthly = _calc_monthly_returns(equity_curve)

    # For metrics, treat each rebalance-period change in equity as a "trade P&L"
    pls = [equity_curve[k]["value"] - equity_curve[k - 1]["value"]
           for k in range(1, len(equity_curve))]
    pseudo_trades = [{"date": equity_curve[k]["date"], "symbol": "—",
                       "pl": round(p, 2), "win": p > 0, "exit_reason": "rebalance"}
                      for k, p in enumerate(pls, start=1) if p != 0]
    metrics = _calc_metrics(pseudo_trades, equity_curve)
    metrics["symbols_traded"] = len(set(s for s in holdings
                                         if not s.endswith("_entry_price")))

    run_id = f"XS-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{model_id}-{len(syms)}syms"
    return {
        "run_id":         run_id,
        "model":          model_id,
        "symbol":         f"{len(syms)} symbols (cross-sectional)",
        "period_days":    period_days,
        "conf_threshold": 0,
        "metrics":        metrics,
        "equity_curve":   equity_curve,
        "monthly_returns": monthly,
        "trades":         pseudo_trades,
        "run_at":         datetime.now().isoformat(),
    }


# ── Walk-forward validation ─────────────────────────────────────────────────

def walk_forward_folds(
    start: pd.Timestamp,
    end:   pd.Timestamp,
    n_folds: int = 4,
    min_is_days: int = 365,
) -> list[tuple[tuple, tuple]]:
    """
    Build expanding-window walk-forward folds across a date range.

    Each fold is ((is_start, is_end), (oos_start, oos_end)).  IS expands
    monotonically; OOS chunks tile the post-burn-in period exactly once
    and don't overlap with each other or with their own IS half.

    Layout for n_folds=4 over 6 years (2020-2026):
        Fold 1: IS = 2020-01-01 .. 2023-01-01   OOS = 2023-01-01 .. 2024-01-01
        Fold 2: IS = 2020-01-01 .. 2024-01-01   OOS = 2024-01-01 .. 2025-01-01
        Fold 3: IS = 2020-01-01 .. 2025-01-01   OOS = 2025-01-01 .. 2026-01-01
        Fold 4: IS = 2020-01-01 .. 2025-07-01   OOS = 2025-07-01 .. 2026-01-01

    Args:
        start:        Earliest available date (Timestamp).
        end:          Latest available date (Timestamp).
        n_folds:      Number of OOS folds.
        min_is_days:  Minimum in-sample window for fold 1 (defaults to 1y).

    Returns:
        List of (is_window, oos_window) tuples.  Empty if the date range
        is too short.
    """
    start = pd.Timestamp(start)
    end   = pd.Timestamp(end)
    total_days = (end - start).days
    if total_days <= min_is_days + 30:
        return []

    # Reserve min_is_days for the first IS window; split the remainder
    # evenly into n_folds OOS chunks.
    available_for_oos = total_days - min_is_days
    chunk = max(1, available_for_oos // n_folds)

    folds = []
    for k in range(n_folds):
        is_end   = start + pd.Timedelta(days=min_is_days + k * chunk)
        oos_end  = start + pd.Timedelta(days=min_is_days + (k + 1) * chunk)
        if k == n_folds - 1:
            oos_end = end       # last fold absorbs any rounding remainder
        folds.append(((start, is_end), (is_end, oos_end)))
    return folds


def run_walk_forward(
    model_id:     str,
    n_folds:      int  = 4,
    symbols:      Optional[list]  = None,
    period_days:  int  = 365 * 6,        # default: full 6yr lookback
    **engine_kwargs,
) -> dict:
    """
    Run a strategy across n walk-forward folds and return per-fold metrics
    plus an aggregate summary.

    The IS half of each fold is currently *not* used to retune any
    parameters — that's Phase 2's Strategy Finder.  Here we just measure
    how the strategy performs on each OOS window so the user can see if
    the edge holds across regimes.

    Returns:
        {
            "model":     str,
            "n_folds":   int,
            "fold_results": [
                {"fold": 1, "is_window": (str, str), "oos_window": (str, str),
                 "metrics": {...}, "trades": [...]},
                ...
            ],
            "aggregate": {
                "mean_oos_sharpe":   float,
                "median_oos_sharpe": float,
                "stdev_oos_sharpe":  float,
                "pct_positive_folds": float,
                "mean_oos_return_pct": float,
            },
            "run_id":  str,
            "run_at":  isoformat string,
        }
    """
    # Determine date range from any one symbol's data
    from bot.universe import select_universe
    syms = symbols or select_universe("top_100", limit=50)
    if not syms:
        return _empty_results()

    sample = _load_features(syms[0], period_days)
    if sample.empty:
        return _empty_results()
    start = sample.index.min()
    end   = sample.index.max()

    folds = walk_forward_folds(start, end, n_folds=n_folds)
    if not folds:
        logger.warning("Date range too short for %d walk-forward folds.", n_folds)
        return _empty_results()

    # Filters apply to every fold equally — pop once, reuse.
    fold_filters = engine_kwargs.pop("filters", [])

    fold_results = []
    sharpes = []
    returns = []
    for k, (_, oos_window) in enumerate(folds, start=1):
        oos_out = run_filtered_backtest(
            model_id     = model_id,
            filters      = fold_filters,
            symbols      = syms,
            period_days  = period_days,
            date_window  = oos_window,
            **engine_kwargs,
        )
        m = oos_out.get("metrics", {})
        sharpes.append(float(m.get("sharpe", 0) or 0))
        returns.append(float(m.get("total_return_pct", 0) or 0))
        fold_results.append({
            "fold":       k,
            "oos_window": (str(oos_window[0])[:10], str(oos_window[1])[:10]),
            "metrics":    m,
            "trades":     oos_out.get("trades", []),
        })

    if sharpes:
        agg = {
            "mean_oos_sharpe":     round(float(np.mean(sharpes)),   3),
            "median_oos_sharpe":   round(float(np.median(sharpes)), 3),
            "stdev_oos_sharpe":    round(float(np.std(sharpes)),    3),
            "pct_positive_folds":  round(100 * sum(1 for s in sharpes if s > 0) / len(sharpes), 1),
            "mean_oos_return_pct": round(float(np.mean(returns)),   2),
        }
    else:
        agg = {}

    run_id = f"WF-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{model_id}-{n_folds}folds"
    return {
        "model":        model_id,
        "n_folds":      n_folds,
        "fold_results": fold_results,
        "aggregate":    agg,
        "run_id":       run_id,
        "run_at":       datetime.now().isoformat(),
    }
