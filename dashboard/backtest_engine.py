"""
dashboard/backtest_engine.py
-----------------------------
Runs a full backtest of an ML strategy on historical data.
Returns a results dict with all metrics, equity curve, and trade log.
Saves/loads results as JSON in dashboard/backtests/.
"""
from __future__ import annotations

import functools
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Benchmark overlay loader (cached) ───────────────────────────────────────

@functools.lru_cache(maxsize=8)
def _benchmark_close(symbol: str) -> tuple[tuple[str, float], ...]:
    """
    Return the symbol's close column as a tuple of (date_iso, close)
    pairs.  LRU-cached so the equity-chart overlay isn't an N+1 read on
    every render.  Returns empty tuple if the symbol's parquet isn't
    on disk (caller renders without overlay).
    """
    path = DATA_DIR / f"{symbol.upper()}_features.parquet"
    if not path.exists():
        path = DATA_DIR / f"{symbol.upper()}_raw.parquet"
    if not path.exists():
        return tuple()
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.warning("Could not read benchmark %s: %s", symbol, exc)
        return tuple()
    if "close" not in df.columns or df.empty:
        return tuple()
    return tuple(
        (str(idx.date()) if hasattr(idx, "date") else str(idx)[:10], float(c))
        for idx, c in df["close"].items()
    )


def load_benchmark_curve(
    symbol:       str,
    start:        Optional[str] = None,
    end:          Optional[str] = None,
    normalize_to: Optional[float] = None,
) -> list[dict]:
    """
    Returns the benchmark close as a list of {date, value} dicts,
    sliced to ``[start, end]`` and (optionally) normalized so the first
    value equals ``normalize_to`` — making it visually comparable to an
    equity curve that started at the same dollar amount.
    """
    series = _benchmark_close(symbol)
    if not series:
        return []
    if start:
        series = tuple((d, c) for d, c in series if d >= start)
    if end:
        series = tuple((d, c) for d, c in series if d <= end)
    if not series:
        return []
    first_close = series[0][1]
    if first_close <= 0:
        return [{"date": d, "value": round(c, 4)} for d, c in series]
    if normalize_to is None:
        return [{"date": d, "value": round(c, 4)} for d, c in series]
    factor = float(normalize_to) / first_close
    return [{"date": d, "value": round(c * factor, 2)} for d, c in series]

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
    equity_curve = _calc_equity_curve(trades, starting_cash=starting_cash)
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


def _sanitize_run_id(run_id: str) -> str:
    """Make a run_id safe for filenames on every platform.

    Colons (commonly appearing in custom-model run_ids like
    ``custom:foo``) are illegal on Windows and confuse some tools on
    macOS/Linux when shell-globbing.  Replace them with underscores.
    Also strips any ``..`` path-traversal attempts.
    """
    safe = run_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    safe = safe.replace("..", "_")
    return safe


def save_backtest(results: dict) -> str:
    """Save backtest results to disk. Returns the (sanitised) run_id."""
    run_id      = results["run_id"]
    safe_run_id = _sanitize_run_id(run_id)
    # Reflect the sanitised id back in the envelope so downstream
    # load_backtest(run_id) lookups work consistently.
    results["run_id"] = safe_run_id
    path = BACKTEST_DIR / f"{safe_run_id}.json"

    serialisable = {
        k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v)
        for k, v in results.items()
    }
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)

    logger.info("Saved backtest to %s", path)
    return safe_run_id


def load_backtest(run_id: str) -> dict:
    """Load a saved backtest by run_id."""
    path = BACKTEST_DIR / f"{run_id}.json"
    if not path.exists():
        logger.warning("Backtest not found: %s", run_id)
        return {}
    with open(path) as f:
        return json.load(f)


def list_saved_backtests() -> list[dict]:
    """
    Return saved-run summaries for the dashboard dropdown.

    Sort order:
      1. Seed runs (SEED-*) at the top — these are the curated examples
         shipped with the project.
      2. User runs sorted newest-first.
    """
    seeds: list[tuple[str, dict]] = []
    user:  list[tuple[str, dict]] = []

    # JSON artefacts in this dir that aren't user-loadable backtest runs
    NON_RUN_FILES = {"seed_leaderboard.json", "tuned_experimentals.json"}

    for path in sorted(BACKTEST_DIR.glob("*.json")):
        if path.name in NON_RUN_FILES:
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue

        run_id   = data.get("run_id", path.stem)
        metrics  = data.get("metrics", {}) or {}
        agg      = data.get("aggregate", {}) or {}

        # Walk-forward results don't have top-level metrics — pull from aggregate
        ret    = metrics.get("total_return_pct", agg.get("mean_oos_return_pct", 0))
        sharpe = metrics.get("sharpe",            agg.get("mean_oos_sharpe", 0))
        trades = int(metrics.get("total_trades", 0) or 0)
        if trades == 0 and data.get("fold_results"):
            trades = sum(int((fr.get("metrics") or {}).get("total_trades", 0) or 0)
                          for fr in data["fold_results"])

        if data.get("seed_meta", {}).get("label"):
            label = (f"⭐ {data['seed_meta']['label']}  "
                     f"·  {ret:+.1f}%  ·  Sharpe {sharpe:+.2f}  ·  {trades} trades")
            seeds.append((run_id, {"label": label, "value": run_id}))
        else:
            label = f"{run_id} · {ret:+.1f}% · Sharpe {sharpe:+.2f} · {trades} trades"
            user.append((path.stat().st_mtime, {"label": label, "value": run_id}))

    seeds.sort(key=lambda t: t[0])                        # alphabetical SEED-tag order
    user.sort(key=lambda t: t[0], reverse=True)           # newest user runs first
    return [s[1] for s in seeds] + [u[1] for u in user]


def _load_features(symbol: str, period_days: int) -> pd.DataFrame:
    """Load features (sentiment-enriched if present) + breakout columns.

    Normalises the index timezone on read: drops any tz info so that
    parquets written by different fetchers (Alpaca → tz-aware UTC,
    yfinance → tz-naive) can be combined without ``Cannot compare
    tz-naive and tz-aware`` errors in the shared-pool simulator.
    """
    from bot.patterns import add_breakout_features

    for tag in ("features_with_sentiment", "features"):
        path = DATA_DIR / f"{symbol.upper()}_{tag}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df.sort_index(inplace=True)
            # Normalise timezone — strip tz info so the shared-pool
            # simulator can merge timelines from yfinance (tz-naive)
            # and legacy Alpaca-fetched parquets (tz-aware UTC).
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            cutoff = df.index.max() - pd.Timedelta(days=period_days)
            df = df[df.index >= cutoff].copy()
            return add_breakout_features(df)
    # Logged at DEBUG, not WARNING — for multi-symbol scans the
    # eligible universe always has more symbols than `data/processed/`
    # holds parquet files for, and printing one warning per absent
    # symbol just spams the console.  The aggregate is captured in
    # `load_report.missing_features` and surfaced in the UI's empty-
    # result diagnostic.
    logger.debug("No features file for %s.", symbol)
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
    market_regime_exit: bool          = False,  # force-sell when SPY/QQQ trending down
    sector_regime_exit: bool          = False,  # force-sell when symbol's sector ETF down
    tax_rate:           float         = 0.0,    # fraction of YTD gains paid in tax (Dec 31)
    omit_top_n_outliers: int          = 0,      # exclude N largest-P&L wins from metrics
    max_per_sector:     Optional[int] = None,   # cap simultaneous positions per GICS sector
) -> dict:
    """
    Run a backtest across many symbols, only entering on bars that
    satisfy ``filters``.  This is what the NL query and screener
    "Send to backtest" feed into.

    Returns the same shape as ``run_backtest`` but aggregated.
    """
    from bot.screener        import _candidate_symbols, OPS
    from bot.models.registry import get_model
    from bot.models.base     import CrossSectionalModel

    if symbols is None:
        symbols = _candidate_symbols()
    symbols = symbols[:max_symbols]

    # ── Single-ticker convenience ──────────────────────────────────────
    # When the user picks a single-symbol scope (etf:QQQ, sym:AAPL, etc.)
    # there's no portfolio-allocation question to answer — they want
    # 100% of capital on that one ticker.  Override the default
    # ``fixed_pct=0.95`` only when the user didn't customise sizing
    # themselves (kwargs empty AND method is the engine default).
    if len(symbols) == 1 and sizing_method == "fixed_pct" and not (sizing_kwargs or {}):
        sizing_kwargs = {"pct": 1.0}
        logger.info("Single-symbol scope %s: defaulting fixed_pct sizing to 100%%.",
                    symbols[0])

    # ── On-the-fly fetch for missing data on small scopes ──────────────
    # If the user picks a small universe (≤5 symbols) and some lack a
    # ``*_features.parquet`` on disk, try to fetch them right now via
    # the same pipeline ``python -m bot.pipeline`` would use.  Bigger
    # scopes still require a manual pipeline run — we don't want a
    # 50-symbol Alpaca blast triggered by a button click.
    if len(symbols) <= 5:
        from pathlib import Path as _Path
        from bot.config import DATA_DIR as _DATA_DIR
        missing_on_disk = [s for s in symbols
                           if not _Path(_DATA_DIR / f"{s.upper()}_features.parquet").exists()]
        if missing_on_disk:
            try:
                from bot.pipeline import run_pipeline
                logger.info("Auto-fetching missing data for %s "
                            "(small-scope convenience fetch)…",
                            ", ".join(missing_on_disk))
                # Use a generous lookback so even long backtests have data
                run_pipeline(symbols=missing_on_disk,
                             lookback_days=max(int(period_days) * 2, 365 * 6))
            except Exception as exc:
                # Most common: .env missing → Alpaca auth fails.  Don't
                # crash the run — just log and let load_report record
                # what didn't load so the dashboard can show a precise
                # diagnostic with the right fix-it command.
                logger.warning("Auto-fetch failed for %s: %s.  "
                               "Run `python -m bot.pipeline` manually.",
                               missing_on_disk, exc)

    try:
        model = get_model(model_id)
    except Exception as exc:
        logger.error("Cannot load model %r: %s", model_id, exc)
        return _empty_results()

    # Cross-sectional models can't be run through the per-symbol pipeline
    # — they emit ranks across the whole universe per bar and need a
    # different runner.  Route automatically so the dashboard's regular
    # Run button works whether the user picks a per-symbol or
    # cross-sectional model.
    if isinstance(model, CrossSectionalModel):
        logger.info("Routing %s to run_cross_sectional_backtest "
                    "(cross-sectional model)", model_id)
        return run_cross_sectional_backtest(
            model_id      = model_id,
            symbols       = symbols,
            period_days   = period_days,
            starting_cash = float(starting_cash),
            slippage_bps  = float(slippage_bps),
        )

    initial_cash = float(starting_cash)

    # Per-symbol load diagnostics — surfaced in the result envelope so
    # the dashboard can show exactly *why* a run produced 0 symbols
    # instead of the generic "data missing" panel.
    load_report = {"requested": len(symbols), "loaded": 0,
                   "missing_features": [], "empty_after_window": []}

    # Failure post-mortem telemetry — accumulated as we score symbols
    # and apply filters.  The dashboard renders this when total_trades
    # comes back 0 so the user can see *exactly* where signals
    # vanished (raw model buys → after confidence cutoff → after
    # filters).  Cost is one Series.sum() per filter per symbol.
    diag = {
        "raw_buy_signals":         0,   # before any masking
        "after_confidence_cutoff": 0,   # after conf_threshold
        "after_filters":           0,   # after screener filters
        "conf_threshold":          float(conf_threshold or 0),
        "filters": [
            {
                "field":                f["field"],
                "op":                   f["op"],
                "value":                f["value"],
                "field_missing_in":     0,   # symbols where col absent
                "field_present_in":     0,   # symbols where col present
                "bars_passing_alone":   0,   # how often this filter alone fires
                "bars_present_total":   0,   # bars where the field is non-null
            }
            for f in (filters or [])
        ],
    }

    # ── Pre-score every symbol; queue scored DataFrames for the shared-pool
    # simulator below.  Each symbol gets its model signals + filter mask
    # applied; the simulator then walks them on a chronologically-merged
    # timeline so cash deployed on AAPL today reduces the cash available
    # for an AMZN entry the same bar.
    scored_per_symbol: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        df = _load_features(symbol, period_days)
        if df.empty:
            load_report["missing_features"].append(symbol)
            continue

        if date_window is not None:
            start, end = date_window
            try:
                df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
            except Exception:
                pass
            if df.empty:
                load_report["empty_after_window"].append(symbol)
                continue
        load_report["loaded"] += 1

        scored = model.predict_batch(df.copy())
        scored["signal"]     = scored.get("signal", "hold").fillna("hold")
        scored["confidence"] = scored.get("confidence", 0.5).fillna(0.5)

        # ── Diagnostic telemetry: count buy signals at each stage ──
        raw_buys = int((scored["signal"] == "buy").sum())
        diag["raw_buy_signals"] += raw_buys
        scored.loc[scored["confidence"] < conf_threshold, "signal"] = "hold"
        after_conf_buys = int((scored["signal"] == "buy").sum())
        diag["after_confidence_cutoff"] += after_conf_buys

        # Apply screener filters per-bar — only allow buys when all match
        if filters:
            mask = pd.Series(True, index=scored.index)
            for fi, f in enumerate(filters):
                fld = f["field"]; op = f["op"]; val = float(f["value"])
                fdiag = diag["filters"][fi]
                if fld not in scored.columns:
                    fdiag["field_missing_in"] += 1
                    mask &= False
                    continue
                fdiag["field_present_in"] += 1
                col = pd.to_numeric(scored[fld], errors="coerce")
                fdiag["bars_present_total"] += int(col.notna().sum())
                opfn = OPS.get(op)
                if opfn is None:
                    continue
                this_filter_pass = opfn(col, val).fillna(False)
                fdiag["bars_passing_alone"] += int(this_filter_pass.sum())
                mask &= this_filter_pass
            scored.loc[~mask & (scored["signal"] == "buy"), "signal"] = "hold"

        diag["after_filters"] += int((scored["signal"] == "buy").sum())
        scored_per_symbol[symbol] = scored

    # Single-line load summary so the user sees a concise view of how
    # many symbols actually had data, instead of one log line per
    # missing symbol.
    miss = load_report["missing_features"]
    if miss:
        sample = ", ".join(miss[:8]) + (f", … (+{len(miss) - 8} more)"
                                          if len(miss) > 8 else "")
        logger.info(
            "Loaded %d/%d symbols (%d missing parquet — run "
            "scripts/refetch_shallow.py --apply to fill them in).  "
            "Missing sample: %s",
            load_report["loaded"], load_report["requested"],
            len(miss), sample,
        )
    else:
        logger.info("Loaded %d/%d symbols.",
                    load_report["loaded"], load_report["requested"])

    # Build the optional regime checker (loads SPY/QQQ + sector ETF
    # parquets ONCE up front so the per-bar lookup in the sim is cheap).
    regime_checker = None
    regime_summary = None
    if market_regime_exit or sector_regime_exit:
        from dashboard.services.regime import RegimeChecker
        regime_checker = RegimeChecker(
            symbols    = list(scored_per_symbol.keys()) or list(symbols),
            use_market = bool(market_regime_exit),
            use_sector = bool(sector_regime_exit),
        )
        regime_summary = regime_checker.status_summary()
        logger.info("Regime check: %s", regime_summary)

    # Build sym→sector lookup if the user has enabled the sector
    # concentration cap.  Same data source the regime checker uses
    # (universe parquet's GICS ``sector`` column).
    sym_to_sector: Optional[dict] = None
    if max_per_sector and int(max_per_sector) > 0:
        try:
            from bot.universe import load_universe
            u = load_universe(eligible_only=False)
            if not u.empty and "sector" in u.columns:
                sub = u[u["symbol"].isin(scored_per_symbol.keys())]
                sym_to_sector = dict(zip(sub["symbol"], sub["sector"]))
                logger.info("Sector cap enabled — max %d positions per sector "
                            "(mapped %d symbols).",
                            int(max_per_sector), len(sym_to_sector))
        except Exception as exc:
            logger.warning("Sector lookup failed — cap disabled: %s", exc)

    # Run the shared-pool simulator over every symbol's scored bars
    sim = _simulate_portfolio(
        scored_per_symbol,
        starting_cash        = initial_cash,
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
        regime_checker       = regime_checker,
        max_per_sector       = int(max_per_sector) if max_per_sector else None,
        sym_to_sector        = sym_to_sector,
    )
    all_trades   = sim["trades"]
    equity_curve = sim["equity_curve"]

    # ── Post-processing: outlier-trim + year-end tax ────────────────
    # Outlier-trim drops the N largest-P&L wins from the trade log
    # before metrics are computed.  Year-end tax deducts a fixed %
    # of YTD gains from the equity curve every Dec 31.  Both are
    # off by default — when enabled, the original trade log is
    # preserved on the envelope alongside the trimmed/taxed view so
    # the dashboard can show "with vs without" panels.
    raw_trades = list(all_trades)
    raw_equity = list(equity_curve)

    kept_trades, omitted_trades = _apply_outlier_trim(all_trades,
                                                       top_n=int(omit_top_n_outliers or 0))
    if omitted_trades:
        all_trades = kept_trades
        # Rebuild the equity curve from the kept trades so metrics
        # reflect the trimmed view.
        equity_curve = _calc_equity_curve(all_trades, starting_cash=initial_cash)

    after_tax_equity, tax_events = _apply_year_end_tax(equity_curve,
                                                        tax_rate=float(tax_rate or 0))
    if tax_events:
        equity_curve = after_tax_equity

    metric_per_symbol = _build_per_symbol_summary(all_trades)

    monthly = _calc_monthly_returns(equity_curve)
    metrics = _calc_metrics(all_trades, equity_curve)
    metrics["symbols_traded"] = len(metric_per_symbol)
    metrics["starting_cash"]  = float(initial_cash)
    # ending_cash reflects the post-tax / post-trim equity if either
    # was applied; raw cash is kept on the envelope for comparison.
    metrics["ending_cash"]    = float(equity_curve[-1]["value"]
                                       if equity_curve else
                                       sim.get("final_cash", initial_cash))

    run_id = f"BT-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{model_id}-multi-{period_days}d"

    # Preset payload — lets the dashboard re-hydrate every form field
    # from this saved run via the "Apply preset" UI.
    preset = {
        "model_id":         model_id,
        "filters":          list(filters),
        "period_days":      int(period_days),
        "min_confidence":   float(conf_threshold),
        "max_symbols":      int(max_symbols),
        "use_signal_exit":  bool(use_signal_exit),
        "take_profit_pct":  take_profit_pct,
        "stop_loss_pct":    stop_loss_pct,
        "time_stop_days":   time_stop_days,
        "atr_stop_mult":    atr_stop_mult,
        "starting_cash":    float(starting_cash),
        "sizing_method":    sizing_method,
        "sizing_kwargs":    dict(sizing_kwargs or {}),
        "execution_model":  execution_model,
        "execution_delay":  int(execution_delay_bars),
        "slippage_bps":     float(slippage_bps),
        "market_regime_exit": bool(market_regime_exit),
        "sector_regime_exit": bool(sector_regime_exit),
        "regime_summary":   regime_summary,
        "tax_rate":         float(tax_rate or 0),
        "omit_top_n_outliers": int(omit_top_n_outliers or 0),
        "max_per_sector":   int(max_per_sector) if max_per_sector else 0,
    }

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
        "load_report":     load_report,
        "failure_diagnostics": diag,
        "preset":          preset,
        "tax_events":      tax_events,
        "omitted_trades":  omitted_trades,
        "raw_trades":      raw_trades if (omitted_trades or tax_events) else None,
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
    df["position"]     = 0.0
    df["cash"]         = initial_cash
    df["portfolio"]    = initial_cash
    df["trade_pl"]     = 0.0
    df["exit_reason"]  = ""
    df["in_trade"]     = False
    # Per-trade attribution columns — populated only on exit bars; the
    # trade-log table reads these to show entry/exit dates, prices, shares.
    df["entry_date"]   = ""
    df["entry_price"]  = 0.0
    df["exit_price"]   = 0.0
    df["trade_shares"] = 0

    position      = 0.0
    cash          = initial_cash
    entry_price   = 0.0
    entry_idx     = None
    entry_atr     = 0.0
    entry_date    = ""
    entry_shares  = 0

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
                    position      = shares
                    cash         -= shares * fill
                    entry_price   = fill
                    entry_idx     = i
                    entry_atr     = atr_now
                    entry_date    = (str(idx.date())
                                     if hasattr(idx, "date") else str(idx))
                    entry_shares  = shares
                    df.at[idx, "in_trade"] = True

            elif pending["type"] == "exit" and position > 0:
                # Sell-side slippage: receive a bit less than the print.
                fill = raw_fill * (1 - bps)
                pl   = (fill - entry_price) * position
                cash += position * fill
                df.at[idx, "trade_pl"]      = pl
                df.at[idx, "exit_reason"]   = pending["reason"]
                df.at[idx, "entry_date"]    = entry_date
                df.at[idx, "entry_price"]   = entry_price
                df.at[idx, "exit_price"]    = fill
                df.at[idx, "trade_shares"]  = entry_shares
                position    = 0
                entry_price = 0.0
                entry_idx   = None
                entry_atr   = 0.0
                entry_date  = ""
                entry_shares = 0

            pending = None

        df.at[idx, "position"]  = position
        df.at[idx, "cash"]      = cash
        df.at[idx, "portfolio"] = cash + position * price

    return df


def _simulate_portfolio(
    scored_per_symbol:    dict,            # symbol -> scored DataFrame
    starting_cash:        float            = 10_000.0,
    use_signal_exit:      bool             = True,
    take_profit_pct:      Optional[float]  = 0.15,
    stop_loss_pct:        Optional[float]  = 0.07,
    time_stop_days:       Optional[int]    = 30,
    atr_stop_mult:        Optional[float]  = None,
    sizing_method:        str              = "fixed_pct",
    sizing_kwargs:        Optional[dict]   = None,
    execution_model:      str              = "next_open",
    execution_delay_bars: int              = 0,
    slippage_bps:         float            = 5.0,
    regime_checker      = None,            # RegimeChecker | None
    max_per_sector:       Optional[int]    = None,    # cap simultaneous positions per sector
    sym_to_sector:        Optional[dict]   = None,    # {symbol: sector} lookup
) -> dict:
    """
    Walk every symbol's scored bars on a chronologically merged
    timeline, using a SINGLE shared cash pool for the whole portfolio.

    Differs from ``_simulate_trades`` (single-symbol):
      * one ``cash`` value spans the whole run; entries decrement,
        exits credit it
      * exits are processed before entries within the same bar so
        freed cash is available for that bar's new positions
      * if available cash can't fund the requested position size, the
        entry is skipped — no margin
      * sizing % refers to the live mark-to-market portfolio value
        (cash + open positions), so 10% means 10% of total equity, not
        10% of cash alone

    Returns:
      {"trades": [...], "equity_curve": [...], "final_cash": float}
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
    required_wait = (1 if execution_model == "next_open" else 0) + max(0, int(execution_delay_bars or 0))
    fill_col       = "open" if execution_model == "next_open" else "close"
    bps            = float(slippage_bps or 0) / 10_000.0

    cash         = float(starting_cash)
    positions:    dict[str, dict] = {}     # symbol -> entry state
    pending:      dict[str, dict] = {}     # symbol -> queued action
    sym_bar_idx:  dict[str, int]  = {}     # per-symbol bar counter
    trades        = []
    equity_curve  = [{"date": "start", "value": cash}]

    if not scored_per_symbol:
        return {"trades": [], "equity_curve": equity_curve, "final_cash": cash}

    # Collect every (symbol, timestamp) event into a chronological iter.
    all_dates = sorted({ts for sd in scored_per_symbol.values() for ts in sd.index})

    def _mark_to_market(ts) -> float:
        """Cash + Σ shares × close at ts for every open position."""
        v = cash
        for s, pos in positions.items():
            sdf = scored_per_symbol.get(s)
            if sdf is None or ts not in sdf.index:
                continue
            close = sdf.at[ts, "close"]
            if pd.isna(close):
                continue
            v += pos["shares"] * float(close)
        return v

    for ts in all_dates:
        # Bucket events on this timestamp
        events = []
        for sym, sdf in scored_per_symbol.items():
            if ts in sdf.index:
                sym_bar_idx[sym] = sym_bar_idx.get(sym, -1) + 1
                events.append((sym, sdf.loc[ts]))
        if not events:
            continue

        # ── Step 1: execute any pending EXIT actions that are due ──
        # Process exits before entries so freed cash is available below.
        for sym, row in events:
            p = pending.get(sym)
            if p is None or p["type"] != "exit" or sym not in positions:
                continue
            if (sym_bar_idx[sym] - p["queued_bar"]) < required_wait:
                continue
            if fill_col not in row or pd.isna(row[fill_col]):
                continue
            raw_fill = float(row[fill_col])
            fill     = raw_fill * (1 - bps)
            pos      = positions[sym]
            pl       = (fill - pos["entry_price"]) * pos["shares"]
            cash    += pos["shares"] * fill
            trades.append({
                "date":         str(ts.date()) if hasattr(ts, "date") else str(ts),
                "entry_date":   pos["entry_date"],
                "symbol":       sym,
                "pl":           round(pl, 2),
                "win":          pl > 0,
                "exit_reason":  p["reason"],
                "entry_price":  round(pos["entry_price"], 2),
                "exit_price":   round(fill, 2),
                "shares":       int(pos["shares"]),
            })
            del positions[sym]
            del pending[sym]
            equity_curve.append({
                "date":  str(ts.date()) if hasattr(ts, "date") else str(ts),
                "value": round(_mark_to_market(ts), 2),
            })

        # ── Step 2: execute any pending BUY actions that are due ──
        # When the user sets a sector cap (max_per_sector > 0), we
        # process pending buys in *confidence-descending* order so
        # the strongest leaders consume cap slots first, matching
        # the "leaders of leaders" intent.  Without the cap, the
        # original event-order behaviour is preserved.
        portfolio_value = _mark_to_market(ts)
        if max_per_sector and max_per_sector > 0 and sym_to_sector:
            # Count current per-sector positions
            sector_count: dict[str, int] = {}
            for held_sym in positions:
                sec = sym_to_sector.get(held_sym)
                if sec:
                    sector_count[sec] = sector_count.get(sec, 0) + 1
            # Sort pending buys for THIS bar by confidence desc
            ordered_events = sorted(
                events,
                key=lambda kv: (
                    pending.get(kv[0], {}).get("confidence", 0.0)
                    if pending.get(kv[0], {}).get("type") == "buy" else -1
                ),
                reverse=True,
            )
        else:
            sector_count = {}
            ordered_events = events

        for sym, row in ordered_events:
            p = pending.get(sym)
            if p is None or p["type"] != "buy" or sym in positions:
                continue
            if (sym_bar_idx[sym] - p["queued_bar"]) < required_wait:
                continue
            # Sector-cap enforcement — silently drop the buy when the
            # symbol's sector already has ``max_per_sector`` open
            # positions.  The "skip and free the slot for a stronger
            # name later" effect emerges naturally from the
            # confidence-desc ordering above.
            if max_per_sector and max_per_sector > 0 and sym_to_sector:
                sec = sym_to_sector.get(sym)
                if sec and sector_count.get(sec, 0) >= max_per_sector:
                    del pending[sym]
                    continue
            if fill_col not in row or pd.isna(row[fill_col]):
                continue
            raw_fill = float(row[fill_col])
            fill     = raw_fill * (1 + bps)
            if cash < fill:
                # Not enough cash for even one share — drop the queued buy.
                del pending[sym]
                continue
            atr_now = float(row.get("atr_14", 0) or 0)
            shares = _position_size_shares(
                method        = sizing_method,
                cash          = cash,
                portfolio     = portfolio_value,
                price         = fill,
                atr           = atr_now,
                sizing_kwargs = sizing_kwargs,
            )
            cost = shares * fill
            if shares <= 0 or cost > cash:
                del pending[sym]
                continue
            cash -= cost
            positions[sym] = {
                "shares":      int(shares),
                "entry_price": fill,
                "entry_atr":   atr_now,
                "entry_date":  str(ts.date()) if hasattr(ts, "date") else str(ts),
                "entry_idx":   sym_bar_idx[sym],
            }
            # Bump per-sector count so subsequent same-bar entries in
            # the same sector get capped correctly.
            if max_per_sector and max_per_sector > 0 and sym_to_sector:
                sec = sym_to_sector.get(sym)
                if sec:
                    sector_count[sec] = sector_count.get(sec, 0) + 1
            del pending[sym]

        # ── Step 3: detect new signals / exits this bar and queue them ──
        for sym, row in events:
            if sym in pending:
                continue
            sig    = row.get("signal", "hold")
            close  = row["close"]
            if pd.isna(close):
                continue
            price = float(close)

            if sym not in positions:
                if sig == "buy":
                    pending[sym] = {
                        "type": "buy",
                        "queued_bar": sym_bar_idx[sym],
                        # Capture confidence at signal time so the
                        # sector-cap entry path can prefer the
                        # strongest leaders when cash/cap-limited.
                        "confidence": float(row.get("confidence", 0.5)
                                              if not pd.isna(row.get("confidence", 0.5))
                                              else 0.5),
                    }
                continue

            pos          = positions[sym]
            pl_per_share = price - pos["entry_price"]
            ret_pct      = pl_per_share / pos["entry_price"] if pos["entry_price"] else 0.0
            held_days    = sym_bar_idx[sym] - pos["entry_idx"]
            atr_floor    = (pos["entry_price"] - atr_stop_mult * pos["entry_atr"]
                             if (atr_stop_mult is not None and pos["entry_atr"] > 0)
                             else None)
            exit_reason = ""
            # Regime exits run FIRST — they're hard "macro" overrides
            # that should win over per-symbol logic.  When the broad
            # market or this symbol's sector ETF is in a clear
            # downtrend (close < EMA10 AND EMA20 AND EMA50) we sell
            # regardless of what the strategy thinks.
            if regime_checker is not None:
                rr = regime_checker.regime_exit_reason(sym, ts)
                if rr:
                    exit_reason = rr
            if exit_reason:
                pass
            elif use_signal_exit and sig == "sell":                                  exit_reason = "signal"
            elif take_profit_pct is not None and ret_pct >=  take_profit_pct:        exit_reason = "take_profit"
            elif stop_loss_pct  is not None and ret_pct <= -stop_loss_pct:           exit_reason = "stop_loss"
            elif atr_floor is not None and price <= atr_floor:                       exit_reason = "atr_stop"
            elif time_stop_days is not None and held_days >= time_stop_days:         exit_reason = "time_stop"
            if exit_reason:
                pending[sym] = {"type": "exit", "queued_bar": sym_bar_idx[sym], "reason": exit_reason}

    # Final liquidation at the last available bar so equity is realised.
    last_ts = all_dates[-1] if all_dates else None
    if last_ts is not None:
        for sym, pos in list(positions.items()):
            sdf = scored_per_symbol.get(sym)
            if sdf is None or last_ts not in sdf.index:
                continue
            row = sdf.loc[last_ts]
            if fill_col not in row or pd.isna(row[fill_col]):
                continue
            raw_fill = float(row[fill_col])
            fill     = raw_fill * (1 - bps)
            pl       = (fill - pos["entry_price"]) * pos["shares"]
            cash    += pos["shares"] * fill
            trades.append({
                "date":         str(last_ts.date()) if hasattr(last_ts, "date") else str(last_ts),
                "entry_date":   pos["entry_date"],
                "symbol":       sym,
                "pl":           round(pl, 2),
                "win":          pl > 0,
                "exit_reason":  "final_liquidation",
                "entry_price":  round(pos["entry_price"], 2),
                "exit_price":   round(fill, 2),
                "shares":       int(pos["shares"]),
            })
            del positions[sym]

    equity_curve.append({
        "date":  str(last_ts.date()) if last_ts is not None and hasattr(last_ts, "date")
                  else (str(last_ts) if last_ts is not None else "end"),
        "value": round(cash, 2),
    })

    # Trades may already be roughly in date order via the timeline, but
    # within a bar exits append before entries.  Sort defensively for the
    # equity-curve-by-date renderer downstream.
    trades.sort(key=lambda t: (t["date"], t["symbol"]))
    return {"trades": trades, "equity_curve": equity_curve, "final_cash": cash}


def _build_per_symbol_summary(trades: list[dict]) -> list[dict]:
    """Aggregate the trade log into one row per traded symbol."""
    by_sym: dict[str, list[dict]] = {}
    for t in trades:
        by_sym.setdefault(t.get("symbol", "—"), []).append(t)
    rows = []
    for sym, syms_trades in by_sym.items():
        wins = sum(1 for t in syms_trades if t.get("win"))
        pl   = sum(float(t.get("pl", 0) or 0) for t in syms_trades)
        rows.append({
            "symbol":   sym,
            "trades":   len(syms_trades),
            "pl":       round(pl, 2),
            "win_rate": round(wins / len(syms_trades) * 100, 1) if syms_trades else 0.0,
        })
    rows.sort(key=lambda r: r["pl"], reverse=True)
    return rows


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


def _apply_outlier_trim(trades: list, top_n: int = 0) -> tuple[list, list]:
    """Optionally remove the ``top_n`` largest-P&L wins from the trade
    list before metrics are computed.  Big lucky wins can dominate the
    aggregate return on a small trade count and obscure whether the
    strategy has a real edge.  Excluded trades are returned separately
    so the dashboard can still show them in a "trimmed outliers" panel.

    Args:
        trades: full trade log (each row has a ``pl`` field).
        top_n:  how many of the largest-P&L wins to omit.

    Returns:
        ``(kept_trades, omitted_trades)``.  Original order preserved
        in ``kept_trades``.  ``omitted_trades`` is sorted by P&L desc.
    """
    n = int(top_n or 0)
    if n <= 0 or not trades:
        return list(trades), []
    # Pick the indices of the top-N P&L wins
    sorted_with_idx = sorted(
        enumerate(trades),
        key=lambda kv: float(kv[1].get("pl", 0) or 0),
        reverse=True,
    )
    omit_indices = {idx for idx, _ in sorted_with_idx[:n]}
    omitted = [trades[i] for i, _ in sorted_with_idx[:n]]
    kept    = [t for i, t in enumerate(trades) if i not in omit_indices]
    return kept, omitted


def _apply_year_end_tax(
    equity_curve: list,
    tax_rate: float = 0.30,
) -> tuple[list, list]:
    """Adjust the equity curve to reflect a year-end capital-gains tax
    payment.  Each Dec 31 (or last bar of the calendar year) we deduct
    ``tax_rate × max(0, ytd_gains)`` from the running equity.

    Args:
        equity_curve: [{date, value}, ...] in chronological order.
        tax_rate:     fraction of YTD gains paid in tax (0.30 = 30%).

    Returns:
        ``(after_tax_curve, tax_events)``.  ``tax_events`` is a list of
        ``{date, ytd_gain, tax_paid, equity_before, equity_after}``
        rows the dashboard can show in a small panel.
    """
    if not equity_curve or float(tax_rate or 0) <= 0:
        return list(equity_curve), []

    rate = float(tax_rate)
    out: list[dict] = []
    events: list[dict] = []
    # year_start_equity tracks the equity at the start of the
    # current calendar year (after-tax basis).  We seed it with the
    # equity_curve's first point — typically the "start" bar with the
    # initial deposit.  This way Y1 gains are measured from the
    # starting capital, not from the second bar.
    year_start_equity: Optional[float] = (
        float(equity_curve[0].get("value", 0)) if equity_curve else None
    )
    current_year: Optional[int] = None
    cumulative_tax_paid = 0.0

    for i, point in enumerate(equity_curve):
        d = point.get("date", "")
        v = float(point.get("value", 0))
        # First-year anchor: when we see the first dated bar, record
        # the calendar year so we can detect year-rollover later.
        if current_year is None and d != "start":
            try:
                current_year = int(str(d)[:4])
            except ValueError:
                pass

        # Detect "last bar of a calendar year" — current bar is in
        # year Y, next bar is in year Y+1 (or this is the final bar)
        try:
            this_year = int(str(d)[:4]) if d != "start" else None
        except ValueError:
            this_year = None
        next_d = equity_curve[i + 1].get("date", "") if i + 1 < len(equity_curve) else ""
        try:
            next_year = int(str(next_d)[:4]) if next_d not in ("", "start") else None
        except ValueError:
            next_year = None

        is_year_end = (
            this_year is not None
            and (next_year is None or next_year > this_year)
        )

        adjusted_v = v - cumulative_tax_paid
        out.append({"date": d, "value": round(adjusted_v, 2)})

        if is_year_end and year_start_equity is not None:
            # ytd_gain measured against last year-end's after-tax
            # equity (or the initial deposit for Y1).
            ytd_gain = max(0.0, adjusted_v - year_start_equity)
            tax = ytd_gain * rate
            if tax > 0:
                events.append({
                    "date":          d,
                    "year":          this_year,
                    "ytd_gain":      round(ytd_gain, 2),
                    "tax_rate":      rate,
                    "tax_paid":      round(tax, 2),
                    "equity_before": round(adjusted_v, 2),
                    "equity_after":  round(adjusted_v - tax, 2),
                })
                cumulative_tax_paid += tax
                # Update the just-appended point with the after-tax value
                out[-1]["value"] = round(adjusted_v - tax, 2)
            # Roll the year-start anchor to this bar's after-tax value
            year_start_equity = adjusted_v - tax
            current_year = (next_year if next_year is not None
                            else (this_year + 1 if this_year else None))

    return out, events


def _calc_equity_curve(trades: list, starting_cash: float = 10_000.0) -> list[dict]:
    """
    Build the cumulative equity curve from a trade log.

    For multi-symbol runs the engine simulates each symbol with its
    own ``starting_cash`` account, then aggregates the trade list.
    The equity curve here is therefore the *aggregate* across all
    those independent per-symbol accounts — pass
    ``starting_cash * n_symbols`` if you want a comparable percentage
    return against total deployed capital.
    """
    equity = float(starting_cash)
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

    # Build the preset block up-front so EVERY return path (including
    # empty-universe / model-load failures) carries the cross-sectional
    # marker.  The dashboard explainer uses ``model_kind`` to branch
    # away from per-symbol exit rules — without it, loading an empty
    # saved run shows a misleading "Model sell signal" row.
    xs_preset = {
        "model_id":       model_id,
        "model_kind":     "cross_sectional",
        "period_days":    int(period_days),
        "starting_cash":  float(starting_cash),
        "slippage_bps":   float(slippage_bps),
        "top_decile":     float(top_decile),
        "rebalance_days": int(rebalance_days),
    }

    # Distinguish None ("use default") from [] ("explicitly nothing")
    if symbols is None:
        syms = select_universe("top_100", limit=50)
    else:
        syms = list(symbols)

    # Per-symbol load diagnostics — populated as we go so the
    # dashboard's render_results can show a precise diagnostic
    # instead of the generic "no data on disk" message when an XS
    # run produces no rebalances.
    load_report = {
        "requested":         len(syms),
        "loaded":            0,
        "missing_features":  [],
        "empty_after_window": [],
        # XS-specific:
        "formation_warmup_days":    0,
        "user_period_first_bar":    None,
        "panel_first_bar":          None,
        "panel_last_bar":           None,
        "first_valid_rank_bar":     None,
        "rebalances_in_user_period": 0,
    }

    if not syms:
        return {**_empty_results(), "preset": xs_preset, "load_report": load_report}

    try:
        model = get_model(model_id)
    except Exception as exc:
        logger.error("Cannot load cross-sectional model %r: %s", model_id, exc)
        return {**_empty_results(), "preset": xs_preset, "load_report": load_report}

    # Auto-extend the loaded history by the model's formation
    # requirement.  Without this, a "12-month backtest" of JT-12-1
    # would only load 252 trading days — exactly the formation
    # window — and `rank_universe` would return NaN for every bar
    # (the strategy needs PRIOR data to compute the trailing 12-1
    # return).  Result: 0 rebalances, dashboard shows misleading
    # "no data on disk" diagnostic.  Fix: load extra warmup, then
    # only count rebalances within the user's requested period.
    formation_months = (
        getattr(model, "FORMATION_MONTHS", 0)
        + getattr(model, "SKIP_MONTHS", 0)
    )
    formation_warmup_days = (
        # Generous calendar-day estimate: ~31 days/month + 30-day buffer
        formation_months * 31 + 30 if formation_months > 0 else 0
    )
    effective_load_days = int(period_days) + int(formation_warmup_days)
    load_report["formation_warmup_days"] = int(formation_warmup_days)

    # Build the wide-format close panel from disk.  Belt-and-suspenders:
    # ``_load_features`` already strips tz info, but we double-check here
    # so a stale parquet that escaped the loader (or a future code path
    # that bypasses _load_features) can't blow up pd.concat with
    # "Cannot join tz-naive with tz-aware DatetimeIndex".
    closes = {}
    for s in syms:
        df = _load_features(s, effective_load_days)
        if df.empty or "close" not in df.columns:
            load_report["missing_features"].append(s)
            continue
        s_close = df["close"]
        if s_close.index.tz is not None:
            s_close = s_close.copy()
            s_close.index = s_close.index.tz_localize(None)
        closes[s] = s_close
        load_report["loaded"] += 1
    if not closes:
        logger.warning("No symbols had features data for cross-sectional run.")
        return {**_empty_results(), "preset": xs_preset, "load_report": load_report}

    panel = pd.concat(closes, axis=1).sort_index()
    load_report["panel_first_bar"] = (
        str(panel.index.min().date()) if hasattr(panel.index.min(), "date")
        else str(panel.index.min())
    )
    load_report["panel_last_bar"] = (
        str(panel.index.max().date()) if hasattr(panel.index.max(), "date")
        else str(panel.index.max())
    )

    # Get cross-sectional ranks per bar
    try:
        ranks = model.rank_universe(panel.copy())
    except Exception as exc:
        logger.error("rank_universe failed: %s", exc, exc_info=True)
        return {**_empty_results(), "preset": xs_preset, "load_report": load_report}

    # Find the first bar where ranks are valid (formation completed).
    # This is the earliest date the strategy could possibly have
    # signalled a rebalance — useful context for the diagnostic
    # panel when the user's period_days is too short for formation.
    if isinstance(ranks, pd.DataFrame) and not ranks.empty:
        valid_mask = ranks.notna().any(axis=1)
        if valid_mask.any():
            first_valid = ranks.index[valid_mask][0]
            load_report["first_valid_rank_bar"] = (
                str(first_valid.date()) if hasattr(first_valid, "date")
                else str(first_valid)
            )

    # Compute the user's requested period start.  Rebalances and
    # equity tracking happen ONLY within [user_period_start, end].
    # Any earlier dates are pure formation-warmup and don't count
    # as backtest activity.
    user_period_start = panel.index.max() - pd.Timedelta(days=int(period_days))
    load_report["user_period_first_bar"] = (
        str(user_period_start.date()) if hasattr(user_period_start, "date")
        else str(user_period_start)
    )

    # Walk the rebalance schedule
    bps           = float(slippage_bps or 0) / 10_000.0
    cash          = float(starting_cash)
    equity        = cash
    holdings:      dict[str, int]   = {}     # symbol -> shares
    entry_prices:  dict[str, float] = {}     # symbol -> entry fill price
    entry_dates:   dict[str, str]   = {}     # symbol -> ISO entry date
    trades        = []
    equity_curve  = [{"date": "start", "value": equity}]

    # Only include rebalance dates within the user-requested period.
    # We anchor the cadence at panel.start so warmup-period rebalances
    # are computed in the same grid (consistent with re-running with
    # a longer period_days), but we don't ACT on the warmup rebalances
    # — the loop skips them silently below.
    rebal_dates = [d for d in panel.index[::rebalance_days]
                    if d >= user_period_start]
    load_report["rebalances_in_user_period"] = len(rebal_dates)

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
        # IMPORTANT: if the symbol's price is NaN at the rebalance date
        # (corp action, halt, delist), fall back to its most recent
        # valid close.  The old code did `continue` here which silently
        # cleared the position from holdings WITHOUT crediting the
        # cash — leaking real money out of the portfolio every time a
        # held symbol had a missing bar.  That's how a long-only
        # momentum run on 100 symbols over 6 years could end at -99%.
        for sym, shares in list(holdings.items()):
            price = panel.at[dt, sym] if sym in panel.columns else None
            if price is None or pd.isna(price):
                # Forward-fill: use the last valid price <= dt.
                if sym in panel.columns:
                    series = panel[sym].loc[:dt].dropna()
                    if not series.empty:
                        price = float(series.iloc[-1])
                if price is None or pd.isna(price):
                    # The symbol genuinely has no price history through
                    # dt — should be impossible since we only entered
                    # on a valid bar, but punt safely: use entry price
                    # to credit the original cost back (zero P&L).
                    price = entry_prices.get(sym, 0.0)
                    if price <= 0:
                        # Last resort: drop the position without leaking.
                        continue
            fill = float(price) * (1 - bps)
            cash += shares * fill
            entry_fill = entry_prices.get(sym, fill)
            pl = round((fill - entry_fill) * shares, 2)
            trades.append({
                "date":        str(dt.date()) if hasattr(dt, "date") else str(dt),
                "entry_date":  entry_dates.get(sym, ""),
                "symbol":      sym,
                "pl":          pl,
                "win":         pl > 0,
                "exit_reason": "rebalance",
                "entry_price": round(entry_fill, 2),
                "exit_price":  round(fill, 2),
                "shares":      int(shares),
            })
        holdings.clear()
        entry_prices.clear()
        entry_dates.clear()

        # ── Allocate equally among picks (buy at close × (1+bps)) ──
        per_position = cash / len(picks)
        entry_iso = str(dt.date()) if hasattr(dt, "date") else str(dt)
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
            holdings[sym]     = shares
            entry_prices[sym] = fill
            entry_dates[sym]  = entry_iso

        # mark-to-market equity — use forward-filled close so a one-day
        # NaN doesn't artificially zero out a position's value.
        m2m = 0.0
        for sym, shares in holdings.items():
            if sym not in panel.columns:
                continue
            cur = panel.at[dt, sym]
            if pd.isna(cur):
                series = panel[sym].loc[:dt].dropna()
                if series.empty:
                    continue
                cur = float(series.iloc[-1])
            m2m += shares * float(cur)
        equity = cash + m2m
        equity_curve.append({
            "date":  str(dt.date()) if hasattr(dt, "date") else str(dt),
            "value": round(float(equity), 2),
        })

    # Final liquidation at panel's last close so the equity curve is
    # closed.  Same forward-fill logic as the rebalance liquidation —
    # a NaN final-bar price must NOT silently drop the position.
    last_dt = panel.index[-1]
    last_iso = str(last_dt.date()) if hasattr(last_dt, "date") else str(last_dt)
    for sym, shares in list(holdings.items()):
        if sym not in panel.columns:
            continue
        price = panel.at[last_dt, sym]
        if pd.isna(price):
            series = panel[sym].dropna()
            if not series.empty:
                price = float(series.iloc[-1])
        if price is None or pd.isna(price):
            price = entry_prices.get(sym, 0.0)
            if price <= 0:
                continue
        fill = float(price) * (1 - bps)
        cash += shares * fill
        entry_fill = entry_prices.get(sym, fill)
        pl = round((fill - entry_fill) * shares, 2)
        trades.append({
            "date":        last_iso,
            "entry_date":  entry_dates.get(sym, ""),
            "symbol":      sym,
            "pl":          pl,
            "win":         pl > 0,
            "exit_reason": "final_liquidation",
            "entry_price": round(entry_fill, 2),
            "exit_price":  round(fill, 2),
            "shares":      int(shares),
        })
    holdings.clear()
    entry_prices.clear()
    entry_dates.clear()
    equity = cash
    equity_curve.append({"date": str(last_dt.date()), "value": round(equity, 2)})

    monthly = _calc_monthly_returns(equity_curve)

    # Metrics use rebalance-period equity changes as the "trade" unit —
    # for a cross-sectional strategy that's the natural P&L granularity
    # (each rebalance is one cohort decision).  The per-symbol trade log
    # is preserved separately on the envelope as `trades` so the UI can
    # show the actual buy/sell history if needed.
    rebalance_pls = [equity_curve[k]["value"] - equity_curve[k - 1]["value"]
                      for k in range(1, len(equity_curve))]
    rebalance_trades = [
        {"date": equity_curve[k]["date"], "symbol": "—",
         "pl":   round(p, 2), "win": p > 0,
         "exit_reason": "rebalance"}
        for k, p in enumerate(rebalance_pls, start=1) if p != 0
    ]
    metrics = _calc_metrics(rebalance_trades, equity_curve)
    metrics["symbols_traded"] = len({t["symbol"] for t in trades})

    run_id = f"XS-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{model_id}-{len(syms)}syms"
    # ``xs_preset`` was built up-front (see top of function) so that
    # every return path — including empty-universe / load-failure early
    # returns — carries the cross-sectional marker for the dashboard.
    return {
        "run_id":            run_id,
        "model":             model_id,
        "symbol":            f"{len(syms)} symbols (cross-sectional)",
        "period_days":       period_days,
        "conf_threshold":    0,
        "metrics":           metrics,
        "equity_curve":      equity_curve,
        "monthly_returns":   monthly,
        "trades":            trades,             # per-symbol buy/sell history
        "rebalance_trades":  rebalance_trades,   # one entry per rebalance period
        "preset":            xs_preset,
        "load_report":       load_report,
        "run_at":            datetime.now().isoformat(),
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
            "fold":            k,
            "oos_window":      (str(oos_window[0])[:10], str(oos_window[1])[:10]),
            "metrics":         m,
            "trades":          oos_out.get("trades", []),
            "equity_curve":    oos_out.get("equity_curve", []),
            "monthly_returns": oos_out.get("monthly_returns", []),
            "run_id":          oos_out.get("run_id", f"fold-{k}"),
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
