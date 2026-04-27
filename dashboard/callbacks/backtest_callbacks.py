"""
dashboard/callbacks/backtest_callbacks.py
------------------------------------------
Registers all backtest-related Dash callbacks.
Imported by ``dashboard/app.py`` so the @callback decorators run.
"""
from __future__ import annotations

import logging

from dash import Input, Output, State, ALL, callback_context, html, no_update, callback

from dashboard.backtest_engine import (
    run_backtest, run_filtered_backtest, run_walk_forward,
    save_backtest, load_backtest, list_saved_backtests,
)
from dashboard.pages.backtest import render_results, _filter_row

logger = logging.getLogger(__name__)


# ── NL query → populate model + filters + period ─────────────────────────────

@callback(
    Output("bt-dd-model",       "value",   allow_duplicate=True),
    Output("bt-input-period",   "value",   allow_duplicate=True),
    Output("bt-input-conf",     "value",   allow_duplicate=True),
    Output("bt-filter-rows",    "children", allow_duplicate=True),
    Output("bt-nl-status",      "children"),
    Input("bt-nl-parse",        "n_clicks"),
    State("bt-nl-query",        "value"),
    prevent_initial_call=True,
)
def parse_nl_query(n_clicks, text):
    if not n_clicks or not text:
        return no_update, no_update, no_update, no_update, "Type a query, then Parse."
    try:
        from bot.nl_query import parse_query
        parsed = parse_query(text)
    except Exception as exc:
        return no_update, no_update, no_update, no_update, f"❌ {exc}"

    rows = []
    for i, f in enumerate(parsed.filters):
        rows.append(_filter_row(i, f["field"], f["op"], f["value"]))

    status = (
        f"✅ {parsed.rationale}  "
        f"(model={parsed.model_id}, {parsed.period_days}d, "
        f"{len(parsed.filters)} filter(s))"
    )
    return (
        parsed.model_id,
        parsed.period_days,
        parsed.min_confidence,
        rows,
        status,
    )


# ── Add manual filter row ───────────────────────────────────────────────────

@callback(
    Output("bt-filter-rows", "children", allow_duplicate=True),
    Input("bt-add-filter",   "n_clicks"),
    State("bt-filter-rows",  "children"),
    prevent_initial_call=True,
)
def add_filter(n, current):
    if not n:
        return no_update
    current = list(current or [])
    current.append(_filter_row(len(current)))
    return current


# ── Run / load backtest ──────────────────────────────────────────────────────

@callback(
    Output("bt-store-results", "data"),
    Input("bt-btn-run",      "n_clicks"),
    Input("bt-dd-saved",     "value"),
    State("bt-dd-model",     "value"),
    State("bt-dd-scope",     "value"),
    State("bt-input-max",    "value"),
    State("bt-dd-tf",        "value"),
    State("bt-input-period", "value"),
    State("bt-input-conf",   "value"),
    State("bt-indicators",   "value"),
    State({"type": "bt-filter-field", "index": ALL}, "value"),
    State({"type": "bt-filter-op",    "index": ALL}, "value"),
    State({"type": "bt-filter-value", "index": ALL}, "value"),
    State("bt-exit-signal-on", "value"),
    State("bt-exit-tp-on",     "value"),
    State("bt-exit-tp-val",    "value"),
    State("bt-exit-sl-on",     "value"),
    State("bt-exit-sl-val",    "value"),
    State("bt-exit-ts-on",     "value"),
    State("bt-exit-ts-val",    "value"),
    State("bt-acct-cash",      "value"),
    State("bt-acct-sizing",    "value"),
    State("bt-acct-arg-a",     "value"),
    State("bt-acct-arg-b",     "value"),
    State("bt-acct-atr-stop",  "value"),
    State("bt-real-em",        "value"),
    State("bt-real-delay",     "value"),
    State("bt-real-slip",      "value"),
    State("bt-real-val",       "value"),
    prevent_initial_call=True,
)
def run_or_load(n_run, saved_id, model, scope, max_syms, tf, period, conf,
                indicators, ff, fo, fv,
                signal_on, tp_on, tp_val, sl_on, sl_val, ts_on, ts_val,
                acct_cash, sizing_method, sizing_a, sizing_b, atr_stop,
                exec_model, exec_delay, slip_bps, val_mode):
    ctx = callback_context.triggered[0]["prop_id"]
    if "bt-dd-saved" in ctx and saved_id:
        return load_backtest(saved_id)
    if "bt-btn-run" not in ctx or not n_run:
        return {}

    # Parse filters
    filters = []
    for fld, op, val in zip(ff or [], fo or [], fv or []):
        if fld is None or op is None or val in (None, ""):
            continue
        try:
            filters.append({"field": fld, "op": op, "value": float(val)})
        except (TypeError, ValueError):
            continue

    # Resolve exit conditions — checklist returns ['on'] when checked
    use_signal_exit = bool(signal_on)
    take_profit_pct = float(tp_val) if (tp_on and tp_val not in (None, ""))      else None
    stop_loss_pct   = float(sl_val) if (sl_on and sl_val not in (None, ""))      else None
    time_stop_days  = int(ts_val)   if (ts_on and ts_val not in (None, "", 0))   else None

    # Resolve symbols from universe scope
    from bot.universe import select_universe
    syms = select_universe(scope or "top_100", limit=int(max_syms or 50))

    # Position sizing kwargs depend on the chosen method
    method = (sizing_method or "fixed_pct").lower()
    a = float(sizing_a) if sizing_a not in (None, "") else 0.95
    b = float(sizing_b) if sizing_b not in (None, "") else 2.0
    if   method == "fixed_pct":           sizing_kwargs = {"pct": a}
    elif method in ("kelly", "half_kelly"):
        sizing_kwargs = {"win_rate": a, "win_loss_ratio": b}
    elif method == "atr_risk":            sizing_kwargs = {"risk_pct": a, "atr_mult": b}
    else:                                  sizing_kwargs = {}

    atr_stop_mult = float(atr_stop) if (atr_stop and float(atr_stop) > 0) else None

    # Common engine kwargs shared between full + walk-forward paths
    engine_kwargs = dict(
        conf_threshold       = float(conf or 0.65),
        max_symbols          = int(max_syms or 50),
        use_signal_exit      = use_signal_exit,
        take_profit_pct      = take_profit_pct,
        stop_loss_pct        = stop_loss_pct,
        time_stop_days       = time_stop_days,
        atr_stop_mult        = atr_stop_mult,
        starting_cash        = float(acct_cash or 10_000),
        sizing_method        = method,
        sizing_kwargs        = sizing_kwargs,
        execution_model      = exec_model or "next_open",
        execution_delay_bars = int(exec_delay or 0),
        slippage_bps         = float(slip_bps or 0),
    )

    # Walk-forward branch
    if val_mode and val_mode.startswith("wf"):
        try:
            n_folds = int((val_mode or "wf4")[2:])
        except ValueError:
            n_folds = 4
        return run_walk_forward(
            model_id    = model or "rsi_macd_v1",
            n_folds     = n_folds,
            symbols     = syms,
            period_days = int(period or 365),
            **engine_kwargs,
        )

    # Single-sample branch (default "full")
    return run_filtered_backtest(
        model_id    = model or "rsi_macd_v1",
        filters     = filters,
        symbols     = syms,
        period_days = int(period or 365),
        **engine_kwargs,
    )


# ── Render results ──────────────────────────────────────────────────────────

@callback(
    Output("bt-results-area", "children"),
    Input("bt-store-results", "data"),
    prevent_initial_call=True,
)
def display_results(results):
    return render_results(results or {})


# ── Save run ────────────────────────────────────────────────────────────────

@callback(
    Output("bt-save-msg",  "children"),
    Output("bt-dd-saved",  "options"),
    Input("bt-btn-save",   "n_clicks"),
    State("bt-store-results", "data"),
    prevent_initial_call=True,
)
def save_run(n_clicks, results):
    if not results or not results.get("run_id"):
        return "Nothing to save — run a backtest first.", list_saved_backtests()
    run_id = save_backtest(results)
    return f"Saved: {run_id}", list_saved_backtests()
