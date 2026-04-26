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
    run_backtest, run_filtered_backtest,
    save_backtest, load_backtest, list_saved_backtests,
)
from dashboard.pages.backtest import render_results, _filter_row

logger = logging.getLogger(__name__)


# ── NL query → populate model + filters + period ─────────────────────────────

@callback(
    Output("bt-dd-model",     "value", allow_duplicate=True),
    Output("bt-dd-period",    "value", allow_duplicate=True),
    Output("bt-dd-conf",      "value", allow_duplicate=True),
    Output("bt-filter-rows",  "children", allow_duplicate=True),
    Output("bt-nl-status",    "children"),
    Input("bt-nl-parse",      "n_clicks"),
    State("bt-nl-query",      "value"),
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
    Input("bt-btn-run",    "n_clicks"),
    Input("bt-dd-saved",   "value"),
    State("bt-dd-model",   "value"),
    State("bt-dd-symbol",  "value"),
    State("bt-dd-tf",      "value"),
    State("bt-dd-period",  "value"),
    State("bt-dd-conf",    "value"),
    State("bt-indicators", "value"),
    State({"type": "bt-filter-field", "index": ALL}, "value"),
    State({"type": "bt-filter-op",    "index": ALL}, "value"),
    State({"type": "bt-filter-value", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def run_or_load(n_run, saved_id, model, symbol, tf, period, conf,
                indicators, ff, fo, fv):
    ctx = callback_context.triggered[0]["prop_id"]
    if "bt-dd-saved" in ctx and saved_id:
        return load_backtest(saved_id)
    if "bt-btn-run" in ctx and n_run:
        filters = []
        for fld, op, val in zip(ff or [], fo or [], fv or []):
            if fld is None or op is None or val in (None, ""):
                continue
            try:
                filters.append({"field": fld, "op": op, "value": float(val)})
            except (TypeError, ValueError):
                continue

        if filters or (symbol or "").lower() == "all":
            return run_filtered_backtest(
                model_id       = model or "rsi_macd_v1",
                filters        = filters,
                symbols        = None if (symbol or "All") == "All" else [symbol],
                period_days    = int(period or 365),
                conf_threshold = float(conf or 0.65),
            )

        return run_backtest(
            model_name      = model or "rsi_macd_v1",
            symbol          = symbol or "AAPL",
            timeframe       = tf or "1d",
            period_days     = int(period or 365),
            conf_threshold  = float(conf or 0.65),
            active_indicators = indicators or [],
        )
    return {}


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
