"""
dashboard/callbacks/backtest_callbacks.py
------------------------------------------
Registers all backtest-related Dash callbacks.
Import this module in app.py after app is created.
"""

from dash import Input, Output, State, callback_context, html, callback
from dashboard.backtest_engine import run_backtest, save_backtest, load_backtest, list_saved_backtests
from dashboard.pages.backtest import render_results


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
    prevent_initial_call=True,
)
def run_or_load(n_run, saved_id, model, symbol, tf, period, conf, indicators):
    ctx = callback_context.triggered[0]["prop_id"]
    if "bt-dd-saved" in ctx and saved_id:
        return load_backtest(saved_id)
    if "bt-btn-run" in ctx and n_run:
        return run_backtest(
            model_name=model or "model_v1",
            symbol=symbol or "AAPL",
            timeframe=tf or "1d",
            period_days=int(period or 365),
            conf_threshold=float(conf or 0.65),
            active_indicators=indicators or [],
        )
    return {}


@callback(
    Output("bt-results-area", "children"),
    Input("bt-store-results", "data"),
    prevent_initial_call=True,
)
def display_results(results):
    return render_results(results or {})


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
