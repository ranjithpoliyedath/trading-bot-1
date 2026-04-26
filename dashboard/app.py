"""
dashboard/app.py
----------------
Main entry point. Run with:  python -m dashboard.app
Then open:  http://localhost:8050
"""

import dash
from dash import dcc, html, Input, Output, State, ALL, callback_context, no_update
import dash_bootstrap_components as dbc
from dashboard.pages import (
    overview, market_overview, screener as screener_page,
    model_builder as builder_page, backtest as bt_page,
)
from dashboard.components.global_controls import render_topbar
from dashboard.callbacks import backtest_callbacks  # noqa: F401  (registers callbacks)
from bot.screener import Filter, run_screener

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Trading Bot Dashboard",
)

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="store-account", data="paper"),
    dcc.Store(id="store-model",   data="model_v1"),
    dcc.Store(id="store-symbol",  data="AAPL"),
    dcc.Store(id="store-view",    data="overview"),

    render_topbar(),

    html.Div(id="page-content", style={"padding": "0 24px 24px"}),
], style={"fontFamily": "system-ui, sans-serif", "background": "#F8F8F7", "minHeight": "100vh"})


@app.callback(
    Output("page-content", "children"),
    Input("store-view", "data"),
    Input("store-account", "data"),
    Input("store-model", "data"),
    Input("store-symbol", "data"),
)
def render_page(view, account, model, symbol):
    if view == "backtest":
        return bt_page.layout(account, model, symbol)
    if view == "screener":
        return screener_page.layout(account, model, symbol)
    if view == "builder":
        return builder_page.layout(account, model, symbol)
    return market_overview.layout(account, model, symbol)


@app.callback(
    Output("store-account", "data"),
    Input("btn-paper", "n_clicks"),
    Input("btn-live",  "n_clicks"),
    State("store-account", "data"),
    prevent_initial_call=True,
)
def toggle_account(n_paper, n_live, current):
    ctx = callback_context.triggered[0]["prop_id"]
    if "btn-paper" in ctx:
        return "paper"
    if "btn-live" in ctx:
        return "live"
    return current


@app.callback(
    Output("store-model",  "data"),
    Input("dd-model", "value"),
)
def set_model(value):
    return value or "model_v1"


@app.callback(
    Output("store-symbol", "data", allow_duplicate=True),
    Input("dd-symbol", "value"),
    prevent_initial_call=True,
)
def set_symbol(value):
    return value or "AAPL"


@app.callback(
    Output("store-view", "data", allow_duplicate=True),
    Input("btn-overview",  "n_clicks"),
    Input("btn-screener",  "n_clicks"),
    Input("btn-builder",   "n_clicks"),
    Input("btn-trades",    "n_clicks"),
    Input("btn-model-tab", "n_clicks"),
    Input("btn-market",    "n_clicks"),
    Input("btn-backtest",  "n_clicks"),
    State("store-view", "data"),
    prevent_initial_call=True,
)
def set_view(ov, sc, bld, tr, md, mk, bt, current):
    ctx = callback_context.triggered[0]["prop_id"]
    mapping = {
        "btn-overview":  "overview",
        "btn-screener":  "screener",
        "btn-builder":   "builder",
        "btn-trades":    "trades",
        "btn-model-tab": "model",
        "btn-market":    "market",
        "btn-backtest":  "backtest",
    }
    for key, val in mapping.items():
        if key in ctx:
            return val
    return current


# ── Screener callbacks ────────────────────────────────────────────────────────

@app.callback(
    Output("screener-filters", "children"),
    Input("btn-add-filter", "n_clicks"),
    State("screener-filters", "children"),
    prevent_initial_call=True,
)
def add_filter_row(n_clicks, current):
    if not n_clicks:
        return no_update
    idx = len(current or [])
    current = list(current or [])
    current.append(screener_page._filter_row(idx, "rsi_14", "<", 30))
    return current


@app.callback(
    Output("screener-results", "children"),
    Input("btn-run-screener", "n_clicks"),
    State({"type": "screener-field", "index": ALL}, "value"),
    State({"type": "screener-op",    "index": ALL}, "value"),
    State({"type": "screener-value", "index": ALL}, "value"),
    State("screener-sort",  "value"),
    State("screener-limit", "value"),
    prevent_initial_call=True,
)
def run_screener_callback(n_clicks, fields, ops, values, sort_by, limit):
    filters = []
    for fld, op, val in zip(fields or [], ops or [], values or []):
        if fld is None or op is None or val is None or val == "":
            continue
        try:
            filters.append(Filter(field=fld, op=op, value=float(val)))
        except (TypeError, ValueError):
            continue
    if not filters:
        return html.P("Add at least one filter with a numeric value.",
                      style={"color": "#aaa", "fontSize": "13px", "padding": "12px"})
    try:
        rows = run_screener(filters=filters, sort_by=sort_by,
                            limit=int(limit or 25))
    except Exception as exc:
        return html.P(f"Screener failed: {exc}",
                      style={"color": "#A32D2D", "fontSize": "13px", "padding": "12px"})
    return screener_page.render_results(rows)


@app.callback(
    Output("store-symbol", "data", allow_duplicate=True),
    Output("store-view",   "data", allow_duplicate=True),
    Input({"type": "screener-send", "symbol": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def send_to_backtest(n_clicks_list):
    ctx = callback_context.triggered[0]
    if not ctx["value"]:
        return no_update, no_update
    import json as _json
    try:
        prop = _json.loads(ctx["prop_id"].split(".")[0])
        symbol = prop.get("symbol")
    except Exception:
        return no_update, no_update
    if not symbol:
        return no_update, no_update
    return symbol, "backtest"


# ── Model builder callbacks ──────────────────────────────────────────────────

@app.callback(
    Output("mb-buy-rows", "children"),
    Input("mb-add-buy", "n_clicks"),
    State("mb-buy-rows", "children"),
    prevent_initial_call=True,
)
def add_buy_row(n, current):
    if not n:
        return no_update
    current = list(current or [])
    current.append(builder_page._rule_row("buy", len(current),
                                            "rsi_14", "<", 30))
    return current


@app.callback(
    Output("mb-sell-rows", "children"),
    Input("mb-add-sell", "n_clicks"),
    State("mb-sell-rows", "children"),
    prevent_initial_call=True,
)
def add_sell_row(n, current):
    if not n:
        return no_update
    current = list(current or [])
    current.append(builder_page._rule_row("sell", len(current),
                                            "rsi_14", ">", 70))
    return current


@app.callback(
    Output("mb-save-status", "children"),
    Output("mb-saved-list",  "children"),
    Input("mb-save", "n_clicks"),
    State("mb-id",   "value"),
    State("mb-name", "value"),
    State("mb-desc", "value"),
    State("mb-conf", "value"),
    State({"type": "mb-buy-field",  "index": ALL}, "value"),
    State({"type": "mb-buy-op",     "index": ALL}, "value"),
    State({"type": "mb-buy-value",  "index": ALL}, "value"),
    State({"type": "mb-sell-field", "index": ALL}, "value"),
    State({"type": "mb-sell-op",    "index": ALL}, "value"),
    State({"type": "mb-sell-value", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def save_custom_model(n, mid, mname, mdesc, mconf,
                       bf, bo, bv, sf, so, sv):
    import json, re
    from pathlib import Path

    if not mid:
        return "❌ Model id required.", no_update
    if not re.match(r"^[a-zA-Z0-9_\-]{1,40}$", mid):
        return "❌ id must be 1-40 chars: letters / digits / _ / -.", no_update

    def _rules(fields, ops, values):
        out = []
        for f, o, v in zip(fields or [], ops or [], values or []):
            if f is None or o is None or v in (None, ""):
                continue
            try:
                out.append({"field": f, "op": o, "value": float(v)})
            except (TypeError, ValueError):
                continue
        return out

    buy_rules  = _rules(bf, bo, bv)
    sell_rules = _rules(sf, so, sv)
    if not buy_rules and not sell_rules:
        return "❌ Add at least one buy or sell condition.", no_update

    spec = {
        "id":             mid,
        "name":           mname or mid,
        "description":    mdesc or "",
        "buy_when":       buy_rules,
        "sell_when":      sell_rules,
        "min_confidence": float(mconf or 0.65),
    }

    target_dir = Path(__file__).resolve().parent / "custom_models"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{mid}.json"
    with open(target, "w") as fh:
        json.dump(spec, fh, indent=2)

    return f"✅ Saved {target.name}", builder_page.render_saved_list()


if __name__ == "__main__":
    app.run(debug=True, port=8050)
