"""
dashboard/app.py
----------------
Main entry point. Run with:  python -m dashboard.app
Then open:  http://localhost:8050
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from dashboard.pages import overview, market_overview, backtest as bt_page
from dashboard.components.global_controls import render_topbar

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
    Output("store-symbol", "data"),
    Input("dd-symbol", "value"),
)
def set_symbol(value):
    return value or "AAPL"


@app.callback(
    Output("store-view", "data"),
    Input("btn-overview",  "n_clicks"),
    Input("btn-trades",    "n_clicks"),
    Input("btn-model-tab", "n_clicks"),
    Input("btn-market",    "n_clicks"),
    Input("btn-backtest",  "n_clicks"),
    State("store-view", "data"),
    prevent_initial_call=True,
)
def set_view(ov, tr, md, mk, bt, current):
    ctx = callback_context.triggered[0]["prop_id"]
    mapping = {
        "btn-overview":  "overview",
        "btn-trades":    "trades",
        "btn-model-tab": "model",
        "btn-market":    "market",
        "btn-backtest":  "backtest",
    }
    for key, val in mapping.items():
        if key in ctx:
            return val
    return current


if __name__ == "__main__":
    app.run(debug=True, port=8050)
