"""
dashboard/components/global_controls.py
----------------------------------------
Top navigation bar — account toggle, model/symbol dropdowns, page nav, backtest button.
All controls write to dcc.Store; pages read from store.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc

MODELS = [
    {"label": "model_v1 — Random Forest", "value": "model_v1"},
    {"label": "model_v2 — XGBoost",       "value": "model_v2"},
    {"label": "model_v3 — LSTM",           "value": "model_v3"},
]

SYMBOLS = [
    {"label": "AAPL", "value": "AAPL"},
    {"label": "TSLA", "value": "TSLA"},
    {"label": "MSFT", "value": "MSFT"},
    {"label": "NVDA", "value": "NVDA"},
    {"label": "SPY",  "value": "SPY"},
]

NAV_STYLE = {
    "fontSize": "13px", "padding": "5px 14px",
    "borderRadius": "20px", "border": "1px solid #ddd",
    "background": "white", "cursor": "pointer", "marginRight": "6px",
}

NAV_ACTIVE = {**NAV_STYLE, "background": "#111", "color": "white", "border": "1px solid #111"}

BT_STYLE = {
    "fontSize": "13px", "padding": "5px 16px",
    "borderRadius": "8px", "border": "none",
    "background": "#1D9E75", "color": "white",
    "fontWeight": "500", "cursor": "pointer",
}

TOGGLE_ON  = {"fontSize": "13px", "padding": "5px 14px", "border": "none", "background": "#111", "color": "white", "cursor": "pointer"}
TOGGLE_OFF = {"fontSize": "13px", "padding": "5px 14px", "border": "none", "background": "white", "color": "#555", "cursor": "pointer"}

DD_STYLE = {"fontSize": "13px", "minWidth": "180px"}


def render_topbar():
    return html.Div([
        html.Div([
            html.Span("Trading bot", style={"fontSize": "15px", "fontWeight": "500", "marginRight": "20px"}),

            html.Div([
                html.Span("Account", style={"fontSize": "11px", "color": "#888", "marginRight": "6px"}),
                html.Div([
                    html.Button("Paper", id="btn-paper", style=TOGGLE_ON),
                    html.Button("Live",  id="btn-live",  style=TOGGLE_OFF),
                ], style={"display": "flex", "border": "1px solid #ddd", "borderRadius": "8px", "overflow": "hidden", "marginRight": "16px"}),
            ], style={"display": "flex", "alignItems": "center"}),

            html.Div([
                html.Span("Model", style={"fontSize": "11px", "color": "#888", "marginRight": "6px"}),
                dcc.Dropdown(id="dd-model", options=MODELS, value="model_v1",
                             clearable=False, style=DD_STYLE),
            ], style={"display": "flex", "alignItems": "center", "marginRight": "16px"}),

            html.Div([
                html.Span("Symbol", style={"fontSize": "11px", "color": "#888", "marginRight": "6px"}),
                dcc.Dropdown(id="dd-symbol", options=SYMBOLS, value="AAPL",
                             clearable=False, style={"fontSize": "13px", "minWidth": "100px"}),
            ], style={"display": "flex", "alignItems": "center", "marginRight": "20px"}),

            html.Div([
                html.Span(style={"width": "8px", "height": "8px", "borderRadius": "50%",
                                 "background": "#1D9E75", "display": "inline-block", "marginRight": "5px"}),
                html.Span("Live", style={"fontSize": "12px", "color": "#27500A",
                                         "background": "#EAF3DE", "padding": "2px 10px",
                                         "borderRadius": "12px"}),
            ], style={"display": "flex", "alignItems": "center", "marginRight": "20px"}),

        ], style={"display": "flex", "alignItems": "center", "flex": "1"}),

        html.Div([
            html.Button("Overview",  id="btn-overview",  style=NAV_ACTIVE),
            html.Button("Trades",    id="btn-trades",    style=NAV_STYLE),
            html.Button("Model",     id="btn-model-tab", style=NAV_STYLE),
            html.Button("Market",    id="btn-market",    style=NAV_STYLE),
            html.Button("Backtest",  id="btn-backtest",  style=BT_STYLE),
        ], style={"display": "flex", "alignItems": "center"}),

    ], style={
        "display": "flex", "alignItems": "center", "justifyContent": "space-between",
        "padding": "12px 24px", "background": "white",
        "borderBottom": "1px solid #eee", "marginBottom": "20px",
        "flexWrap": "wrap", "gap": "10px",
    })
