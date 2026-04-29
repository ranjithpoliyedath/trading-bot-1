"""
dashboard/components/global_controls.py
----------------------------------------
Top navigation bar — account toggle, model/symbol dropdowns, page nav, backtest button.
All controls write to dcc.Store; pages read from store.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc

def _load_models():
    """Pull options from the registry — built-in + saved custom models.
    Cross-sectional models are excluded — they need their own runner
    (``dashboard.backtest_engine.run_cross_sectional_backtest``)."""
    try:
        from bot.models.registry import list_models
        opts = []
        for m in list_models():
            if m.type == "cross_sectional":
                continue
            tag = "custom" if m.id.startswith("custom:") else "builtin"
            opts.append({
                "label": f"{m.name}  [{tag}]",
                "value": m.id,
            })
        if opts:
            return opts
    except Exception:
        pass
    return [{"label": "rsi_macd_v1", "value": "rsi_macd_v1"}]


MODELS = _load_models()

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

            # Model + Symbol selectors are only used by the Overview
            # page (signal panel + per-symbol equity chart).  Every
            # other page either has its own selector (Backtest,
            # Screener, Builder, Finder) or doesn't use these inputs
            # at all (Market, Data).  We wrap them in a dedicated
            # container so a callback (in app.py) can hide them on
            # pages where they're dead UI — clears clutter.
            html.Div([
                html.Div([
                    html.Span("Model", style={"fontSize": "11px",
                                                "color": "#888",
                                                "marginRight": "6px"}),
                    dcc.Dropdown(id="dd-model", options=MODELS,
                                  value=(MODELS[0]["value"] if MODELS else None),
                                  clearable=False, style=DD_STYLE),
                ], style={"display": "flex", "alignItems": "center",
                          "marginRight": "16px"}),

                html.Div([
                    html.Span("Symbol", style={"fontSize": "11px",
                                                 "color": "#888",
                                                 "marginRight": "6px"}),
                    dcc.Dropdown(id="dd-symbol", options=SYMBOLS, value="AAPL",
                                  clearable=False,
                                  style={"fontSize": "13px",
                                         "minWidth": "100px"}),
                ], style={"display": "flex", "alignItems": "center",
                          "marginRight": "20px"}),
            ], id="topbar-model-symbol",
                style={"display": "flex", "alignItems": "center"}),

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
            html.Button("Screener",  id="btn-screener",  style=NAV_STYLE),
            html.Button("Builder",   id="btn-builder",   style=NAV_STYLE),
            html.Button("Finder",    id="btn-finder",    style=NAV_STYLE),
            html.Button("Trades",    id="btn-trades",    style=NAV_STYLE),
            html.Button("Model",     id="btn-model-tab", style=NAV_STYLE),
            html.Button("Market",    id="btn-market",    style=NAV_STYLE),
            html.Button("Data",      id="btn-data",      style=NAV_STYLE),
            html.Button("Backtest",  id="btn-backtest",  style=BT_STYLE),
        ], style={"display": "flex", "alignItems": "center"}),

    ], style={
        "display": "flex", "alignItems": "center", "justifyContent": "space-between",
        "padding": "12px 24px", "background": "white",
        "borderBottom": "1px solid #eee", "marginBottom": "20px",
        "flexWrap": "wrap", "gap": "10px",
    })
