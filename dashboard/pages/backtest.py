"""
dashboard/pages/backtest.py
----------------------------
Full backtest view: NL-query box, manual filter rows, controls, run
button, equity curve, monthly returns, metrics row, trade distribution,
risk metrics, saved runs dropdown.
"""
from __future__ import annotations

import plotly.graph_objects as go
from dash import html, dcc, Input, Output, State, callback_context, callback
import dash_bootstrap_components as dbc

from dashboard.backtest_engine import (
    run_backtest, save_backtest, load_backtest, list_saved_backtests
)
from bot.screener import SCREENER_FIELDS

CARD  = {"background": "white", "borderRadius": "12px", "border": "1px solid #eee", "padding": "14px"}
METRIC = {"background": "#F8F8F7", "borderRadius": "8px", "padding": "10px 12px"}

INDICATORS = ["RSI", "MACD", "EMA cross", "Bollinger", "ATR", "Volume ratio", "VWAP"]


def _model_options():
    """Pull live options from the registry (built-in + custom)."""
    try:
        from bot.models.registry import list_models
        return [{"label": f"{m.name} [{m.type}]", "value": m.id}
                for m in list_models()]
    except Exception:
        return [{"label": "rsi_macd_v1", "value": "rsi_macd_v1"}]


MODELS  = _model_options()
SYMBOLS = [{"label": s, "value": s} for s in ["All", "AAPL", "TSLA", "MSFT", "NVDA", "SPY"]]
FIELD_OPTIONS = [
    {"label": f"{meta['group']} — {meta['label']}", "value": key}
    for key, meta in SCREENER_FIELDS.items()
]
FILTER_OPS = [{"label": op, "value": op} for op in [">", ">=", "<", "<=", "==", "!="]]
TIMEFRAMES = [{"label": l, "value": v} for l, v in [("1 day","1d"),("4 hour","4h"),("1 hour","1h"),("15 min","15m")]]
PERIODS    = [{"label": l, "value": v} for l, v in [("365 days",365),("180 days",180),("90 days",90),("30 days",30)]]
CONFS      = [{"label": f"{c}%", "value": c/100} for c in [60, 65, 70, 75, 80]]

DD = {"fontSize": "13px"}


def layout(account: str, model: str, symbol: str):
    saved = list_saved_backtests()
    # Make sure the model picker has a valid default — if the topbar-stored
    # model id isn't in the live registry, fall back to the first option.
    valid_ids = {m["value"] for m in MODELS}
    initial_model = model if model in valid_ids else (MODELS[0]["value"] if MODELS else None)
    return html.Div([
        _section_label("Backtest — natural-language query"),
        _nl_query_panel(),
        _section_label("Backtest — manual filters"),
        _filter_panel(),
        _section_label("Backtest — configure & run"),
        _controls_panel(initial_model, symbol, saved),
        html.Div(id="bt-results-area"),
    ])


def _nl_query_panel():
    return html.Div([
        html.P(
            "Describe the backtest in plain English — Claude will translate "
            "it into a model, filters and period.",
            style={"fontSize": "12px", "color": "#666", "margin": "0 0 8px"},
        ),
        dcc.Textarea(
            id="bt-nl-query",
            placeholder=("e.g. \"backtest the qullamaggie breakout on stocks with "
                         "volume > 2x avg and combined sentiment > 0.2 over the "
                         "last year\""),
            style={"width": "100%", "minHeight": "70px",
                   "fontSize": "13px", "padding": "8px 10px",
                   "border": "1px solid #ddd", "borderRadius": "6px"},
        ),
        html.Div([
            html.Button("Parse with Claude", id="bt-nl-parse", n_clicks=0, style={
                "fontSize": "13px", "padding": "6px 14px",
                "border": "none", "borderRadius": "8px",
                "background": "#3C3489", "color": "white",
                "fontWeight": "500", "cursor": "pointer", "marginRight": "10px",
            }),
            html.Span(id="bt-nl-status", style={"fontSize": "12px", "color": "#666"}),
        ], style={"marginTop": "8px", "display": "flex", "alignItems": "center"}),
    ], style={**CARD, "marginBottom": "16px"})


def _filter_row(idx: int, default_field="rsi_14", default_op="<", default_value=30):
    return html.Div([
        dcc.Dropdown(
            id={"type": "bt-filter-field", "index": idx},
            options=FIELD_OPTIONS, value=default_field, clearable=False,
            style={"flex": "2", "fontSize": "13px"},
        ),
        dcc.Dropdown(
            id={"type": "bt-filter-op", "index": idx},
            options=FILTER_OPS, value=default_op, clearable=False,
            style={"flex": "0 0 80px", "fontSize": "13px", "marginLeft": "8px"},
        ),
        dcc.Input(
            id={"type": "bt-filter-value", "index": idx},
            type="number", value=default_value,
            style={"flex": "0 0 120px", "marginLeft": "8px",
                    "fontSize": "13px", "padding": "6px 10px",
                    "border": "1px solid #ddd", "borderRadius": "6px"},
        ),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"})


def _filter_panel():
    return html.Div([
        html.Div([
            html.Span("Per-bar filters (all must match for a buy)",
                      style={"fontSize": "12px", "color": "#666",
                              "marginRight": "auto"}),
            html.Button("+ Add filter", id="bt-add-filter", n_clicks=0, style={
                "fontSize": "12px", "padding": "5px 12px",
                "border": "1px solid #ddd", "borderRadius": "6px",
                "background": "white", "cursor": "pointer",
            }),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}),
        html.Div(id="bt-filter-rows", children=[]),
    ], style={**CARD, "marginBottom": "16px"})


def _section_label(text):
    return html.P(text.upper(), style={
        "fontSize": "10px", "fontWeight": "500", "color": "#aaa",
        "letterSpacing": "0.07em", "margin": "0 0 10px",
    })


def _controls_panel(model: str, symbol: str, saved: list):
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.P("Saved runs", style={"fontSize": "11px", "color": "#888", "margin": "0 0 4px"}),
                dcc.Dropdown(id="bt-dd-saved", options=saved,
                             placeholder="Select a saved backtest...",
                             style=DD, clearable=True),
            ], md=4),
            dbc.Col([
                html.P("Model", style={"fontSize": "11px", "color": "#888", "margin": "0 0 4px"}),
                dcc.Dropdown(id="bt-dd-model", options=MODELS, value=model, clearable=False, style=DD),
            ], md=3),
            dbc.Col([
                html.P("Symbol", style={"fontSize": "11px", "color": "#888", "margin": "0 0 4px"}),
                dcc.Dropdown(id="bt-dd-symbol", options=SYMBOLS, value=symbol, clearable=False, style=DD),
            ], md=2),
            dbc.Col([
                html.Div([
                    html.Button("Run backtest", id="bt-btn-run", n_clicks=0, style={
                        "fontSize": "13px", "padding": "8px 20px", "borderRadius": "8px",
                        "border": "none", "background": "#1D9E75", "color": "white",
                        "fontWeight": "500", "cursor": "pointer", "marginRight": "8px",
                    }),
                    html.Button("Save run", id="bt-btn-save", n_clicks=0, style={
                        "fontSize": "13px", "padding": "8px 16px", "borderRadius": "8px",
                        "border": "1px solid #ddd", "background": "white",
                        "cursor": "pointer",
                    }),
                ], style={"display": "flex", "alignItems": "flex-end", "height": "100%", "paddingTop": "18px"}),
            ], md=3),
        ], className="g-3 mb-3"),

        dbc.Row([
            dbc.Col([
                html.P("Timeframe", style={"fontSize": "11px", "color": "#888", "margin": "0 0 4px"}),
                dcc.Dropdown(id="bt-dd-tf", options=TIMEFRAMES, value="1d", clearable=False, style=DD),
            ], md=2),
            dbc.Col([
                html.P("Period", style={"fontSize": "11px", "color": "#888", "margin": "0 0 4px"}),
                dcc.Dropdown(id="bt-dd-period", options=PERIODS, value=365, clearable=False, style=DD),
            ], md=2),
            dbc.Col([
                html.P("Confidence threshold", style={"fontSize": "11px", "color": "#888", "margin": "0 0 4px"}),
                dcc.Dropdown(id="bt-dd-conf", options=CONFS, value=0.65, clearable=False, style=DD),
            ], md=2),
            dbc.Col([
                html.P("Indicators", style={"fontSize": "11px", "color": "#888", "margin": "0 0 4px"}),
                dcc.Checklist(
                    id="bt-indicators",
                    options=[{"label": html.Span(i, style={"fontSize": "12px", "marginRight": "6px"}), "value": i}
                             for i in INDICATORS],
                    value=["RSI", "MACD", "EMA cross", "ATR", "VWAP"],
                    inline=True,
                    style={"display": "flex", "flexWrap": "wrap", "gap": "6px"},
                ),
            ], md=6),
        ], className="g-3"),

        dcc.Store(id="bt-store-results"),
        html.Div(id="bt-save-msg", style={"fontSize": "12px", "color": "#27500A", "marginTop": "8px"}),
    ], style={**CARD, "marginBottom": "16px"})


def render_results(results: dict):
    if not results or not results.get("metrics"):
        return html.P("Run a backtest to see results.", style={"color": "#aaa", "fontSize": "13px"})

    m  = results["metrics"]
    ec = results.get("equity_curve", [])
    mr = results.get("monthly_returns", [])

    return html.Div([
        _section_label(f"Results — {results.get('run_id', '')}"),
        _metrics_row(m),
        dbc.Row([
            dbc.Col(_equity_chart(ec),   md=7),
            dbc.Col(_monthly_chart(mr),  md=5),
        ], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(_trade_dist(m),  md=4),
            dbc.Col(_risk_card(m),   md=4),
            dbc.Col(_signal_quality(m), md=4),
        ], className="g-3"),
    ])


def _metrics_row(m: dict):
    items = [
        ("Total return",  f"{'+' if m['total_return_pct'] >= 0 else ''}{m['total_return_pct']:.1f}%",  m['total_return_pct'] >= 0),
        ("Sharpe ratio",  f"{m['sharpe']:.2f}",  m['sharpe'] >= 1),
        ("Expectancy",    f"${m['expectancy']:.2f}", m['expectancy'] >= 0),
        ("Edge ratio",    f"{m['edge_ratio']:.2f}x", m['edge_ratio'] >= 1),
        ("Max drawdown",  f"{m['max_drawdown_pct']:.1f}%", False),
        ("Win rate",      f"{m['win_rate_pct']:.1f}%", m['win_rate_pct'] >= 50),
    ]
    return dbc.Row([
        dbc.Col(html.Div([
            html.P(label, style={"fontSize": "11px", "color": "#888", "margin": "0 0 3px"}),
            html.P(value, style={"fontSize": "18px", "fontWeight": "500", "margin": "0",
                                  "color": ("#27500A" if positive else "#A32D2D") if label != "Sharpe ratio" and label != "Edge ratio" else "#111"}),
        ], style=METRIC), md=2)
        for label, value, positive in items
    ], className="g-3 mb-3")


def _equity_chart(equity_curve: list):
    if not equity_curve:
        return html.Div("No equity data", style={**CARD, "color": "#aaa", "fontSize": "12px"})

    dates  = [e["date"]  for e in equity_curve]
    values = [e["value"] for e in equity_curve]
    up     = values[-1] >= values[0] if values else True
    color  = "#1D9E75" if up else "#E24B4A"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=values, mode="lines",
        line=dict(color=color, width=1.5),
        fill="tozeroy",
        fillcolor=f"rgba(29,158,117,0.08)" if up else "rgba(226,75,74,0.08)",
    ))
    fig.update_layout(
        margin=dict(l=0,r=0,t=28,b=0), height=200,
        paper_bgcolor="white", plot_bgcolor="white",
        title=dict(text="Equity curve", font=dict(size=12)),
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=10), tickprefix="$"),
        showlegend=False,
    )
    return html.Div([dcc.Graph(figure=fig, config={"displayModeBar": False})], style=CARD)


def _monthly_chart(monthly: list):
    if not monthly:
        return html.Div("No monthly data", style={**CARD, "color": "#aaa", "fontSize": "12px"})

    months  = [m["month"]  for m in monthly]
    returns = [m["return"] for m in monthly]
    colors  = ["#1D9E75" if r >= 0 else "#E24B4A" for r in returns]

    fig = go.Figure(go.Bar(x=months, y=returns, marker_color=colors, name="Monthly return"))
    fig.update_layout(
        margin=dict(l=0,r=0,t=28,b=0), height=200,
        paper_bgcolor="white", plot_bgcolor="white",
        title=dict(text="Monthly returns (%)", font=dict(size=12)),
        xaxis=dict(showgrid=False, tickfont=dict(size=9)),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=10), ticksuffix="%"),
        showlegend=False,
    )
    return html.Div([dcc.Graph(figure=fig, config={"displayModeBar": False})], style=CARD)


def _stat_row(label, value, color="#111"):
    return html.Div([
        html.Span(label, style={"fontSize": "12px", "color": "#888", "flex": "1"}),
        html.Span(value, style={"fontSize": "12px", "fontWeight": "500", "color": color}),
    ], style={"display": "flex", "padding": "5px 0", "borderBottom": "1px solid #f5f5f5"})


def _trade_dist(m: dict):
    return html.Div([
        html.P("Trade distribution", style={"fontSize": "12px", "fontWeight": "500", "margin": "0 0 10px"}),
        _stat_row("Total trades",  str(m["total_trades"])),
        _stat_row("Wins",          str(int(m["total_trades"] * m["win_rate_pct"] / 100)), "#27500A"),
        _stat_row("Losses",        str(int(m["total_trades"] * (1 - m["win_rate_pct"] / 100))), "#A32D2D"),
        _stat_row("Profit factor", f"{m['profit_factor']:.2f}"),
    ], style=CARD)


def _risk_card(m: dict):
    return html.Div([
        html.P("Risk metrics", style={"fontSize": "12px", "fontWeight": "500", "margin": "0 0 10px"}),
        _stat_row("Avg win",      f"${m['avg_win']:,.2f}",   "#27500A"),
        _stat_row("Avg loss",     f"${m['avg_loss']:,.2f}",  "#A32D2D"),
        _stat_row("Largest win",  f"${m['largest_win']:,.2f}","#27500A"),
        _stat_row("Largest loss", f"${m['largest_loss']:,.2f}","#A32D2D"),
        _stat_row("Sortino",      f"{m['sortino']:.2f}"),
    ], style=CARD)


def _signal_quality(m: dict):
    return html.Div([
        html.P("Signal quality", style={"fontSize": "12px", "fontWeight": "500", "margin": "0 0 10px"}),
        _stat_row("Win rate",       f"{m['win_rate_pct']:.1f}%", "#27500A" if m["win_rate_pct"] >= 50 else "#A32D2D"),
        _stat_row("Expectancy",     f"${m['expectancy']:.2f}",   "#27500A" if m["expectancy"] >= 0 else "#A32D2D"),
        _stat_row("Edge ratio",     f"{m['edge_ratio']:.2f}x"),
        _stat_row("Max drawdown",   f"{m['max_drawdown_pct']:.1f}%", "#A32D2D"),
    ], style=CARD)
