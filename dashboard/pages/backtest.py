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
from bot.screener  import SCREENER_FIELDS
from bot.universe  import UNIVERSE_SCOPES

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
SCOPE_OPTIONS = [{"label": label, "value": key}
                 for key, label in UNIVERSE_SCOPES.items()]
FIELD_OPTIONS = [
    {"label": f"{meta['group']} — {meta['label']}", "value": key}
    for key, meta in SCREENER_FIELDS.items()
]
FILTER_OPS = [{"label": op, "value": op} for op in [">", ">=", "<", "<=", "==", "!="]]
TIMEFRAMES = [{"label": l, "value": v} for l, v in [("1 day","1d"),("4 hour","4h"),("1 hour","1h"),("15 min","15m")]]

DD     = {"fontSize": "13px"}
NUMBOX = {
    "fontSize":     "13px",
    "padding":      "6px 10px",
    "border":       "1px solid #ddd",
    "borderRadius": "6px",
    "width":        "100%",
}


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
                dcc.Dropdown(id="bt-dd-model", options=MODELS, value=model,
                             clearable=False, style=DD),
            ], md=4),
            dbc.Col([
                html.P("Universe scope",
                       title="Pick which slice of the universe to scan.",
                       style={"fontSize": "11px", "color": "#888", "margin": "0 0 4px"}),
                dcc.Dropdown(id="bt-dd-scope", options=SCOPE_OPTIONS,
                             value="top_100", clearable=False, style=DD),
            ], md=4),
            dbc.Col([
                html.P("Symbol limit",
                       title="Cap how many symbols from the scope are scanned.",
                       style={"fontSize": "11px", "color": "#888", "margin": "0 0 4px"}),
                dcc.Input(id="bt-input-max", type="number", value=50,
                          min=1, max=2000, step=1, style=NUMBOX),
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
            ], md=2),
        ], className="g-3 mb-3"),

        dbc.Row([
            dbc.Col([
                html.P("Timeframe",
                       title="Bar size used for the backtest.  Daily only for now.",
                       style={"fontSize": "11px", "color": "#888", "margin": "0 0 4px"}),
                dcc.Dropdown(id="bt-dd-tf", options=TIMEFRAMES, value="1d",
                             clearable=False, style=DD),
            ], md=2),
            dbc.Col([
                html.P("Period (days)",
                       title="Number of days of history to backtest over.",
                       style={"fontSize": "11px", "color": "#888", "margin": "0 0 4px"}),
                dcc.Input(id="bt-input-period", type="number", value=365,
                          min=30, max=2000, step=1, style=NUMBOX),
            ], md=2),
            dbc.Col([
                html.P("Confidence threshold",
                       title="Minimum model confidence (0-1) to act on a signal.",
                       style={"fontSize": "11px", "color": "#888", "margin": "0 0 4px"}),
                dcc.Input(id="bt-input-conf", type="number", value=0.65,
                          min=0, max=1, step=0.01, style=NUMBOX),
            ], md=2),
            dbc.Col([
                html.P("Indicators",
                       title="Subset of feature columns to keep.  Affects only ML models.",
                       style={"fontSize": "11px", "color": "#888", "margin": "0 0 4px"}),
                dcc.Checklist(
                    id="bt-indicators",
                    options=[{"label": html.Span(i, style={"fontSize": "12px", "marginRight": "6px"}), "value": i}
                             for i in INDICATORS],
                    value=["RSI", "MACD", "EMA cross", "ATR", "VWAP"],
                    inline=True,
                    style={"display": "flex", "flexWrap": "wrap", "gap": "6px"},
                ),
            ], md=6),
        ], className="g-3 mb-3"),

        # Hidden — kept for backwards compatibility with existing callbacks.
        # Symbol selection is now driven by the Universe scope dropdown.
        dcc.Store(id="bt-dd-symbol", data="All"),

        # ── Exit conditions ─────────────────────────────────────────
        _exit_conditions_panel(),

        dcc.Store(id="bt-store-results"),
        html.Div(id="bt-save-msg", style={"fontSize": "12px", "color": "#27500A", "marginTop": "8px"}),
    ], style={**CARD, "marginBottom": "16px"})


def _exit_conditions_panel():
    """The four exit-rule toggles + numeric values."""
    row = lambda label, tip, toggle_id, val_id, default_val, suffix, **inp_props: html.Div([
        dcc.Checklist(
            id=toggle_id,
            options=[{"label": html.Span(label, style={
                "fontSize": "12px", "fontWeight": "500", "marginLeft": "4px",
            }), "value": "on"}],
            value=["on"],
            style={"flex": "0 0 220px"},
        ),
        dcc.Input(
            id=val_id, type="number", value=default_val,
            style={**NUMBOX, "flex": "0 0 110px", "marginLeft": "8px"},
            **inp_props,
        ),
        html.Span(suffix, style={"fontSize": "12px", "color": "#888",
                                  "marginLeft": "8px"}),
        html.Span(tip, style={"fontSize": "11px", "color": "#aaa",
                                "marginLeft": "16px", "flex": "1"}),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"})

    return html.Div([
        html.P("Exit conditions",
               title="Whichever rule fires first closes the trade.  Disable any rule to ignore it.",
               style={"fontSize": "12px", "fontWeight": "500", "margin": "10px 0 8px"}),
        row("Model sell signal", "Close when the model emits its own sell.",
            "bt-exit-signal-on", "bt-exit-signal-val", 0,
            "(no value)", min=0, max=0, step=1, disabled=True),
        row("Take-profit", "Close when price rises this much above entry.",
            "bt-exit-tp-on", "bt-exit-tp-val", 0.15, "× entry (e.g. 0.15 = +15%)",
            min=0.005, max=2, step=0.01),
        row("Stop-loss", "Close when price drops this much below entry.",
            "bt-exit-sl-on", "bt-exit-sl-val", 0.07, "× entry (e.g. 0.07 = -7%)",
            min=0.005, max=1, step=0.01),
        row("Time stop", "Close after holding this many trading days.",
            "bt-exit-ts-on", "bt-exit-ts-val", 30, "days",
            min=1, max=500, step=1),
    ], style={**CARD, "marginTop": "12px"})


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


# Hover tooltips — same exact text the html `title` attribute renders.
METRIC_TIPS = {
    "Total return":   "Sum of all closed-trade P&L expressed as % of starting capital.",
    "Sharpe ratio":   "Annualised excess return per unit of total volatility (daily equity returns × √252). >1 is decent, >2 is exceptional, <0.5 is noise.",
    "Sortino":        "Like Sharpe but only penalises downside volatility — better for asymmetric strategies.",
    "Expectancy":     "Average dollar P&L per trade: win_rate × avg_win + loss_rate × avg_loss. Positive = profitable on average.",
    "Edge ratio":     "Average win / average loss (in absolute $). >1 means winners are bigger than losers; combined with win rate gives expectancy.",
    "Profit factor":  "Sum of winning P&L / sum of losing P&L (absolute). >1 is profitable, >1.5 is healthy.",
    "Max drawdown":   "Largest peak-to-trough drop in equity, in %. Smaller (closer to 0%) is better.",
    "Win rate":       "% of trades with positive P&L. Strategies with low win rate can still be profitable if profit factor > 1.",
    "Loss rate":      "% of trades with negative P&L. Equal to 100% − win rate.",
    "Total trades":   "Total number of completed buy-and-sell pairs in this backtest.",
    "Wins":           "Count of trades with positive P&L.",
    "Losses":         "Count of trades with negative P&L.",
    "Avg win %":      "Average winning trade return as % of starting position size ($10k notional).",
    "Avg loss %":     "Average losing trade return as % of starting position size (negative).",
    "Avg win":        "Average dollar P&L of a winning trade.",
    "Avg loss":       "Average dollar P&L of a losing trade (negative).",
    "Largest win":    "Single largest winning trade in dollars.",
    "Largest loss":   "Single largest losing trade in dollars.",
}


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
        dbc.Col(html.Div(
            [
                html.P(label, title=METRIC_TIPS.get(label, ""),
                       style={"fontSize": "11px", "color": "#888", "margin": "0 0 3px",
                                "cursor": "help"}),
                html.P(value, style={"fontSize": "18px", "fontWeight": "500", "margin": "0",
                                      "color": ("#27500A" if positive else "#A32D2D")
                                                if label not in ("Sharpe ratio", "Edge ratio") else "#111"}),
            ], style=METRIC, title=METRIC_TIPS.get(label, "")), md=2)
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
        html.Span(label, title=METRIC_TIPS.get(label, ""),
                  style={"fontSize": "12px", "color": "#888", "flex": "1",
                          "cursor": "help"}),
        html.Span(value, style={"fontSize": "12px", "fontWeight": "500", "color": color}),
    ], style={"display": "flex", "padding": "5px 0", "borderBottom": "1px solid #f5f5f5"})


def _trade_dist(m: dict):
    wins   = m.get("wins",   int(m["total_trades"] * m["win_rate_pct"]      / 100))
    losses = m.get("losses", int(m["total_trades"] * (1 - m["win_rate_pct"] / 100)))
    return html.Div([
        html.P("Trade distribution", style={"fontSize": "12px", "fontWeight": "500", "margin": "0 0 10px"}),
        _stat_row("Total trades",  str(m["total_trades"])),
        _stat_row("Wins",          str(wins),   "#27500A"),
        _stat_row("Losses",        str(losses), "#A32D2D"),
        _stat_row("Win rate",      f"{m['win_rate_pct']:.1f}%",
                                    "#27500A" if m["win_rate_pct"] >= 50 else "#A32D2D"),
        _stat_row("Loss rate",     f"{m.get('loss_rate_pct', 100 - m['win_rate_pct']):.1f}%"),
        _stat_row("Profit factor", f"{m['profit_factor']:.2f}"),
    ], style=CARD)


def _risk_card(m: dict):
    return html.Div([
        html.P("Risk metrics", style={"fontSize": "12px", "fontWeight": "500", "margin": "0 0 10px"}),
        _stat_row("Avg win",      f"${m['avg_win']:,.2f}",                 "#27500A"),
        _stat_row("Avg loss",     f"${m['avg_loss']:,.2f}",                "#A32D2D"),
        _stat_row("Avg win %",    f"{m.get('avg_win_pct', 0):+.2f}%",      "#27500A"),
        _stat_row("Avg loss %",   f"{m.get('avg_loss_pct', 0):+.2f}%",     "#A32D2D"),
        _stat_row("Largest win",  f"${m['largest_win']:,.2f}",             "#27500A"),
        _stat_row("Largest loss", f"${m['largest_loss']:,.2f}",            "#A32D2D"),
        _stat_row("Sortino",      f"{m['sortino']:.2f}"),
        _stat_row("Max drawdown", f"{m['max_drawdown_pct']:.2f}%",         "#A32D2D"),
    ], style=CARD)


def _signal_quality(m: dict):
    return html.Div([
        html.P("Signal quality", style={"fontSize": "12px", "fontWeight": "500", "margin": "0 0 10px"}),
        _stat_row("Win rate",       f"{m['win_rate_pct']:.1f}%", "#27500A" if m["win_rate_pct"] >= 50 else "#A32D2D"),
        _stat_row("Expectancy",     f"${m['expectancy']:.2f}",   "#27500A" if m["expectancy"] >= 0 else "#A32D2D"),
        _stat_row("Edge ratio",     f"{m['edge_ratio']:.2f}x"),
        _stat_row("Max drawdown",   f"{m['max_drawdown_pct']:.1f}%", "#A32D2D"),
    ], style=CARD)
