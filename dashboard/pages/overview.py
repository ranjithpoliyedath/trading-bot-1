"""
dashboard/pages/overview.py
----------------------------
Main overview page: portfolio summary, live signals, recent trades, model info.
All data driven by account + model + symbol from global store.
"""

import plotly.graph_objects as go
from dash import html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc

from dashboard.alpaca_client import get_account_summary, get_positions, get_recent_orders, get_bars
from dashboard.components.signal_panel import render_signals
from dashboard.components.model_summary import render_model_summary


CARD = {
    "background": "white", "borderRadius": "12px",
    "border": "1px solid #eee", "padding": "14px",
}

METRIC = {
    "background": "#F8F8F7", "borderRadius": "8px", "padding": "12px",
}


def layout(account: str, model: str, symbol: str):
    acct      = get_account_summary(account)
    positions = get_positions(account)
    orders    = get_recent_orders(account, limit=8)
    bars      = get_bars(symbol, days=60)

    return html.Div([
        _section_label("Portfolio & P&L summary"),
        _portfolio_metrics(acct, account),

        dbc.Row([
            dbc.Col(_equity_chart(bars, symbol), md=7),
            dbc.Col(_positions_panel(positions), md=5),
        ], className="g-3 mb-3"),

        _section_label("Trade history & signals"),
        dbc.Row([
            dbc.Col(render_signals(model, symbol), md=5),
            dbc.Col(_orders_panel(orders), md=7),
        ], className="g-3 mb-3"),

        _section_label("Model summary"),
        render_model_summary(model),

        dcc.Interval(id="interval-refresh", interval=30_000, n_intervals=0),
    ])


def _section_label(text: str):
    return html.P(text.upper(), style={
        "fontSize": "10px", "fontWeight": "500", "color": "#aaa",
        "letterSpacing": "0.07em", "margin": "0 0 8px",
    })


def _portfolio_metrics(acct: dict, account: str):
    badge_color = "#E6F1FB" if account == "paper" else "#EAF3DE"
    badge_text  = "#0C447C" if account == "paper" else "#27500A"
    badge_label = "Paper account" if account == "paper" else "Live account"

    pv  = acct["portfolio_value"]
    pl  = acct["daily_pl"]
    plp = acct["daily_pl_pct"]
    pl_color  = "#27500A" if pl >= 0 else "#A32D2D"
    pl_prefix = "+" if pl >= 0 else ""

    return dbc.Row([
        dbc.Col(_metric_card("Portfolio value",
            f"${pv:,.2f}",
            html.Span(badge_label, style={"fontSize": "11px", "padding": "2px 8px",
                      "borderRadius": "10px", "background": badge_color, "color": badge_text})), md=3),
        dbc.Col(_metric_card("Today's P&L",
            f"{pl_prefix}${pl:,.2f}",
            f"{pl_prefix}{plp:.2f}%", val_color=pl_color), md=3),
        dbc.Col(_metric_card("Cash available",
            f"${acct['cash']:,.2f}", "Buying power"), md=3),
        dbc.Col(_metric_card("Open positions",
            str(len(get_positions("paper"))), "Across all symbols"), md=3),
    ], className="g-3 mb-3")


def _metric_card(label, value, sub=None, val_color="#111"):
    return html.Div([
        html.P(label, style={"fontSize": "11px", "color": "#888", "margin": "0 0 4px"}),
        html.P(value, style={"fontSize": "22px", "fontWeight": "500", "color": val_color, "margin": "0"}),
        html.P(sub,   style={"fontSize": "11px", "color": "#aaa", "margin": "4px 0 0"}) if sub else None,
    ], style=METRIC)


def _equity_chart(bars: list, symbol: str):
    if not bars:
        fig = go.Figure()
        fig.update_layout(title="No data", height=200)
    else:
        dates  = [b["date"]  for b in bars]
        closes = [b["close"] for b in bars]
        color  = "#1D9E75" if closes[-1] >= closes[0] else "#E24B4A"
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=closes, mode="lines",
            line=dict(color=color, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(29,158,117,0.08)" if color == "#1D9E75" else "rgba(226,75,74,0.08)",
            name=symbol,
        ))
        fig.update_layout(
            margin=dict(l=0, r=0, t=28, b=0), height=180,
            paper_bgcolor="white", plot_bgcolor="white",
            title=dict(text=f"{symbol} — 60 day price", font=dict(size=12)),
            xaxis=dict(showgrid=False, showticklabels=True, tickfont=dict(size=10)),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=10)),
            showlegend=False,
        )

    return html.Div([dcc.Graph(figure=fig, config={"displayModeBar": False})], style=CARD)


def _positions_panel(positions: list):
    if not positions:
        rows = [html.P("No open positions", style={"fontSize": "12px", "color": "#aaa", "margin": "0"})]
    else:
        rows = []
        for p in positions[:6]:
            pl_color = "#27500A" if p["unrealized_pl"] >= 0 else "#A32D2D"
            prefix   = "+" if p["unrealized_pl"] >= 0 else ""
            rows.append(html.Div([
                html.Span(p["symbol"], style={"fontWeight": "500", "fontSize": "13px", "width": "50px", "display": "inline-block"}),
                html.Span(p["side"].upper(), style={
                    "fontSize": "11px", "padding": "2px 8px", "borderRadius": "10px",
                    "background": "#EAF3DE" if p["side"] == "long" else "#FCEBEB",
                    "color": "#27500A" if p["side"] == "long" else "#A32D2D",
                    "marginRight": "8px",
                }),
                html.Span(f"{int(p['qty'])} shares", style={"fontSize": "12px", "color": "#888", "marginRight": "8px"}),
                html.Span(f"{prefix}${p['unrealized_pl']:,.2f}", style={"fontSize": "12px", "fontWeight": "500", "color": pl_color}),
            ], style={"display": "flex", "alignItems": "center", "padding": "6px 0",
                      "borderBottom": "1px solid #f5f5f5"}))

    return html.Div([
        html.P("Open positions", style={"fontSize": "12px", "fontWeight": "500", "margin": "0 0 10px"}),
        *rows,
    ], style=CARD)


def _orders_panel(orders: list):
    if not orders:
        rows = [html.P("No recent trades", style={"fontSize": "12px", "color": "#aaa"})]
    else:
        rows = []
        for o in orders:
            side_bg    = "#EAF3DE" if o["side"] == "buy" else "#FCEBEB"
            side_color = "#27500A" if o["side"] == "buy" else "#A32D2D"
            rows.append(html.Div([
                html.Span(o["symbol"],    style={"fontWeight": "500", "fontSize": "13px", "width": "50px", "display": "inline-block"}),
                html.Span(o["side"].upper(), style={
                    "fontSize": "11px", "padding": "2px 8px", "borderRadius": "10px",
                    "background": side_bg, "color": side_color, "marginRight": "8px",
                }),
                html.Span(o["filled_at"], style={"fontSize": "11px", "color": "#aaa", "marginRight": "8px", "width": "40px"}),
                html.Span(f"@ ${o['filled_avg_price']:,.2f}", style={"fontSize": "12px", "color": "#888"}),
            ], style={"display": "flex", "alignItems": "center", "padding": "6px 0",
                      "borderBottom": "1px solid #f5f5f5"}))

    return html.Div([
        html.P("Recent trades", style={"fontSize": "12px", "fontWeight": "500", "margin": "0 0 10px"}),
        *rows,
    ], style=CARD)
