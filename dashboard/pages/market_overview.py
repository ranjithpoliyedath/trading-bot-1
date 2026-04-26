"""
dashboard/pages/market_overview.py
------------------------------------
The landing page of the dashboard. Shows the six market panels:
  1. Fear & Greed gauge
  2. Index snapshot (SPY, QQQ, DIA, IWM, VTI)
  3. Sector leaders (top 3 by daily change)
  4. Volume movers (abnormal volume today)
  5. Sentiment heatmap (universe colored by sentiment)
  6. News headlines (recent 10)
"""

import logging

import plotly.graph_objects as go
from dash import html, dcc
import dash_bootstrap_components as dbc

from bot.market_overview import get_market_overview

logger = logging.getLogger(__name__)


CARD = {
    "background":   "white",
    "borderRadius": "12px",
    "border":       "1px solid #eee",
    "padding":      "14px",
    "height":       "100%",
}

SECTION_LABEL = {
    "fontSize":      "10px",
    "fontWeight":    "500",
    "color":         "#aaa",
    "letterSpacing": "0.07em",
    "margin":        "0 0 10px",
}


def layout(account: str = "paper", model: str = "model_v1", symbol: str = "AAPL"):
    """Render the full overview page."""
    try:
        data = get_market_overview()
    except Exception as exc:
        logger.error("Overview data load failed: %s", exc)
        data = {}

    return html.Div([
        html.P("MARKET OVERVIEW", style=SECTION_LABEL),
        dbc.Row([
            dbc.Col(_fear_greed_panel(data.get("fear_greed", {})), md=4),
            dbc.Col(_index_snapshot_panel(data.get("indexes", [])), md=4),
            dbc.Col(_sector_leaders_panel(data.get("sectors", [])), md=4),
        ], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(_volume_movers_panel(data.get("volume_movers", [])),  md=6),
            dbc.Col(_sentiment_heatmap_panel(data.get("sentiment_heatmap", [])), md=6),
        ], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(_news_panel(data.get("news", [])), md=12),
        ], className="g-3"),

        dcc.Interval(id="interval-overview-refresh", interval=300_000, n_intervals=0),
    ])


# ── Panel 1: Fear & Greed ─────────────────────────────────────────────────────

def _fear_greed_panel(fg: dict):
    score = fg.get("score", 50)
    label = fg.get("label", "Neutral")

    color  = _fg_color(score)

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = score,
        number = {"font": {"size": 32}},
        gauge = {
            "axis":  {"range": [0, 100], "tickwidth": 1, "tickcolor": "#ddd"},
            "bar":   {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 25],   "color": "#FCEBEB"},
                {"range": [25, 45],  "color": "#FFF3E6"},
                {"range": [45, 55],  "color": "#F1EFE8"},
                {"range": [55, 75],  "color": "#E1F5EE"},
                {"range": [75, 100], "color": "#EAF3DE"},
            ],
        },
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10), height=170,
        paper_bgcolor="white", plot_bgcolor="white",
    )

    yesterday = fg.get("yesterday", score)
    week_ago  = fg.get("week_ago",  score)

    return html.Div([
        html.P("Market mood", style={"fontSize": "12px", "fontWeight": "500", "margin": "0 0 4px"}),
        html.P(label, style={"fontSize": "16px", "fontWeight": "500", "color": color, "margin": "0 0 4px"}),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        html.Div([
            html.Span(f"Yesterday: {yesterday:.0f}", style={"fontSize": "11px", "color": "#888", "marginRight": "12px"}),
            html.Span(f"Week ago: {week_ago:.0f}", style={"fontSize": "11px", "color": "#888"}),
        ], style={"textAlign": "center", "marginTop": "-8px"}),
    ], style=CARD)


def _fg_color(score: float) -> str:
    if score < 25:  return "#A32D2D"
    if score < 45:  return "#D97706"
    if score < 55:  return "#737373"
    if score < 75:  return "#16A34A"
    return "#27500A"


# ── Panel 2: Index snapshot ───────────────────────────────────────────────────

def _index_snapshot_panel(indexes: list[dict]):
    rows = []
    for ix in indexes:
        chg    = ix["change_pct"]
        color  = "#27500A" if chg >= 0 else "#A32D2D"
        prefix = "+" if chg >= 0 else ""
        rows.append(html.Div([
            html.Span(ix["symbol"],     style={"fontWeight": "500", "fontSize": "13px", "width": "50px"}),
            html.Span(f"${ix['close']:,.2f}", style={"fontSize": "13px", "color": "#444", "flex": "1"}),
            html.Span(f"{prefix}{chg:.2f}%", style={"fontSize": "13px", "fontWeight": "500", "color": color}),
        ], style={"display": "flex", "padding": "7px 0", "borderBottom": "1px solid #f5f5f5"}))

    if not rows:
        rows = [html.P("No index data — run pipeline.", style={"color": "#aaa", "fontSize": "12px"})]

    return html.Div([
        html.P("Index snapshot", style={"fontSize": "12px", "fontWeight": "500", "margin": "0 0 10px"}),
        *rows,
    ], style=CARD)


# ── Panel 3: Sector leaders ───────────────────────────────────────────────────

def _sector_leaders_panel(sectors: list[dict]):
    rows = []
    for s in sectors:
        chg    = s["change_pct"]
        color  = "#27500A" if chg >= 0 else "#A32D2D"
        prefix = "+" if chg >= 0 else ""
        rows.append(html.Div([
            html.Div([
                html.Span(s["sector"], style={"fontSize": "12px", "fontWeight": "500"}),
                html.Span(s["symbol"], style={"fontSize": "10px", "color": "#888", "marginLeft": "6px"}),
            ], style={"flex": "1"}),
            html.Span(f"{prefix}{chg:.2f}%", style={"fontSize": "13px", "fontWeight": "500", "color": color}),
        ], style={"display": "flex", "padding": "8px 0", "borderBottom": "1px solid #f5f5f5"}))

    if not rows:
        rows = [html.P("No sector data — fetch sector ETFs.", style={"color": "#aaa", "fontSize": "12px"})]

    return html.Div([
        html.P("Top sectors today", style={"fontSize": "12px", "fontWeight": "500", "margin": "0 0 10px"}),
        *rows,
        html.P("Need: XLK, XLV, XLF, XLE, XLY, XLP, XLI, XLU, XLB, XLRE, XLC",
               style={"fontSize": "10px", "color": "#aaa", "marginTop": "10px"}) if not rows else None,
    ], style=CARD)


# ── Panel 4: Volume movers ────────────────────────────────────────────────────

def _volume_movers_panel(movers: list[dict]):
    rows = []
    for m in movers[:8]:
        chg    = m["change_pct"]
        color  = "#27500A" if chg >= 0 else "#A32D2D"
        prefix = "+" if chg >= 0 else ""
        rows.append(html.Div([
            html.Span(m["symbol"], style={"fontWeight": "500", "fontSize": "13px", "width": "70px"}),
            html.Span(f"${m['close']:.2f}", style={"fontSize": "12px", "color": "#666", "width": "70px"}),
            html.Span(f"{m['volume_ratio']:.1f}x vol", style={"fontSize": "12px", "color": "#3C3489", "background": "#EEEDFE", "padding": "1px 8px", "borderRadius": "10px", "marginRight": "auto"}),
            html.Span(f"{prefix}{chg:.2f}%", style={"fontSize": "13px", "fontWeight": "500", "color": color}),
        ], style={"display": "flex", "alignItems": "center", "padding": "6px 0", "borderBottom": "1px solid #f5f5f5"}))

    if not rows:
        rows = [html.P("No abnormal volume detected today.", style={"color": "#aaa", "fontSize": "12px"})]

    return html.Div([
        html.P("Volume movers (>2× avg)", style={"fontSize": "12px", "fontWeight": "500", "margin": "0 0 10px"}),
        *rows,
    ], style=CARD)


# ── Panel 5: Sentiment heatmap ────────────────────────────────────────────────

def _sentiment_heatmap_panel(items: list[dict]):
    """Render a grid of symbol tiles colored by sentiment."""
    if not items:
        return html.Div([
            html.P("Sentiment heatmap", style={"fontSize": "12px", "fontWeight": "500", "margin": "0 0 10px"}),
            html.P("No sentiment data — run sentiment pipeline.",
                   style={"color": "#aaa", "fontSize": "12px"}),
        ], style=CARD)

    tiles = []
    for it in items:
        s     = it["sentiment"]
        color = _sentiment_color(s)
        tiles.append(html.Div([
            html.Div(it["symbol"], style={"fontSize": "11px", "fontWeight": "500", "color": "white"}),
            html.Div(f"{s:+.2f}", style={"fontSize": "10px", "color": "rgba(255,255,255,0.85)"}),
        ], style={
            "background":   color,
            "borderRadius": "6px",
            "padding":      "8px 4px",
            "textAlign":    "center",
            "minWidth":     "55px",
            "flex":         "0 0 calc(20% - 4px)",
        }))

    return html.Div([
        html.P("Sentiment heatmap", style={"fontSize": "12px", "fontWeight": "500", "margin": "0 0 10px"}),
        html.Div(tiles, style={
            "display":  "flex",
            "flexWrap": "wrap",
            "gap":      "5px",
        }),
    ], style=CARD)


def _sentiment_color(s: float) -> str:
    if s >  0.3:  return "#16A34A"
    if s >  0.1:  return "#84CC16"
    if s > -0.1:  return "#888888"
    if s > -0.3:  return "#EA580C"
    return "#A32D2D"


# ── Panel 6: News headlines ───────────────────────────────────────────────────

def _news_panel(news: list[dict]):
    if not news:
        body = html.P("No high-confidence news right now.",
                      style={"color": "#aaa", "fontSize": "12px"})
    else:
        rows = []
        for n in news:
            ts        = str(n.get("published_at", ""))[:16].replace("T", " ")
            stars     = int(n.get("stars", 1))
            direction = n.get("direction", "neutral")
            flags     = n.get("flags", []) or []
            insight   = n.get("insight", "")

            arrow_color = {"bullish": "#16A34A", "bearish": "#A32D2D"}.get(direction, "#888")
            arrow       = {"bullish": "▲", "bearish": "▼"}.get(direction, "•")

            star_str = "★" * stars + "☆" * (5 - stars)

            badge_text = []
            badge_color = "#F1EFE8"
            badge_fg    = "#444"
            if "strong-divergence" in flags:
                badge_text.append("DIVERGES")
                badge_color = "#FFE7E0"; badge_fg = "#A32D2D"
            elif "vs-sector" in flags:
                badge_text.append("vs sector")
                badge_color = "#FFF3E6"; badge_fg = "#92400E"
            elif "vs-market" in flags:
                badge_text.append("vs market")
                badge_color = "#FFF3E6"; badge_fg = "#92400E"

            kw_flags = [f.replace("kw:", "") for f in flags if f.startswith("kw:")][:2]

            rows.append(html.Div([
                html.Div([
                    html.Span(n["symbol"], style={
                        "fontSize": "11px", "fontWeight": "500",
                        "background": "#F1EFE8", "padding": "1px 8px",
                        "borderRadius": "10px", "minWidth": "55px",
                        "textAlign": "center", "marginRight": "8px",
                    }),
                    html.Span(arrow, style={"color": arrow_color, "marginRight": "4px",
                                              "fontSize": "13px"}),
                    html.Span(star_str, style={"color": "#D97706", "fontSize": "12px",
                                                 "letterSpacing": "1px",
                                                 "marginRight": "8px"}),
                    html.A(n["headline"], href=n.get("url", "#"), target="_blank",
                           style={"fontSize": "13px", "color": "#222",
                                   "textDecoration": "none", "flex": "1"}),
                    *([html.Span(t, style={
                        "fontSize": "10px", "fontWeight": "500",
                        "color": badge_fg, "background": badge_color,
                        "padding": "1px 7px", "borderRadius": "10px",
                        "marginLeft": "6px",
                    }) for t in badge_text]),
                    html.Span(n.get("source", ""), style={
                        "fontSize": "11px", "color": "#888", "marginLeft": "8px",
                    }),
                    html.Span(ts, style={"fontSize": "11px", "color": "#aaa",
                                           "marginLeft": "10px"}),
                ], style={"display": "flex", "alignItems": "center"}),
                html.Div([
                    html.Span(insight, style={"fontSize": "11px", "color": "#666"}),
                    *([html.Span(f"  ·  {', '.join(kw_flags)}",
                                  style={"fontSize": "11px", "color": "#888"})]
                      if kw_flags else []),
                ], style={"marginLeft": "75px", "marginTop": "2px"}),
            ], style={"padding": "10px 0", "borderBottom": "1px solid #f5f5f5"}))
        body = html.Div(rows)

    return html.Div([
        html.Div([
            html.P("Recent news",
                   style={"fontSize": "12px", "fontWeight": "500",
                           "margin": "0", "flex": "1"}),
            html.Span("4★+ or contrarian only",
                     style={"fontSize": "10px", "color": "#aaa"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}),
        body,
    ], style=CARD)
