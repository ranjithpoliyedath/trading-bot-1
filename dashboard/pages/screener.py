"""
dashboard/pages/screener.py
----------------------------
Stock discovery / screener page.

The user picks any number of filter rows (field + operator + value),
hits "Run screener", and we display matching universe symbols ranked
by the first filter's field.  Each row exposes a "Send to backtest"
button (currently just routes the symbol through the global symbol
store and switches the view).
"""
from __future__ import annotations

import logging

import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table

from bot.screener import SCREENER_FIELDS

logger = logging.getLogger(__name__)


CARD = {
    "background":   "white",
    "borderRadius": "12px",
    "border":       "1px solid #eee",
    "padding":      "18px",
    "marginBottom": "16px",
}

LABEL = {
    "fontSize":      "10px",
    "fontWeight":    "500",
    "color":         "#aaa",
    "letterSpacing": "0.07em",
    "margin":        "0 0 10px",
}

OPS = [
    {"label": ">",  "value": ">"},
    {"label": ">=", "value": ">="},
    {"label": "<",  "value": "<"},
    {"label": "<=", "value": "<="},
    {"label": "==", "value": "=="},
    {"label": "!=", "value": "!="},
]

# Build field options grouped by category for readability.
FIELD_OPTIONS = [
    {"label": f"{meta['group']} — {meta['label']}", "value": key}
    for key, meta in SCREENER_FIELDS.items()
]


def _filter_row(idx: int, default_field: str, default_op: str, default_value: float):
    """Render a single filter row."""
    return html.Div([
        dcc.Dropdown(
            id={"type": "screener-field", "index": idx},
            options=FIELD_OPTIONS,
            value=default_field,
            clearable=False,
            style={"flex": "2", "fontSize": "13px"},
        ),
        dcc.Dropdown(
            id={"type": "screener-op", "index": idx},
            options=OPS,
            value=default_op,
            clearable=False,
            style={"flex": "0 0 80px", "fontSize": "13px", "marginLeft": "8px"},
        ),
        dcc.Input(
            id={"type": "screener-value", "index": idx},
            type="number",
            value=default_value,
            style={
                "flex": "0 0 120px", "marginLeft": "8px",
                "fontSize": "13px", "padding": "6px 10px",
                "border": "1px solid #ddd", "borderRadius": "6px",
            },
        ),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"})


def _default_rows():
    return [
        _filter_row(0, "rsi_14",             "<", 30),
        _filter_row(1, "combined_sentiment", ">", 0.1),
        _filter_row(2, "volume_ratio",       ">", 1.0),
    ]


def layout(account: str = "paper", model: str = "model_v1", symbol: str = "AAPL"):
    return html.Div([
        html.P("STOCK SCREENER", style=LABEL),

        html.Div([
            html.Div([
                html.Span("Filters", style={
                    "fontSize": "13px", "fontWeight": "500", "marginRight": "auto",
                }),
                html.Button("+ Add filter", id="btn-add-filter", n_clicks=0, style={
                    "fontSize": "12px", "padding": "5px 12px",
                    "border": "1px solid #ddd", "borderRadius": "6px",
                    "background": "white", "cursor": "pointer", "marginRight": "8px",
                }),
                html.Button("Run screener", id="btn-run-screener", n_clicks=0, style={
                    "fontSize": "13px", "padding": "6px 16px",
                    "border": "none", "borderRadius": "8px",
                    "background": "#1D9E75", "color": "white",
                    "fontWeight": "500", "cursor": "pointer",
                }),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "14px"}),

            html.Div(id="screener-filters", children=_default_rows()),

            html.Div([
                html.Span("Sort by", style={"fontSize": "11px", "color": "#888",
                                              "marginRight": "8px"}),
                dcc.Dropdown(
                    id="screener-sort",
                    options=FIELD_OPTIONS,
                    placeholder="(first filter)",
                    style={"flex": "0 0 260px", "fontSize": "12px"},
                ),
                html.Span("Limit", style={"fontSize": "11px", "color": "#888",
                                            "marginLeft": "16px", "marginRight": "8px"}),
                dcc.Input(id="screener-limit", type="number", value=25, min=1, max=200,
                          style={"flex": "0 0 80px", "fontSize": "12px",
                                  "padding": "5px 8px", "border": "1px solid #ddd",
                                  "borderRadius": "6px"}),
            ], style={"display": "flex", "alignItems": "center", "marginTop": "6px"}),
        ], style=CARD),

        html.Div(id="screener-results", children=html.P(
            "Set filters and click Run screener.",
            style={"color": "#aaa", "fontSize": "13px", "padding": "12px"},
        ), style=CARD),
    ])


# ── Result rendering ─────────────────────────────────────────────────────────

def render_results(rows: list[dict]):
    """Render the matched-rows table.  Imported by the callback in app.py."""
    if not rows:
        return html.P("No matches — loosen your filters.",
                      style={"color": "#aaa", "fontSize": "13px", "padding": "12px"})

    # Determine the union of matched keys for column headers.
    matched_keys: list[str] = []
    for r in rows:
        for k in r.get("matched", {}):
            if k not in matched_keys:
                matched_keys.append(k)

    header = html.Tr([
        html.Th("Symbol",  style=_th()),
        html.Th("Close",   style=_th()),
        *[html.Th(SCREENER_FIELDS.get(k, {}).get("label", k), style=_th())
          for k in matched_keys],
        html.Th("Action",  style=_th()),
    ])

    body = []
    for r in rows:
        cells = [
            html.Td(r["symbol"], style=_td(weight="500")),
            html.Td(f"${r['close']:.2f}", style=_td()),
        ]
        for k in matched_keys:
            v = r.get("matched", {}).get(k)
            cells.append(html.Td(_fmt(v), style=_td()))
        cells.append(html.Td(
            html.Button("Send to backtest",
                        id={"type": "screener-send", "symbol": r["symbol"]},
                        style={
                            "fontSize": "11px", "padding": "3px 10px",
                            "border": "1px solid #1D9E75", "borderRadius": "6px",
                            "background": "white", "color": "#1D9E75", "cursor": "pointer",
                        }),
            style=_td(),
        ))
        body.append(html.Tr(cells))

    summary = html.P(
        f"{len(rows)} match{'es' if len(rows) != 1 else ''}.",
        style={"fontSize": "12px", "color": "#666", "margin": "0 0 10px"},
    )
    return html.Div([
        summary,
        html.Table([html.Thead(header), html.Tbody(body)],
                   style={"width": "100%", "borderCollapse": "collapse"}),
    ])


def _th():
    return {
        "textAlign": "left", "fontSize": "11px", "color": "#888",
        "padding": "8px 6px", "borderBottom": "1px solid #eee",
        "fontWeight": "500", "letterSpacing": "0.04em",
        "textTransform": "uppercase",
    }


def _td(weight: str = "400"):
    return {
        "fontSize":     "13px",
        "padding":      "8px 6px",
        "borderBottom": "1px solid #f5f5f5",
        "color":        "#222",
        "fontWeight":   weight,
    }


def _fmt(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)
