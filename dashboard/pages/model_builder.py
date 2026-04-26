"""
dashboard/pages/model_builder.py
---------------------------------
Custom rule-based model builder.

Lets the user compose a model from buy/sell condition rows and save it
to ``dashboard/custom_models/<id>.json``.  The registry picks it up
automatically (`bot.models.registry.list_models`).
"""
from __future__ import annotations

import logging

from dash import html, dcc

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
    "fontSize": "10px", "fontWeight": "500", "color": "#aaa",
    "letterSpacing": "0.07em", "margin": "0 0 10px",
}

OPS = [
    {"label": ">",  "value": ">"},
    {"label": ">=", "value": ">="},
    {"label": "<",  "value": "<"},
    {"label": "<=", "value": "<="},
    {"label": "==", "value": "=="},
    {"label": "!=", "value": "!="},
]
FIELD_OPTIONS = [
    {"label": f"{meta['group']} — {meta['label']}", "value": key}
    for key, meta in SCREENER_FIELDS.items()
]


def _rule_row(group: str, idx: int, default_field: str, default_op: str, default_value: float):
    return html.Div([
        dcc.Dropdown(
            id={"type": f"mb-{group}-field", "index": idx},
            options=FIELD_OPTIONS, value=default_field, clearable=False,
            style={"flex": "2", "fontSize": "13px"},
        ),
        dcc.Dropdown(
            id={"type": f"mb-{group}-op", "index": idx},
            options=OPS, value=default_op, clearable=False,
            style={"flex": "0 0 80px", "fontSize": "13px", "marginLeft": "8px"},
        ),
        dcc.Input(
            id={"type": f"mb-{group}-value", "index": idx},
            type="number", value=default_value,
            style={
                "flex": "0 0 120px", "marginLeft": "8px",
                "fontSize": "13px", "padding": "6px 10px",
                "border": "1px solid #ddd", "borderRadius": "6px",
            },
        ),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"})


def _default_buy_rows():
    return [_rule_row("buy", 0, "rsi_14", "<", 30)]


def _default_sell_rows():
    return [_rule_row("sell", 0, "rsi_14", ">", 70)]


def _section(title: str, rows_id: str, add_btn_id: str, default_rows):
    return html.Div([
        html.Div([
            html.Span(title, style={
                "fontSize": "13px", "fontWeight": "500", "marginRight": "auto",
            }),
            html.Button("+ Add condition", id=add_btn_id, n_clicks=0, style={
                "fontSize": "12px", "padding": "5px 12px",
                "border": "1px solid #ddd", "borderRadius": "6px",
                "background": "white", "cursor": "pointer",
            }),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}),
        html.Div(id=rows_id, children=default_rows),
    ], style=CARD)


def layout(account: str = "paper", model: str = "model_v1", symbol: str = "AAPL"):
    return html.Div([
        html.P("CUSTOM MODEL BUILDER", style=LABEL),

        # ── Identity ─────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Span("Model id", style={"fontSize": "11px", "color": "#888",
                                              "marginRight": "8px", "minWidth": "70px"}),
                dcc.Input(id="mb-id", type="text", placeholder="my_strategy",
                          style={"flex": "1", "fontSize": "13px",
                                  "padding": "6px 10px", "border": "1px solid #ddd",
                                  "borderRadius": "6px"}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}),
            html.Div([
                html.Span("Name", style={"fontSize": "11px", "color": "#888",
                                          "marginRight": "8px", "minWidth": "70px"}),
                dcc.Input(id="mb-name", type="text", placeholder="Oversold bounce",
                          style={"flex": "1", "fontSize": "13px",
                                  "padding": "6px 10px", "border": "1px solid #ddd",
                                  "borderRadius": "6px"}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}),
            html.Div([
                html.Span("Description", style={"fontSize": "11px", "color": "#888",
                                                  "marginRight": "8px", "minWidth": "70px"}),
                dcc.Input(id="mb-desc", type="text", placeholder="Optional",
                          style={"flex": "1", "fontSize": "13px",
                                  "padding": "6px 10px", "border": "1px solid #ddd",
                                  "borderRadius": "6px"}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}),
            html.Div([
                html.Span("Min confidence", style={"fontSize": "11px", "color": "#888",
                                                     "marginRight": "8px", "minWidth": "100px"}),
                dcc.Input(id="mb-conf", type="number", value=0.65, min=0, max=1, step=0.05,
                          style={"flex": "0 0 100px", "fontSize": "13px",
                                  "padding": "6px 10px", "border": "1px solid #ddd",
                                  "borderRadius": "6px"}),
            ], style={"display": "flex", "alignItems": "center"}),
        ], style=CARD),

        # ── Conditions ───────────────────────────────────────────────────
        _section("Buy when (all conditions match)",
                 "mb-buy-rows", "mb-add-buy", _default_buy_rows()),
        _section("Sell when (all conditions match)",
                 "mb-sell-rows", "mb-add-sell", _default_sell_rows()),

        # ── Save / load / list ──────────────────────────────────────────
        html.Div([
            html.Button("Save model", id="mb-save", n_clicks=0, style={
                "fontSize": "13px", "padding": "8px 18px",
                "border": "none", "borderRadius": "8px",
                "background": "#1D9E75", "color": "white",
                "fontWeight": "500", "cursor": "pointer", "marginRight": "10px",
            }),
            html.Button("Reset", id="mb-reset", n_clicks=0, style={
                "fontSize": "13px", "padding": "8px 18px",
                "border": "1px solid #ddd", "borderRadius": "8px",
                "background": "white", "cursor": "pointer",
            }),
            html.Span(id="mb-save-status", style={
                "marginLeft": "16px", "fontSize": "12px", "color": "#666",
            }),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "16px"}),

        html.Div([
            html.P("Saved custom models", style={"fontSize": "13px",
                                                   "fontWeight": "500", "margin": "0 0 10px"}),
            html.Div(id="mb-saved-list",
                     children=render_saved_list()),
        ], style=CARD),
    ])


# ── Helpers used by callbacks ────────────────────────────────────────────────

def render_saved_list():
    """Render the list of saved custom models."""
    from bot.models.registry import _list_custom_models  # type: ignore
    items = _list_custom_models()
    if not items:
        return html.P("No custom models saved yet.",
                      style={"color": "#aaa", "fontSize": "12px"})
    rows = []
    for m in items:
        rows.append(html.Div([
            html.Span(m.id.replace("custom:", ""),
                      style={"fontFamily": "monospace", "fontSize": "12px",
                              "background": "#F1EFE8", "padding": "2px 8px",
                              "borderRadius": "6px", "marginRight": "10px"}),
            html.Span(m.name, style={"fontSize": "13px", "fontWeight": "500",
                                       "marginRight": "10px"}),
            html.Span(m.description or "",
                      style={"fontSize": "12px", "color": "#888",
                              "flex": "1"}),
            html.Span(", ".join((m.required_features or [])[:4]),
                      style={"fontSize": "11px", "color": "#666"}),
        ], style={"display": "flex", "alignItems": "center",
                  "padding": "8px 0", "borderBottom": "1px solid #f5f5f5"}))
    return html.Div(rows)
