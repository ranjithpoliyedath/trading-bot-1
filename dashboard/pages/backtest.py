"""
dashboard/pages/backtest.py
----------------------------
Full backtest view: NL-query box, manual filter rows, controls, run
button, equity curve, monthly returns, metrics row, trade distribution,
risk metrics, saved runs dropdown.
"""
from __future__ import annotations

import logging
from typing import Optional

import plotly.graph_objects as go
from dash import html, dcc, Input, Output, State, callback_context, callback
import dash_bootstrap_components as dbc

from dashboard.backtest_engine import (
    run_backtest, save_backtest, load_backtest, list_saved_backtests
)
from bot.screener  import SCREENER_FIELDS
from bot.universe  import UNIVERSE_SCOPES

logger = logging.getLogger(__name__)

CARD  = {"background": "white", "borderRadius": "12px", "border": "1px solid #eee", "padding": "14px"}
METRIC = {"background": "#F8F8F7", "borderRadius": "8px", "padding": "10px 12px"}

INDICATORS = ["RSI", "MACD", "EMA cross", "Bollinger", "ATR", "Volume ratio", "VWAP"]


def _model_options():
    """Pull live options from the registry — exclude cross-sectional
    strategies (they need ``run_cross_sectional_backtest``, not the
    per-symbol runner this page wires up)."""
    try:
        from bot.models.registry import list_models
        return [{"label": f"{m.name} [{m.type}]", "value": m.id}
                for m in list_models() if m.type != "cross_sectional"]
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

    # ── Left column: every configuration panel, scrollable independently ──
    left = html.Div([
        _section_label("Natural-language query"),
        _nl_query_panel(),
        _section_label("Manual filters"),
        _filter_panel(),
        _section_label("Configure & run"),
        _controls_panel(initial_model, symbol, saved),
    ], style=_LEFT_COLUMN_STYLE)

    # ── Right column: results, scrollable independently ──
    right = html.Div([
        html.Div(id="bt-results-anchor"),
        html.Div(id="bt-results-area",
                 children=html.P(
                    "▼ Run backtest to see results here.",
                    style={"color": "#aaa", "fontSize": "12px",
                            "padding": "24px", "fontStyle": "italic",
                            "background": "#FAFAFA", "borderRadius": "8px",
                            "border": "1px dashed #DDD",
                            "textAlign": "center"})),
    ], style=_RIGHT_COLUMN_STYLE)

    return dbc.Row([
        dbc.Col(left,  md=3, xs=12),     # ~25% on md+, full width on small screens
        dbc.Col(right, md=9, xs=12),
    ], className="g-3", style={"minHeight": "calc(100vh - 80px)"})


# Independent-scroll columns.  Sticky to viewport so the right pane keeps
# rolling while the left pane (controls) stays in place.
_LEFT_COLUMN_STYLE = {
    "position":    "sticky",
    "top":         "0",
    "height":      "calc(100vh - 80px)",
    "overflowY":   "auto",
    "paddingRight": "8px",
}
_RIGHT_COLUMN_STYLE = {
    "position":    "sticky",
    "top":         "0",
    "height":      "calc(100vh - 80px)",
    "overflowY":   "auto",
    "paddingLeft": "8px",
}


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


def _labeled(text: str, tip: str, id_: str):
    """Single-form-field label with hover tooltip.  Returns a list of
    components (label + tooltip) to splat into the parent layout — keep
    consistent with the rest of the panels' visual style."""
    return html.Div([
        html.P(text, id=id_, style={"fontSize": "11px", "color": "#888",
                                      "margin": "0 0 4px", "cursor": "help"}),
        dbc.Tooltip(tip, target=id_, placement="top",
                    style={"maxWidth": "320px", "fontSize": "12px"}),
    ])


# Per-indicator hover descriptions for the checklist.
INDICATOR_TIPS = {
    "RSI":          "Relative Strength Index (14-period). Momentum oscillator: <30 oversold, >70 overbought.",
    "MACD":         "Moving Average Convergence Divergence. Fast EMA(12) − Slow EMA(26), with a signal-line trigger. Trend / momentum.",
    "EMA cross":    "Short EMA(9) vs long EMA(21) crossover. Classic short-term trend signal.",
    "Bollinger":    "Bollinger Bands (20-period, 2σ). Mean-reversion at the band extremes.",
    "ATR":          "Average True Range (14). Volatility — drives ATR-based position sizing and stops.",
    "Volume ratio": "Today's volume vs the 20-day average. >2× often confirms a breakout.",
    "VWAP":         "Volume-Weighted Average Price. Intraday support/resistance benchmark — also used as a fair-value gauge on daily bars.",
}


def _indicators_checklist():
    """Indicators checklist with a hover tooltip per item.  Dash's
    ``dcc.Checklist`` doesn't directly accept per-option tooltips, so we
    render each label as an ``html.Span`` carrying its own DOM id and
    pair it with a sibling ``dbc.Tooltip``.
    """
    options = []
    tooltips = []
    for ind in INDICATORS:
        slug = ind.lower().replace(" ", "-")
        opt_id = f"bt-ind-tip-{slug}"
        options.append({
            "label": html.Span([
                ind,
                html.Span(" ⓘ", id=opt_id,
                          style={"color": "#bbb", "fontSize": "10px",
                                  "cursor": "help"}),
            ], style={"fontSize": "12px", "marginRight": "10px"}),
            "value": ind,
        })
        tooltips.append(dbc.Tooltip(
            INDICATOR_TIPS.get(ind, ""),
            target=opt_id, placement="top",
            style={"maxWidth": "320px", "fontSize": "12px"},
        ))
    return html.Div([
        dcc.Checklist(
            id="bt-indicators",
            options=options,
            value=["RSI", "MACD", "EMA cross", "ATR", "VWAP"],
            inline=False,
            style={"display": "flex", "flexWrap": "wrap", "gap": "4px"},
        ),
        *tooltips,
    ])


def _controls_panel(model: str, symbol: str, saved: list):
    """Render the configure-and-run panel as a vertical stack so it fits
    inside the 25%-width left column."""
    return html.Div([
        # Saved-run / preset picker — separate row so it stands out.
        _labeled("Saved run / preset",
                 "Pick a checked-in SEED or any user-saved run.  Selecting "
                 "loads its results into the right pane; click Apply "
                 "preset to also fill the form below from its saved "
                 "configuration so you can tweak and re-run.",
                 "bt-lbl-saved"),
        dcc.Dropdown(id="bt-dd-saved", options=saved,
                     placeholder="Select a saved backtest...",
                     style=DD, clearable=True),
        html.Div([
            html.Button("⤴ Apply preset", id="bt-btn-apply-preset", n_clicks=0,
                        style={
                "fontSize": "12px", "padding": "5px 12px",
                "border": "1px solid #3C3489", "borderRadius": "6px",
                "background": "white", "color": "#3C3489",
                "cursor": "pointer", "marginRight": "8px",
            }),
            html.Span(id="bt-preset-status", style={
                "fontSize": "11px", "color": "#666",
            }),
        ], style={"display": "flex", "alignItems": "center", "marginTop": "6px"}),
        html.Div(style={"height": "10px"}),

        _labeled("Model",
                 "Trading strategy that emits buy / sell signals.  "
                 "Built-in rule models, ML models (when trained), and any "
                 "custom JSON-spec models you've saved appear here.",
                 "bt-lbl-model"),
        dcc.Dropdown(id="bt-dd-model", options=MODELS, value=model,
                     clearable=False, style=DD),
        html.Div(style={"height": "10px"}),

        _labeled("Universe scope",
                 "Slice of the universe to scan.  Includes categorical "
                 "groupings (S&P 500, top 100 by liquidity), broad-market "
                 "ETFs (SPY/QQQ/etc.), and individual mega-cap tickers.",
                 "bt-lbl-scope"),
        dcc.Dropdown(id="bt-dd-scope", options=SCOPE_OPTIONS,
                     value="top_100", clearable=False, style=DD),
        html.Div(style={"height": "10px"}),

        _labeled("Symbol limit",
                 "Cap how many symbols from the chosen scope get scanned.  "
                 "Lower = faster runs.  Ignored for single-ETF / single-"
                 "stock scopes.",
                 "bt-lbl-max"),
        dcc.Input(id="bt-input-max", type="number", value=50,
                  min=1, max=2000, step=1, style=NUMBOX),
        html.Div(style={"height": "12px"}),

        # Run / save buttons
        html.Div([
            html.Button("Run backtest", id="bt-btn-run", n_clicks=0, style={
                "fontSize": "13px", "padding": "8px 20px", "borderRadius": "8px",
                "border": "none", "background": "#1D9E75", "color": "white",
                "fontWeight": "500", "cursor": "pointer", "marginRight": "8px",
                "flex": "1",
            }),
            html.Button("Save run", id="bt-btn-save", n_clicks=0, style={
                "fontSize": "13px", "padding": "8px 16px", "borderRadius": "8px",
                "border": "1px solid #ddd", "background": "white",
                "cursor": "pointer", "flex": "0 0 auto",
            }),
        ], style={"display": "flex", "gap": "6px"}),
        html.Div(style={"height": "16px"}),

        _labeled("Timeframe",
                 "Bar size for the backtest.  Daily only for now; intraday "
                 "support is parked behind the news look-ahead Phase.",
                 "bt-lbl-tf"),
        dcc.Dropdown(id="bt-dd-tf", options=TIMEFRAMES, value="1d",
                     clearable=False, style=DD),
        html.Div(style={"height": "10px"}),

        _labeled("Period (days)",
                 "Number of trading days of history to backtest over.  "
                 "Strategies needing SMA(200) want at least 365.",
                 "bt-lbl-period"),
        dcc.Input(id="bt-input-period", type="number", value=365,
                  min=30, max=2000, step=1, style=NUMBOX),
        html.Div(style={"height": "10px"}),

        _labeled("Confidence threshold",
                 "Minimum signal confidence (0-1) for the engine to act.  "
                 "0.55-0.65 is typical; 0.80+ trades only the highest-"
                 "conviction signals.",
                 "bt-lbl-conf"),
        dcc.Input(id="bt-input-conf", type="number", value=0.65,
                  min=0, max=1, step=0.01, style=NUMBOX),
        html.Div(style={"height": "10px"}),

        _labeled("Indicators",
                 "Subset of feature columns to keep.  Affects only ML "
                 "models — the rule-based strategies always read every "
                 "column they need.  Tooltips on each check explain the "
                 "indicator.",
                 "bt-lbl-ind"),
        _indicators_checklist(),
        html.Div(style={"height": "16px"}),

        # Hidden — kept for backwards compatibility with existing callbacks.
        # Symbol selection is now driven by the Universe scope dropdown.
        dcc.Store(id="bt-dd-symbol", data="All"),

        # ── Sample trading account ──────────────────────────────────
        _account_panel(),

        # ── Realism: execution model, slippage, walk-forward ────────
        _realism_panel(),

        # ── Exit conditions ─────────────────────────────────────────
        _exit_conditions_panel(),

        dcc.Store(id="bt-store-results"),
        html.Div(id="bt-save-msg", style={"fontSize": "12px", "color": "#27500A", "marginTop": "8px"}),
    ], style={**CARD, "marginBottom": "16px"})


def _realism_panel():
    """Execution-model / slippage / walk-forward controls."""
    return html.Div([
        html.P("Realism settings",
               style={"fontSize": "12px", "fontWeight": "500", "margin": "10px 0 8px"}),

        dbc.Row([
            dbc.Col([
                html.Span("Execution model", id="bt-real-em-tip",
                          style={"fontSize": "11px", "color": "#888"}),
                dcc.Dropdown(
                    id="bt-real-em",
                    options=[
                        {"label": "Next bar's open (realistic)", "value": "next_open"},
                        {"label": "Same bar close (legacy)",     "value": "same_close"},
                    ],
                    value="next_open", clearable=False,
                    style={"fontSize": "13px", "marginTop": "4px"},
                ),
                dbc.Tooltip(
                    "When the model emits a signal at bar t, when does the "
                    "trade actually fill?  next_open = open of bar t+1 "
                    "(industry-standard, prevents look-ahead bias).  "
                    "same_close = close of bar t (legacy, faster-but-fake).",
                    target="bt-real-em-tip", placement="top",
                    style={"maxWidth": "360px", "fontSize": "12px"}),
            ], md=12),
            dbc.Col([
                html.Span("Execution delay (bars)", id="bt-real-delay-tip",
                          style={"fontSize": "11px", "color": "#888"}),
                dcc.Input(id="bt-real-delay", type="number", value=0,
                          min=0, max=10, step=1,
                          style={**NUMBOX, "marginTop": "4px"}),
                dbc.Tooltip(
                    "Extra bars between signal and fill.  0 = act on the "
                    "next bar (per execution model).  >0 simulates slow "
                    "decision pipelines.  Reserved for intraday strategies.",
                    target="bt-real-delay-tip", placement="top",
                    style={"maxWidth": "360px", "fontSize": "12px"}),
            ], md=12),
            dbc.Col([
                html.Span("Slippage (bps)", id="bt-real-slip-tip",
                          style={"fontSize": "11px", "color": "#888"}),
                dcc.Input(id="bt-real-slip", type="number", value=5,
                          min=0, max=200, step=1,
                          style={**NUMBOX, "marginTop": "4px"}),
                dbc.Tooltip(
                    "Per-trade adverse fill, in basis points.  5 bps = 0.05% "
                    "worse than the printed price.  Buy fills are bumped up, "
                    "sell fills bumped down — round-trip cost is 2× this.  "
                    "Liquid large-caps: 1-5 bps. Small-caps: 10-30 bps.",
                    target="bt-real-slip-tip", placement="top",
                    style={"maxWidth": "360px", "fontSize": "12px"}),
            ], md=12),
            dbc.Col([
                html.Span("Validation mode", id="bt-real-val-tip",
                          style={"fontSize": "11px", "color": "#888"}),
                dcc.Dropdown(
                    id="bt-real-val",
                    options=[
                        {"label": "Full history (single run)",         "value": "full"},
                        {"label": "Walk-forward (4 folds)",             "value": "wf4"},
                        {"label": "Walk-forward (3 folds)",             "value": "wf3"},
                        {"label": "Walk-forward (2 folds)",             "value": "wf2"},
                    ],
                    value="full", clearable=False,
                    style={"fontSize": "13px", "marginTop": "4px"},
                ),
                dbc.Tooltip(
                    "Walk-forward splits history into N OOS chunks and "
                    "tests the strategy on each.  Use it to spot strategies "
                    "that only worked in one regime.  4 folds = ~4 OOS "
                    "Sharpe scores you can compare for consistency.",
                    target="bt-real-val-tip", placement="top",
                    style={"maxWidth": "360px", "fontSize": "12px"}),
            ], md=12),
        ], className="g-3"),
    ], style={**CARD, "marginTop": "12px"})


def _account_panel():
    """Starting cash + position sizing method + sizing params."""
    return html.Div([
        html.P("Sample trading account",
               style={"fontSize": "12px", "fontWeight": "500", "margin": "10px 0 8px"}),

        dbc.Row([
            dbc.Col([
                html.Span("Starting cash ($)",
                          id="bt-acct-cash-tip",
                          style={"fontSize": "11px", "color": "#888"}),
                dcc.Input(id="bt-acct-cash", type="number", value=10000,
                          min=100, step=100, style={**NUMBOX, "marginTop": "4px"}),
                dbc.Tooltip(
                    "Initial portfolio value used for position sizing. All P&L "
                    "and equity-curve metrics are computed against this base.",
                    target="bt-acct-cash-tip", placement="top",
                    style={"maxWidth": "320px", "fontSize": "12px"}),
            ], md=12),
            dbc.Col([
                html.Span("Position sizing",
                          id="bt-acct-sizing-tip",
                          style={"fontSize": "11px", "color": "#888"}),
                dcc.Dropdown(
                    id="bt-acct-sizing",
                    options=[
                        {"label": "Fixed % of portfolio", "value": "fixed_pct"},
                        {"label": "Kelly criterion",      "value": "kelly"},
                        {"label": "Half-Kelly (safer)",   "value": "half_kelly"},
                        {"label": "ATR risk (volatility-normalised)",
                                                          "value": "atr_risk"},
                    ],
                    value="fixed_pct", clearable=False,
                    style={"fontSize": "13px", "marginTop": "4px"},
                ),
                dbc.Tooltip(
                    "How much of your portfolio to deploy on each trade. "
                    "ATR-risk sizes positions so a stop-out costs no more "
                    "than `risk %` of your capital — pro standard for "
                    "volatile names. Kelly sizes by edge; half-Kelly is the "
                    "safer practical default.",
                    target="bt-acct-sizing-tip", placement="top",
                    style={"maxWidth": "360px", "fontSize": "12px"}),
            ], md=12),
            dbc.Col([
                html.Span("Sizing arg A", id="bt-acct-arg-a-tip",
                          style={"fontSize": "11px", "color": "#888"}),
                dcc.Input(id="bt-acct-arg-a", type="number", value=0.95,
                          min=0, max=1, step=0.01,
                          style={**NUMBOX, "marginTop": "4px"}),
                dbc.Tooltip(
                    "fixed_pct: fraction of portfolio per trade (0-1). "
                    "kelly / half_kelly: assumed historical win rate (0-1). "
                    "atr_risk: % of capital risked per trade (e.g. 0.01 = 1%).",
                    target="bt-acct-arg-a-tip", placement="top",
                    style={"maxWidth": "360px", "fontSize": "12px"}),
            ], md=12),
            dbc.Col([
                html.Span("Sizing arg B", id="bt-acct-arg-b-tip",
                          style={"fontSize": "11px", "color": "#888"}),
                dcc.Input(id="bt-acct-arg-b", type="number", value=2.0,
                          step=0.1, style={**NUMBOX, "marginTop": "4px"}),
                dbc.Tooltip(
                    "fixed_pct: ignored. "
                    "kelly / half_kelly: win/loss ratio (avg win $ / avg loss $). "
                    "atr_risk: ATR multiple for stop distance (e.g. 2.0 = 2×ATR).",
                    target="bt-acct-arg-b-tip", placement="top",
                    style={"maxWidth": "360px", "fontSize": "12px"}),
            ], md=12),
            dbc.Col([
                html.Span("ATR exit (× ATR)", id="bt-acct-atr-tip",
                          style={"fontSize": "11px", "color": "#888"}),
                dcc.Input(id="bt-acct-atr-stop", type="number", value=0,
                          min=0, step=0.5,
                          style={**NUMBOX, "marginTop": "4px"}),
                dbc.Tooltip(
                    "Close the trade when price falls below entry minus N × "
                    "ATR (using ATR on the entry day). 0 = disabled. "
                    "Volatility-aware stop — adapts to each symbol.",
                    target="bt-acct-atr-tip", placement="top",
                    style={"maxWidth": "360px", "fontSize": "12px"}),
            ], md=12),
        ], className="g-3"),
    ], style={**CARD, "marginTop": "12px"})


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
        html.P("Exit conditions", id="bt-lbl-exits",
               style={"fontSize": "12px", "fontWeight": "500",
                       "margin": "10px 0 8px", "cursor": "help"}),
        dbc.Tooltip(
            "Whichever rule fires first closes the trade.  Disable any "
            "rule (uncheck) to ignore it.  At least one must stay "
            "enabled or the engine forces a default time-stop.",
            target="bt-lbl-exits", placement="top",
            style={"maxWidth": "320px", "fontSize": "12px"}),
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


def render_walk_forward(results: dict):
    """Per-fold table + aggregate Sharpe summary for a walk-forward run."""
    folds   = results.get("fold_results", [])
    agg     = results.get("aggregate", {})
    if not folds:
        return html.Div([
            _section_label("Walk-forward results — no folds"),
            html.P("Walk-forward returned no folds — date range too short?",
                   style={"color": "#aaa", "fontSize": "13px"}),
        ])

    total_trades = sum(int((fr.get("metrics") or {}).get("total_trades", 0) or 0)
                        for fr in folds)
    if total_trades == 0:
        return html.Div([
            _section_label(f"Walk-forward results — {results.get('run_id', '')}"),
            html.Div([
                html.P("⚠ No trades fired in any fold.",
                       style={"fontSize": "14px", "fontWeight": "500",
                              "color": "#A32D2D", "margin": "0 0 10px"}),
                html.P("The strategy / filter combination didn't trigger a single buy across the chosen universe and period.  Common causes:",
                       style={"fontSize": "13px", "color": "#444", "margin": "0 0 8px"}),
                html.Ul([
                    html.Li("Confidence threshold too tight — try lowering it (e.g. 0.55)."),
                    html.Li("Filters too restrictive — for custom rule models, AND-only conditions multiply quickly."),
                    html.Li("Period too short for warm-up — strategies that need SMA(200) or 90-day lookback need ≥ 1y per fold."),
                    html.Li("Sentiment data sparse — strategies gating on sentiment > 0 fire only when news data is present (currently ~10% of bars)."),
                    html.Li("Universe too small — try a broader scope or higher symbol limit."),
                ], style={"fontSize": "13px", "color": "#444", "marginBottom": "0"}),
            ], style=CARD),
        ])

    # Aggregate tiles up top
    agg_items = [
        ("Mean OOS Sharpe",   f"{agg.get('mean_oos_sharpe', 0):.2f}",
         agg.get("mean_oos_sharpe", 0) >= 1),
        ("Median OOS Sharpe", f"{agg.get('median_oos_sharpe', 0):.2f}",
         agg.get("median_oos_sharpe", 0) >= 1),
        ("Sharpe stdev",      f"{agg.get('stdev_oos_sharpe', 0):.2f}",
         agg.get("stdev_oos_sharpe", 1) <= 1),
        ("% positive folds",  f"{agg.get('pct_positive_folds', 0):.0f}%",
         agg.get("pct_positive_folds", 0) >= 50),
        ("Mean OOS return",
         f"{'+' if agg.get('mean_oos_return_pct', 0) >= 0 else ''}{agg.get('mean_oos_return_pct', 0):.2f}%",
         agg.get("mean_oos_return_pct", 0) >= 0),
    ]
    agg_row = dbc.Row([
        dbc.Col(_metric_tile(label, value, positive, ns="agg"),
                md=int(12 / len(agg_items)))
        for label, value, positive in agg_items
    ], className="g-3 mb-3")

    # Per-fold summary table (compact overview)
    header = html.Tr([
        html.Th("Fold",      style=_th()),
        html.Th("OOS window", style=_th()),
        html.Th("Trades",    style=_th()),
        html.Th("Return",    style=_th()),
        html.Th("Sharpe",    style=_th()),
        html.Th("Win rate",  style=_th()),
        html.Th("Max DD",    style=_th()),
    ])
    summary_rows = []
    for fr in folds:
        m = fr.get("metrics", {})
        sharpe = m.get("sharpe", 0)
        ret    = m.get("total_return_pct", 0)
        summary_rows.append(html.Tr([
            html.Td(f"#{fr['fold']}",                     style=_td(weight="500")),
            html.Td(f"{fr['oos_window'][0]} → {fr['oos_window'][1]}",
                                                          style=_td()),
            html.Td(str(m.get("total_trades", 0)),        style=_td()),
            html.Td(f"{'+' if ret >= 0 else ''}{ret:.2f}%",
                    style={**_td(), "color": "#27500A" if ret >= 0 else "#A32D2D"}),
            html.Td(f"{sharpe:.2f}",                       style=_td()),
            html.Td(f"{m.get('win_rate_pct', 0):.1f}%",    style=_td()),
            html.Td(f"{m.get('max_drawdown_pct', 0):.2f}%", style=_td()),
        ]))

    sections = [
        _section_label(f"Walk-forward results — {results.get('run_id', '')}"),
        agg_row,
        html.Div([
            html.P("Per-fold breakdown",
                   style={"fontSize": "12px", "fontWeight": "500", "margin": "0 0 10px"}),
            html.Table([html.Thead(header), html.Tbody(summary_rows)],
                       style={"width": "100%", "borderCollapse": "collapse"}),
        ], style=CARD),
    ]

    # Per-fold full result section — every fold gets its own equity curve,
    # monthly chart, trade-distribution + risk + signal-quality cards.
    # Each fold's tooltip ids are namespaced by fold number to dodge
    # duplicate-DOM-id warnings.
    for fr in folds:
        fold_n = fr["fold"]
        fold_ns = f"f{fold_n}"
        fold_label = (f"Fold #{fold_n} — {fr['oos_window'][0]} → "
                      f"{fr['oos_window'][1]}")
        fold_payload = {
            "run_id":          fr.get("run_id", f"fold-{fold_n}"),
            "metrics":         fr.get("metrics", {}),
            "equity_curve":    fr.get("equity_curve", []),
            "monthly_returns": fr.get("monthly_returns", []),
            "trades":          fr.get("trades", []),
        }
        sections.append(html.Hr(style={"margin": "24px 0",
                                         "borderColor": "#eee"}))
        sections.append(render_results(
            fold_payload, ns=fold_ns, section_label=fold_label,
        ))
    return html.Div(sections)


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


def render_results(results: dict, ns: str = "", section_label: Optional[str] = None):
    """Render a single-period results envelope.

    ``ns`` namespaces every tooltip / stat-row id so the same payload can
    be rendered multiple times on the same page (per fold).  Pass an
    explicit ``section_label`` to override the default "Results — <run_id>"
    heading (useful for fold sections).
    """
    # Walk-forward results have a different shape
    if results and "fold_results" in results:
        return render_walk_forward(results)

    if not results or not results.get("metrics"):
        return html.P("Run a backtest to see results.", style={"color": "#aaa", "fontSize": "13px"})

    m  = results["metrics"]
    ec = results.get("equity_curve", [])
    mr = results.get("monthly_returns", [])

    label = section_label or f"Results — {results.get('run_id', '')}"

    # Empty-trade case — show clear diagnostic instead of zero-everything tiles
    if int(m.get("total_trades", 0) or 0) == 0:
        return html.Div([
            _section_label(label),
            html.Div([
                html.P("⚠ No trades fired with these parameters.",
                       style={"fontSize": "14px", "fontWeight": "500",
                              "color": "#A32D2D", "margin": "0 0 10px"}),
                html.P("Common causes:", style={"fontSize": "13px",
                                                  "color": "#444", "margin": "0 0 8px"}),
                html.Ul([
                    html.Li("Confidence threshold too tight — try 0.55 to start."),
                    html.Li("Filter combination too restrictive — AND-only filters multiply quickly."),
                    html.Li("Strategy needs sentiment / sma_200 / etc. that may be sparse on the chosen universe."),
                    html.Li("Universe too small or period too short."),
                ], style={"fontSize": "13px", "color": "#444", "marginBottom": "0"}),
            ], style=CARD),
        ])

    return html.Div([
        _section_label(label),
        _metrics_row(m, ns=ns),
        dbc.Row([
            dbc.Col(_equity_chart(ec, title=f"Equity curve — {ns}" if ns else "Equity curve"),  md=7),
            dbc.Col(_monthly_chart(mr),  md=5),
        ], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(_trade_dist(m, ns=ns),    md=4),
            dbc.Col(_risk_card(m, ns=ns),     md=4),
            dbc.Col(_signal_quality(m, ns=ns), md=4),
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


def _slug(label: str) -> str:
    """Stable DOM-id slug from a metric label, e.g. 'Sharpe ratio' → 'sharpe-ratio'."""
    return "bt-tip-" + label.lower().replace(" ", "-").replace("%", "pct").replace("/", "-")


def _metric_tile(label: str, value: str, positive: bool, ns: str = ""):
    """Single metric tile with a real Bootstrap tooltip on hover.

    ``ns`` is an optional namespace prefix added to the tooltip id so
    the same tile can be rendered N times on the page (per-fold walk-
    forward results) without producing duplicate DOM ids.
    """
    tip_id = _slug(label) + (f"-{ns}" if ns else "")
    color  = ("#27500A" if positive else "#A32D2D") \
                if label not in ("Sharpe ratio", "Edge ratio") else "#111"
    return html.Div([
        html.Div([
            html.P(label, style={"fontSize": "11px", "color": "#888",
                                  "margin": "0", "flex": "1"}),
            html.Span("ⓘ", id=tip_id, style={
                "fontSize": "11px", "color": "#bbb", "cursor": "help",
                "marginLeft": "4px",
            }),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "3px"}),
        html.P(value, style={"fontSize": "18px", "fontWeight": "500",
                              "margin": "0", "color": color}),
        dbc.Tooltip(METRIC_TIPS.get(label, ""), target=tip_id,
                    placement="top",
                    style={"maxWidth": "320px", "fontSize": "12px"}),
    ], style=METRIC)


def _metrics_row(m: dict, ns: str = ""):
    items = [
        ("Total return",  f"{'+' if m['total_return_pct'] >= 0 else ''}{m['total_return_pct']:.1f}%",  m['total_return_pct'] >= 0),
        ("Sharpe ratio",  f"{m['sharpe']:.2f}",  m['sharpe'] >= 1),
        ("Expectancy",    f"${m['expectancy']:.2f}", m['expectancy'] >= 0),
        ("Edge ratio",    f"{m['edge_ratio']:.2f}x", m['edge_ratio'] >= 1),
        ("Max drawdown",  f"{m['max_drawdown_pct']:.1f}%", False),
        ("Win rate",      f"{m['win_rate_pct']:.1f}%", m['win_rate_pct'] >= 50),
    ]
    return dbc.Row([
        dbc.Col(_metric_tile(label, value, positive, ns=ns), md=2)
        for label, value, positive in items
    ], className="g-3 mb-3")


def _equity_chart(
    equity_curve:    list,
    benchmark_symbol: str = "SPY",
    title:           str = "Equity curve",
):
    """Equity curve plus a half-opacity benchmark overlay (default SPY)
    normalised to the equity curve's starting dollar value.  When the
    benchmark file isn't on disk the overlay silently disappears.
    """
    if not equity_curve:
        return html.Div("No equity data", style={**CARD, "color": "#aaa", "fontSize": "12px"})

    dates  = [e["date"]  for e in equity_curve]
    values = [e["value"] for e in equity_curve]
    up     = values[-1] >= values[0] if values else True
    color  = "#1D9E75" if up else "#E24B4A"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=values, mode="lines",
        name="Strategy",
        line=dict(color=color, width=1.6),
        fill="tozeroy",
        fillcolor="rgba(29,158,117,0.08)" if up else "rgba(226,75,74,0.08)",
    ))

    # ── Benchmark overlay (half-opaque, dashed) ─────────────────────────
    bench: list = []
    if benchmark_symbol and dates:
        try:
            from dashboard.backtest_engine import load_benchmark_curve
            # Skip the curve's "start" sentinel that doesn't parse as a date.
            real_dates = [d for d in dates if d and d != "start"]
            if real_dates:
                bench = load_benchmark_curve(
                    benchmark_symbol,
                    start         = real_dates[0],
                    end           = real_dates[-1],
                    normalize_to  = float(values[0]) if values else None,
                )
        except Exception as exc:                # noqa: BLE001 — render must not break
            logger.debug("Benchmark load failed for %s: %s", benchmark_symbol, exc)
            bench = []

    if bench:
        fig.add_trace(go.Scatter(
            x    = [b["date"]  for b in bench],
            y    = [b["value"] for b in bench],
            mode = "lines",
            name = f"{benchmark_symbol} (benchmark)",
            line = dict(color="rgba(60,52,137,0.5)", width=1.4, dash="dot"),
            opacity = 0.5,
        ))

    fig.update_layout(
        margin = dict(l=0, r=0, t=28, b=0), height=220,
        paper_bgcolor = "white", plot_bgcolor = "white",
        title  = dict(text=title, font=dict(size=12)),
        xaxis  = dict(showgrid=False, tickfont=dict(size=10)),
        yaxis  = dict(showgrid=True,  gridcolor="#f0f0f0",
                       tickfont=dict(size=10), tickprefix="$"),
        showlegend = bool(bench),
        legend     = dict(orientation="h", y=1.10, x=0,
                            font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
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


_stat_row_counter = {"n": 0}


def _stat_row(label, value, color="#111", ns: str = ""):
    """Stat row with a Bootstrap tooltip — id is unique per (label, ns,
    counter) so the same label can appear in many cards / many folds."""
    _stat_row_counter["n"] += 1
    suffix = f"{ns}-{_stat_row_counter['n']}" if ns else str(_stat_row_counter["n"])
    tip_id = f"{_slug(label)}-{suffix}"
    tip = METRIC_TIPS.get(label, "")
    return html.Div([
        html.Span(label, style={"fontSize": "12px", "color": "#888", "flex": "1"}),
        *([html.Span("ⓘ", id=tip_id, style={
            "fontSize": "11px", "color": "#bbb", "cursor": "help",
            "marginRight": "8px",
        }), dbc.Tooltip(tip, target=tip_id, placement="left",
                          style={"maxWidth": "320px", "fontSize": "12px"})]
          if tip else []),
        html.Span(value, style={"fontSize": "12px", "fontWeight": "500", "color": color}),
    ], style={"display": "flex", "alignItems": "center",
              "padding": "5px 0", "borderBottom": "1px solid #f5f5f5"})


def _trade_dist(m: dict, ns: str = ""):
    wins   = m.get("wins",   int(m["total_trades"] * m["win_rate_pct"]      / 100))
    losses = m.get("losses", int(m["total_trades"] * (1 - m["win_rate_pct"] / 100)))
    return html.Div([
        html.P("Trade distribution", style={"fontSize": "12px", "fontWeight": "500", "margin": "0 0 10px"}),
        _stat_row("Total trades",  str(m["total_trades"]),               ns=ns),
        _stat_row("Wins",          str(wins),   "#27500A",                ns=ns),
        _stat_row("Losses",        str(losses), "#A32D2D",                ns=ns),
        _stat_row("Win rate",      f"{m['win_rate_pct']:.1f}%",
                                    "#27500A" if m["win_rate_pct"] >= 50 else "#A32D2D",
                                                                          ns=ns),
        _stat_row("Loss rate",     f"{m.get('loss_rate_pct', 100 - m['win_rate_pct']):.1f}%",
                                                                          ns=ns),
        _stat_row("Profit factor", f"{m['profit_factor']:.2f}",            ns=ns),
    ], style=CARD)


def _risk_card(m: dict, ns: str = ""):
    return html.Div([
        html.P("Risk metrics", style={"fontSize": "12px", "fontWeight": "500", "margin": "0 0 10px"}),
        _stat_row("Avg win",      f"${m['avg_win']:,.2f}",                 "#27500A", ns=ns),
        _stat_row("Avg loss",     f"${m['avg_loss']:,.2f}",                "#A32D2D", ns=ns),
        _stat_row("Avg win %",    f"{m.get('avg_win_pct', 0):+.2f}%",      "#27500A", ns=ns),
        _stat_row("Avg loss %",   f"{m.get('avg_loss_pct', 0):+.2f}%",     "#A32D2D", ns=ns),
        _stat_row("Largest win",  f"${m['largest_win']:,.2f}",             "#27500A", ns=ns),
        _stat_row("Largest loss", f"${m['largest_loss']:,.2f}",            "#A32D2D", ns=ns),
        _stat_row("Sortino",      f"{m['sortino']:.2f}",                                ns=ns),
        _stat_row("Max drawdown", f"{m['max_drawdown_pct']:.2f}%",         "#A32D2D", ns=ns),
    ], style=CARD)


def _signal_quality(m: dict, ns: str = ""):
    return html.Div([
        html.P("Signal quality", style={"fontSize": "12px", "fontWeight": "500", "margin": "0 0 10px"}),
        _stat_row("Win rate",       f"{m['win_rate_pct']:.1f}%",
                                    "#27500A" if m["win_rate_pct"] >= 50 else "#A32D2D",
                                                                                       ns=ns),
        _stat_row("Expectancy",     f"${m['expectancy']:.2f}",
                                    "#27500A" if m["expectancy"]  >= 0  else "#A32D2D", ns=ns),
        _stat_row("Edge ratio",     f"{m['edge_ratio']:.2f}x",                          ns=ns),
        _stat_row("Max drawdown",   f"{m['max_drawdown_pct']:.1f}%",      "#A32D2D",   ns=ns),
    ], style=CARD)
