"""
dashboard/pages/strategy_finder.py
-----------------------------------
Strategy Finder — Optuna-driven hyperparameter search across the
registered strategies, with an optional "🤖 Ask Claude" button for
unconventional ideas.

Layout:
  1. Strategy + scope picker
  2. Search settings (n_trials, n_folds, seed)
  3. Run / Save run / Ask Claude buttons
  4. Leaderboard table — sortable, with "Save as new strategy" per row
  5. Saved-as panel (recent saves)
"""
from __future__ import annotations

import logging

import dash_bootstrap_components as dbc
from dash import html, dcc

from bot.universe import UNIVERSE_SCOPES

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
NUMBOX = {
    "fontSize":     "13px",
    "padding":      "6px 10px",
    "border":       "1px solid #ddd",
    "borderRadius": "6px",
    "width":        "100%",
}


def _strategy_options():
    """Only registered strategies that have a declared param_space."""
    from bot.strategy_finder import PARAM_SPACES
    try:
        from bot.models.registry import list_models
        models = list_models()
    except Exception:
        models = []
    out = []
    for m in models:
        if m.id in PARAM_SPACES:
            out.append({"label": f"{m.name} [{m.type}]", "value": m.id})
    return out or [{"label": "rsi_macd_v1 [rule]", "value": "rsi_macd_v1"}]


SCOPE_OPTIONS = [{"label": label, "value": key}
                 for key, label in UNIVERSE_SCOPES.items()]


def layout(account: str = "paper", model: str = "rsi_macd_v1", symbol: str = "AAPL"):
    strategies = _strategy_options()
    initial = model if model in {s["value"] for s in strategies} else strategies[0]["value"]

    return html.Div([
        html.P("STRATEGY FINDER", style=LABEL),

        html.Div([
            html.P(
                "Pick a strategy and let Optuna search its parameter space "
                "across walk-forward folds.  Each trial is one full "
                "backtest — the leaderboard is sorted by mean OOS Sharpe so "
                "you can save the most regime-robust configuration.",
                style={"fontSize": "12px", "color": "#666", "margin": "0 0 10px"},
            ),

            # ── Strategy + scope ──────────────────────────────────────
            dbc.Row([
                dbc.Col([
                    html.Span("Strategy",
                              style={"fontSize": "11px", "color": "#888"}),
                    dcc.Dropdown(id="sf-strategy", options=strategies,
                                 value=initial, clearable=False,
                                 style={"fontSize": "13px", "marginTop": "4px"}),
                ], md=4),
                dbc.Col([
                    html.Span("Universe scope",
                              style={"fontSize": "11px", "color": "#888"}),
                    dcc.Dropdown(id="sf-scope", options=SCOPE_OPTIONS,
                                 value="top_100", clearable=False,
                                 style={"fontSize": "13px", "marginTop": "4px"}),
                ], md=4),
                dbc.Col([
                    html.Span("Symbol limit",
                              style={"fontSize": "11px", "color": "#888"}),
                    dcc.Input(id="sf-max-syms", type="number", value=20,
                              min=1, max=200, step=1,
                              style={**NUMBOX, "marginTop": "4px"}),
                ], md=2),
                dbc.Col([
                    html.Span("Period (days)",
                              style={"fontSize": "11px", "color": "#888"}),
                    dcc.Input(id="sf-period", type="number", value=365 * 6,
                              min=180, max=2500, step=1,
                              style={**NUMBOX, "marginTop": "4px"}),
                ], md=2),
            ], className="g-3 mb-3"),

            # ── Search settings ──────────────────────────────────────
            dbc.Row([
                dbc.Col([
                    html.Span("Trials", id="sf-trials-tip",
                              style={"fontSize": "11px", "color": "#888"}),
                    dcc.Input(id="sf-trials", type="number", value=20,
                              min=5, max=200, step=1,
                              style={**NUMBOX, "marginTop": "4px"}),
                    dbc.Tooltip(
                        "Number of parameter combinations Optuna will try. "
                        "TPE typically converges in 30-50 trials.  Each "
                        "trial runs a full walk-forward backtest, so this "
                        "is the main cost lever.",
                        target="sf-trials-tip", placement="top",
                        style={"maxWidth": "320px", "fontSize": "12px"}),
                ], md=3),
                dbc.Col([
                    html.Span("Walk-forward folds", id="sf-folds-tip",
                              style={"fontSize": "11px", "color": "#888"}),
                    dcc.Input(id="sf-folds", type="number", value=4,
                              min=2, max=8, step=1,
                              style={**NUMBOX, "marginTop": "4px"}),
                    dbc.Tooltip(
                        "Number of OOS chunks per trial.  More folds = "
                        "more honest Sharpe estimate but slower.  4 is a "
                        "good default for 6-year history.",
                        target="sf-folds-tip", placement="top",
                        style={"maxWidth": "320px", "fontSize": "12px"}),
                ], md=3),
                dbc.Col([
                    html.Span("Seed", id="sf-seed-tip",
                              style={"fontSize": "11px", "color": "#888"}),
                    dcc.Input(id="sf-seed", type="number", value=42,
                              min=0, max=99999, step=1,
                              style={**NUMBOX, "marginTop": "4px"}),
                    dbc.Tooltip(
                        "Random seed for the TPE sampler.  Same seed + same "
                        "data = same leaderboard, every time.",
                        target="sf-seed-tip", placement="top",
                        style={"maxWidth": "320px", "fontSize": "12px"}),
                ], md=2),
                dbc.Col([
                    html.Span("Early stop after", id="sf-early-tip",
                              style={"fontSize": "11px", "color": "#888"}),
                    dcc.Input(id="sf-early", type="number", value=10,
                              min=2, max=50, step=1,
                              style={**NUMBOX, "marginTop": "4px"}),
                    dbc.Tooltip(
                        "Stop the search if the best Sharpe hasn't improved "
                        "in this many consecutive trials.  Saves wall time "
                        "on plateaus.",
                        target="sf-early-tip", placement="top",
                        style={"maxWidth": "320px", "fontSize": "12px"}),
                ], md=2),
                dbc.Col([
                    html.Div([
                        html.Button("🔍 Run search", id="sf-run", n_clicks=0,
                                    style={
                            "fontSize": "13px", "padding": "8px 18px",
                            "borderRadius": "8px", "border": "none",
                            "background": "#1D9E75", "color": "white",
                            "fontWeight": "500", "cursor": "pointer",
                            "marginRight": "8px",
                        }),
                        html.Button("🤖 Ask Claude", id="sf-claude",
                                    n_clicks=0, title="Optional: query the "
                                    "Anthropic API for unconventional "
                                    "parameter combinations. ~$0.05 per "
                                    "click. Requires ANTHROPIC_API_KEY.",
                                    style={
                            "fontSize": "13px", "padding": "8px 14px",
                            "borderRadius": "8px",
                            "border": "1px solid #3C3489",
                            "background": "white", "color": "#3C3489",
                            "fontWeight": "500", "cursor": "pointer",
                        }),
                    ], style={"display": "flex", "alignItems": "flex-end",
                                "height": "100%", "paddingTop": "18px"}),
                ], md=2),
            ], className="g-3"),
        ], style=CARD),

        # ── Status + leaderboard ─────────────────────────────────────────
        html.Div([
            html.Span(id="sf-status", style={
                "fontSize": "12px", "color": "#666",
            }),
            html.Span(id="sf-claude-status", style={
                "fontSize": "12px", "color": "#3C3489", "marginLeft": "12px",
            }),
        ], style={"marginBottom": "12px"}),

        dcc.Store(id="sf-store-results"),

        html.Div(id="sf-leaderboard-area",
                 children=html.P(
                    "Configure search and click Run.",
                    style={"color": "#aaa", "fontSize": "13px", "padding": "12px"}
                 ),
                 style=CARD),

        # Save modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Save as new strategy")),
            dbc.ModalBody([
                html.P("Snapshot the chosen trial as a CustomRuleModel — "
                       "it will appear in the Backtest model dropdown.",
                       style={"fontSize": "12px", "color": "#666",
                              "margin": "0 0 12px"}),
                html.Div([
                    html.Span("Strategy id (no spaces)",
                              style={"fontSize": "11px", "color": "#888"}),
                    dcc.Input(id="sf-save-id", type="text",
                              placeholder="e.g. rsi_macd_tuned_2026q1",
                              style={**NUMBOX, "marginTop": "4px"}),
                ], style={"marginBottom": "10px"}),
                html.Div([
                    html.Span("Description",
                              style={"fontSize": "11px", "color": "#888"}),
                    dcc.Textarea(id="sf-save-desc",
                                 placeholder="Short description (optional)",
                                 style={**NUMBOX, "marginTop": "4px",
                                          "minHeight": "60px"}),
                ]),
                dcc.Store(id="sf-save-trial-id"),
                html.Div(id="sf-save-msg", style={
                    "fontSize": "12px", "color": "#27500A", "marginTop": "10px",
                }),
            ]),
            dbc.ModalFooter([
                html.Button("Cancel", id="sf-modal-cancel", n_clicks=0, style={
                    "fontSize": "13px", "padding": "6px 14px",
                    "border": "1px solid #ddd", "borderRadius": "6px",
                    "background": "white", "cursor": "pointer",
                    "marginRight": "8px",
                }),
                html.Button("Save", id="sf-modal-save", n_clicks=0, style={
                    "fontSize": "13px", "padding": "6px 14px",
                    "border": "none", "borderRadius": "6px",
                    "background": "#1D9E75", "color": "white",
                    "fontWeight": "500", "cursor": "pointer",
                }),
            ]),
        ], id="sf-modal", is_open=False),
    ])


# ── Render helpers ──────────────────────────────────────────────────────────

def render_leaderboard(results: dict):
    if not results:
        return html.P("No results yet.",
                      style={"color": "#aaa", "fontSize": "13px"})

    if results.get("error"):
        return html.P(f"❌ {results['error']}",
                      style={"color": "#A32D2D", "fontSize": "13px"})

    rows = results.get("leaderboard", [])
    if not rows:
        return html.P("Search produced no trials — check the data is loaded.",
                      style={"color": "#aaa", "fontSize": "13px"})

    state = results.get("study_state", {})
    summary = (
        f"Strategy: {results.get('strategy', '')}  ·  "
        f"trials run: {results.get('n_trials', 0)}  ·  "
        f"best Sharpe: {state.get('best_value', 0)}  "
        + ("· (early-stopped)" if state.get("early_stopped") else "")
    )

    header = html.Tr([
        html.Th("Rank",     style=_th()),
        html.Th("Trial",    style=_th()),
        html.Th("OOS Sharpe", style=_th()),
        html.Th("OOS return", style=_th()),
        html.Th("% positive folds", style=_th()),
        html.Th("Trades",   style=_th()),
        html.Th("Params",   style=_th()),
        html.Th("Action",   style=_th()),
    ])

    body = []
    for rank, r in enumerate(rows, start=1):
        sharpe = r["mean_oos_sharpe"]
        ret    = r["mean_oos_return_pct"]
        params_text = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        body.append(html.Tr([
            html.Td(f"#{rank}",                 style=_td(weight="500")),
            html.Td(str(r["trial"]),            style=_td()),
            html.Td(f"{sharpe:+.3f}",
                    style={**_td(), "color": "#27500A" if sharpe >= 0 else "#A32D2D"}),
            html.Td(f"{ret:+.2f}%",
                    style={**_td(), "color": "#27500A" if ret >= 0 else "#A32D2D"}),
            html.Td(f"{r['pct_positive_folds']:.0f}%", style=_td()),
            html.Td(str(r["total_trades"]),     style=_td()),
            html.Td(params_text,                style={**_td(), "fontFamily": "monospace",
                                                        "fontSize": "11px"}),
            html.Td(html.Button("Save", id={"type": "sf-save-row", "trial": r["trial"]},
                                style={
                "fontSize": "11px", "padding": "3px 10px",
                "border": "1px solid #1D9E75", "borderRadius": "6px",
                "background": "white", "color": "#1D9E75",
                "cursor": "pointer",
            }), style=_td()),
        ]))

    return html.Div([
        html.P(summary, style={"fontSize": "12px", "color": "#666",
                                 "margin": "0 0 10px"}),
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
