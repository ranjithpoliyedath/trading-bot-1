"""
dashboard/callbacks/backtest_callbacks.py
------------------------------------------
Registers all backtest-related Dash callbacks.
Imported by ``dashboard/app.py`` so the @callback decorators run.
"""
from __future__ import annotations

import logging

from dash import Input, Output, State, ALL, callback_context, html, no_update, callback

from dashboard.backtest_engine import (
    run_backtest, run_filtered_backtest, run_walk_forward,
    save_backtest, load_backtest, list_saved_backtests,
)
from dashboard.pages.backtest import render_results, _filter_row

logger = logging.getLogger(__name__)


# ── NL query → populate model + filters + period ─────────────────────────────

@callback(
    Output("bt-dd-model",       "value",   allow_duplicate=True),
    Output("bt-input-period",   "value",   allow_duplicate=True),
    Output("bt-input-conf",     "value",   allow_duplicate=True),
    Output("bt-filter-rows",    "children", allow_duplicate=True),
    Output("bt-nl-status",      "children"),
    Input("bt-nl-parse",        "n_clicks"),
    State("bt-nl-query",        "value"),
    prevent_initial_call=True,
)
def parse_nl_query(n_clicks, text):
    if not n_clicks or not text:
        return no_update, no_update, no_update, no_update, "Type a query, then Parse."
    try:
        from bot.nl_query import parse_query
        parsed = parse_query(text)
    except Exception as exc:
        return no_update, no_update, no_update, no_update, f"❌ {exc}"

    rows = []
    for i, f in enumerate(parsed.filters):
        rows.append(_filter_row(i, f["field"], f["op"], f["value"]))

    status = (
        f"✅ {parsed.rationale}  "
        f"(model={parsed.model_id}, {parsed.period_days}d, "
        f"{len(parsed.filters)} filter(s))"
    )
    return (
        parsed.model_id,
        parsed.period_days,
        parsed.min_confidence,
        rows,
        status,
    )


# ── Show / hide the regime-exit conflict warning ────────────────────────
#
# When the user toggles market-regime OR sector-regime exit on, surface
# a yellow inline warning: these exits override per-symbol logic so
# strategy buy signals on a downtrend day are silently dropped.

@callback(
    Output("bt-regime-conflict-warning", "style"),
    Input("bt-exit-market-regime-on",     "value"),
    Input("bt-exit-sector-regime-on",     "value"),
    prevent_initial_call=False,
)
def toggle_regime_conflict_warning(market_on, sector_on):
    if market_on or sector_on:
        return {"display": "block"}
    return {"display": "none"}


# ── Add manual filter row ───────────────────────────────────────────────────

@callback(
    Output("bt-filter-rows", "children", allow_duplicate=True),
    Input("bt-add-filter",   "n_clicks"),
    State("bt-filter-rows",  "children"),
    prevent_initial_call=True,
)
def add_filter(n, current):
    if not n:
        return no_update
    current = list(current or [])
    current.append(_filter_row(len(current)))
    return current


# ── Expand "Indicator preset" picks into filter rows ─────────────────────
#
# When the user selects a preset from the bt-dd-indicator-preset dropdown,
# the preset's ``filters`` definition is appended to the existing filter
# rows.  Append (not replace) so users can stack presets — e.g. "Bull
# stack" + "RSI oversold" — and still hand-edit fields after.

@callback(
    Output("bt-filter-rows",            "children", allow_duplicate=True),
    Output("bt-dd-indicator-preset",    "value",    allow_duplicate=True),
    Input("bt-dd-indicator-preset",     "value"),
    State("bt-filter-rows",             "children"),
    prevent_initial_call=True,
)
def apply_indicator_preset(preset_key, current_rows):
    """Append the preset's filter rows to the existing list, then
    reset the dropdown so the same preset can be re-selected later."""
    if not preset_key:
        return no_update, no_update
    from bot.screener import INDICATOR_PRESETS
    preset = INDICATOR_PRESETS.get(preset_key)
    if not preset:
        return no_update, ""
    current_rows = list(current_rows or [])
    base_idx = len(current_rows)
    for i, f in enumerate(preset["filters"]):
        current_rows.append(_filter_row(
            base_idx + i,
            default_field = f["field"],
            default_op    = f["op"],
            default_value = f["value"],
        ))
    return current_rows, ""   # reset dropdown to placeholder


# ── Run / load backtest ──────────────────────────────────────────────────────

@callback(
    Output("bt-store-results", "data"),
    Input("bt-btn-run",      "n_clicks"),
    Input("bt-dd-saved",     "value"),
    State("bt-dd-model",     "value"),
    State("bt-dd-scope",     "value"),
    State("bt-input-max",    "value"),
    State("bt-dd-tf",        "value"),
    State("bt-input-period", "value"),
    State("bt-input-conf",   "value"),
    State("bt-indicators",   "value"),
    State({"type": "bt-filter-field", "index": ALL}, "value"),
    State({"type": "bt-filter-op",    "index": ALL}, "value"),
    State({"type": "bt-filter-value", "index": ALL}, "value"),
    State("bt-exit-signal-on", "value"),
    State("bt-exit-tp-on",     "value"),
    State("bt-exit-tp-val",    "value"),
    State("bt-exit-sl-on",     "value"),
    State("bt-exit-sl-val",    "value"),
    State("bt-exit-ts-on",     "value"),
    State("bt-exit-ts-val",    "value"),
    State("bt-exit-market-regime-on",  "value"),
    State("bt-exit-sector-regime-on",  "value"),
    State("bt-acct-cash",      "value"),
    State("bt-acct-sizing",    "value"),
    State("bt-acct-arg-a",     "value"),
    State("bt-acct-arg-b",     "value"),
    State("bt-acct-atr-stop",  "value"),
    State("bt-real-em",        "value"),
    State("bt-real-delay",     "value"),
    State("bt-real-slip",      "value"),
    State("bt-real-val",       "value"),
    State("bt-real-tax",       "value"),
    State("bt-real-omit",      "value"),
    prevent_initial_call=True,
)
def run_or_load(n_run, saved_id, model, scope, max_syms, tf, period, conf,
                indicators, ff, fo, fv,
                signal_on, tp_on, tp_val, sl_on, sl_val, ts_on, ts_val,
                market_regime_on, sector_regime_on,
                acct_cash, sizing_method, sizing_a, sizing_b, atr_stop,
                exec_model, exec_delay, slip_bps, val_mode,
                tax_pct, omit_top_n):
    ctx = callback_context.triggered[0]["prop_id"]
    if "bt-dd-saved" in ctx and saved_id:
        data = load_backtest(saved_id)
        if data:
            # Tag so the renderer can show a "loaded from saved" banner
            data = {**data, "_loaded_from_saved": saved_id}
        return data or {}
    if "bt-btn-run" not in ctx or not n_run:
        return {}

    # Parse filters
    filters = []
    for fld, op, val in zip(ff or [], fo or [], fv or []):
        if fld is None or op is None or val in (None, ""):
            continue
        try:
            filters.append({"field": fld, "op": op, "value": float(val)})
        except (TypeError, ValueError):
            continue

    # Resolve exit conditions — checklist returns ['on'] when checked
    use_signal_exit = bool(signal_on)
    take_profit_pct = float(tp_val) if (tp_on and tp_val not in (None, ""))      else None
    stop_loss_pct   = float(sl_val) if (sl_on and sl_val not in (None, ""))      else None
    time_stop_days  = int(ts_val)   if (ts_on and ts_val not in (None, "", 0))   else None

    # Resolve symbols from universe scope
    from bot.universe import select_universe
    syms = select_universe(scope or "top_100", limit=int(max_syms or 50))

    # Position sizing kwargs depend on the chosen method
    method = (sizing_method or "fixed_pct").lower()
    a = float(sizing_a) if sizing_a not in (None, "") else 0.95
    b = float(sizing_b) if sizing_b not in (None, "") else 2.0
    if   method == "fixed_pct":           sizing_kwargs = {"pct": a}
    elif method in ("kelly", "half_kelly"):
        sizing_kwargs = {"win_rate": a, "win_loss_ratio": b}
    elif method == "atr_risk":            sizing_kwargs = {"risk_pct": a, "atr_mult": b}
    else:                                  sizing_kwargs = {}

    atr_stop_mult = float(atr_stop) if (atr_stop and float(atr_stop) > 0) else None

    # Common engine kwargs shared between full + walk-forward paths
    engine_kwargs = dict(
        conf_threshold       = float(conf or 0.65),
        max_symbols          = int(max_syms or 50),
        use_signal_exit      = use_signal_exit,
        take_profit_pct      = take_profit_pct,
        stop_loss_pct        = stop_loss_pct,
        time_stop_days       = time_stop_days,
        atr_stop_mult        = atr_stop_mult,
        starting_cash        = float(acct_cash or 10_000),
        sizing_method        = method,
        sizing_kwargs        = sizing_kwargs,
        execution_model      = exec_model or "next_open",
        execution_delay_bars = int(exec_delay or 0),
        slippage_bps         = float(slip_bps or 0),
    )

    # Regime-exit kwargs are only honoured by run_filtered_backtest's
    # single-sample path (the walk-forward + cross-sectional runners
    # don't take them yet — adding silently would crash with
    # TypeError on unexpected kwarg).
    regime_kwargs = dict(
        market_regime_exit  = bool(market_regime_on),
        sector_regime_exit  = bool(sector_regime_on),
        tax_rate            = float(tax_pct or 0) / 100.0,
        omit_top_n_outliers = int(omit_top_n or 0),
    )

    # Walk-forward branch
    if val_mode and val_mode.startswith("wf"):
        try:
            n_folds = int((val_mode or "wf4")[2:])
        except ValueError:
            n_folds = 4
        out = run_walk_forward(
            model_id    = model or "rsi_macd_v1",
            n_folds     = n_folds,
            symbols     = syms,
            period_days = int(period or 365),
            **engine_kwargs,
        )
    else:
        # Single-sample branch (default "full")
        out = run_filtered_backtest(
            model_id    = model or "rsi_macd_v1",
            filters     = filters,
            symbols     = syms,
            period_days = int(period or 365),
            **engine_kwargs,
            **regime_kwargs,
        )

    # Persist scope + indicators + val_mode in the preset payload so a
    # later "selecting this saved run" can hydrate every form field.
    if isinstance(out, dict):
        preset = out.get("preset") or {}
        preset.update({
            "scope":      scope or "top_100",
            "indicators": list(indicators or []),
            "val_mode":   val_mode or "full",
            "filters":    filters,
        })
        out["preset"] = preset
    return out


# ── Render results ──────────────────────────────────────────────────────────

@callback(
    Output("bt-results-area", "children"),
    Input("bt-store-results", "data"),
    prevent_initial_call=True,
)
def display_results(results):
    return render_results(results or {})


# ── Save run ────────────────────────────────────────────────────────────────

@callback(
    Output("bt-save-msg",  "children"),
    Output("bt-dd-saved",  "options"),
    Input("bt-btn-save",   "n_clicks"),
    State("bt-store-results", "data"),
    prevent_initial_call=True,
)
def save_run(n_clicks, results):
    if not results or not results.get("run_id"):
        return "Nothing to save — run a backtest first.", list_saved_backtests()
    run_id = save_backtest(results)
    return f"Saved: {run_id}", list_saved_backtests()


# ── Apply preset: hydrate every form field from a saved-run's preset ────────
#
# The user picks a SEED or BT-* run from the saved dropdown and clicks
# "⤴ Apply preset".  The saved JSON's `preset` payload holds the exact
# configuration that produced that run; we splat it back into every
# form input so the user can tweak and re-run.
#
# Old saved runs without a `preset` payload synthesize one from the
# top-level `model`, `period_days`, `conf_threshold`, `filters` so the
# feature degrades cleanly across snapshots.

def _synthesize_preset_if_missing(data: dict) -> dict:
    """Backfill a preset payload from top-level fields if absent."""
    p = data.get("preset")
    if p:
        return p
    return {
        "model_id":        data.get("model", "rsi_macd_v1"),
        "filters":         data.get("filters", []) or [],
        "period_days":     int(data.get("period_days", 365) or 365),
        "min_confidence":  float(data.get("conf_threshold", 0.65) or 0.65),
    }


@callback(
    Output("bt-dd-model",       "value",    allow_duplicate=True),
    Output("bt-input-period",   "value",    allow_duplicate=True),
    Output("bt-input-conf",     "value",    allow_duplicate=True),
    Output("bt-input-max",      "value",    allow_duplicate=True),
    Output("bt-acct-cash",      "value",    allow_duplicate=True),
    Output("bt-acct-sizing",    "value",    allow_duplicate=True),
    Output("bt-acct-arg-a",     "value",    allow_duplicate=True),
    Output("bt-acct-arg-b",     "value",    allow_duplicate=True),
    Output("bt-acct-atr-stop",  "value",    allow_duplicate=True),
    Output("bt-real-em",        "value",    allow_duplicate=True),
    Output("bt-real-delay",     "value",    allow_duplicate=True),
    Output("bt-real-slip",      "value",    allow_duplicate=True),
    Output("bt-exit-signal-on", "value",    allow_duplicate=True),
    Output("bt-exit-tp-on",     "value",    allow_duplicate=True),
    Output("bt-exit-tp-val",    "value",    allow_duplicate=True),
    Output("bt-exit-sl-on",     "value",    allow_duplicate=True),
    Output("bt-exit-sl-val",    "value",    allow_duplicate=True),
    Output("bt-exit-ts-on",     "value",    allow_duplicate=True),
    Output("bt-exit-ts-val",    "value",    allow_duplicate=True),
    Output("bt-exit-market-regime-on", "value", allow_duplicate=True),
    Output("bt-exit-sector-regime-on", "value", allow_duplicate=True),
    Output("bt-real-tax",       "value",    allow_duplicate=True),
    Output("bt-real-omit",      "value",    allow_duplicate=True),
    Output("bt-real-val",       "value",    allow_duplicate=True),
    Output("bt-dd-tf",          "value",    allow_duplicate=True),
    Output("bt-indicators",     "value",    allow_duplicate=True),
    Output("bt-dd-scope",       "value",    allow_duplicate=True),
    Output("bt-filter-rows",    "children", allow_duplicate=True),
    Output("bt-preset-status",  "children"),
    Input("bt-dd-saved",         "value"),
    prevent_initial_call=True,
)
def apply_preset(run_id):
    """Hydrate every form field from a saved-run's preset payload.

    Fires the moment the user picks a saved run from the dropdown — no
    confirmation button needed.  The result envelope itself is loaded
    by `run_or_load` (also keyed off `bt-dd-saved.value`); the two
    callbacks run in parallel so the right pane updates simultaneously.
    """
    # 29 outputs total — keep this in sync with the @callback decorator.
    n_outputs = 29
    if not run_id:
        # Cleared the dropdown — leave the form alone, just clear status
        return (*([no_update] * (n_outputs - 1)), "")

    data = load_backtest(run_id)
    if not data:
        return (*([no_update] * (n_outputs - 1)),
                f"❌ Couldn't load {run_id}.")

    p = _synthesize_preset_if_missing(data)
    sk = p.get("sizing_kwargs", {}) or {}

    # Sizing args A/B depend on the chosen method
    method = (p.get("sizing_method") or "fixed_pct").lower()
    if   method == "fixed_pct":            arg_a, arg_b = sk.get("pct", 0.95), 2.0
    elif method in ("kelly", "half_kelly"):
        arg_a = sk.get("win_rate", 0.5)
        arg_b = sk.get("win_loss_ratio", 1.5)
    elif method == "atr_risk":
        arg_a = sk.get("risk_pct", 0.01)
        arg_b = sk.get("atr_mult", 2.0)
    else:
        arg_a, arg_b = 0.95, 2.0

    # Exit toggles: ['on'] / [] convention used by dcc.Checklist
    on   = ["on"]
    off  = []
    tp   = p.get("take_profit_pct")
    sl   = p.get("stop_loss_pct")
    ts   = p.get("time_stop_days")
    atr  = p.get("atr_stop_mult") or 0

    filters = p.get("filters", []) or []
    filter_rows = [
        _filter_row(i, f.get("field", "rsi_14"), f.get("op", "<"),
                    float(f.get("value", 30)))
        for i, f in enumerate(filters)
    ]

    # Validation mode comes from the preset payload first; fall back to
    # inferring from the result shape (walk-forward runs have fold_results).
    val_mode = p.get("val_mode")
    if not val_mode:
        if "fold_results" in data:
            n_folds = data.get("n_folds") or len(data.get("fold_results") or [])
            val_mode = f"wf{n_folds}" if n_folds in (2, 3, 4) else "full"
        else:
            val_mode = "full"

    scope_val      = p.get("scope") or "top_100"
    indicators_val = p.get("indicators") or no_update

    status = (
        f"✅ Loaded preset from {run_id}: model={p.get('model_id')}, "
        f"period={p.get('period_days')}d, {len(filters)} filter(s)."
    )

    return (
        p.get("model_id", "rsi_macd_v1"),                          # bt-dd-model
        int(p.get("period_days", 365) or 365),                     # bt-input-period
        float(p.get("min_confidence", 0.65) or 0.65),              # bt-input-conf
        int(p.get("max_symbols", 50) or 50),                       # bt-input-max
        float(p.get("starting_cash", 10_000) or 10_000),           # bt-acct-cash
        method,                                                    # bt-acct-sizing
        float(arg_a),                                              # bt-acct-arg-a
        float(arg_b),                                              # bt-acct-arg-b
        float(atr),                                                # bt-acct-atr-stop
        p.get("execution_model", "next_open"),                     # bt-real-em
        int(p.get("execution_delay", 0) or 0),                     # bt-real-delay
        float(p.get("slippage_bps", 5) or 5),                      # bt-real-slip
        on if p.get("use_signal_exit", True) else off,             # bt-exit-signal-on
        on if tp is not None else off,                             # bt-exit-tp-on
        float(tp) if tp is not None else 0.15,                     # bt-exit-tp-val
        on if sl is not None else off,                             # bt-exit-sl-on
        float(sl) if sl is not None else 0.07,                     # bt-exit-sl-val
        on if ts is not None else off,                             # bt-exit-ts-on
        int(ts) if ts is not None else 30,                         # bt-exit-ts-val
        on if p.get("market_regime_exit") else off,                # bt-exit-market-regime-on
        on if p.get("sector_regime_exit") else off,                # bt-exit-sector-regime-on
        round(float(p.get("tax_rate", 0) or 0) * 100, 0),          # bt-real-tax (% display)
        int(p.get("omit_top_n_outliers", 0) or 0),                 # bt-real-omit
        val_mode,                                                  # bt-real-val
        "1d",                                                      # bt-dd-tf (only daily for now)
        indicators_val,                                            # bt-indicators
        scope_val,                                                 # bt-dd-scope
        filter_rows,                                               # bt-filter-rows
        status,                                                    # bt-preset-status
    )


# ── AI run analyzer callbacks ─────────────────────────────────────────────
#
# The "Analyze this backtest" button kicks off a background analysis
# (Claude when ANTHROPIC_API_KEY is set, local heuristics otherwise).
# An Interval polls the analyzer's state every 2 seconds and re-renders
# the panel when the result lands.

@callback(
    Output("ai-analyzer-panel", "children", allow_duplicate=True),
    Input("btn-analyze-run", "n_clicks"),
    State("bt-store-results", "data"),
    prevent_initial_call=True,
)
def trigger_run_analysis(n_clicks, results):
    """Click handler: kick off the analyzer thread, render an
    immediate "running…" indicator.  The Interval poll below
    refreshes the panel every 2 seconds until status flips to
    done / failed."""
    if not n_clicks:
        return no_update
    from dashboard.services.run_analyzer import start_analysis
    state = start_analysis(results or {})
    return _render_analyzer_state(state)


@callback(
    Output("ai-analyzer-panel", "children", allow_duplicate=True),
    Input("ai-analyzer-poll", "n_intervals"),
    prevent_initial_call=True,
)
def poll_run_analysis(_n_intervals):
    """Background poll while analyzer is running.  When the state is
    'done', the panel snaps to the markdown result and stops changing
    on subsequent polls."""
    from dashboard.services.run_analyzer import get_analyzer_status
    state = get_analyzer_status()
    return _render_analyzer_state(state)


def _render_analyzer_state(state: dict):
    """Translate analyzer state → Dash component tree."""
    from dash import dcc

    s = state.get("status", "idle")

    if s == "idle":
        return html.Div("Click the button above to analyze this run.",
                         style={"fontSize": "12px", "color": "#888",
                                "fontStyle": "italic"})

    if s == "running":
        return html.Div([
            html.Div([
                html.Span("● ", style={"color": "#D97706",
                                        "fontSize": "16px"}),
                html.Span("Analyzing…",
                          style={"fontSize": "13px", "fontWeight": "600",
                                 "color": "#D97706"}),
            ]),
            html.Div("This usually takes 5–15 seconds with Claude, "
                     "or under a second with local heuristics.",
                     style={"fontSize": "11px", "color": "#888",
                            "marginTop": "4px"}),
        ], style={"padding": "12px 14px",
                  "background": "#FFF7ED",
                  "border":     "1px solid #FED7AA",
                  "borderRadius": "8px"})

    if s == "done":
        result = state.get("result") or {}
        text   = result.get("text", "(empty result)")
        source = result.get("source", "?")
        model  = result.get("model")
        dur    = state.get("duration_s") or 0

        source_label = (
            f"Claude ({model})" if source == "claude" else
            "Local heuristics (no API key)"
        )

        return html.Div([
            dcc.Markdown(text, link_target="_blank",
                          style={"fontSize": "13px",
                                 "color":    "#1f2937",
                                 "lineHeight": "1.6"}),
            html.Div(f"— Generated by {source_label} in {dur:.1f}s",
                     style={"fontSize": "11px", "color": "#888",
                            "fontStyle": "italic",
                            "marginTop": "12px",
                            "paddingTop": "8px",
                            "borderTop": "1px solid #eee"}),
        ])

    if s == "failed":
        err = state.get("error", "(no error message)")
        return html.Div([
            html.Div([
                html.Span("✕ ", style={"color": "#A32D2D",
                                        "fontSize": "16px"}),
                html.Span("Analysis failed",
                          style={"fontSize": "13px", "fontWeight": "600",
                                 "color": "#A32D2D"}),
            ]),
            html.Div(err,
                     style={"fontSize": "12px", "color": "#A32D2D",
                            "marginTop": "6px",
                            "fontFamily": "monospace",
                            "wordBreak": "break-word"}),
        ], style={"padding": "12px 14px",
                  "background": "#FEF2F2",
                  "border":     "1px solid #FECACA",
                  "borderRadius": "8px"})

    return html.Div(f"Unknown state: {s}",
                     style={"fontSize": "12px", "color": "#888"})
