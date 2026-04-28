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
    State("bt-acct-cash",      "value"),
    State("bt-acct-sizing",    "value"),
    State("bt-acct-arg-a",     "value"),
    State("bt-acct-arg-b",     "value"),
    State("bt-acct-atr-stop",  "value"),
    State("bt-real-em",        "value"),
    State("bt-real-delay",     "value"),
    State("bt-real-slip",      "value"),
    State("bt-real-val",       "value"),
    prevent_initial_call=True,
)
def run_or_load(n_run, saved_id, model, scope, max_syms, tf, period, conf,
                indicators, ff, fo, fv,
                signal_on, tp_on, tp_val, sl_on, sl_val, ts_on, ts_val,
                acct_cash, sizing_method, sizing_a, sizing_b, atr_stop,
                exec_model, exec_delay, slip_bps, val_mode):
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

    # Walk-forward branch
    if val_mode and val_mode.startswith("wf"):
        try:
            n_folds = int((val_mode or "wf4")[2:])
        except ValueError:
            n_folds = 4
        return run_walk_forward(
            model_id    = model or "rsi_macd_v1",
            n_folds     = n_folds,
            symbols     = syms,
            period_days = int(period or 365),
            **engine_kwargs,
        )

    # Single-sample branch (default "full")
    return run_filtered_backtest(
        model_id    = model or "rsi_macd_v1",
        filters     = filters,
        symbols     = syms,
        period_days = int(period or 365),
        **engine_kwargs,
    )


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
    Output("bt-exit-tp-on",     "value",    allow_duplicate=True),
    Output("bt-exit-tp-val",    "value",    allow_duplicate=True),
    Output("bt-exit-sl-on",     "value",    allow_duplicate=True),
    Output("bt-exit-sl-val",    "value",    allow_duplicate=True),
    Output("bt-exit-ts-on",     "value",    allow_duplicate=True),
    Output("bt-exit-ts-val",    "value",    allow_duplicate=True),
    Output("bt-filter-rows",    "children", allow_duplicate=True),
    Output("bt-preset-status",  "children"),
    Input("bt-btn-apply-preset", "n_clicks"),
    State("bt-dd-saved",         "value"),
    prevent_initial_call=True,
)
def apply_preset(n_clicks, run_id):
    """Hydrate every form field from a saved-run's preset payload."""
    n_outputs = 20
    if not n_clicks or not run_id:
        return (*([no_update] * (n_outputs - 1)),
                "Pick a saved run, then click Apply preset.")

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
        on if tp is not None else off,                             # bt-exit-tp-on
        float(tp) if tp is not None else 0.15,                     # bt-exit-tp-val
        on if sl is not None else off,                             # bt-exit-sl-on
        float(sl) if sl is not None else 0.07,                     # bt-exit-sl-val
        on if ts is not None else off,                             # bt-exit-ts-on
        int(ts) if ts is not None else 30,                         # bt-exit-ts-val
        filter_rows,                                               # bt-filter-rows
        status,                                                    # bt-preset-status
    )
