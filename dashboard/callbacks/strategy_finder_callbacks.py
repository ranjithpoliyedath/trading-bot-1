"""
dashboard/callbacks/strategy_finder_callbacks.py
-------------------------------------------------
Wires the Strategy Finder page UI to ``bot.strategy_finder`` and the
"Save as new strategy" modal that writes a CustomRuleModel JSON.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from dash import Input, Output, State, ALL, callback_context, callback, no_update

from dashboard.pages.strategy_finder import render_leaderboard

logger = logging.getLogger(__name__)


# ── Run search (Optuna, free) ───────────────────────────────────────────────

@callback(
    Output("sf-store-results", "data"),
    Output("sf-status",        "children"),
    Input("sf-run", "n_clicks"),
    State("sf-strategy", "value"),
    State("sf-scope",    "value"),
    State("sf-max-syms", "value"),
    State("sf-period",   "value"),
    State("sf-trials",   "value"),
    State("sf-folds",    "value"),
    State("sf-seed",     "value"),
    State("sf-early",    "value"),
    prevent_initial_call=True,
)
def run_search(n_clicks, strategy, scope, max_syms, period,
                n_trials, n_folds, seed, early_stop):
    if not n_clicks or not strategy:
        return no_update, "Pick a strategy and click Run."

    from bot.universe        import select_universe
    from bot.strategy_finder import run_optuna

    syms = select_universe(scope or "top_100", limit=int(max_syms or 20))
    if not syms:
        return {}, "❌ Universe scope returned no symbols."

    try:
        results = run_optuna(
            strategy_id      = strategy,
            n_trials         = int(n_trials or 20),
            n_folds          = int(n_folds or 4),
            symbols          = syms,
            period_days      = int(period or 365 * 6),
            seed             = int(seed or 42),
            early_stop_after = int(early_stop or 10),
        )
    except Exception as exc:
        logger.exception("Optuna search failed")
        return {}, f"❌ Search failed: {exc}"

    n = results.get("n_trials", 0)
    best = results.get("best", {})
    sharpe = best.get("mean_oos_sharpe", 0)
    msg = (
        f"✅ Searched {n} trials over {len(syms)} symbols. "
        f"Best mean OOS Sharpe: {sharpe:+.3f}."
    )
    return results, msg


# ── Render leaderboard from store ───────────────────────────────────────────

@callback(
    Output("sf-leaderboard-area", "children"),
    Input("sf-store-results", "data"),
    prevent_initial_call=True,
)
def show_leaderboard(results):
    return render_leaderboard(results or {})


# ── Optional: Ask Claude (paid, opt-in) ─────────────────────────────────────

@callback(
    Output("sf-claude-status", "children"),
    Input("sf-claude", "n_clicks"),
    State("sf-strategy",      "value"),
    State("sf-store-results", "data"),
    prevent_initial_call=True,
)
def ask_claude(n_clicks, strategy, results):
    if not n_clicks:
        return ""

    if not results or not results.get("leaderboard"):
        return "🤖 Run a search first so Claude has a leaderboard to react to."

    from bot.strategy_finder import suggest_with_claude
    suggestions = suggest_with_claude(
        strategy_id = strategy,
        leaderboard = results.get("leaderboard", []),
        top_n_seed  = 5,
        n_proposals = 3,
    )
    if not suggestions:
        return ("🤖 No suggestions returned — check ANTHROPIC_API_KEY is set "
                "and the network is reachable.")

    bullets = " · ".join(
        f"{s.params}" for s in suggestions[:3]
    )
    return (f"🤖 Claude suggests {len(suggestions)} unconventional combos: "
            f"{bullets}.  Re-run to incorporate them.")


# ── Save modal ─────────────────────────────────────────────────────────────

@callback(
    Output("sf-modal",          "is_open"),
    Output("sf-save-trial-id",  "data"),
    Output("sf-save-id",        "value"),
    Output("sf-save-desc",      "value"),
    Output("sf-save-msg",       "children"),
    Input({"type": "sf-save-row", "trial": ALL}, "n_clicks"),
    Input("sf-modal-cancel",    "n_clicks"),
    Input("sf-modal-save",      "n_clicks"),
    State("sf-store-results",   "data"),
    State("sf-save-trial-id",   "data"),
    State("sf-save-id",         "value"),
    State("sf-save-desc",       "value"),
    State("sf-strategy",        "value"),
    State("sf-modal",           "is_open"),
    prevent_initial_call=True,
)
def handle_save_modal(row_clicks, cancel_clicks, save_clicks,
                       results, trial_id, save_id, save_desc,
                       strategy, is_open):
    ctx = callback_context.triggered[0]
    prop = ctx["prop_id"]

    # Open modal (a "Save" row button was clicked)
    if "sf-save-row" in prop:
        if not ctx["value"]:
            return no_update, no_update, no_update, no_update, no_update
        try:
            trial_no = int(json.loads(prop.split(".")[0])["trial"])
        except Exception:
            return no_update, no_update, no_update, no_update, no_update

        rows = (results or {}).get("leaderboard", [])
        chosen = next((r for r in rows if r["trial"] == trial_no), None)
        if not chosen:
            return False, None, "", "", ""

        suggested_id = f"{strategy}_t{trial_no:03d}"
        suggested_desc = (
            f"Tuned {strategy} (trial {trial_no})  "
            f"OOS Sharpe {chosen['mean_oos_sharpe']:+.3f}, "
            f"return {chosen['mean_oos_return_pct']:+.2f}%"
        )
        return True, trial_no, suggested_id, suggested_desc, ""

    # Cancel
    if "sf-modal-cancel" in prop:
        return False, no_update, "", "", ""

    # Save
    if "sf-modal-save" in prop:
        if not save_id or not re.match(r"^[a-zA-Z0-9_\-]{1,50}$", save_id):
            return True, no_update, no_update, no_update, \
                "❌ id must be 1-50 chars: letters / digits / _ / -."

        rows = (results or {}).get("leaderboard", [])
        chosen = next((r for r in rows if r["trial"] == trial_id), None)
        if not chosen:
            return False, None, "", "", ""

        from bot.strategy_finder import apply_params
        spec = apply_params(strategy, chosen["params"], save_id, save_desc or "")

        target_dir = Path(__file__).resolve().parents[1] / "custom_models"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{save_id}.json"
        with open(target, "w") as f:
            json.dump(spec, f, indent=2)

        # Reset registry so the next list_models() pickup the new file
        return False, None, "", "", f"✅ Saved {target.name}"

    return no_update, no_update, no_update, no_update, no_update
