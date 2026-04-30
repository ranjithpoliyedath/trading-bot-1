"""
bot/strategy_finder.py
-----------------------
Hyperparameter search across registered strategies.

Default driver: **Optuna** (Tree-structured Parzen Estimator).  Free,
deterministic with a seed, no API key required.  Each trial backtests
the strategy through Phase 1's walk-forward folds and reports
mean-OOS-Sharpe — so we optimise for *cross-regime robustness*, not
luck on any one window.

Optional driver: ``suggest_with_claude`` calls the Anthropic API when
the user clicks "🤖 Ask Claude" in the dashboard.  It proposes 3
unconventional combinations (e.g. cross-strategy hybrids) which are
appended to the running Optuna study as fixed trials.  This path is
gated behind an explicit user click so the default cost is $0.

Public surface used by the dashboard:
  * ``param_space(strategy_id)``    — Optuna trial spec for a strategy
  * ``run_optuna(strategy_id, …)``  — main loop, returns leaderboard
  * ``confirm_holdout(top_k, …)``   — re-rank top configs on a held-out tail
  * ``suggest_with_claude(…)``      — opt-in only; returns ParamSuggestions
  * ``apply_params(strategy_id, params)`` — convert a row of trial params
                                            into a CustomRuleModel JSON spec
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Parameter spaces per strategy ──────────────────────────────────────────
#
# Each entry maps a strategy_id to the Optuna trial-builder.  We declare
# search spaces declaratively as a list of tuples so the same metadata
# can be reused for the JSON-spec output and the Anthropic prompt.
#
# Keep ranges grounded in trader heuristics — we're searching the
# neighbourhood of the published default, not from scratch.

PARAM_SPACES: dict[str, list[dict]] = {
    "rsi_macd_v1": [
        {"name": "buy_rsi",        "type": "int",   "low": 15, "high": 40},
        # sell_rsi removed: exits aren't gated by the screener filter
        # chain (only buys are), so an entry here was inert and wasted
        # Optuna trials.  Sell threshold is a class-level constant.
        {"name": "min_confidence", "type": "float", "low": 0.50, "high": 0.85, "step": 0.01},
    ],
    "bollinger_v1": [
        {"name": "lower_band_pct", "type": "float", "low": 0.0,  "high": 0.30, "step": 0.01},
        # upper_band_pct removed: same reason as sell_rsi — sell-side
        # threshold isn't filter-tunable.
        {"name": "min_confidence", "type": "float", "low": 0.50, "high": 0.85, "step": 0.01},
    ],
    "sentiment_v1": [
        {"name": "buy_sentiment",  "type": "float", "low": 0.20, "high": 0.80, "step": 0.05},
        # sell_sentiment removed: filter chain only gates buys, so a
        # sell-threshold param was inert and wasted Optuna trials.
        {"name": "min_news_count", "type": "int",   "low": 1, "high": 15},
        {"name": "min_confidence", "type": "float", "low": 0.50, "high": 0.85, "step": 0.01},
    ],
    "qullamaggie_v1": [
        {"name": "runup_min_pct",  "type": "float", "low": 0.20, "high": 0.50, "step": 0.05},
        {"name": "consol_max_rng", "type": "float", "low": 0.08, "high": 0.20, "step": 0.01},
        {"name": "vol_drop_max",   "type": "float", "low": 0.70, "high": 0.95, "step": 0.05},
        {"name": "min_confidence", "type": "float", "low": 0.55, "high": 0.85, "step": 0.01},
    ],
    "vcp_v1": [
        {"name": "min_contractions", "type": "int",   "low": 2, "high": 4},
        {"name": "consol_max_rng",   "type": "float", "low": 0.08, "high": 0.20, "step": 0.01},
        {"name": "min_confidence",   "type": "float", "low": 0.55, "high": 0.85, "step": 0.01},
    ],
    "golden_cross_v1": [
        # trend_buffer + breakout_period removed: would require model-
        # internal hooks that don't exist yet.  Confidence-only tuning
        # for these strategies until per-strategy param injection ships.
        {"name": "min_confidence", "type": "float", "low": 0.55, "high": 0.85, "step": 0.01},
    ],
    "donchian_v1": [
        {"name": "min_confidence",  "type": "float", "low": 0.55, "high": 0.85, "step": 0.01},
    ],
    "connors_rsi2_v1": [
        {"name": "buy_rsi2",       "type": "int",   "low": 3,  "high": 20},
        {"name": "min_confidence", "type": "float", "low": 0.55, "high": 0.85, "step": 0.01},
    ],
    "ibs_v1": [
        {"name": "buy_ibs_max",    "type": "float", "low": 0.05, "high": 0.30, "step": 0.01},
        {"name": "min_confidence", "type": "float", "low": 0.55, "high": 0.85, "step": 0.01},
    ],
    "adx_trend_v1": [
        {"name": "buy_adx",        "type": "int",   "low": 18, "high": 35},
        {"name": "min_confidence", "type": "float", "low": 0.55, "high": 0.85, "step": 0.01},
    ],
    "keltner_breakout_v1": [
        # atr_multiplier removed: it's consumed at indicator-computation
        # time inside add_keltner, not at signal time, so Optuna can't
        # tune it through the filter chain.  Confidence tuning only.
        {"name": "min_confidence", "type": "float", "low": 0.55, "high": 0.85, "step": 0.01},
    ],
    "obv_momentum_v1": [
        {"name": "min_5d_return",  "type": "float", "low": 0.0,  "high": 0.05, "step": 0.005},
        {"name": "min_confidence", "type": "float", "low": 0.55, "high": 0.85, "step": 0.01},
    ],
    "zscore_reversion_v1": [
        {"name": "buy_zscore",     "type": "float", "low": -3.0, "high": -1.0, "step": 0.1},
        {"name": "min_confidence", "type": "float", "low": 0.55, "high": 0.85, "step": 0.01},
    ],
    # ── 2026-04-29 additions ─────────────────────────────────────
    "tsmom_v1": [
        # The strategy itself uses fixed 252-bar / 50-SMA rules; we
        # tune the entry confidence + an extra "min_mom_threshold"
        # filter so we only buy on stronger momentum readings.
        {"name": "min_mom_12m",     "type": "float", "low": 0.0,  "high": 0.30, "step": 0.02},
        {"name": "min_confidence",  "type": "float", "low": 0.55, "high": 0.85, "step": 0.01},
    ],
    "pct52w_high_v1": [
        # The published rule uses 95% / 85% thresholds.  Tune the
        # buy threshold (within X% of the 52w high) and confidence.
        {"name": "min_pct_52w_high", "type": "float", "low": 0.85, "high": 0.99, "step": 0.01},
        {"name": "min_volume_ratio", "type": "float", "low": 0.80, "high": 1.50, "step": 0.05},
        {"name": "min_confidence",   "type": "float", "low": 0.55, "high": 0.85, "step": 0.01},
    ],
    "recovery_rally_v1": [
        # The entry needs both SMA reclaim and macd_hist > 0; tune
        # how strongly we need each to fire.
        {"name": "min_macd_hist",    "type": "float", "low": 0.0,  "high": 0.50, "step": 0.05},
        {"name": "min_confidence",   "type": "float", "low": 0.55, "high": 0.85, "step": 0.01},
    ],
    "weinstein_v1": [
        # Tune the slope threshold (how aggressively the 30-week MA
        # has to be rising) and how extended above the MA we tolerate.
        {"name": "min_sma_slope",    "type": "float", "low": 0.0,  "high": 0.50, "step": 0.05},
        {"name": "max_extension",    "type": "float", "low": 0.10, "high": 0.40, "step": 0.05},
        {"name": "min_confidence",   "type": "float", "low": 0.55, "high": 0.85, "step": 0.01},
    ],
    "quantitativo_mr_v1": [
        # The article uses fixed 25/10/2.5 + IBS<0.3.  Tune around
        # these neighbourhoods to see if the QQQ-tuned defaults
        # transfer or need re-fitting on the user's universe.
        {"name": "max_ibs",          "type": "float", "low": 0.10, "high": 0.40, "step": 0.05},
        {"name": "min_band_dive",    "type": "float", "low": 0.0,  "high": 0.05, "step": 0.005},
        {"name": "min_confidence",   "type": "float", "low": 0.55, "high": 0.85, "step": 0.01},
    ],
    "leaders_breakout_v1": [
        # All four entry knobs: volume spike multiplier, 5d return
        # threshold, breakout lookback aren't filter-tunable (they
        # determine whether the strategy fires AT ALL), so we tune
        # the post-firing thresholds: volume spike strength,
        # required 5d return, and confidence.  These read from the
        # strategy's emitted columns.
        {"name": "min_volume_spike", "type": "float", "low": 1.5,  "high": 5.0,  "step": 0.25},
        {"name": "min_5d_return",    "type": "float", "low": 0.01, "high": 0.10, "step": 0.005},
        {"name": "min_confidence",   "type": "float", "low": 0.55, "high": 0.85, "step": 0.01},
    ],
}


def param_space(strategy_id: str) -> list[dict]:
    """Return the declared search space for a strategy, or [] if unknown."""
    return PARAM_SPACES.get(strategy_id, [])


def _suggest_value(trial, p: dict):
    """Bridge our declarative spec to the Optuna trial API."""
    name = p["name"]
    t    = p["type"]
    if t == "int":
        return trial.suggest_int(name, p["low"], p["high"])
    if t == "float":
        step = p.get("step")
        if step:
            return trial.suggest_float(name, p["low"], p["high"], step=step)
        return trial.suggest_float(name, p["low"], p["high"])
    if t == "categorical":
        return trial.suggest_categorical(name, p["choices"])
    raise ValueError(f"Unsupported param type: {t}")


# ── Mapping trial params → backtest engine kwargs ───────────────────────────
#
# Most strategies expose their tunables either via class attributes
# (rsi_macd thresholds), filter conditions on the screener, or the
# CustomRuleModel JSON spec.  For Phase 2 v1 we treat them as filter
# overrides on top of the model — this avoids monkey-patching class
# attributes and keeps the loop reproducible.

def params_to_filters(strategy_id: str, params: dict) -> list[dict]:
    """
    Convert a trial's chosen param dict into per-bar screener filters
    that gate the model's buy signals.  This is the "tunable layer" we
    optimise without touching the strategy's published rules.
    """
    if strategy_id == "rsi_macd_v1":
        return [
            {"field": "rsi_14", "op": "<", "value": params["buy_rsi"]},
            # sell_rsi gates exits via the model — skipped here, applied via min_confidence
        ]
    if strategy_id == "bollinger_v1":
        return [
            {"field": "bb_pct", "op": "<", "value": params["lower_band_pct"]},
        ]
    if strategy_id == "sentiment_v1":
        return [
            {"field": "combined_sentiment", "op": ">=", "value": params["buy_sentiment"]},
            {"field": "news_count",         "op": ">=", "value": params["min_news_count"]},
        ]
    if strategy_id == "qullamaggie_v1":
        return [
            {"field": "prior_runup_pct",        "op": ">=", "value": params["runup_min_pct"]},
            {"field": "consolidation_range",    "op": "<=", "value": params["consol_max_rng"]},
            {"field": "consolidation_vol_drop", "op": "<=", "value": params["vol_drop_max"]},
        ]
    if strategy_id == "vcp_v1":
        return [
            {"field": "contraction_count",   "op": ">=", "value": params["min_contractions"]},
            {"field": "consolidation_range", "op": "<=", "value": params["consol_max_rng"]},
        ]
    if strategy_id == "donchian_v1":
        # Donchian period is a class-level tunable; here we approximate by
        # filtering on the existing donchian_high_20 column.  Specific period
        # tuning is left to a future param-injected version.
        return []
    if strategy_id == "connors_rsi2_v1":
        return [
            {"field": "rsi_2", "op": "<", "value": params.get("buy_rsi2", 10)},
        ]
    if strategy_id == "ibs_v1":
        return [
            {"field": "ibs", "op": "<", "value": params.get("buy_ibs_max", 0.20)},
        ]
    if strategy_id == "adx_trend_v1":
        return [
            {"field": "adx_14", "op": ">", "value": params.get("buy_adx", 25)},
        ]
    if strategy_id == "obv_momentum_v1":
        return [
            {"field": "price_change_5d", "op": ">", "value": params.get("min_5d_return", 0.0)},
        ]
    if strategy_id == "zscore_reversion_v1":
        return [
            {"field": "zscore_close_20", "op": "<", "value": params.get("buy_zscore", -1.5)},
        ]
    # ── 2026-04-29 additions ─────────────────────────────────────
    if strategy_id == "tsmom_v1":
        # Require 12-month momentum to clear the tuned threshold
        return [
            {"field": "mom_12m", "op": ">=", "value": params.get("min_mom_12m", 0.0)},
        ]
    if strategy_id == "pct52w_high_v1":
        return [
            {"field": "pct_52w_high", "op": ">=", "value": params.get("min_pct_52w_high", 0.95)},
            {"field": "volume_ratio", "op": ">=", "value": params.get("min_volume_ratio", 1.0)},
        ]
    if strategy_id == "recovery_rally_v1":
        return [
            {"field": "macd_hist", "op": ">=", "value": params.get("min_macd_hist", 0.0)},
        ]
    if strategy_id == "weinstein_v1":
        # Use the price-normalised slope + extension columns the
        # strategy emits in predict_batch.  These are dimensionless
        # ratios so the Optuna thresholds work uniformly across
        # symbols (no $10-stock-vs-$1000-stock scaling issue).
        return [
            {"field": "sma_150_slope_pct",
             "op": ">", "value": params.get("min_sma_slope", 0.0)},
            {"field": "sma_150_extension",
             "op": "<", "value": params.get("max_extension", 0.30)},
        ]
    if strategy_id == "quantitativo_mr_v1":
        # Tune the IBS ceiling + how deeply below the band we want to
        # see before buying.  qmr_band_dive is the strategy-emitted
        # column "(lower_band - close) / close, clipped to >=0",
        # so larger values = more oversold.
        return [
            {"field": "ibs",
             "op":    "<",
             "value": params.get("max_ibs", 0.30)},
            {"field": "qmr_band_dive",
             "op":    ">=",
             "value": params.get("min_band_dive", 0.0)},
        ]
    if strategy_id == "leaders_breakout_v1":
        # Tune the volume-spike floor + 5-day return floor.  Both
        # are emitted by the strategy as filter-friendly columns.
        return [
            {"field": "volume_spike_5d_max",
             "op":    ">=",
             "value": params.get("min_volume_spike", 2.5)},
            {"field": "price_change_5d",
             "op":    ">=",
             "value": params.get("min_5d_return", 0.03)},
        ]
    # golden_cross_v1, keltner_breakout_v1: tuning happens via min_confidence
    # only — the strategies' published rules are kept intact.
    return []


def apply_params(strategy_id: str, params: dict, name: str, description: str = "") -> dict:
    """
    Build a CustomRuleModel JSON spec from a trial result.  This is what
    the dashboard "Save as new strategy" button writes to disk.
    """
    filters = params_to_filters(strategy_id, params)
    spec = {
        "id":             name,
        "name":           name.replace("_", " "),
        "description":    description or f"Tuned {strategy_id} via Strategy Finder",
        "buy_when":       filters,
        "sell_when":      [],          # exits handled by engine rules
        "min_confidence": float(params.get("min_confidence", 0.65)),
        "tuned_from":     strategy_id,
        "tuned_params":   {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                           for k, v in params.items()},
    }
    return spec


# ── Optuna driver ───────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    trial_id:           int
    params:             dict
    mean_oos_sharpe:    float
    mean_oos_return:    float
    pct_positive_folds: float
    total_trades:       int
    fold_metrics:       list[dict] = field(default_factory=list)


def run_optuna(
    strategy_id:    str,
    n_trials:       int             = 30,
    n_folds:        int             = 4,
    symbols:        Optional[list]  = None,
    period_days:    int             = 365 * 6,
    seed:           int             = 42,
    early_stop_after: int           = 10,         # stop if no improvement for N trials
    extra_engine_kwargs: Optional[dict] = None,
    progress_cb:    Optional[Callable[[int, int, float], None]] = None,
) -> dict:
    """
    Run an Optuna TPE search over the strategy's parameter space.

    Each trial is one full walk-forward backtest.  Objective: mean OOS
    Sharpe across folds (maximised).  Early-stop kicks in when the
    running best hasn't improved for ``early_stop_after`` trials —
    saves time on plateaus.

    Returns:
        {
            "strategy":     strategy_id,
            "n_trials":     n_actually_run,
            "best":         {"params": {...}, "mean_oos_sharpe": ...},
            "leaderboard":  pd.DataFrame.to_dict("records") sorted by sharpe desc,
            "study_state":  brief summary (best, n_pruned, n_complete),
            "run_id":       str,
            "run_at":       isoformat,
        }
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    space = param_space(strategy_id)
    if not space:
        return {
            "strategy": strategy_id, "n_trials": 0, "best": {},
            "leaderboard": [], "study_state": {},
            "run_id": "", "run_at": datetime.now().isoformat(),
            "error": f"No parameter space declared for {strategy_id!r}.",
        }

    extra_engine_kwargs = extra_engine_kwargs or {}

    # Local import to avoid heavy module init on cold paths
    from dashboard.backtest_engine import run_walk_forward

    results: list[TrialResult] = []
    best_so_far = -np.inf
    no_improve  = 0

    def objective(trial):
        nonlocal best_so_far, no_improve
        params = {p["name"]: _suggest_value(trial, p) for p in space}
        filters = params_to_filters(strategy_id, params)

        out = run_walk_forward(
            model_id     = strategy_id,
            n_folds      = n_folds,
            symbols      = symbols,
            period_days  = period_days,
            filters      = filters,
            conf_threshold = float(params.get("min_confidence", 0.6)),
            **{k: v for k, v in extra_engine_kwargs.items()
               if k not in {"filters", "conf_threshold"}},
        )
        agg = out.get("aggregate", {}) or {}
        sharpe = float(agg.get("mean_oos_sharpe", 0) or 0)
        ret    = float(agg.get("mean_oos_return_pct", 0) or 0)
        pct_pos = float(agg.get("pct_positive_folds", 0) or 0)

        total_trades = sum(
            int((fr.get("metrics") or {}).get("total_trades", 0) or 0)
            for fr in out.get("fold_results", [])
        )

        results.append(TrialResult(
            trial_id           = trial.number,
            params             = params,
            mean_oos_sharpe    = sharpe,
            mean_oos_return    = ret,
            pct_positive_folds = pct_pos,
            total_trades       = total_trades,
            fold_metrics       = [fr.get("metrics", {}) for fr in out.get("fold_results", [])],
        ))

        if progress_cb:
            try:
                progress_cb(trial.number + 1, n_trials, sharpe)
            except Exception:
                pass

        if sharpe > best_so_far + 1e-9:
            best_so_far = sharpe
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= early_stop_after:
                trial.study.stop()

        return sharpe

    sampler = optuna.samplers.TPESampler(seed=seed)
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    # Build leaderboard sorted by mean OOS Sharpe desc
    leaderboard_rows = []
    for r in sorted(results, key=lambda r: r.mean_oos_sharpe, reverse=True):
        leaderboard_rows.append({
            "trial":                r.trial_id,
            "mean_oos_sharpe":      round(r.mean_oos_sharpe, 3),
            "mean_oos_return_pct":  round(r.mean_oos_return, 2),
            "pct_positive_folds":   round(r.pct_positive_folds, 1),
            "total_trades":         r.total_trades,
            "params":               r.params,
        })

    best = leaderboard_rows[0] if leaderboard_rows else {}

    # study.best_value raises ValueError if no trial completed successfully —
    # use try/except instead of trying to predict it from study.trials state.
    try:
        best_value = round(study.best_value, 3)
    except Exception:
        best_value = 0

    return {
        "strategy":    strategy_id,
        "n_trials":    len(results),
        "best":        best,
        "leaderboard": leaderboard_rows,
        "study_state": {
            "best_value":   best_value,
            "n_complete":   len([t for t in study.trials
                                 if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_pruned":     len([t for t in study.trials
                                 if t.state == optuna.trial.TrialState.PRUNED]),
            "n_failed":     len([t for t in study.trials
                                 if t.state == optuna.trial.TrialState.FAIL]),
            "early_stopped": len(results) < n_trials,
        },
        "run_id":   f"SF-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{strategy_id}",
        "run_at":   datetime.now().isoformat(),
    }


# ── Holdout confirmation ───────────────────────────────────────────────────

def confirm_holdout(
    strategy_id: str,
    top_k_params: list[dict],
    holdout_start: pd.Timestamp,
    holdout_end:   pd.Timestamp,
    symbols:       Optional[list]  = None,
    period_days:   int             = 365 * 6,
    extra_engine_kwargs: Optional[dict] = None,
) -> list[dict]:
    """
    Re-run the top-K configurations on a held-out date range that wasn't
    used by Optuna's walk-forward search.  This is the final
    "did-it-actually-survive-OOS" check.
    """
    from dashboard.backtest_engine import run_filtered_backtest

    extra = extra_engine_kwargs or {}
    rows = []
    for cfg in top_k_params:
        params = cfg.get("params", cfg)
        filters = params_to_filters(strategy_id, params)
        out = run_filtered_backtest(
            model_id        = strategy_id,
            filters         = filters,
            symbols         = symbols,
            period_days     = period_days,
            conf_threshold  = float(params.get("min_confidence", 0.6)),
            date_window     = (holdout_start, holdout_end),
            **{k: v for k, v in extra.items()
               if k not in {"filters", "conf_threshold", "date_window"}},
        )
        m = out.get("metrics", {})
        rows.append({
            "params":       params,
            "holdout_sharpe": round(float(m.get("sharpe", 0) or 0), 3),
            "holdout_return": round(float(m.get("total_return_pct", 0) or 0), 2),
            "holdout_trades": int(m.get("total_trades", 0) or 0),
        })
    return rows


# ── Optional: Anthropic suggestion path (opt-in) ────────────────────────────

@dataclass
class ParamSuggestion:
    params:    dict
    rationale: str


def suggest_with_claude(
    strategy_id:  str,
    leaderboard:  list[dict],
    top_n_seed:   int = 5,
    n_proposals:  int = 3,
    model:        str = "claude-sonnet-4-5",
) -> list[ParamSuggestion]:
    """
    Ask Claude for ``n_proposals`` unconventional parameter combinations,
    given the current leaderboard.  Caller is expected to feed each
    proposal back into a future Optuna trial via ``study.enqueue_trial``.

    Requires ``ANTHROPIC_API_KEY`` in env.  Costs ~$0.05 per call with
    prompt caching.

    Returns an empty list if the SDK or API key are unavailable — the
    dashboard treats that as a soft failure.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — Claude suggestions skipped.")
        return []

    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic SDK not installed — Claude suggestions skipped.")
        return []

    space = param_space(strategy_id)
    if not space:
        return []

    client = anthropic.Anthropic(api_key=api_key)

    # Top trials seed Claude with what worked.  Cache the static system prompt.
    seed_rows = leaderboard[:top_n_seed]

    space_blob = "\n".join(
        f"  - {p['name']:20s} {p['type']:6s} "
        f"range=[{p.get('low','-')}, {p.get('high','-')}]"
        f"{(' step=' + str(p['step'])) if p.get('step') else ''}"
        for p in space
    )
    leaderboard_blob = "\n".join(
        f"  trial {r['trial']:3d}  sharpe={r['mean_oos_sharpe']:+.3f}  "
        f"ret={r['mean_oos_return_pct']:+.2f}%  params={r['params']}"
        for r in seed_rows
    )

    system = [{
        "type": "text",
        "text": (
            "You are a quantitative trading strategy parameter advisor.  "
            "You suggest unconventional parameter combinations the user "
            "might not try otherwise — cross-regime hedges, contrarian "
            "settings, hybrid filters.  Always call the "
            "`propose_params` tool exactly once.  Each proposal must "
            "be a dict whose keys match the declared parameter space "
            "exactly, with values inside the declared ranges.\n\n"
            f"Strategy: {strategy_id}\n"
            f"Parameter space:\n{space_blob}\n"
        ),
        "cache_control": {"type": "ephemeral"},
    }]

    tool = {
        "name": "propose_params",
        "description": f"Propose {n_proposals} parameter combinations to try next.",
        "input_schema": {
            "type": "object",
            "properties": {
                "proposals": {
                    "type": "array",
                    "minItems": n_proposals,
                    "maxItems": n_proposals,
                    "items": {
                        "type": "object",
                        "properties": {
                            "params":    {"type": "object"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["params", "rationale"],
                    },
                },
            },
            "required": ["proposals"],
        },
    }

    user_msg = (
        "Top trials so far on this strategy:\n"
        f"{leaderboard_blob or '  (none yet — propose 3 starter combinations)'}\n\n"
        f"Propose exactly {n_proposals} unconventional next-tries.  "
        "Don't just interpolate the leaderboard — try edge cases, "
        "regime hedges, or under-explored corners of the space."
    )

    try:
        response = client.messages.create(
            model       = model,
            max_tokens  = 1024,
            system      = system,
            tools       = [tool],
            tool_choice = {"type": "tool", "name": "propose_params"},
            messages    = [{"role": "user", "content": user_msg}],
        )
    except Exception as exc:
        logger.warning("Anthropic call failed: %s", exc)
        return []

    block = next(
        (b for b in response.content if getattr(b, "type", None) == "tool_use"),
        None,
    )
    if block is None:
        return []
    proposals = (block.input or {}).get("proposals", [])

    # Coerce + clip values into the declared ranges
    valid: list[ParamSuggestion] = []
    for prop in proposals:
        params = prop.get("params", {}) or {}
        clean = {}
        for p in space:
            if p["name"] not in params:
                continue
            v = params[p["name"]]
            try:
                if p["type"] == "int":
                    v = int(round(float(v)))
                else:
                    v = float(v)
                if "low"  in p: v = max(v, p["low"])
                if "high" in p: v = min(v, p["high"])
                clean[p["name"]] = v
            except (TypeError, ValueError):
                continue
        if clean:
            valid.append(ParamSuggestion(
                params    = clean,
                rationale = prop.get("rationale", ""),
            ))
    return valid
