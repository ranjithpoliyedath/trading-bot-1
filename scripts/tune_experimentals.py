"""
scripts/tune_experimentals.py
------------------------------
Run Optuna against each "experimental" strategy with a full-period
objective (Sharpe + positive-return filter), so we can see whether
non-default params would have cleared the production gate.

Why full-period instead of walk-forward: with the current ~2yr of
on-disk data, 4 walk-forward folds work out to ~29-day OOS chunks.
Most of the experimental strategies don't fire often enough on a
single 29-day window to produce a meaningful Sharpe.  Once the OHLCV
history is deepened (future direction #2), the Finder's walk-forward
default becomes the right choice and this script becomes redundant.

Run:
    python scripts/tune_experimentals.py
    python scripts/tune_experimentals.py --trials 30 --max-symbols 30
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import optuna   # noqa: E402

from bot.universe              import select_universe        # noqa: E402
from bot.strategy_finder       import (                       # noqa: E402
    PARAM_SPACES, _suggest_value, params_to_filters, apply_params,
)
from dashboard.backtest_engine import run_filtered_backtest   # noqa: E402


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("tune_experimentals")


# Strategies flagged as "experimental" in CATALOGUE.md.  Edit if more
# get demoted (or pull from the catalogue programmatically later).
EXPERIMENTAL = [
    "donchian_v1",
    "adx_trend_v1",
    "keltner_breakout_v1",
    "obv_momentum_v1",
]

OUTPUT = Path(__file__).resolve().parents[1] / "dashboard" / "backtests" / "tuned_experimentals.json"


def tune_one(strategy_id: str, symbols: list, n_trials: int,
             period_days: int, seed: int) -> dict:
    space = PARAM_SPACES.get(strategy_id, [])
    if not space:
        return {"strategy": strategy_id, "error": "no param space"}

    best_seen = {"sharpe": -1e9}

    def objective(trial):
        params  = {p["name"]: _suggest_value(trial, p) for p in space}
        filters = params_to_filters(strategy_id, params)
        out = run_filtered_backtest(
            model_id        = strategy_id,
            filters         = filters,
            symbols         = symbols,
            period_days     = period_days,
            conf_threshold  = float(params.get("min_confidence", 0.55)),
        )
        m       = out.get("metrics", {}) or {}
        sharpe  = float(m.get("sharpe", 0)            or 0)
        ret     = float(m.get("total_return_pct", 0)  or 0)
        trades  = int(  m.get("total_trades", 0)      or 0)

        # Only count Sharpe when the strategy actually traded *and* didn't
        # blow up on cumulative return — otherwise return -inf so Optuna
        # avoids that region.
        score = sharpe if (trades >= 5 and ret > -50) else -1e3

        if score > best_seen["sharpe"]:
            best_seen.update({
                "sharpe":   sharpe,
                "return":   ret,
                "trades":   trades,
                "win_rate": float(m.get("win_rate_pct", 0) or 0),
                "params":   params,
            })
        return score

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    if best_seen["sharpe"] == -1e9:
        return {"strategy": strategy_id, "error": "no valid trial"}
    return {"strategy": strategy_id, **best_seen}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials",       type=int, default=20)
    ap.add_argument("--max-symbols",  type=int, default=20)
    ap.add_argument("--period-days",  type=int, default=730)
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    syms = select_universe("top_100", limit=args.max_symbols)
    if not syms:
        logger.error("Universe scan returned no symbols.")
        return 1

    logger.info("Tuning %d strategies on %d symbols, %d trials each",
                len(EXPERIMENTAL), len(syms), args.trials)

    results = []
    for sid in EXPERIMENTAL:
        logger.info("  — %s …", sid)
        r = tune_one(sid, syms, args.trials, args.period_days, args.seed)
        if r.get("error"):
            logger.warning("    %s: %s", sid, r["error"])
        else:
            logger.info("    %s: best Sharpe %+.3f, return %+.2f%%, %d trades, win %.1f%% with %s",
                        sid, r["sharpe"], r["return"], r["trades"],
                        r["win_rate"], r["params"])
        results.append(r)

    # Persist for inspection / later code-update
    payload = {
        "tuned_at": datetime.utcnow().isoformat(),
        "n_symbols": len(syms),
        "n_trials":  args.trials,
        "results":   results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote tuned defaults → %s", OUTPUT)

    print()
    print(f"{'Strategy':24s}  {'Sharpe':>7s}  {'Return':>9s}  {'Trades':>7s}  Pass?  {'Best params'}")
    print("-" * 110)
    for r in results:
        if r.get("error"):
            print(f"{r['strategy']:24s}  {'—':>7s}  {'—':>9s}  {'—':>7s}  fail   {r['error']}")
            continue
        passes = (r["sharpe"] > 0.3) and (r["return"] > 0)
        flag = "PASS" if passes else "fail"
        print(f"{r['strategy']:24s}  {r['sharpe']:+7.3f}  {r['return']:+8.2f}%  "
              f"{r['trades']:7d}  {flag}   {r['params']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
