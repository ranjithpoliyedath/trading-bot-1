"""
scripts/rank_strategies.py
---------------------------
Bulk-rank every registered strategy through Phase-1 walk-forward
validation and write a leaderboard json to
``dashboard/backtests/seed_leaderboard.json``.

This file is what the Strategy Finder reads on cold open so the user
sees a ranked table of strategies before running their first search.

Run:
    python scripts/rank_strategies.py
    python scripts/rank_strategies.py --scope sp500 --max-symbols 30 --folds 4
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Project root on sys.path so this works as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.models.registry              import list_models   # noqa: E402
from bot.universe                      import select_universe  # noqa: E402
from dashboard.backtest_engine         import (                # noqa: E402
    run_filtered_backtest, run_walk_forward, run_cross_sectional_backtest,
)


warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rank_strategies")


OUTPUT = Path(__file__).resolve().parents[1] / "dashboard" / "backtests" / "seed_leaderboard.json"


def _format_metrics(out: dict, strategy_id: str, kind: str) -> dict:
    if kind == "walk_forward":
        agg = out.get("aggregate", {}) or {}
        n_folds = len(out.get("fold_results", []))
        return {
            "strategy":            strategy_id,
            "validation":          "walk_forward",
            "n_folds":             n_folds,
            "mean_oos_sharpe":     float(agg.get("mean_oos_sharpe",    0) or 0),
            "median_oos_sharpe":   float(agg.get("median_oos_sharpe",  0) or 0),
            "stdev_oos_sharpe":    float(agg.get("stdev_oos_sharpe",   0) or 0),
            "pct_positive_folds":  float(agg.get("pct_positive_folds", 0) or 0),
            "mean_oos_return_pct": float(agg.get("mean_oos_return_pct",0) or 0),
            "trades":              sum(int((fr.get("metrics") or {}).get("total_trades", 0) or 0)
                                        for fr in out.get("fold_results", [])),
        }
    # fallback: full-period
    m = out.get("metrics", {}) or {}
    return {
        "strategy":          strategy_id,
        "validation":        "full_period",
        "n_folds":           1,
        "mean_oos_sharpe":   float(m.get("sharpe", 0) or 0),
        "median_oos_sharpe": float(m.get("sharpe", 0) or 0),
        "stdev_oos_sharpe":  0.0,
        "pct_positive_folds": 100.0 if m.get("total_return_pct", 0) > 0 else 0.0,
        "mean_oos_return_pct": float(m.get("total_return_pct", 0) or 0),
        "trades":            int(m.get("total_trades", 0) or 0),
        "sharpe":            float(m.get("sharpe",       0) or 0),
        "return_pct":        float(m.get("total_return_pct",0) or 0),
        "win_rate_pct":      float(m.get("win_rate_pct",  0) or 0),
        "max_drawdown_pct":  float(m.get("max_drawdown_pct",0) or 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope",       default="top_100",
                    help="Universe scope (default: top_100)")
    ap.add_argument("--max-symbols", type=int, default=20,
                    help="Cap symbols scanned per strategy")
    ap.add_argument("--folds",       type=int, default=4,
                    help="Walk-forward folds (set 0 to use full-period instead)")
    ap.add_argument("--period-days", type=int, default=365 * 6)
    ap.add_argument("--conf-threshold", type=float, default=0.55)
    args = ap.parse_args()

    syms = select_universe(args.scope, limit=args.max_symbols)
    if not syms:
        logger.error("No symbols matched scope=%s", args.scope)
        return 1

    logger.info("Ranking %d strategies on %d symbols (scope=%s)",
                len(list_models()), len(syms), args.scope)

    rows = []
    for meta in list_models():
        # Skip user-saved custom models (those get scored individually).
        if meta.id.startswith("custom:"):
            continue
        try:
            if meta.type == "cross_sectional":
                # Cross-sectional models have a different runner.  No
                # walk-forward yet — the panel-rank approach makes IS/OOS
                # split more involved; left as a future extension.
                out = run_cross_sectional_backtest(
                    model_id     = meta.id,
                    symbols      = syms,
                    period_days  = args.period_days,
                )
                row = _format_metrics(out, meta.id, "full_period")
                row["validation"] = "cross_sectional"
            elif args.folds > 0:
                out = run_walk_forward(
                    model_id        = meta.id,
                    n_folds         = args.folds,
                    symbols         = syms,
                    period_days     = args.period_days,
                    conf_threshold  = args.conf_threshold,
                )
                row = _format_metrics(out, meta.id, "walk_forward")
            else:
                out = run_filtered_backtest(
                    model_id        = meta.id,
                    filters         = [],
                    symbols         = syms,
                    period_days     = args.period_days,
                    conf_threshold  = args.conf_threshold,
                )
                row = _format_metrics(out, meta.id, "full_period")

            row["model_name"]        = meta.name
            row["model_description"] = meta.description
            rows.append(row)
            logger.info("  %s  sharpe=%+5.3f  return=%+6.2f%%  trades=%d",
                        meta.id, row["mean_oos_sharpe"],
                        row["mean_oos_return_pct"], row["trades"])
        except Exception as exc:
            logger.warning("  %s failed: %s", meta.id, exc)

    rows.sort(key=lambda r: r["mean_oos_sharpe"], reverse=True)

    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "scope":        args.scope,
        "n_symbols":    len(syms),
        "folds":        args.folds,
        "period_days":  args.period_days,
        "leaderboard":  rows,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote leaderboard → %s", OUTPUT)

    print()
    print(f"{'Rank':4s}  {'Strategy':24s}  {'Sharpe':>7s}  {'Return':>9s}  {'Trades':>7s}")
    print("-" * 70)
    for k, r in enumerate(rows, start=1):
        print(f"{k:4d}  {r['strategy']:24s}  {r['mean_oos_sharpe']:+7.3f}  "
              f"{r['mean_oos_return_pct']:+8.2f}%  {r['trades']:7d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
