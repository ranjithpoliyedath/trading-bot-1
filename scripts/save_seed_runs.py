"""
scripts/save_seed_runs.py
--------------------------
Run each leaderboard-winning strategy with the production-validated
config and save the result as a ``SEED-*.json`` snapshot in
``dashboard/backtests/``.  These show up in the dashboard's
"Saved runs" dropdown so the user can replay any of them with one
click.

Each snapshot includes the run's full config, equity curve, trade log,
and metrics — same shape as a user-saved run, but tagged ``SEED-`` so
.gitignore can keep them checked in alongside seed_leaderboard.json.

Run:
    python scripts/save_seed_runs.py
    python scripts/save_seed_runs.py --max-symbols 50 --period-days 2190
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

from bot.universe              import select_universe                 # noqa: E402
from dashboard.backtest_engine import (                                # noqa: E402
    run_filtered_backtest, run_walk_forward,
    run_cross_sectional_backtest,
    save_backtest,
)


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("save_seed_runs")


# ── Seed-run roster ────────────────────────────────────────────────────────
#
# Each entry is the production-validated config for one strategy.
# These were the configurations that produced the snapshot leaderboard
# in dashboard/backtests/seed_leaderboard.json.
#
# Tag is what the user sees in the saved-runs dropdown — keep it short.

SEEDS = [
    # Top-5 from the Phase-5 leaderboard
    {
        "tag":          "TOP1-jt-momentum",
        "kind":         "cross_sectional",
        "model_id":     "jt_momentum_v1",
        "label":        "JT 12-1 momentum  ·  Sharpe 0.90, +800% (cross-sectional)",
        "kwargs":       dict(top_decile=0.20, rebalance_days=21,
                              starting_cash=10_000, slippage_bps=5),
    },
    {
        "tag":          "TOP2-adx-trend",
        "kind":         "single_period",
        "model_id":     "adx_trend_v1",
        "label":        "ADX trend filter  ·  Sharpe 0.66, +381%",
        "kwargs":       dict(filters=[], conf_threshold=0.55),
    },
    {
        "tag":          "TOP3-golden-cross",
        "kind":         "single_period",
        "model_id":     "golden_cross_v1",
        "label":        "Golden Cross 50/200  ·  Sharpe 0.63, +344%",
        "kwargs":       dict(filters=[], conf_threshold=0.55),
    },
    {
        "tag":          "TOP4-connors-rsi2",
        "kind":         "single_period",
        "model_id":     "connors_rsi2_v1",
        "label":        "Connors RSI(2)  ·  Sharpe 0.61, +312%",
        "kwargs":       dict(filters=[], conf_threshold=0.55),
    },
    {
        "tag":          "TOP5-ibs",
        "kind":         "single_period",
        "model_id":     "ibs_v1",
        "label":        "Internal Bar Strength  ·  Sharpe 0.61, +331%",
        "kwargs":       dict(filters=[], conf_threshold=0.55),
    },

    # Phase 4 — RSI/MACD baseline
    {
        "tag":          "P4-rsi-macd-baseline",
        "kind":         "single_period",
        "model_id":     "rsi_macd_v1",
        "label":        "Phase 4: RSI+MACD baseline  ·  Sharpe 0.41, +63%",
        "kwargs":       dict(filters=[], conf_threshold=0.55),
    },

    # Phase 3 — Connors RSI(2) walk-forward (showcases Phase 1's validation)
    {
        "tag":          "P3-connors-walkforward",
        "kind":         "walk_forward",
        "model_id":     "connors_rsi2_v1",
        "label":        "Phase 3: Connors RSI(2) walk-forward (4 folds)",
        "kwargs":       dict(n_folds=4, conf_threshold=0.55,
                              execution_model="next_open", slippage_bps=5),
    },

    # Phase 2 — Strategy Finder showcase (rsi_macd tuned with Optuna defaults)
    {
        "tag":          "P2-rsi-macd-tuned",
        "kind":         "single_period",
        "model_id":     "rsi_macd_v1",
        "label":        "Phase 2: RSI+MACD tuned via Strategy Finder",
        "kwargs":       dict(filters=[
            {"field": "rsi_14", "op": "<", "value": 30},
        ], conf_threshold=0.55),
    },

    # Phase 1 — realism showcase: same model, both execution paths to compare
    {
        "tag":          "P1-realism-next-open",
        "kind":         "single_period",
        "model_id":     "ibs_v1",
        "label":        "Phase 1: IBS with next-open + 5bps slippage",
        "kwargs":       dict(filters=[], conf_threshold=0.55,
                              execution_model="next_open", slippage_bps=5),
    },
    {
        "tag":          "P1-realism-same-close",
        "kind":         "single_period",
        "model_id":     "ibs_v1",
        "label":        "Phase 1: IBS with legacy same-close (compare)",
        "kwargs":       dict(filters=[], conf_threshold=0.55,
                              execution_model="same_close", slippage_bps=0),
    },

    # Bollinger + sentiment legacy
    {
        "tag":          "LEGACY-bollinger-sentiment",
        "kind":         "single_period",
        "model_id":     "bollinger_v1",
        "label":        "Legacy: Bollinger + sentiment combo",
        "kwargs":       dict(filters=[], conf_threshold=0.55),
    },

    # Z-score reversion
    {
        "tag":          "ZSCORE-reversion",
        "kind":         "single_period",
        "model_id":     "zscore_reversion_v1",
        "label":        "Z-score reversion (stat-arb mean reversion)",
        "kwargs":       dict(filters=[], conf_threshold=0.55),
    },

    # OBV momentum (promoted from experimental)
    {
        "tag":          "OBV-momentum",
        "kind":         "single_period",
        "model_id":     "obv_momentum_v1",
        "label":        "OBV momentum (promoted from experimental)",
        "kwargs":       dict(filters=[], conf_threshold=0.55),
    },
]

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "dashboard" / "backtests"


def _override_run_id(out: dict, tag: str, label: str) -> dict:
    """Rewrite the run_id so the saved-runs dropdown sorts our seeds
    together at the top, and use the human-readable label as the
    saved-run summary text."""
    if not out:
        return out
    out["run_id"] = f"SEED-{tag}"
    out["label"]  = label
    return out


def _run_one(seed: dict, syms: list, period_days: int) -> dict | None:
    kind = seed["kind"]
    try:
        if kind == "cross_sectional":
            out = run_cross_sectional_backtest(
                model_id    = seed["model_id"],
                symbols     = syms,
                period_days = period_days,
                **seed["kwargs"],
            )
        elif kind == "walk_forward":
            out = run_walk_forward(
                model_id    = seed["model_id"],
                symbols     = syms,
                period_days = period_days,
                **seed["kwargs"],
            )
        else:  # single_period
            out = run_filtered_backtest(
                model_id    = seed["model_id"],
                symbols     = syms,
                period_days = period_days,
                **seed["kwargs"],
            )
    except Exception as exc:
        logger.exception("  %s failed: %s", seed["tag"], exc)
        return None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-symbols", type=int, default=30)
    ap.add_argument("--period-days", type=int, default=365 * 6)
    args = ap.parse_args()

    syms = select_universe("top_100", limit=args.max_symbols)
    if not syms:
        logger.error("Universe scan returned no symbols.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Wipe any existing SEED-*.json so we don't accumulate stale ones
    for p in OUTPUT_DIR.glob("SEED-*.json"):
        p.unlink()

    logger.info("Generating %d seed runs over %d symbols, %d-day lookback",
                len(SEEDS), len(syms), args.period_days)

    saved = []
    for seed in SEEDS:
        logger.info("  %-26s — %s", seed["tag"], seed["model_id"])
        out = _run_one(seed, syms, args.period_days)
        if out is None:
            logger.warning("    skipped (engine returned None)")
            continue
        # Walk-forward outputs have fold_results / aggregate instead of metrics
        has_payload = bool(out.get("metrics")) or bool(out.get("fold_results"))
        if not has_payload:
            logger.warning("    skipped (empty result envelope)")
            continue

        out = _override_run_id(out, seed["tag"], seed["label"])
        out["seed_meta"] = {"tag": seed["tag"], "label": seed["label"],
                            "kind": seed["kind"], "model_id": seed["model_id"]}

        path = OUTPUT_DIR / f"{out['run_id']}.json"
        path.write_text(json.dumps(out, indent=2, default=str))
        saved.append((seed["tag"], seed["label"]))
        logger.info("    saved  → %s", path.name)

    print()
    print("Saved seed runs (visible in the Backtest tab's Saved runs dropdown):")
    print("-" * 90)
    for tag, label in saved:
        print(f"  SEED-{tag:24s}  {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
