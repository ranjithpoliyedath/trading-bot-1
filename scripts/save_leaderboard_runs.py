"""
scripts/save_leaderboard_runs.py
---------------------------------
Run every per-symbol strategy on a common universe + period and save
each result to ``dashboard/backtests/`` so they all show up in the
saved-runs dropdown.

Two passes:
  1. **Default-config sweep** — each strategy with its published rules.
     Saved as ``LBRD-YYYYMMDD-<strategy_id>.json``.
  2. **Tuned best configs** — the Optuna-walk-forward winners with
     their best params applied via the screener filter chain.
     Saved as ``TUNED-YYYYMMDD-<name>.json``.

Run:
    python scripts/save_leaderboard_runs.py
    python scripts/save_leaderboard_runs.py --symbols=AAPL,MSFT,NVDA  # smaller universe
    python scripts/save_leaderboard_runs.py --period-days=730        # 2-year window

The dropdown will surface the new runs immediately.  Restart the
dashboard if it's already running.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib  import Path

# Make the project root importable when the script is run as
# ``python scripts/save_leaderboard_runs.py`` (no -m).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Quiet the noisy startup output before any heavy import
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.basicConfig(level=logging.ERROR, format="%(message)s")
logging.getLogger("transformers").setLevel(logging.WARNING)

from dashboard.backtest_engine import run_filtered_backtest, save_backtest
from bot.models.registry         import list_models


# Default top-30 large-cap universe.  All have ≥6 years of data.
DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA",
    "AVGO", "AMD",  "NFLX", "ORCL",  "ADBE", "INTC", "BAC",
    "JPM",  "WFC",  "C",    "XOM",   "CVX",  "COP",  "JNJ",
    "UNH",  "PFE",  "WMT",  "COST",  "HD",   "MCD",  "DIS",
    "NKE",  "SPY",
]


# Tuned-best configs from the Optuna 4-fold walk-forward sweep
# (2026-04-29).  Each entry is a complete backtest spec.
TUNED_CONFIGS = [
    {
        "name":     "keltner_tuned",
        "model_id": "keltner_breakout_v1",
        "filters":  [],
        "conf":     0.81,
        "note":     "Best persistent edge — OOS Sharpe 1.155 (4-fold WF)",
    },
    {
        "name":     "ibs_tuned",
        "model_id": "ibs_v1",
        "filters":  [{"field": "ibs", "op": "<", "value": 0.24}],
        "conf":     0.73,
        "note":     "OOS Sharpe 1.020 (4-fold WF)",
    },
    {
        "name":     "weinstein_tuned",
        "model_id": "weinstein_v1",
        "filters":  [
            {"field": "sma_150_slope_pct", "op": ">", "value": 0.35},
            {"field": "sma_150_extension", "op": "<", "value": 0.10},
        ],
        "conf":     0.85,
        "note":     "OOS Sharpe 0.695 (4-fold WF, after Optuna fix)",
    },
    {
        # Quantitativo mean-reversion tuned for high-precision QQQ
        # entries (84.6% win rate, max DD only -5.9%).  Article:
        # quantitativo.com "A Mean Reversion Strategy with 2.11 Sharpe".
        "name":     "quantitativo_mr_qqq",
        "model_id": "quantitativo_mr_v1",
        "filters":  [
            {"field": "ibs",           "op": "<",  "value": 0.10},
            {"field": "qmr_band_dive", "op": ">=", "value": 0.015},
        ],
        "conf":     0.66,
        "note":     ("Quantitativo MR tuned on QQQ — 4-fold WF OOS "
                     "Sharpe 0.838.  IBS<0.10 + 1.5% below band for "
                     "rare high-precision setups."),
    },
    {
        # User-designed Leaders Breakout, tuned via Optuna walk-forward.
        # OOS Sharpe 1.592 — strongest persistent edge in the suite.
        "name":     "leaders_breakout",
        "model_id": "leaders_breakout_v1",
        "filters":  [
            {"field": "volume_spike_5d_max", "op": ">=", "value": 1.5},
            {"field": "price_change_5d",     "op": ">=", "value": 0.09},
        ],
        "conf":     0.73,
        "note":     ("Leaders Breakout tuned via Optuna 4-fold WF.  "
                     "OOS Sharpe 1.592.  Lower volume floor (1.5x) + "
                     "bigger price-move requirement (9% in 5d) = "
                     "stronger leaders-of-leaders filtering."),
    },
]


def _common_kwargs(symbols, period_days, conf):
    return dict(
        symbols         = symbols,
        period_days     = period_days,
        conf_threshold  = conf,
        starting_cash   = 100_000,
        sizing_method   = "fixed_pct",
        sizing_kwargs   = {"pct": 0.10},
        stop_loss_pct   = 0.07,
        time_stop_days  = 30,
        use_signal_exit = True,
    )


def run_default_sweep(symbols: list[str], period_days: int, today: str):
    """Pass 1 — every per-symbol RULE strategy with default config.

    Custom models are skipped here: they're filter-spec only (no own
    predict_batch) so running them with empty filters produces 0
    trades.  The TUNED- runs cover the curated tuned configs.
    """
    per_symbol = [m.id for m in list_models() if m.type == "rule"]
    print(f"=== Default sweep: {len(per_symbol)} strategies ===")
    for mid in per_symbol:
        try:
            out = run_filtered_backtest(
                model_id=mid, filters=[],
                **_common_kwargs(symbols, period_days, 0.55),
            )
            # Sanitise colons in run_id (custom: prefix → custom_)
            safe = mid.replace(":", "_")
            out["run_id"] = f"LBRD-{today}-{safe}"
            rid = save_backtest(out)
            m = out.get("metrics", {})
            print(f"  ✓ {rid}  "
                  f"trades={m.get('total_trades', 0)}  "
                  f"return={m.get('total_return_pct', 0):+.1f}%  "
                  f"sharpe={m.get('sharpe', 0)}")
        except Exception as exc:
            print(f"  ✗ {mid}: {exc}")


def run_tuned_sweep(symbols: list[str], period_days: int, today: str):
    """Pass 2 — Optuna-tuned best configs."""
    print(f"\n=== Tuned best configs ===")
    for spec in TUNED_CONFIGS:
        try:
            out = run_filtered_backtest(
                model_id = spec["model_id"],
                filters  = spec["filters"],
                **_common_kwargs(symbols, period_days, spec["conf"]),
            )
            out["run_id"] = f"TUNED-{today}-{spec['name']}"
            out.setdefault("preset", {})["seed_note"] = spec["note"]
            rid = save_backtest(out)
            m = out.get("metrics", {})
            print(f"  ✓ {rid}")
            print(f"      {spec['note']}")
            print(f"      In-sample: sharpe={m.get('sharpe', 0)}  "
                  f"return={m.get('total_return_pct', 0):+.1f}%  "
                  f"max_dd={m.get('max_drawdown_pct', 0):+.1f}%  "
                  f"trades={m.get('total_trades', 0)}")
        except Exception as exc:
            print(f"  ✗ {spec['name']}: {exc}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS),
                        help="Comma-separated tickers (default: top-30 large-caps)")
    parser.add_argument("--period-days", type=int, default=365 * 6,
                        help="Lookback in days (default: 6 years)")
    parser.add_argument("--default-only", action="store_true",
                        help="Skip the tuned-config pass")
    parser.add_argument("--tuned-only", action="store_true",
                        help="Skip the default sweep")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    today   = datetime.now().strftime("%Y%m%d")

    print(f"Universe: {len(symbols)} symbols  ·  Period: {args.period_days} days  ·  "
          f"Today: {today}")
    print()

    if not args.tuned_only:
        run_default_sweep(symbols, args.period_days, today)
    if not args.default_only:
        run_tuned_sweep(symbols, args.period_days, today)

    print()
    print("Done.  Saved runs are now visible in the Backtest tab's saved-runs dropdown.")
    print("(Restart `python -m dashboard.app` if the dashboard was already open.)")


if __name__ == "__main__":
    main()
