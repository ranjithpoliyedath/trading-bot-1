"""
dashboard/services/data_status.py
---------------------------------
Service layer for the Data Status dashboard page.  Two responsibilities:

  1. ``get_coverage_summary()``  — synchronous read of ``data/processed/``.
     Returns a dict the page renders (total parquets, depth buckets,
     freshness — when was the most recent bar fetched / when was the
     newest parquet last written).

  2. ``start_pipeline_async()`` / ``get_pipeline_status()`` — kicks off
     ``bot.pipeline.run_pipeline`` in a background thread so the dashboard
     stays responsive while the ~1-minute incremental refresh runs.
     A module-level singleton tracks state across requests.

The async runner is single-flight: clicking "Update Now" while a run is
already in progress is a no-op (the second click returns the existing
job's status).  This is what we want — we don't need to parallelise
yfinance fetches, and avoiding race conditions on the data store is
worth more than letting a refresh run twice.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from bot.config import DATA_DIR

logger = logging.getLogger(__name__)


# ── Coverage summary ────────────────────────────────────────────────────────

def get_coverage_summary() -> dict:
    """Walk ``data/processed/`` and compute a one-shot snapshot of the
    on-disk OHLCV state.

    Returns a dict with:

        total_features    int   — number of *_features.parquet files
        total_raw         int   — number of *_raw.parquet files
        depth_buckets     dict  — {label: count} bar-count distribution
        deepest_symbol    str   — example of a 9y+ symbol
        shallowest_count  int   — bars in the shortest non-empty parquet
        earliest_data     str   — ISO date of the oldest bar across all
        latest_bar_date   str   — ISO date of the newest bar across all
        latest_mtime      str   — ISO timestamp of the most-recent fetch
                                  write (i.e., "data was refreshed at X")
        size_mb           float — total disk footprint (MB)
    """
    data_dir = Path(DATA_DIR)
    if not data_dir.exists():
        return _empty_summary()

    feat_files = sorted(data_dir.glob("*_features.parquet"))
    raw_files  = sorted(data_dir.glob("*_raw.parquet"))

    if not feat_files:
        return _empty_summary()

    # Walk each features parquet.  Reading just the index is ~1ms each
    # — fine for 1500 files.  We read just the close column to avoid
    # materialising the full feature set.
    depths      = []
    deepest     = ("", 0)
    shallowest  = ("", 10**9)
    earliest    = None        # oldest bar across all parquets
    latest_bar  = None        # newest bar across all parquets
    latest_mtime = None       # most recent file write
    total_bytes = 0

    for p in feat_files:
        try:
            df = pd.read_parquet(p, columns=["close"])
        except Exception as exc:
            logger.debug("Skipping %s — %s", p, exc)
            continue
        if df.empty:
            continue
        sym = p.stem.replace("_features", "")
        n   = len(df)
        depths.append(n)
        if n > deepest[1]:
            deepest = (sym, n)
        if n < shallowest[1]:
            shallowest = (sym, n)

        # tz-naive normalise so min/max comparisons across all parquets work
        idx_min = df.index.min()
        idx_max = df.index.max()
        if hasattr(idx_min, "tz_localize") and idx_min.tz is not None:
            idx_min = idx_min.tz_localize(None)
            idx_max = idx_max.tz_localize(None)
        if earliest is None or idx_min < earliest:
            earliest = idx_min
        if latest_bar is None or idx_max > latest_bar:
            latest_bar = idx_max

        try:
            total_bytes += p.stat().st_size
            mt = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            if latest_mtime is None or mt > latest_mtime:
                latest_mtime = mt
        except OSError:
            pass

    for p in raw_files:
        try:
            total_bytes += p.stat().st_size
        except OSError:
            pass

    # Bucket depths
    buckets = {
        "<2y (recent IPO)":       0,
        "2-6y":                   0,
        "6-8y":                   0,
        "8y+ (deep)":             0,
    }
    for d in depths:
        if d < 500:    buckets["<2y (recent IPO)"]  += 1
        elif d < 1500: buckets["2-6y"]               += 1
        elif d < 2000: buckets["6-8y"]               += 1
        else:          buckets["8y+ (deep)"]         += 1

    return {
        "total_features":   len(feat_files),
        "total_raw":        len(raw_files),
        "depth_buckets":    buckets,
        "deepest_symbol":   f"{deepest[0]} ({deepest[1]:,} bars)",
        "shallowest":       f"{shallowest[0]} ({shallowest[1]:,} bars)",
        "earliest_data":    earliest.strftime("%Y-%m-%d") if earliest else "",
        "latest_bar_date":  latest_bar.strftime("%Y-%m-%d") if latest_bar else "",
        "latest_mtime":     latest_mtime.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
                              if latest_mtime else "",
        "size_mb":          total_bytes / (1024 * 1024),
    }


def _empty_summary() -> dict:
    return {
        "total_features":  0, "total_raw":       0,
        "depth_buckets":   {},
        "deepest_symbol":  "—",  "shallowest":  "—",
        "earliest_data":   "",   "latest_bar_date":  "",
        "latest_mtime":    "",
        "size_mb":         0.0,
    }


# ── Async pipeline runner (single-flight) ───────────────────────────────────

# Module-level state — simple is fine since we only ever have one runner.
# Lock guards transitions between idle / running / done.
_run_lock     = threading.Lock()
_run_thread:  Optional[threading.Thread] = None
_run_state    = {
    "status":      "idle",      # idle / running / done / failed
    "started_at":  None,        # ISO timestamp
    "finished_at": None,
    "duration_s":  None,
    "counts":      None,        # dict from run_pipeline
    "error":       None,        # str if failed
    "trigger":     None,        # "manual" / "scheduled"
}


def get_pipeline_status() -> dict:
    """Return a copy of the current pipeline run state.  Safe to call
    from any thread (including Dash callbacks)."""
    with _run_lock:
        return dict(_run_state)


def start_pipeline_async(trigger: str = "manual") -> dict:
    """Kick off ``run_pipeline(--full-universe --source=yfinance)`` in
    a background thread.  Returns the new run state dict.

    If a run is already in progress this is a no-op — the existing
    run's state is returned.  Callers should poll ``get_pipeline_status``
    to see progress; the dashboard does this via ``dcc.Interval``.
    """
    global _run_thread

    with _run_lock:
        if _run_state["status"] == "running":
            logger.info("Pipeline run requested but one is already in progress.")
            return dict(_run_state)

        # Reset state for the new run
        _run_state.update({
            "status":      "running",
            "started_at":  datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "duration_s":  None,
            "counts":      None,
            "error":       None,
            "trigger":     trigger,
        })

    t = threading.Thread(target=_run_pipeline_safe, daemon=True,
                          name="data-pipeline")
    t.start()
    _run_thread = t

    return get_pipeline_status()


def _run_pipeline_safe() -> None:
    """Thread body — call run_pipeline, capture result, update state.
    All exceptions are caught and reported via the state dict so the
    dashboard always sees a deterministic terminal state."""
    started = datetime.now(timezone.utc)

    try:
        # Lazy import — keeps the dashboard cold-start fast and avoids
        # pulling Alpaca/yfinance / pyarrow into the Dash worker until
        # someone actually clicks "Update Now".
        from bot.pipeline import run_pipeline
        from bot.universe import get_all_for_data_fetch
        from bot.config   import DATA_LOOKBACK_DAYS

        symbols = get_all_for_data_fetch()
        logger.info("Manual pipeline run started — %d symbols at %dy depth.",
                    len(symbols), DATA_LOOKBACK_DAYS // 365)

        counts = run_pipeline(
            symbols=symbols,
            lookback_days=DATA_LOOKBACK_DAYS,
            incremental=True,
            source="yfinance",
        )

        finished = datetime.now(timezone.utc)
        with _run_lock:
            _run_state.update({
                "status":      "done",
                "finished_at": finished.isoformat(),
                "duration_s":  (finished - started).total_seconds(),
                "counts":      counts,
                "error":       None,
            })
        logger.info("Manual pipeline run done — %s", counts)

    except Exception as exc:
        finished = datetime.now(timezone.utc)
        logger.exception("Manual pipeline run FAILED")
        with _run_lock:
            _run_state.update({
                "status":      "failed",
                "finished_at": finished.isoformat(),
                "duration_s":  (finished - started).total_seconds(),
                "counts":      None,
                "error":       str(exc),
            })


# ── Schedule introspection ──────────────────────────────────────────────────

def get_schedule_info() -> dict:
    """Look up the configured launchd / cron schedule.  Returns ``None``
    fields when neither is installed — the page renders accordingly."""
    info: dict = {"launchd_loaded": False, "cron_installed": False,
                  "next_run_hint": None}

    # launchd
    plist = Path.home() / "Library" / "LaunchAgents" / "com.trading-bot-1.pipeline.plist"
    if plist.exists():
        info["launchd_loaded"] = True
        info["launchd_plist"]  = str(plist)

    # cron — read crontab, look for our tag
    try:
        import subprocess
        out = subprocess.run(["crontab", "-l"], capture_output=True,
                              text=True, timeout=2)
        if out.returncode == 0 and "trading-bot-1" in out.stdout:
            info["cron_installed"] = True
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    # If anything is scheduled, the next-run hint is "next 06:30 weekday"
    # — both installers use the same time.
    if info["launchd_loaded"] or info["cron_installed"]:
        now    = datetime.now()
        target = now.replace(hour=6, minute=30, second=0, microsecond=0)
        if target <= now:
            target = target.replace(day=target.day + 1)
        # Skip Saturday (weekday=5) and Sunday (weekday=6)
        while target.weekday() >= 5:
            target = target.replace(day=target.day + 1)
        info["next_run_hint"] = target.strftime("%a %Y-%m-%d %H:%M")

    return info
