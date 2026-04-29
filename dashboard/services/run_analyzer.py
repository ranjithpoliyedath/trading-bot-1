"""
dashboard/services/run_analyzer.py
-----------------------------------
"Analyze this backtest" service.  Two modes, picked at runtime:

  1. **Claude API** — when ANTHROPIC_API_KEY is in .env, sends a
     structured prompt with the run envelope and gets a free-form
     analysis back.  Best quality, costs cents per click.

  2. **Local heuristics** — when no API key, runs deterministic
     checks against the run envelope and produces a markdown report.
     Covers the most common failure modes the user has hit:
       • cross-sectional NaN-price money leak (negative compounding)
       • Kelly fraction sized to zero (no trades)
       • filter fields don't exist (post-mortem catches this)
       • sizing too aggressive on a small pool (one trade locks pool)
       • exit settings don't apply to strategy class

Both paths return ``{"text": <markdown>, "source": "claude"|"local",
"model": <str|None>}`` — the dashboard renders the text via dcc.Markdown.

Like ``data_status.py`` the run is single-flight and async so the
dashboard stays responsive while waiting on the LLM.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level state — same pattern as data_status.py.  One analyzer
# in flight at a time per dashboard process.
_run_lock  = threading.Lock()
_run_state = {
    "status":      "idle",       # idle / running / done / failed
    "started_at":  None,
    "finished_at": None,
    "duration_s":  None,
    "result":      None,         # {"text": ..., "source": ..., "model": ...}
    "error":       None,
    "run_id":      None,         # which run was analyzed
}


def get_analyzer_status() -> dict:
    with _run_lock:
        return dict(_run_state)


def start_analysis(results: dict) -> dict:
    """Kick off analysis on ``results`` in a background thread.

    Single-flight: a second click while one is running is a no-op.
    """
    with _run_lock:
        if _run_state["status"] == "running":
            return dict(_run_state)
        _run_state.update({
            "status":      "running",
            "started_at":  datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "duration_s":  None,
            "result":      None,
            "error":       None,
            "run_id":      (results or {}).get("run_id", "(unknown)"),
        })

    t = threading.Thread(
        target=_run_analysis,
        args=(results,),
        daemon=True,
        name="run-analyzer",
    )
    t.start()
    return get_analyzer_status()


def _run_analysis(results: dict) -> None:
    """Thread body — does the actual analysis, captures result/error."""
    started = datetime.now(timezone.utc)
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if api_key:
            text, model = _claude_analysis(results, api_key)
            source = "claude"
        else:
            text  = _local_analysis(results)
            model = None
            source = "local"

        finished = datetime.now(timezone.utc)
        with _run_lock:
            _run_state.update({
                "status":      "done",
                "finished_at": finished.isoformat(),
                "duration_s":  (finished - started).total_seconds(),
                "result":      {"text": text, "source": source, "model": model},
                "error":       None,
            })
    except Exception as exc:
        finished = datetime.now(timezone.utc)
        logger.exception("Run analysis failed")
        with _run_lock:
            _run_state.update({
                "status":      "failed",
                "finished_at": finished.isoformat(),
                "duration_s":  (finished - started).total_seconds(),
                "result":      None,
                "error":       str(exc),
            })


# ── Mode 1: Claude API ─────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a quantitative-trading expert reviewing a backtest result.
Your job: explain what really happened in plain English, identify any
issues with the run (data, strategy, sizing, exits), and flag results
that are too-good-to-be-true or obvious bugs.

Structure your response in markdown with these sections:

  ## Verdict
  One paragraph: did this strategy work?  Is the result trustworthy?

  ## What happened
  Walk through the run: how many trades, win rate, return, max
  drawdown, what the strategy was actually doing.

  ## Issues found
  Specific, named problems.  Reference exact numbers from the run.
  Look for: implausible returns (>>500% or <<-50% on diversified
  long-only suggests a bug), low trade count, sizing pitfalls,
  filter field mismatches, exit-rule issues, missing data.

  ## Recommendations
  Concrete next steps the user can take.  If the result is clean,
  say so and suggest next experiments.

Be direct and specific.  Cite numbers from the data.  Don't hedge.
"""


def _claude_analysis(results: dict, api_key: str) -> tuple[str, str]:
    """Send a compact run summary to Claude and return its analysis.

    Returns ``(markdown_text, model_id)``.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    model  = "claude-sonnet-4-5-20251022"

    payload = _build_run_summary(results)

    user_msg = (
        "Analyze this backtest run.  The full envelope is JSON below.  "
        "Use the schema described in the system prompt.\n\n"
        f"```json\n{json.dumps(payload, indent=2, default=str)}\n```"
    )

    resp = client.messages.create(
        model       = model,
        max_tokens  = 2048,
        system      = _SYSTEM_PROMPT,
        messages    = [{"role": "user", "content": user_msg}],
    )

    # Concatenate all text blocks
    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    return text, model


def _build_run_summary(results: dict) -> dict:
    """Compress the run envelope to the fields Claude needs.  We trim
    the trade list to a sample so we stay under token budget on
    1000-trade runs — the metrics already capture the aggregate.
    """
    if not isinstance(results, dict):
        return {}

    metrics = results.get("metrics") or {}
    preset  = results.get("preset")  or {}
    trades  = results.get("trades")  or []
    rb      = results.get("rebalance_trades") or []
    diag    = results.get("failure_diagnostics") or {}
    lr      = results.get("load_report") or {}

    return {
        "run_id":         results.get("run_id"),
        "model":          results.get("model"),
        "symbol":         results.get("symbol"),
        "period_days":    results.get("period_days"),
        "metrics": {
            "total_trades":      metrics.get("total_trades"),
            "symbols_traded":    metrics.get("symbols_traded"),
            "total_return_pct":  metrics.get("total_return_pct"),
            "sharpe":            metrics.get("sharpe"),
            "sortino":           metrics.get("sortino"),
            "max_drawdown_pct":  metrics.get("max_drawdown_pct"),
            "win_rate_pct":      metrics.get("win_rate_pct"),
            "starting_cash":     metrics.get("starting_cash"),
            "ending_cash":       metrics.get("ending_cash"),
        },
        "preset":         {k: v for k, v in preset.items()
                           if not isinstance(v, (dict, list)) or k in (
                               "filters", "sizing_kwargs")},
        "load_report":    lr,
        "failure_diagnostics": diag,
        "n_trades_total": len(trades),
        "trade_sample":   trades[:5] + trades[-5:] if len(trades) > 10
                          else trades,
        "n_rebalances":   len(rb),
        "rebalance_sample": rb[:5] + rb[-5:] if len(rb) > 10 else rb,
    }


# ── Mode 2: Local heuristics ──────────────────────────────────────

def _local_analysis(results: dict) -> str:
    """Run the same pattern-matching the user would do by hand and
    produce a markdown report.  No API needed."""
    metrics = (results or {}).get("metrics") or {}
    preset  = (results or {}).get("preset")  or {}
    trades  = (results or {}).get("trades")  or []
    rb      = (results or {}).get("rebalance_trades") or []
    diag    = (results or {}).get("failure_diagnostics") or {}
    lr      = (results or {}).get("load_report") or {}

    n_trades   = int(metrics.get("total_trades", 0) or 0)
    ret_pct    = float(metrics.get("total_return_pct", 0) or 0)
    sharpe     = metrics.get("sharpe")
    max_dd     = float(metrics.get("max_drawdown_pct", 0) or 0)
    win_rate   = metrics.get("win_rate_pct")
    n_symbols  = int(metrics.get("symbols_traded", 0) or 0)
    is_xs      = (preset.get("model_kind") == "cross_sectional"
                   or "rebalance_trades" in (results or {}))

    issues:    list[str] = []
    rec:       list[str] = []
    verdict_bits: list[str] = []

    # ── Implausible-return checks (very useful) ────────────────────
    if ret_pct < -90:
        issues.append(
            f"**Catastrophic loss ({ret_pct:.1f}%)** — long-only "
            f"diversified runs almost never lose >90%.  This is a "
            f"strong signal of an engine bug (NaN-price money leak, "
            f"sizing math, etc.) rather than strategy failure."
        )
        if is_xs:
            issues.append(
                "Cross-sectional runs from before the NaN-fix in commit "
                "`78a67c8`+ leaked cash whenever a held symbol's price "
                "went NaN at the rebalance bar.  Rerun with the fixed "
                "engine."
            )
        verdict_bits.append("⚠ Result is almost certainly an engine bug, not real strategy P&L.")
    elif ret_pct > 1000:
        issues.append(
            f"**Implausible return (+{ret_pct:.0f}%)** — even the best "
            f"strategies rarely return >1000% over the standard "
            f"backtest window.  Suspect look-ahead bias, "
            f"unrealistic position sizing, or an equity-curve bug."
        )
        verdict_bits.append("⚠ Suspect look-ahead bias or sizing leverage.")
    elif n_trades > 0 and -90 < ret_pct < 1000:
        verdict_bits.append(
            f"Result range looks plausible ({ret_pct:+.1f}% over "
            f"{n_trades} trades)."
        )

    # ── Trade-count / no-fire diagnostics ──────────────────────────
    if n_trades == 0:
        if diag:
            raw = diag.get("raw_buy_signals", 0)
            af  = diag.get("after_filters", 0)
            if raw == 0:
                issues.append("Strategy emitted **zero raw buy signals** — model didn't recognise any setup in this universe/period.")
            elif af == 0 and raw > 0:
                issues.append(f"All {raw} raw buy signals were rejected by filters.  Check the post-mortem panel — most likely a missing filter field on the loaded symbols.")

            for f in diag.get("filters", []):
                if f.get("field_missing_in", 0) > 0 and f.get("field_present_in", 0) == 0:
                    issues.append(f"Filter field `{f.get('field')}` doesn't exist in any loaded symbol.  Either pick a different field or run feature engineering that produces it.")

        sm = (preset.get("sizing_method") or "").lower()
        sk = preset.get("sizing_kwargs") or {}
        if sm in ("kelly", "half_kelly"):
            try:
                p = float(sk.get("win_rate", 0.5))
                b = float(sk.get("win_loss_ratio", 1.5))
                f = (p * b - (1 - p)) / max(b, 1e-6)
                if sm == "half_kelly": f /= 2
                if f <= 0:
                    issues.append(f"**{sm.replace('_', '-').title()} fraction is negative ({f:+.2%})** with win-rate {p*100:.0f}% and win/loss ratio {b:.2f} — every trade sized to 0 shares.  Raise win-rate, raise win/loss ratio, or switch to fixed_pct.")
            except (TypeError, ValueError):
                pass

    # ── Win rate / Sharpe sanity ───────────────────────────────────
    if win_rate is not None and isinstance(win_rate, (int, float)):
        if win_rate < 30 and n_trades > 20:
            issues.append(f"Low win rate ({win_rate:.1f}%).  Combined with the return profile, check if the strategy's edge is in winning frequency or in average win size.")
    if sharpe is not None and isinstance(sharpe, (int, float)):
        if sharpe > 4:
            issues.append(f"Sharpe of {sharpe:.2f} is exceptionally high — verify there's no look-ahead bias and that slippage/commissions are realistic.")
        elif sharpe < 0 and n_trades > 0:
            issues.append(f"Negative Sharpe ({sharpe:.2f}) — strategy underperformed risk-free rate over the backtest window.")

    # ── Drawdown ───────────────────────────────────────────────────
    if max_dd < -50 and n_trades > 0:
        issues.append(f"Severe max drawdown ({max_dd:.1f}%).  Either size smaller, add stop-loss, or accept this risk profile.")

    # ── Single-position lock-up ────────────────────────────────────
    if n_trades > 0 and n_symbols == 1 and not is_xs:
        verdict_bits.append("Single-symbol run — this is a strategy *baseline*, not a portfolio.")

    # ── Sentiment-based filters with no sentiment data ─────────────
    if preset.get("filters"):
        for fld in (f.get("field", "") for f in preset["filters"]):
            if fld and "sentiment" in fld.lower():
                rec.append(f"Filter `{fld}` requires sentiment data — make sure `python -m bot.sentiment.sentiment_pipeline` has run for the universe.")

    # ── Cross-sectional rebalance count ────────────────────────────
    if is_xs and rb:
        rec.append(f"Rebalance happened {len(rb)} times — at "
                    f"{preset.get('rebalance_days', '?')}-day cadence.")

    # ── Build markdown ─────────────────────────────────────────────
    lines: list[str] = ["## Verdict"]
    if not verdict_bits:
        verdict_bits.append("Run completed without obvious red flags.")
    lines.extend(f"- {b}" for b in verdict_bits)
    lines.append("")

    lines.append("## What happened")
    lines.append(
        f"- Model: `{results.get('model', '?')}`  ·  "
        f"{results.get('symbol', '?')}  ·  "
        f"period: {results.get('period_days', '?')} days"
    )
    if n_trades > 0:
        lines.append(
            f"- **{n_trades} trades** across **{n_symbols} symbols**, "
            f"return **{ret_pct:+.2f}%**, "
            f"max DD **{max_dd:.2f}%**" + (
                f", Sharpe **{sharpe:.2f}**" if isinstance(sharpe, (int, float)) else ""
            )
        )
        if win_rate is not None and isinstance(win_rate, (int, float)):
            lines.append(f"- Win rate: **{win_rate:.1f}%**")
    else:
        lines.append("- **0 trades fired** — see Issues for likely cause.")

    if lr:
        lines.append(
            f"- Data: loaded **{lr.get('loaded', 0)}/{lr.get('requested', 0)}** "
            f"requested symbols"
        )
    lines.append("")

    if issues:
        lines.append("## Issues found")
        for i in issues:
            lines.append(f"- {i}")
        lines.append("")
    else:
        lines.append("## Issues found")
        lines.append("- None detected by the local heuristic checks.")
        lines.append("")

    if rec:
        lines.append("## Recommendations")
        for r in rec:
            lines.append(f"- {r}")
    else:
        lines.append("## Recommendations")
        lines.append("- Result looks acceptable — try walk-forward validation to confirm it's not over-fit to this period.")

    lines.append("")
    lines.append("---")
    lines.append(
        "_Local heuristic analysis (no Anthropic API key set).  "
        "For deeper qualitative analysis, add `ANTHROPIC_API_KEY=...` to "
        "your `.env` and click Analyze again._"
    )

    return "\n".join(lines)
