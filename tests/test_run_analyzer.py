"""
tests/test_run_analyzer.py
--------------------------
Tests for the AI run-analyzer service.  Covers:

  • Local heuristic mode (no API key) — produces markdown with the
    expected sections and catches the known failure modes.
  • Async runner state transitions.
  • Single-flight semantics.
  • Failures captured in state without exceptions escaping.
"""
from __future__ import annotations

import time
from unittest.mock import patch


# ── Local heuristic analysis ───────────────────────────────────────

class TestLocalAnalysis:

    def test_zero_trade_run_flags_no_signals(self):
        from dashboard.services.run_analyzer import _local_analysis
        results = {
            "run_id": "BT-test",
            "model":  "ibs_v1",
            "metrics": {"total_trades": 0, "symbols_traded": 0,
                         "total_return_pct": 0},
            "preset":  {"sizing_method": "fixed_pct"},
            "failure_diagnostics": {
                "raw_buy_signals":         0,
                "after_confidence_cutoff": 0,
                "after_filters":           0,
                "filters": [],
            },
        }
        md = _local_analysis(results)
        assert "## Verdict"            in md
        assert "## What happened"      in md
        assert "## Issues found"       in md
        assert "## Recommendations"    in md
        assert "0 trades"              in md
        assert "zero raw buy signals" in md.lower()

    def test_catastrophic_loss_flags_engine_bug(self):
        """The exact case the user hit — XS run with -99% return.
        Analyzer must call this out as an engine bug, not strategy."""
        from dashboard.services.run_analyzer import _local_analysis
        results = {
            "run_id": "XS-test",
            "model":  "jt_momentum_v1",
            "metrics": {"total_trades": 13, "symbols_traded": 0,
                         "total_return_pct": -99.64,
                         "max_drawdown_pct": -99.64,
                         "sharpe": -0.69},
            "preset":  {"model_kind": "cross_sectional",
                         "rebalance_days": 21},
            "rebalance_trades": [{"pl": -100} for _ in range(13)],
        }
        md = _local_analysis(results)
        assert "Catastrophic loss"             in md
        assert "engine bug"                     in md
        # Cross-sectional NaN-price hint
        assert "NaN-fix"                        in md or "78a67c8" in md

    def test_negative_kelly_flags_zero_sizing(self):
        from dashboard.services.run_analyzer import _local_analysis
        results = {
            "run_id": "BT-kelly",
            "metrics": {"total_trades": 0, "symbols_traded": 0,
                         "total_return_pct": 0},
            "preset":  {"sizing_method": "half_kelly",
                         "sizing_kwargs": {"win_rate": 0.10,
                                            "win_loss_ratio": 2.0}},
        }
        md = _local_analysis(results)
        assert "Half-Kelly" in md or "half-kelly" in md.lower()
        assert "negative"   in md.lower() or "0 shares" in md

    def test_implausible_high_return_flags_lookahead(self):
        from dashboard.services.run_analyzer import _local_analysis
        results = {
            "run_id": "BT-toogood",
            "metrics": {"total_trades": 100, "symbols_traded": 5,
                         "total_return_pct": 50_000},
            "preset":  {},
        }
        md = _local_analysis(results)
        assert "Implausible" in md or "look-ahead" in md.lower()

    def test_sentiment_filter_recommends_pipeline(self):
        from dashboard.services.run_analyzer import _local_analysis
        results = {
            "run_id": "BT-sent",
            "metrics": {"total_trades": 0, "symbols_traded": 0,
                         "total_return_pct": 0},
            "preset": {
                "sizing_method": "fixed_pct",
                "filters": [{"field": "combined_sentiment", "op": ">",
                              "value": 0}],
            },
        }
        md = _local_analysis(results)
        assert "sentiment_pipeline" in md

    def test_clean_run_says_no_red_flags(self):
        from dashboard.services.run_analyzer import _local_analysis
        results = {
            "run_id": "BT-clean",
            "model":  "ibs_v1",
            "metrics": {"total_trades": 200, "symbols_traded": 10,
                         "total_return_pct": 25.5,
                         "sharpe": 1.2,
                         "max_drawdown_pct": -8.3,
                         "win_rate_pct": 56.0},
            "preset":  {"sizing_method": "fixed_pct"},
        }
        md = _local_analysis(results)
        assert "## Issues found" in md
        # Clean run should have no major issue text
        assert "Catastrophic"  not in md
        assert "Implausible"   not in md


# ── Async runner ────────────────────────────────────────────────────

class TestAsyncAnalyzer:

    def setup_method(self):
        from dashboard.services import run_analyzer as ra
        with ra._run_lock:
            ra._run_state.update({
                "status": "idle", "started_at": None, "finished_at": None,
                "duration_s": None, "result": None, "error": None,
                "run_id": None,
            })

    def test_initial_state_is_idle(self):
        from dashboard.services.run_analyzer import get_analyzer_status
        s = get_analyzer_status()
        assert s["status"] == "idle"

    def test_start_runs_to_done_with_local_analysis(self, monkeypatch):
        """No API key → local heuristic path → state hits 'done'."""
        from dashboard.services import run_analyzer as ra
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        ra.start_analysis({"run_id": "BT-x",
                            "metrics": {"total_trades": 50,
                                         "total_return_pct": 12.0}})

        deadline = time.time() + 5
        while time.time() < deadline:
            s = ra.get_analyzer_status()
            if s["status"] in ("done", "failed"):
                break
            time.sleep(0.05)

        assert s["status"]               == "done"
        assert s["result"]["source"]      == "local"
        assert s["result"]["model"]       is None
        assert "## Verdict"               in s["result"]["text"]

    def test_single_flight(self, monkeypatch):
        """Second start while one running is a no-op."""
        from dashboard.services import run_analyzer as ra
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        # Slow stub
        calls = {"n": 0}
        def slow(results):
            calls["n"] += 1
            time.sleep(0.3)
            return "## Verdict\nslow"
        monkeypatch.setattr("dashboard.services.run_analyzer._local_analysis", slow)

        ra.start_analysis({"run_id": "BT-1"})
        ra.start_analysis({"run_id": "BT-2"})  # while first running
        time.sleep(0.6)
        assert calls["n"] == 1, f"expected single-flight, got {calls['n']} calls"

    def test_failure_captured_in_state(self, monkeypatch):
        """Exception in _local_analysis → state.failed, no propagation."""
        from dashboard.services import run_analyzer as ra
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr("dashboard.services.run_analyzer._local_analysis",
                             lambda r: 1/0)

        ra.start_analysis({"run_id": "BT-fail"})
        deadline = time.time() + 3
        while time.time() < deadline:
            s = ra.get_analyzer_status()
            if s["status"] in ("done", "failed"):
                break
            time.sleep(0.05)
        assert s["status"] == "failed"
        assert s["error"]
