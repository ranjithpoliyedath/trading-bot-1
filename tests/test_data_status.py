"""
tests/test_data_status.py
-------------------------
Tests for the Data Status page and its async pipeline runner.

Covered:
  • get_coverage_summary handles empty + populated data dirs
  • Coverage buckets sum to total
  • Async pipeline runner is single-flight (second click is a no-op)
  • Async runner tracks state transitions correctly (idle→running→done)
  • Async runner reports failures via status dict (no exceptions
    leak out to the dashboard)
  • Page layout renders without raising
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


# ── Coverage summary ────────────────────────────────────────────────

class TestCoverageSummary:

    def test_empty_data_dir_returns_zero_summary(self, tmp_path):
        """No parquets → zero counts, no crash."""
        from dashboard.services import data_status as ds
        with patch.object(ds, "DATA_DIR", tmp_path):
            s = ds.get_coverage_summary()
        assert s["total_features"]   == 0
        assert s["total_raw"]        == 0
        assert s["earliest_data"]    == ""
        assert s["latest_bar_date"]  == ""

    def test_populated_dir_reports_correct_counts(self, tmp_path):
        """Three feature parquets → counts + buckets reflect them."""
        from dashboard.services import data_status as ds

        # Synthesise three parquets at different depths
        configs = [
            ("AAPL", 2500),  # 8y+
            ("PLTR",  500),  # 2-6y
            ("GEV",   200),  # <2y
        ]
        for sym, n_bars in configs:
            idx = pd.date_range("2020-01-01", periods=n_bars, freq="B")
            df = pd.DataFrame({"close": range(n_bars)}, index=idx)
            df.to_parquet(tmp_path / f"{sym}_features.parquet")
            df.to_parquet(tmp_path / f"{sym}_raw.parquet")

        with patch.object(ds, "DATA_DIR", tmp_path):
            s = ds.get_coverage_summary()

        assert s["total_features"] == 3
        assert s["total_raw"]      == 3
        # Buckets must sum to the total feature parquets
        bucket_sum = sum(s["depth_buckets"].values())
        assert bucket_sum == 3
        # AAPL is the deepest
        assert "AAPL" in s["deepest_symbol"]
        # GEV is the shallowest
        assert "GEV" in s["shallowest"]
        # Disk-MB is non-negative and finite
        assert s["size_mb"] >= 0


# ── Async pipeline runner ────────────────────────────────────────────

class TestAsyncRunner:

    def setup_method(self):
        # Reset module state between tests
        from dashboard.services import data_status as ds
        with ds._run_lock:
            ds._run_state.update({
                "status": "idle", "started_at": None, "finished_at": None,
                "duration_s": None, "counts": None, "error": None,
                "trigger": None,
            })

    def test_initial_state_is_idle(self):
        from dashboard.services.data_status import get_pipeline_status
        s = get_pipeline_status()
        assert s["status"] == "idle"
        assert s["counts"] is None

    def test_start_transitions_to_running_then_done(self, monkeypatch):
        """Successful run: state goes idle → running → done with counts."""
        from dashboard.services import data_status as ds

        # Fast-fake the heavy import so the test doesn't hit Alpaca/yfinance
        fake_counts = {"processed": 5, "skipped": 0, "failed": 0, "new_rows": 100}
        monkeypatch.setattr("bot.pipeline.run_pipeline",
                             lambda *a, **kw: fake_counts)
        monkeypatch.setattr("bot.universe.get_all_for_data_fetch",
                             lambda **kw: ["AAPL", "MSFT"])

        ds.start_pipeline_async(trigger="manual")

        # Wait briefly for the thread to finish
        deadline = time.time() + 5
        while time.time() < deadline:
            s = ds.get_pipeline_status()
            if s["status"] in ("done", "failed"):
                break
            time.sleep(0.05)

        assert s["status"]      == "done"
        assert s["counts"]      == fake_counts
        assert s["error"]       is None
        assert s["trigger"]     == "manual"
        assert s["duration_s"]  is not None
        assert s["finished_at"] is not None

    def test_failure_is_captured_in_state(self, monkeypatch):
        """run_pipeline raising should NOT propagate — state.failed."""
        from dashboard.services import data_status as ds

        def boom(*a, **kw):
            raise RuntimeError("simulated yfinance outage")
        monkeypatch.setattr("bot.pipeline.run_pipeline", boom)
        monkeypatch.setattr("bot.universe.get_all_for_data_fetch",
                             lambda **kw: ["AAPL"])

        ds.start_pipeline_async(trigger="manual")

        deadline = time.time() + 5
        while time.time() < deadline:
            s = ds.get_pipeline_status()
            if s["status"] in ("done", "failed"):
                break
            time.sleep(0.05)

        assert s["status"] == "failed"
        assert "yfinance outage" in (s["error"] or "")
        assert s["counts"] is None

    def test_single_flight_second_click_is_noop(self, monkeypatch):
        """Clicking Update Now while one is already running must NOT
        spawn a second thread — it returns the running state."""
        from dashboard.services import data_status as ds

        # Pin run_pipeline to a slow stub so we can race a second start
        slow_marker = {"calls": 0}
        def slow_run(*a, **kw):
            slow_marker["calls"] += 1
            time.sleep(0.5)
            return {"processed": 1, "skipped": 0, "failed": 0, "new_rows": 10}
        monkeypatch.setattr("bot.pipeline.run_pipeline", slow_run)
        monkeypatch.setattr("bot.universe.get_all_for_data_fetch",
                             lambda **kw: ["AAPL"])

        first  = ds.start_pipeline_async()
        second = ds.start_pipeline_async()  # while first still running
        assert first["status"]  == "running"
        assert second["status"] == "running"

        # Wait for the first to finish — only ONE actual call should fire
        deadline = time.time() + 3
        while time.time() < deadline:
            if ds.get_pipeline_status()["status"] in ("done", "failed"):
                break
            time.sleep(0.05)

        assert slow_marker["calls"] == 1, \
            f"second start spawned a duplicate run (calls={slow_marker['calls']})"


# ── Schedule introspection ────────────────────────────────────────────

class TestScheduleInfo:

    def test_returns_dict_with_required_keys(self):
        from dashboard.services.data_status import get_schedule_info
        info = get_schedule_info()
        # Always returns these regardless of install state
        for k in ("launchd_loaded", "cron_installed", "next_run_hint"):
            assert k in info
        # Bool flags
        assert isinstance(info["launchd_loaded"], bool)
        assert isinstance(info["cron_installed"], bool)


# ── Page layout ────────────────────────────────────────────────────────

class TestPageLayout:

    def test_layout_renders_without_error(self):
        from dashboard.pages.data_status import layout
        out = layout()
        assert out is not None
        # Top-level Div with two rows (cards + button)
        assert hasattr(out, "children")

    def test_layout_includes_update_button(self):
        from dashboard.pages.data_status import layout
        s = str(layout())
        assert "btn-update-data"      in s
        assert "data-update-status"   in s
        assert "data-update-poll"     in s

    def test_status_renderer_handles_all_states(self):
        from dashboard.pages.data_status import _render_status
        for s in ["idle", "running", "done", "failed"]:
            state = {"status": s, "counts": {"processed": 1, "skipped": 0,
                                              "failed": 0, "new_rows": 1},
                     "started_at": "2026-04-28T10:00:00",
                     "finished_at": "2026-04-28T10:01:00",
                     "duration_s": 60, "error": "boom",
                     "trigger": "manual"}
            out = _render_status(state)
            assert out is not None  # never crashes regardless of state
