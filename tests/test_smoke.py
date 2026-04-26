"""
tests/test_smoke.py
--------------------
End-to-end smoke test (SPEC step 9).

Imports every dashboard page, runs the screener and a backtest, and
exercises every registered model on real on-disk feature data.  Catches
import-time regressions and surface-level breakage that unit tests
don't see.

Skips gracefully when no processed data is available so it can run in
empty CI environments.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bot.config import DATA_DIR


@pytest.fixture(scope="module")
def sample_symbols():
    found = sorted(p.name.split("_")[0]
                   for p in Path(DATA_DIR).glob("*_features.parquet"))
    if not found:
        pytest.skip("No processed feature files on disk.")
    return found[:5]


def test_dashboard_pages_import():
    from dashboard.pages import (
        market_overview, screener, model_builder, backtest,
    )
    layout = market_overview.layout("paper", "rsi_macd_v1", "AAPL")
    assert layout is not None
    assert screener.layout()       is not None
    assert model_builder.layout()  is not None
    assert backtest.layout("paper", "rsi_macd_v1", "AAPL") is not None


def test_app_imports_without_error():
    import dashboard.app  # noqa: F401


def test_registry_has_breakout_models():
    from bot.models.registry import list_models
    ids = {m.id for m in list_models()}
    assert {"rsi_macd_v1", "bollinger_v1", "sentiment_v1",
            "qullamaggie_v1", "vcp_v1"} <= ids


def test_every_model_predicts(sample_symbols):
    from bot.models.registry import list_models, get_model
    from bot.patterns import add_breakout_features

    df = pd.read_parquet(DATA_DIR / f"{sample_symbols[0]}_features.parquet")
    df = add_breakout_features(df).tail(100)

    for meta in list_models():
        model = get_model(meta.id)
        out = model.predict_batch(df.copy())
        assert "signal" in out.columns
        assert "confidence" in out.columns
        assert set(out["signal"].dropna().unique()) <= {"buy", "sell", "hold"}


def test_screener_runs_against_real_data(sample_symbols):
    from bot.screener import Filter, run_screener
    rows = run_screener(
        filters=[Filter("rsi_14", ">", 0)],
        symbols=sample_symbols,
        limit=5,
    )
    assert isinstance(rows, list)


def test_filtered_backtest_runs(sample_symbols):
    from dashboard.backtest_engine import run_filtered_backtest
    out = run_filtered_backtest(
        model_id      = "rsi_macd_v1",
        filters       = [],
        symbols       = sample_symbols,
        period_days   = 365,
        conf_threshold = 0.55,
    )
    assert "metrics" in out
    assert "equity_curve" in out
    assert "per_symbol" in out


def test_news_ranker_handles_empty():
    from bot.sentiment.news_ranker import rank_articles
    assert rank_articles([]) == []


def test_nl_query_module_loads_without_key():
    """Importing nl_query without an API key should not crash."""
    import importlib
    mod = importlib.import_module("bot.nl_query")
    assert hasattr(mod, "parse_query")
