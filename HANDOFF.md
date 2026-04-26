# Trading bot — handoff to Claude Code

A complete state-of-the-project document. Drop this in your repo root and tell
Claude Code to read it first before doing anything else.

---

## Project repo

```
github.com/ranjithpoliyedath/trading-bot-1
```

Local: `~/Documents/trading-bot-1`
Python venv: `venv/` (Python 3.9)
Shell: csh

---

## What's working today

### Phase 1 — Environment ✅
- Alpaca paper API connected, `.env` configured
- `requirements.txt` and `.gitignore` set up
- Repo synced with GitHub

### Phase 2 — Data pipeline ✅
- 117 symbols have 6 years of daily OHLCV bars in `data/processed/`
- Each symbol has 27 features: 17 technical + 10 sentiment columns
  - Technical: rsi_14, macd, macd_hist, bb_upper, bb_lower, bb_pct, ema_9, ema_21,
    ema_cross, atr_14, volume_ratio, price_change_1d, price_change_5d,
    high_low_ratio, close_to_vwap (and a few more)
  - Sentiment: news_sentiment_mean/std/count, st_sentiment_mean/std,
    st_bullish_ratio, st_likes_sum, combined_sentiment, sentiment_momentum,
    sentiment_change, sentiment_accel
- Sentiment columns currently all zeros — pipeline never successfully ran end-to-end

### Universe ✅
- 539 eligible symbols out of 1,502 S&P 500/400/600 candidates
- Filters: 14-day avg volume > 100K (IEX feed), price ≥ $5, no penny stocks
- Saved to `data/universe.parquet`
- Refreshable: `python -m bot.universe`

### Models ✅
- 3 rule-based models registered: `rsi_macd_v1`, `bollinger_v1`, `sentiment_v1`
- `CustomRuleModel` loads JSON specs from `dashboard/custom_models/`
- All accessible via `from bot.models.registry import get_model, list_models`
- Tests passing: `tests/test_models.py` (19 tests)

### Dashboard ✅ (partially)
- Dash app at `dashboard/app.py` runs on `localhost:8050`
- Market Overview page is the landing page
- Six panels built: fear & greed, indexes, sectors, volume movers,
  sentiment heatmap, news headlines
- Account/model/symbol switcher in topbar

### Scrapers ✅
- `bot/scrapers/sp_constituents.py` — Wikipedia S&P lists
- `bot/scrapers/fear_greed.py` — CNN F&G index, 1hr cache

---

## What's broken — pick up here

### Bug 1 — Sentiment heatmap empty
**Symptom:** Heatmap panel says "No sentiment data — run sentiment pipeline."
**Cause:** Sentiment pipeline has run but produced zero `*_features_with_sentiment.parquet`
files. The 117 existing `*_features.parquet` files all have `combined_sentiment = 0.0`.
**To fix:**
1. Run `python -m bot.sentiment.sentiment_pipeline |& tee logs/sentiment_run.log`
   (csh syntax — note `|&` not `2>&1 |`)
2. Read the log to see where it's failing
3. The pipeline lives in `bot/sentiment/sentiment_pipeline.py`

### Bug 2 — News headlines all blank
**Symptom:** News panel shows entries but headlines are empty strings.
**Cause:** Old `news_fetcher.py` was iterating `for _, item in news` which
pulled the dict envelope, not the actual articles.
**Fix already written:** A patched `bot/sentiment/news_fetcher.py` was created
with the correct logic — uses `response.data["news"]` to get the list of
`News` objects. **Apply this fix and clear the cache:**
```csh
rm -f data/cache/news.json
```

### Bug 3 — Sentiment pipeline never produced output
**Symptom:** No `*_features_with_sentiment.parquet` files exist anywhere.
**Likely cause:** The pipeline reads `*_features.parquet`, fetches news +
StockTwits, scores them, then merges. Something is failing silently. Once
news_fetcher is fixed (Bug 2), this should work — the article fetcher was
returning empty headlines, so nothing to score.

---

## Architecture reference

```
trading-bot-1/
├── bot/
│   ├── config.py                 ← Central config (universe size, lookback, paths)
│   ├── universe.py               ← S&P universe builder
│   ├── data_fetcher.py           ← Alpaca OHLCV bars
│   ├── data_store.py             ← Parquet save/load
│   ├── feature_engineer.py       ← 17 technical + 10 sentiment features
│   ├── pipeline.py               ← Main data pipeline (incremental, resumable)
│   ├── market_overview.py        ← Aggregator for dashboard overview page
│   ├── scrapers/
│   │   ├── sp_constituents.py    ← Wikipedia
│   │   └── fear_greed.py         ← CNN
│   ├── sentiment/
│   │   ├── news_fetcher.py       ← Alpaca News  ⚠ NEEDS PATCH
│   │   ├── stocktwits_fetcher.py ← StockTwits public API
│   │   ├── scorer.py             ← FinBERT/VADER
│   │   ├── aggregator.py         ← Daily roll-up
│   │   ├── sentiment_features.py ← Merge into features
│   │   └── sentiment_pipeline.py ← Orchestrator
│   └── models/
│       ├── base.py               ← BaseModel interface
│       ├── registry.py           ← @register_model + lookup
│       ├── custom.py             ← JSON-spec rules
│       └── builtin/
│           ├── rsi_macd_v1.py
│           ├── bollinger_v1.py
│           └── sentiment_v1.py
│
├── dashboard/
│   ├── app.py                    ← Dash entry point
│   ├── alpaca_client.py
│   ├── backtest_engine.py        ← Built but not yet hooked to model registry
│   ├── pages/
│   │   ├── market_overview.py    ← 6-panel landing
│   │   ├── overview.py           ← Old hardcoded version (not used)
│   │   └── backtest.py           ← Old backtest layout
│   ├── components/
│   │   ├── global_controls.py
│   │   ├── signal_panel.py
│   │   └── model_summary.py
│   └── custom_models/            ← User JSON files (gitignored)
│
├── tests/                        ← All tests passing as of last run
│   ├── test_data_fetcher.py
│   ├── test_feature_engineer.py
│   ├── test_sentiment.py
│   ├── test_universe.py
│   ├── test_models.py
│   └── test_market_overview.py
│
├── scripts/
│   ├── setup_cron.sh             ← Daily refresh cron jobs
│   ├── rollout_next_batch.py     ← Fetch next 100 symbols nightly
│   ├── wire_overview_page.py     ← Patch script (already run)
│   ├── diagnose_overview.py      ← Diagnostic
│   └── diagnose_overview2.py     ← Deeper diagnostic
│
├── data/
│   ├── universe.parquet          ← 1,502 S&P symbols, 539 eligible
│   ├── processed/                ← 117 *_features.parquet + *_raw.parquet
│   └── cache/
│       ├── fear_greed.json
│       └── news.json             ← Has 10 entries with empty headlines
│
├── logs/                         ← gitignored
├── CLAUDE.md                     ← Project guidelines
├── SPEC.md                       ← Platform specification
├── HANDOFF.md                    ← This file
├── requirements.txt
└── .env                          ← gitignored
```

---

## .env required keys

```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

(no Reddit or X keys — we use StockTwits instead, which needs no auth)

---

## Roadmap — where we are in the build

From `SPEC.md`:

| Step | Status |
|---|---|
| 1. Universe module | ✅ Done — 539 eligible symbols |
| 2. Pipeline extension (6yr, incremental, resumable) | ✅ Done |
| 3. Initial bulk fetch (top 100) | ✅ Done — 117 symbols processed |
| 4. Cron jobs for daily refresh + nightly rollout | ✅ Built, not yet installed |
| 5. Model registry + 3 rule-based models | ✅ Done |
| 6. Market overview page | ⚠ Built but news + sentiment panels broken |
| 7. Stock screener page | ⏳ Not started |
| 8. Custom model builder UI | ⏳ Not started |
| 9. End-to-end smoke test | ⏳ Pending |

Backtesting is **deliberately out of scope** for this foundation phase — it
will be specified separately once the platform is solid.

---

## Immediate next actions for Claude Code

When you start your Claude Code session, paste this:

> Read HANDOFF.md, CLAUDE.md, and SPEC.md first. Then fix the three open
> bugs in this priority order:
>
> 1. Apply the news_fetcher patch from outputs (use response.data["news"])
> 2. Clear data/cache/news.json
> 3. Run the sentiment pipeline and tail the log to find why no
>    *_features_with_sentiment.parquet files are being produced
> 4. Once sentiment data is real, refresh the dashboard and verify the
>    sentiment heatmap and news headlines populate
>
> After bugs are fixed, proceed with Step 7 (stock screener page) per SPEC.md.

---

## Conventions

- **csh syntax** for all shell commands (`|&` not `2>&1 |`)
- **No `print()`** in production code — use `logging`
- **All Alpaca calls** must use `feed="iex"` (free tier limit)
- **Type hints** use `from __future__ import annotations` for Python 3.9 compat
- **Tests** must mock external APIs — never hit real Alpaca/Wikipedia/CNN in tests
- **Never commit** `.env`, `data/processed/`, `models/saved/`, `logs/`, or
  `dashboard/custom_models/`

---

## How to run things

```csh
# Activate venv
source venv/bin/activate.csh

# Refresh universe
python -m bot.universe

# Pipeline — top 100 symbols, incremental
python -m bot.pipeline

# Pipeline — specific symbols
python -m bot.pipeline --symbols AAPL,TSLA,SPY

# Pipeline — next batch (skips already-fetched)
python scripts/rollout_next_batch.py

# Sentiment pipeline
python -m bot.sentiment.sentiment_pipeline

# Run all tests
python -m pytest tests/ -v

# Launch dashboard
python -m dashboard.app
# Then open http://localhost:8050

# Diagnostic
python scripts/diagnose_overview2.py

# csh redirect to log
python -m bot.pipeline |& tee logs/run.log
```

---

## Known platform quirks

- **Alpaca free tier** = IEX feed only. Real volume is ~5% of full market.
  Volume thresholds set accordingly (100K not 1M).
- **Multi-class shares** (BRK-B, MOG-A, etc.) are filtered out of the
  universe — Alpaca free tier rejects them.
- **Python 3.9** does NOT support `dict | None` syntax — use
  `from __future__ import annotations` at the top of any file with new union types.
- **csh shell** — comments with `#` only work at line start, not inline.
