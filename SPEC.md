# Trading bot platform — specification (v0.1)

## Vision

A personal trading platform connected to my Alpaca paper and live accounts.
Multiple models — both ML-based and rule-based — can be selected through the
dashboard to generate signals, execute trades, and be backtested. The
dashboard shows me the current market picture, helps me discover stocks via
filters, and lets me build my own custom models without writing code.

## Guiding principles

- **Ship something I can use first.** A working backtest of one simple
  rule-based model beats a half-built platform with five model types.
- **Backtest before live.** Every model must be backtested before it can
  generate live signals. The dashboard enforces this.
- **One model abstraction.** ML models, rule-based models, and custom user-
  built models all expose the same interface — predict signal + confidence.
- **Reuse what we already built.** Phase 2 data pipeline, sentiment pipeline,
  feature engineer, and dashboard scaffolding all stay. We're adding on top.

---

## Scope of this spec

This spec covers **the platform foundation** — universe selection, data
infrastructure, model registry, screener, market overview, and custom model
builder. Backtesting is intentionally **out of scope** for this build —
it will be planned and specified separately once the foundation is in place.

---

## Stock universe

The bot runs on a curated universe of US equities, refreshed daily.

**Eligibility rules (applied in order):**

1. Listed in **S&P 500**, **S&P MidCap 400**, or **S&P SmallCap 600**
2. **14-day average daily volume > 1,000,000 shares** (liquidity floor)
3. **Current price ≥ $5** (no penny stocks)
4. Listed on NYSE, NASDAQ, or AMEX (no OTC)

**Universe membership lists** are pulled from Wikipedia's S&P 500 / 400 / 600
constituent tables.

**Universe target:** ~700 symbols after filters (out of ~1,500 candidates).
The remaining ~800 candidates that pass eligibility are stored too but only
fetched on demand if needed later.

**Refresh:** Daily at 6 AM via cron. New symbols entering the indexes get
picked up automatically; delisted ones get flagged but their historical data
is retained.

---

## Historical data

We fetch **6 years of daily OHLCV bars** for symbols in the filtered universe.
Daily bars only — no intraday data.

**Phased rollout for the initial fetch:**

- **Now:** Top 100 symbols by market cap → ~1 hour fetch
- **Scheduled:** Remaining ~600 symbols → cron job runs over the next several
  nights, batched to respect Alpaca's 200 req/min limit

This means the platform is usable end-to-end after the first 100 symbols are
in. Later rollout happens in the background.

**Numbers:**
- ~700 symbols × ~1,500 trading days (6 years) = ~1M rows total
- ~3-5 MB per symbol after compression, ~2-3 GB total
- Stored as one Parquet file per symbol under `data/processed/`

The fetch is incremental — once a symbol has data, subsequent runs only
fetch new bars since the last stored timestamp. Daily refresh keeps everything
current.

Sentiment data stays at its current lookback (60 days for news, latest 30
posts for StockTwits). Sentiment history older than 60 days isn't available
from Alpaca's free tier anyway.

---

## What we build now (v1)

### 1. Model registry

A central place where every model — ML or rule-based — registers itself with
a unique name, description, and a `predict()` method that takes a feature row
and returns `(signal, confidence)`.

The dashboard reads the registry to populate the model dropdown. Adding a
new model means writing a Python class that implements the interface — no
dashboard changes needed.

**Models we implement for v1:**

| ID | Type | Strategy |
|---|---|---|
| `rsi_macd_v1` | Rule-based | Buy when RSI < 30 + MACD histogram positive. Sell when RSI > 70 + MACD negative. |
| `bollinger_v1` | Rule-based | Buy on lower band touch with positive sentiment. Sell on upper band touch. |
| `sentiment_v1` | Rule-based | Buy when combined_sentiment > 0.5 and news_count > 5. Sell when < -0.5. |
| `ml_rf_v1` | ML (later) | Random Forest trained on 27 features. Built in Phase 3. |

Three rule-based models give us something to backtest immediately — no model
training required. The ML model comes after.

### 2. Custom model builder (v1 — minimal)

A page in the dashboard where I can build a new rule-based model by combining
filter conditions with AND/OR logic. v1 supports a small fixed set of fields:

- Technical — RSI, MACD histogram, EMA cross, BB %B, ATR
- Sentiment — combined_sentiment, news_count, st_bullish_ratio
- Price — price_change_1d, price_change_5d, volume_ratio

A model definition is a JSON file under `dashboard/custom_models/`:

```json
{
  "id": "my_rsi_strategy",
  "name": "Oversold bounce",
  "buy_when":  [{"field": "rsi_14",   "op": "<", "value": 30}],
  "sell_when": [{"field": "rsi_14",   "op": ">", "value": 70}],
  "min_confidence": 0.6
}
```

Save = file written to disk. Edit = file overwritten. Saved custom models
appear in the model dropdown alongside built-in ones. Each can be backtested
the same way as any other model.

### 3. Stock discovery / screener

A dashboard page with filter controls. I pick criteria, the screener scans my
12 tracked symbols (extends to S&P 500 later), returns a ranked list.

v1 filters — must be ones we already compute in feature_engineer:

- Technical — RSI range, MACD direction, EMA cross, BB position
- Sentiment — combined_sentiment threshold, news volume, StockTwits bullish %
- Price action — % change over 1/5 days, volume vs average

Each result shows the symbol, current price, the filter values that matched,
and a "send to backtest" button.

### 4. Market overview page

Lands when I open the dashboard. Six panels:

1. **Market mood** — fear & greed index (CNN's number scraped from
   `cnn.com/markets/fear-and-greed`, cached 1 hour)
2. **Index snapshot** — SPY, QQQ, DIA, IWM, VTI with daily change
3. **Sector leaders** — top 3 sector ETFs by daily % change
4. **Volume movers** — symbols in my universe with volume_ratio > 2.0
5. **Sentiment heatmap** — my 12 symbols colored by today's combined_sentiment
6. **News headlines** — last 10 headlines from Alpaca News across my universe

Data sources we already have: Alpaca bars, Alpaca News, computed sentiment.
Only the fear & greed number is scraped externally. Everything else reads
from Parquet files.

### 5. Universe & data infrastructure

A new `bot.universe` module that pulls S&P 500/400/600 constituent lists,
applies the eligibility rules above, and writes `data/universe.parquet`.
The data pipeline reads this file instead of a hardcoded SYMBOLS list.

The data pipeline gains:
- **8-year lookback** (configurable via `config.py`)
- **Incremental fetch** — only pull new bars since last stored timestamp
- **Resumable runs** — if a fetch crashes mid-way, restart skips completed symbols
- **Parallel batching** — Alpaca allows 200 req/min, batch fetches accordingly
- **Progress logging** — clear "X of Y symbols complete" output for long runs

A new `bot.universe.refresh_universe()` command is run weekly (or manually)
to update membership lists. A daily `bot.pipeline` run keeps OHLCV current.

---

## Architecture

```
bot/
  universe.py             S&P 500/400/600 fetcher + eligibility filter
  pipeline.py              Existing — extended for 8yr incremental fetch
  models/
    base.py               Model interface (predict, metadata)
    registry.py           Model discovery and lookup
    builtin/
      rsi_macd_v1.py
      bollinger_v1.py
      sentiment_v1.py
  screener.py             Stock discovery from feature data
  market_overview.py      Aggregates fear/greed, sector data, news
  scrapers/
    fear_greed.py         CNN fear & greed scraper
    sp_constituents.py    Wikipedia S&P index membership scraper

dashboard/
  pages/
    overview.py           Market overview (replaces current overview)
    screener.py           Stock discovery page
    model_builder.py      Custom rule-based model UI
  custom_models/          User-saved model JSON files (gitignored)

data/
  universe.parquet        Filtered universe — refreshed weekly
  processed/              ~1,000 *_features.parquet files (8 years each)
```

---

## What gets built first (foundation milestone)

To prove the platform end-to-end before any backtesting work, the foundation
is complete when this works:

> **Open the dashboard. Universe is loaded (~1,000 symbols). Market
> overview shows fear & greed, sector leaders, and live sentiment. Screener
> filters return matching symbols. Pick a built-in or custom model from the
> dropdown and see what signals it generates today.**

Backtesting is then planned as a separate spec on top of this foundation.

---

## Non-goals for v1 (deferred)

These are explicitly **not** in v1. Listed so the spec is honest about scope.

- **Backtesting itself** — out of scope for this spec, planned separately.
- **Multi-source data scraping** — Barchart, Finviz, etc. v1 uses only
  Alpaca + StockTwits + the single CNN fear & greed scrape and Wikipedia
  for S&P constituents. Adding more scrapers is a separate project.
- **Live trade execution from dashboard** — v1 only generates signals. Live
  execution comes after we trust the signals via backtesting.
- **ML model training pipeline** — Phase 3. v1 ships with rule-based models
  only; ML comes after backtesting validates the rule-based ones.
- **Sector heatmaps and emerging-trends discovery** — needs sector
  classification data we don't have yet. v1 shows top 3 sector ETFs only.
- **Intraday timeframes** — v1 is daily bars only. Hourly/15min data hits
  Alpaca rate limits and is heavier to store at this universe size.
- **International equities, options, crypto** — US equities only.

---

## Decisions made

These were open questions in earlier drafts. Now decided:

1. **Custom model UI** — single AND-only condition list. Simple, ships fast,
   covers most strategies. AND/OR with grouping deferred.
2. **Fear & greed source** — CNN's fear & greed index, scraped from
   `cnn.com/markets/fear-and-greed`. Cached 1 hour to be polite.
3. **Universe refresh** — daily auto-refresh via cron at 6 AM.
4. **Initial data fetch** — top 100 symbols by market cap immediately
   (~1 hour), remaining ~600 scheduled in the background over subsequent
   nights via cron. 6 years of daily bars per symbol.

---

## Build order

The order I'll build in, smallest meaningful piece first:

1. **Universe module** — fetch S&P constituents from Wikipedia, apply
   filters, save `universe.parquet` (~1 hour)
2. **Pipeline extension** — 6-year lookback, incremental fetch, resumable
   runs, progress logging (~2 hours)
3. **Initial bulk fetch — top 100 symbols** (~1 hour, runs unattended)
4. **Cron jobs** — daily universe refresh + scheduled batches for the
   remaining ~600 symbols (~30 min setup)
5. **Model registry + 3 rule-based models** (~1 hour)
6. **Market overview page** — fear & greed, indices, sector leaders, live
   sentiment, news headlines (~2 hours)
7. **Stock screener page** (~1 hour)
8. **Custom model builder UI** — single AND-only condition list (~1.5 hours)
9. **End-to-end smoke test** — open dashboard, see overview, run screener,
   pick a model, verify signals on today's data

After step 9 the platform is ready. Backtesting plan comes next as a
separate spec.
