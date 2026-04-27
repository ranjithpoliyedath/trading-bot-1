# Natural-language backtest queries — usage model

The Backtest tab has a textarea labelled **"Natural-language query"**.
You type a plain-English description of the backtest you want, click
**Parse with Claude**, and the form below the textarea fills itself in
— model, period, confidence threshold, filter rows.  Then you tweak
anything you don't like and click **Run backtest**.

This document explains every step of that pipeline end-to-end so you
know what's happening and how to drive it.

---

## The pipeline at a glance

```
┌────────────┐     ┌─────────────┐     ┌────────────┐     ┌──────────────┐
│  Textarea  │ →   │  parse_query│  →  │  Anthropic │  →  │  ParsedQuery │
│ (your text)│     │   ()        │     │   API      │     │  dataclass   │
└────────────┘     └─────────────┘     └────────────┘     └──────────────┘
                                                                  │
                                                                  ▼
                                                  ┌────────────────────────┐
                                                  │   Dashboard callback   │
                                                  │ (parse_nl_query)       │
                                                  └────────────────────────┘
                                                                  │
                                  ┌──────────────────┬────────────┴────────────────────┐
                                  ▼                  ▼                                 ▼
                           bt-dd-model       bt-input-period                   bt-filter-rows
                           bt-input-conf                                       (existing rows replaced)
```

Files involved:

| Component | File | Responsibility |
|---|---|---|
| Textarea + Parse button | `dashboard/pages/backtest.py::_nl_query_panel` | UI |
| Callback that fires on click | `dashboard/callbacks/backtest_callbacks.py::parse_nl_query` | Hands the textarea text to `parse_query`, splats result into the form |
| API integration | `bot/nl_query.py::parse_query` | Calls Anthropic with a forced tool-use, returns a `ParsedQuery` |

---

## What gets filled in

When the parse succeeds, these form inputs update automatically:

| ParsedQuery field | Form input it sets |
|---|---|
| `model_id` | **Model** dropdown |
| `period_days` | **Period (days)** input |
| `min_confidence` | **Confidence threshold** input |
| `filters[]` | **Manual filters** rows (existing rows are replaced) |
| `rationale` | Status line below the textarea (✅ banner) |

The rest of the form (universe scope, sample-account settings, exits,
realism settings) is **not** changed — Claude doesn't try to guess
those.  You set them yourself via the form, or load a preset.

---

## What Claude is allowed to return

Behind the scenes the API call uses **forced tool-use**: Claude is
required to call exactly one tool called `configure_backtest`, whose
input schema is locked down to:

```python
{
    "model_id":       <one of the registered model ids>,
    "filters":        [ {field, op, value}, ...]      # AND-combined
    "period_days":    int between 30 and 2000,
    "symbols":        [str, ...]                      # optional, currently unused
    "min_confidence": float between 0 and 1,
    "rationale":      str                              # one-sentence echo
}
```

`field` is constrained to the keys of `bot.screener.SCREENER_FIELDS`
(the screener's filter catalogue): `rsi_14`, `macd_hist`, `bb_pct`,
`combined_sentiment`, `volume_ratio`, etc.  `op` is one of
`> >= < <= == !=`.

If Claude returns anything outside that vocabulary, the defensive
sanity check in `parse_query` silently drops the offending filter.
You'll see fewer filter rows than expected — re-phrase the prompt
to use names from the catalogue.

---

## Worked examples

### Example 1 — pure mean-reversion sketch

> "RSI under 30, MACD bullish, over the last year on the most-liquid
> stocks"

Claude returns:

```json
{
  "model_id":       "rsi_macd_v1",
  "filters":        [
    {"field": "rsi_14",    "op": "<", "value": 30},
    {"field": "macd_hist", "op": ">", "value": 0}
  ],
  "period_days":    365,
  "min_confidence": 0.6,
  "rationale":      "RSI(14) < 30 with MACD histogram above 0 over 1 year — RSI/MACD model."
}
```

The form fills in: Model = RSI + MACD, Period = 365, Confidence = 0.6,
two filter rows.  You then pick "Top 100 by liquidity" yourself and
click Run.

### Example 2 — pattern strategy with no filter

> "qullamaggie breakout last 6 months"

```json
{
  "model_id":       "qullamaggie_v1",
  "filters":        [],
  "period_days":    180,
  "min_confidence": 0.6,
  "rationale":      "Qullamaggie breakout over 6 months."
}
```

The pattern model self-gates on its own breakout columns — you usually
don't need additional filters.

### Example 3 — sentiment filter on top of a model

> "VCP setups where combined sentiment is above 0.2"

```json
{
  "model_id":       "vcp_v1",
  "filters":        [
    {"field": "combined_sentiment", "op": ">", "value": 0.2}
  ],
  "period_days":    365,
  "min_confidence": 0.6,
  "rationale":      "VCP breakouts gated by combined sentiment > 0.2."
}
```

---

## Heuristics Claude follows

Pulled verbatim from the system prompt in
`bot/nl_query.py::_build_system_prompt`:

- **"Qullamaggie" / "stage-2 breakout"** → `model_id = qullamaggie_v1`
- **"VCP" / "volatility contraction"** → `model_id = vcp_v1`
- **"RSI / MACD oversold"** → `rsi_macd_v1`
- **"sentiment-driven"** → `sentiment_v1`
- **"Bollinger"** → `bollinger_v1`
- **"last year"** → 365 days; **"last 6 months"** → 180; **"last 3 months"** → 90
- Each indicator threshold the user names becomes a separate filter
- Default `min_confidence = 0.6` unless user specifies otherwise
- Empty filters when the user gives no specific criteria — the model
  alone gates signals
- Sector mentions ("tech", "energy") → can map to sector ETFs but
  prefer per-stock filters

---

## Failure modes

| Failure | Status banner | What to do |
|---|---|---|
| `ANTHROPIC_API_KEY` missing | `❌ ANTHROPIC_API_KEY is not set in .env` | Add the key to `.env` and restart |
| Claude declines the tool call | `❌ Claude did not call the configure_backtest tool` | Rare; usually means the request was blocked.  Re-phrase. |
| Filter field outside the catalogue | Silently dropped (no banner) | Use canonical field names listed below |
| Empty / nonsense query | The form often comes back with a sensible default model + empty filters | Be more specific |

---

## Cost & caching

Each parse is one `messages.create` call to Anthropic.  The system
prompt is wrapped with `cache_control: {"type": "ephemeral"}` so
the (large) static portion — model catalogue, field catalogue,
heuristics — only counts toward cost the first time per ~5 minutes.
A typical call costs **<$0.005** with cache; **~$0.01** without.

If you don't have an Anthropic key and don't want one, ignore the
NL panel entirely — every form field can be filled in manually.
The Strategy Finder also has an opt-in **🤖 Ask Claude** button on
its own page; same cost model.

---

## Filter field catalogue

These are the keys Claude is allowed to use as `filter.field`.  They
correspond to columns the screener / engine know how to look up on
each bar:

| Group | Keys |
|---|---|
| Technical | `rsi_14`, `macd_hist`, `ema_cross`, `bb_pct`, `atr_14` |
| Sentiment | `combined_sentiment`, `news_count`, `st_bullish_ratio` |
| Price | `price_change_1d`, `price_change_5d`, `volume_ratio` |
| Breakout | `prior_runup_pct`, `consolidation_range`, `consolidation_vol_drop`, `contraction_count`, `breakout_today` |

Source of truth: `bot/screener.py::SCREENER_FIELDS`.  When the
catalogue grows, the system prompt re-renders automatically — no
code change needed in `nl_query.py`.

---

## When NOT to use NL queries

- **Reproducible scripted backtests.** The Optuna driver in the
  Strategy Finder is deterministic with a seed; NL queries depend on
  a paid API and aren't bit-reproducible. Use the Finder for
  parameter sweeps, NL for one-off exploration.
- **Large parameter combinations.** Each NL query is one parse; if
  you're sweeping 50 RSI thresholds, just script it.
- **Production trading decisions.** This is a backtest config helper,
  not a strategy author.  The model picks a strategy *id* — it
  doesn't write new strategy code.
