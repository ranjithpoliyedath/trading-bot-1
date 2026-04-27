# Plan — strategy library + Strategy Finder + realistic backtesting

This plan covers three asks in one rollout, in dependency order:

1. **Realistic backtesting upgrades** — look-ahead-bias prevention, execution
   delay, slippage, in-sample / out-of-sample (IS/OOS) split.
2. **Top-15 well-tested strategies** — research them on the internet, register
   each as a `BaseModel` in `bot/models/builtin/`, ranked by expectancy /
   Sharpe over the OOS half of our data.
3. **Strategy Finder dashboard tab** — Claude (via the Anthropic API) probes
   parameter combinations of every registered strategy on a slice of the
   universe, surfaces a sorted leaderboard, and lets the user one-click into
   a full backtest with custom name/description.

Phase 1 must land first because the library evaluation and the Strategy
Finder both rely on the new realistic-backtest plumbing for trustworthy
numbers.

---

## Skills + agents I'll use

I'm calling out the skill / agent for each phase up front so the work plan
is honest about what executes where.  Only skills / agents I have available
are listed; nothing speculative.

| Phase | Skill / Agent | Why |
|---|---|---|
| 1 | `Plan` agent (review only) + `general-purpose` agent | Architecture sanity-check on the look-ahead refactor; then implementation across `dashboard/backtest_engine.py` and tests. |
| 2 | `general-purpose` agent with **`WebSearch` + `WebFetch`** tools | Research the top-15 strategies on the open internet (Investopedia, SSRN, QuantPedia, papers, well-known traders' write-ups). Then implement each as a `BaseModel`. |
| 3 | **`claude-api` skill** + `general-purpose` agent | Strategy Finder uses the Anthropic API with prompt caching for the suggestion loop ("here are the latest results — suggest 3 parameter tweaks worth trying"). The `claude-api` skill ensures we do prompt caching, model versioning, and structured tool calls correctly. |
| 4 | `general-purpose` agent | End-to-end smoke run + tests. |

**Tools I'll need to load up front via `ToolSearch`:** `WebSearch`, `WebFetch`,
`Plan` (already an agent type, no schema needed).  All other tools are
already loaded.

---

## Phase 1 — Realistic backtest plumbing

**Goal.** Make backtest results honest enough to compare strategies against
each other.  Three flaws in the current engine:

- It looks at the same-bar `signal` to enter at the same-bar `close`.  That
  is technically OK for **daily** bars (signal computed at close, trade at
  next open is a tiny improvement), but tomorrow when we add intraday or
  news-event-driven signals it leaks the future.
- No execution delay — orders fill instantly at the bar close.
- No slippage — we get the printed price, not a real-world fill.
- No IS / OOS split — we tune on the same data we test on, so any "winning"
  strategy is suspect.

### Deliverables

1. **Look-ahead guard** in `dashboard/backtest_engine.py`:
   - Default execution model: signal at bar `t`, fill at `open` of bar `t+1`.
     This is the standard "realistic" daily backtest convention.
   - Toggleable via `execution_model: Literal["next_open", "same_close"]` on
     `run_backtest` / `run_filtered_backtest`.

2. **Execution delay**: a configurable `execution_delay_bars` (int, default 0
   for daily, 1+ for intraday) shifts the fill bar further.  For
   future-news strategies the delay is in seconds and applies inside the
   sentiment scoring path — out of scope here, but the param is reserved.

3. **Slippage**: `slippage_bps` (basis points, default 5).  Buy fills at
   `price × (1 + bps/10000)`; sells at `price × (1 - bps/10000)`.

4. **IS / OOS split**: a new helper
   `dashboard/backtest_engine.py::split_is_oos(df, oos_pct=0.30)` that
   returns `(in_sample_df, out_of_sample_df)`.  `run_backtest` /
   `run_filtered_backtest` gain a `sample: Literal["all", "in", "out"]`
   parameter; metrics for `"all"` runs additionally include
   `is_metrics` and `oos_metrics` blocks computed off the same trade log,
   split by date.

5. **Dashboard exposure** — three new controls on the Backtest page:
   - "Execution model" dropdown (`Next-bar open` / `Same-bar close`)
   - "Slippage (bps)" numeric input (default 5)
   - "Split" radio (`Full history` / `In-sample (70%)` / `Out-of-sample (30%)`)
     plus a hover tooltip explaining "Use in-sample to tune, OOS to validate".

### Files

- `dashboard/backtest_engine.py` — add execution model, slippage, IS/OOS,
  metrics-by-sample.
- `dashboard/pages/backtest.py` — three new controls.
- `dashboard/callbacks/backtest_callbacks.py` — plumb through.
- `tests/test_backtest_realism.py` — new file. Tests:
  - Slippage shifts both legs of the trade (round-trip P&L drops by 2× bps).
  - Next-open fill differs from same-close fill on a contrived 2-bar series.
  - IS/OOS split by date is monotonic, sums match, OOS share = 30 ± 1%.

### Estimated effort
**~2h.**  The engine refactor is the meat; UI is straightforward.

---

## Phase 2 — Top-15 well-tested strategies

**Goal.** Have a curated, documented strategy library shipped as importable
`BaseModel` subclasses.  Each strategy comes with: paper / source citation,
default params, recommended universe, expected behaviour, and a one-line
description shown in the dashboard model dropdown.

### How I'll pick the 15

Run the `general-purpose` agent with `WebSearch` + `WebFetch` to survey:

- **Trend-following classics**: 50/200 SMA crossover, Donchian channel
  (Turtle), Donchian breakout w/ ATR sizing, ADX-trend filter.
- **Mean-reversion classics**: RSI(2) by Larry Connors, Bollinger Band
  reversal, Z-score reversion, internal-bar-strength.
- **Momentum**: dual-momentum (Antonacci), Jegadeesh & Titman 12-1 cross-
  sectional, SPY 200d trend filter w/ momentum stocks.
- **Volatility**: Keltner channel breakout, Chandelier exit, supertrend.
- **Volume / order-flow**: OBV trend confirmation, VWAP reversion intraday
  (NB: we only have daily — defer if intraday-only).
- **Sentiment / news**: news-shock fade, post-earnings drift (PEAD).
- **Pattern**: VCP / Qullamaggie (already in registry — counts toward 15),
  ORB (opening-range breakout — daily-friendly only on the long side),
  episodic pivot.

The agent ranks them on **published OOS expectancy / Sharpe** (we'll tag
each with what the source paper / book / blog reported), not on its own
backtest, because backtest comparability requires Phase 1 to exist first.
After Phase 1 lands we re-rank using our own OOS Sharpe on our universe.

### Deliverables

For each of the 15:

- A new file `bot/models/builtin/<strategy_id>.py` implementing
  `BaseModel.predict_batch` (most are vectorisable on the existing 27
  feature columns; some need new columns).
- New helper columns added to `bot/feature_engineer.py` or
  `bot/patterns.py` when the strategy needs them (e.g. ADX, Donchian,
  Keltner, supertrend, OBV).
- One-line registration via `@register_model`.
- Auto-import line in `bot/models/registry.py::_ensure_builtin_imports`.
- A short docstring with: source citation (URL or book chapter), what the
  signal means, default exit philosophy.

Plus a single `bot/strategies/CATALOGUE.md` markdown file summarising all
15 with their source links — for the dashboard "Help" hover and so I (the
maintainer) can audit later.

### Files

- 15 × `bot/models/builtin/*.py`
- `bot/feature_engineer.py` (extend), `bot/patterns.py` (extend)
- `bot/models/registry.py` (auto-imports)
- `bot/strategies/CATALOGUE.md` (new)
- `tests/test_strategy_library.py` — for each registered strategy:
  - imports cleanly,
  - has metadata,
  - `predict_batch` returns valid signals on a synthetic OHLCV df.
- A bulk-rank script `scripts/rank_strategies.py` that runs all 15 on the
  full eligible universe (Phase-1 OOS) and writes a leaderboard json to
  `dashboard/backtests/leaderboard.json`.

### Estimated effort
**~4h.**  Most of the time is the research + careful implementation; the
test scaffold is shared.

---

## Phase 3 — Strategy Finder tab

**Goal.** A new dashboard tab where Claude proposes parameter combinations
for a chosen strategy (or all of them), backtests them on a small universe
slice, sorts by gains / Sharpe / expectancy, and lets the user save the
winner with a custom name + description.

### How it works

1. User picks a strategy (or "All") + a universe scope + a date range.
2. The Strategy Finder backend builds an initial parameter grid (the
   strategy's tunable params with sensible ranges).  E.g. for `rsi_macd_v1`:
   `BUY_RSI_THRESHOLD ∈ {20, 25, 30, 35}`, `SELL_RSI_THRESHOLD ∈ {65, 70,
   75, 80}`.
3. We run all combinations on the **in-sample** half (Phase 1 enables
   this), record metrics.
4. **Anthropic suggestion loop** (claude-api skill): the leaderboard so
   far is shown to Claude with a system prompt explaining the param
   space and the goal ("propose 3 next-tries that look promising").
   Claude returns a structured `ParamSuggestion` via tool-use; we run
   them, append to the leaderboard.
5. After N rounds (default 3) or when no suggestion improves Sharpe by
   more than 5%, we stop.
6. Final leaderboard re-runs the top-K configurations on the **OOS** half
   to confirm the edge holds.
7. UI shows: ranked table (by total return, Sharpe, expectancy — sortable
   columns), each row has a "Save as new strategy" button → opens a
   modal asking for name + description, writes a JSON spec under
   `dashboard/custom_models/` (existing `CustomRuleModel` machinery).

### Deliverables

- `bot/strategy_finder.py` — orchestrator.  Functions:
  - `param_grid(strategy_id) -> dict[str, list]`
  - `run_grid(strategy_id, grid, ...) -> pd.DataFrame leaderboard`
  - `suggest_next(leaderboard_df, strategy_meta) -> list[ParamSuggestion]`
    — uses Anthropic API, tool-choice forced, prompt caching.
  - `confirm_oos(top_k_configs, ...) -> pd.DataFrame`
- `dashboard/pages/strategy_finder.py` — new page, similar layout to
  Backtest.
- `dashboard/callbacks/strategy_finder_callbacks.py` — wire UI.
- `dashboard/components/global_controls.py` — add "Finder" topbar button.
- `tests/test_strategy_finder.py` — mocks the Anthropic client, tests the
  grid runner deterministically.

### claude-api specifics (per the skill)

- Model: `claude-sonnet-4-5` (default for tool-use; cheap + capable enough).
- System prompt cached with `cache_control: ephemeral` so the (large)
  strategy-catalogue context only counts once per session.
- Tool: `propose_parameter_tries` with strict JSON schema.
- Force `tool_choice = {"type": "tool", "name": "propose_parameter_tries"}`
  for guaranteed structured output.
- Per-call cost target: < $0.05 for a typical 3-round suggestion loop.

### Estimated effort
**~3h.**

---

## Phase 4 — Validate end-to-end

- Run `scripts/rank_strategies.py` against the universe for the IS half;
  re-run top-5 on OOS; persist the winner snapshot to
  `dashboard/backtests/seed_leaderboard.json` so the dashboard has data the
  first time the user opens the Finder tab.
- Add an end-to-end smoke test exercising the Finder: mock Anthropic,
  confirm leaderboard generation, confirm OOS confirmation step runs.
- Update `HANDOFF.md` with the new components and run instructions.

### Estimated effort
**~30min.**

---

## Risks / open questions

1. **Data depth.** Several strategies (12-1 momentum, PEAD) want 7+ years
   of history.  We have 6.  Acceptable for v1; the strategy doc will
   flag any that need more.
2. **No intraday data.** ORB / VWAP-reversion / news-shock-fade are
   really intraday strategies.  We'll either (a) implement a daily-bar
   approximation (works only for ORB long-side + PEAD), or (b) flag them
   as "intraday — needs minute bars" and skip implementation.  I'll
   default to (a) and document.
3. **Anthropic API costs.** Strategy Finder makes paid API calls.  The
   skill's prompt-caching brings the marginal cost to ~$0.005 per
   suggestion round.  Worst case for the UI: if a user kicks off 10
   searches per day with 5 rounds each, ~$0.25/day.  Acceptable; will
   show running cost estimate in the UI.
4. **Look-ahead bias in news-driven models.**  Phase 1 plumbing handles
   bar-level look-ahead but news timestamps need an explicit
   "available-at" check — out of scope here, captured as Phase 5
   (future work).

---

## Total timeline

| Phase | Effort | Cumulative |
|---|---|---|
| 1. Realistic backtesting | 2h | 2h |
| 2. Top-15 strategy library | 4h | 6h |
| 3. Strategy Finder tab | 3h | 9h |
| 4. Validate + handoff update | 30min | ~9.5h |

I'll execute one phase per session so each one ships cleanly with tests
and a commit.
