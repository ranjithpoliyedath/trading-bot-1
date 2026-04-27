# Plan — strategy library + Strategy Finder + realistic backtesting

This plan covers three asks in one rollout.  Build order:

1. **Realistic backtesting upgrades** — look-ahead-bias prevention,
   execution delay, slippage, **walk-forward IS/OOS validation (4 folds)**.
2. **Strategy Finder dashboard tab** — Optuna-driven parameter search on
   every registered strategy (default, free), with an opt-in
   "🤖 Ask Claude" button for unconventional ideas.  Built first against
   the existing 5 strategies so the auto-tuner ships quickly.
3. **Quality-gated ≤10 well-tested strategies** — research them on the
   internet, register each as a `BaseModel` in `bot/models/builtin/`,
   ranked by walk-forward Sharpe.  These plug into the already-built
   Finder for free.

Order rationale:
- Phase 1 first — everything else depends on honest numbers.
- Phase 2 (Finder) before Phase 3 (more strategies) because the Finder
  works against any registered strategy; users see the auto-tuner
  light up immediately on the existing 5 models, and each new strategy
  added in Phase 3 just shows up in the dropdown.

---

## Skills + agents I'll use

I'm calling out the skill / agent for each phase up front so the work plan
is honest about what executes where.  Only skills / agents I have available
are listed; nothing speculative.

| Phase | Skill / Agent | Why |
|---|---|---|
| 1 | `Plan` agent (review only) + `general-purpose` agent | Architecture sanity-check on the look-ahead + walk-forward refactor; then implementation across `dashboard/backtest_engine.py` and tests. |
| 2 | `general-purpose` agent (default) — **`claude-api` skill only for opt-in button** | Strategy Finder uses **Optuna** (free, deterministic Bayesian optimization) for the core suggestion loop.  Optional "🤖 Ask Claude" button keeps the Anthropic path for unconventional ideas — that's where the `claude-api` skill (prompt caching, model versioning, forced tool-use) applies.  Default cost: $0. |
| 3 | `general-purpose` agent with **`WebSearch` + `WebFetch`** tools | Research a quality-gated ≤10 strategies on the open internet (Investopedia, SSRN, QuantPedia, papers, well-known traders' write-ups).  Then implement each as a `BaseModel` — they auto-appear in the Finder built in Phase 2. |
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
- No IS / OOS validation — we tune on the same data we test on, so any
  "winning" strategy is suspect.

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

4. **Walk-forward validation (4 folds)** — new helper
   `dashboard/backtest_engine.py::walk_forward_folds(df, n_folds=4)`
   that yields a list of (in_sample_df, out_of_sample_df) tuples.  For
   our 6-year dataset:

       Fold 1:  IS = 2020-2022       OOS = 2023
       Fold 2:  IS = 2020-2023       OOS = 2024
       Fold 3:  IS = 2020-2024       OOS = 2025
       Fold 4:  IS = 2020-2025       OOS = 2026

   `run_backtest` / `run_filtered_backtest` gain:
   - `sample: Literal["all", "in", "out"]` for single-fold runs.
   - A new `run_walk_forward(strategy_id, n_folds=4, ...) -> dict` that
     wraps the loop, returns per-fold metrics + aggregate (mean Sharpe,
     dispersion, % of folds positive).  This is what the Strategy
     Finder calls in Phase 2.

   Why walk-forward over static 70/30: a single 70/30 split tests
   robustness against one specific 1.8-year slice — strategies that
   work only in that regime look great.  Walk-forward gives 4 OOS
   scores so we can see if the edge holds across different market
   conditions, not just the most recent one.

5. **Dashboard exposure** — four new controls on the Backtest page:
   - "Execution model" dropdown (`Next-bar open` / `Same-bar close`)
   - "Slippage (bps)" numeric input (default 5)
   - "Validation mode" radio (`Full history` / `Walk-forward (4 folds)`)
   - When walk-forward selected, results show a small per-fold table
     (Fold 1 Sharpe, Fold 2 Sharpe, ..., aggregate) above the existing
     metrics row.

### Files

- `dashboard/backtest_engine.py` — execution model, slippage,
  walk-forward folds, per-sample metrics.
- `dashboard/pages/backtest.py` — four new controls + per-fold result
  view.
- `dashboard/callbacks/backtest_callbacks.py` — plumb through.
- `tests/test_backtest_realism.py` — new file. Tests:
  - Slippage shifts both legs of the trade (round-trip P&L drops by 2× bps).
  - Next-open fill differs from same-close fill on a contrived 2-bar series.
  - Walk-forward folds are date-monotonic, no IS/OOS overlap inside a
    fold, OOS chunks tile the post-burn-in period exactly once.

### Estimated effort
**~2.5h** (was 2h; walk-forward adds about 30 minutes for the loop +
per-fold UI display).  The engine refactor is the meat; UI is
straightforward.

---

## Phase 2 — Strategy Finder tab

**Goal.** A new dashboard tab where the system proposes parameter
combinations for a chosen strategy (or all of them), backtests them on a
universe slice with walk-forward validation, sorts by gains / Sharpe /
expectancy, and lets the user save the winner with a custom name +
description.

This phase ships before more strategies because the Finder works against
any registered model — the existing 5 (rsi_macd_v1, bollinger_v1,
sentiment_v1, qullamaggie_v1, vcp_v1) are enough to validate the loop.
New strategies added in Phase 3 plug into this Finder for free.

### How it works (default: Optuna, $0)

1. User picks a strategy (or "All") + a universe scope + a date range.
2. The Strategy Finder backend declares the strategy's parameter search
   space (e.g. `BUY_RSI_THRESHOLD ∈ [15, 40]` int, `SELL_RSI_THRESHOLD ∈
   [60, 85]` int, `min_confidence ∈ [0.50, 0.85]` float).
3. **Optuna** (TPE sampler, seed for reproducibility) drives the loop.
   The objective function for each trial runs **walk-forward (4 folds)**
   from Phase 1 and returns mean-OOS-Sharpe across folds — so we
   optimise for *robustness across regimes*, not "looked great in 2024".
4. After N trials (default 50, configurable) or when the running best
   walk-forward Sharpe hasn't improved for 10 trials, we stop.
5. Final leaderboard re-runs the top-K configurations on a held-out
   final segment to confirm.
6. UI shows: ranked table (by mean walk-forward Sharpe, total return,
   expectancy — sortable columns), per-fold breakdown on click, each
   row has a "Save as new strategy" button → opens a modal asking for
   name + description, writes a JSON spec under
   `dashboard/custom_models/` (existing `CustomRuleModel` machinery).

### Optional: "🤖 Ask Claude for unconventional ideas" button

Optuna is great at *interpolating* within a defined param space, but
won't suggest cross-strategy hybrids or reason about the trading domain
("what if we add a volume filter to RSI(2)?"). For that creative escape
hatch:

- One button on the Finder page.  Default state: not clicked, $0 spent.
- When clicked, sends the current leaderboard + the registry catalogue
  to Anthropic with a forced tool-call asking for 3 unconventional
  combinations to try.
- Each suggestion is appended to the Optuna study as a manual trial.
- Cost ~$0.05 per click.  No background calls, no surprise bills.
- This is where the **`claude-api` skill** applies — prompt caching of
  the static catalogue so per-click cost stays low, forced tool-use
  for structured output, model name (`claude-sonnet-4-5`).

### Deliverables

- `bot/strategy_finder.py` — orchestrator.  Functions:
  - `param_space(strategy_id) -> dict` — Optuna distribution objects
    keyed by param name.
  - `run_optuna(strategy_id, n_trials=50, seed=42, ...) -> Study + leaderboard df`
    — uses Phase 1's walk-forward folds inside the objective.
  - `suggest_with_claude(leaderboard_df, strategy_meta) -> list[ParamSuggestion]`
    — opt-in only, called when the button is clicked.  Anthropic API,
    tool-choice forced, prompt caching.
  - `confirm_holdout(top_k_configs, ...) -> pd.DataFrame`
- `dashboard/pages/strategy_finder.py` — new page, similar layout to
  Backtest.
- `dashboard/callbacks/strategy_finder_callbacks.py` — wire UI.
- `dashboard/components/global_controls.py` — add "Finder" topbar button.
- `requirements.txt` — add `optuna>=3.4.0`.
- `tests/test_strategy_finder.py` — covers the Optuna path
  deterministically (no mocking needed) and mocks the Anthropic client
  for the opt-in path.

### Estimated effort
**~3.5h** (Optuna setup, walk-forward objective wiring, opt-in Anthropic
path, UI).

---

## Phase 3 — Top ≤10 well-tested strategies (quality-gated)

**Goal.** Have a curated, documented strategy library shipped as
importable `BaseModel` subclasses.  Each strategy comes with: paper /
source citation, default params, recommended universe, expected
behaviour, and a one-line description shown in the dashboard model
dropdown.  Each one auto-appears in the Finder built in Phase 2.

**Quality bar.** Each strategy must (a) have a peer-reviewed paper or
well-known book chapter as primary source, (b) be implementable on our
6-year daily-bar dataset without compromise, (c) clear an absolute mean
walk-forward Sharpe > 0.3 on at least one S&P slice in our own
validation (Phase 1 + Phase 2 pipeline).  Strategies that fail any gate
get dropped — we ship 6 solid ones over 10 mediocre.

### How I'll pick them

Run the `general-purpose` agent with `WebSearch` + `WebFetch` to survey:

- **Trend-following classics**: 50/200 SMA crossover, Donchian channel
  (Turtle), Donchian breakout w/ ATR sizing, ADX-trend filter.
- **Mean-reversion classics**: RSI(2) by Larry Connors, Bollinger Band
  reversal, Z-score reversion, internal-bar-strength.
- **Momentum**: dual-momentum (Antonacci), Jegadeesh & Titman 12-1
  cross-sectional, SPY 200d trend filter w/ momentum stocks.
- **Volatility**: Keltner channel breakout, Chandelier exit, supertrend.
- **Volume / order-flow**: OBV trend confirmation, VWAP reversion
  intraday (NB: we only have daily — defer if intraday-only).
- **Sentiment / news**: news-shock fade, post-earnings drift (PEAD).
- **Pattern**: ORB (opening-range breakout — daily-friendly only on
  the long side), episodic pivot.  (VCP / Qullamaggie already shipped.)

For each candidate I run a quick walk-forward sanity-check in the
Phase 2 Finder to confirm it clears the Sharpe gate before writing
the production implementation — this catches "famous on the internet
but doesn't actually work on our universe" failures cheaply.

### Deliverables

For each of the ≤10 that clear the quality gate:

- A new file `bot/models/builtin/<strategy_id>.py` implementing
  `BaseModel.predict_batch` (most are vectorisable on the existing 27
  feature columns; some need new columns).
- New helper columns added to `bot/feature_engineer.py` or
  `bot/patterns.py` when the strategy needs them (e.g. ADX, Donchian,
  Keltner, supertrend, OBV).
- One-line registration via `@register_model`.
- Auto-import line in `bot/models/registry.py::_ensure_builtin_imports`.
- A short docstring with: source citation (URL or book chapter), what
  the signal means, default exit philosophy.

Plus a single `bot/strategies/CATALOGUE.md` markdown file summarising
every strategy that shipped with its source links — for the dashboard
"Help" hover and so I (the maintainer) can audit later.

### Files

- Up to 10 × `bot/models/builtin/*.py` (the ones that pass the gate)
- `bot/feature_engineer.py` (extend), `bot/patterns.py` (extend)
- `bot/models/registry.py` (auto-imports)
- `bot/strategies/CATALOGUE.md` (new)
- `tests/test_strategy_library.py` — for each registered strategy:
  imports cleanly, has metadata, `predict_batch` returns valid signals
  on a synthetic OHLCV df.
- A bulk-rank script `scripts/rank_strategies.py` that runs every
  shipped strategy through Phase 1 walk-forward + Phase 2 grid on the
  full eligible universe and writes a leaderboard json to
  `dashboard/backtests/leaderboard.json`.

### Estimated effort
**~3h** (was 4h; quality gate caps the implementation set).  Most of
the time is the research + careful implementation; the test scaffold
is shared.

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
3. **Anthropic API costs.** Default Strategy Finder uses Optuna and
   costs $0.  The optional "🤖 Ask Claude" button costs ~$0.05 per
   click (prompt caching keeps it down).  No background calls, no
   surprise bills.  Click-to-spend model so the user always knows.
4. **Look-ahead bias in news-driven models.**  Phase 1 plumbing handles
   bar-level look-ahead but news timestamps need an explicit
   "available-at" check — out of scope here, captured as Phase 5
   (future work).

---

## Total timeline

| Phase | Effort | Cumulative |
|---|---|---|
| 1. Realistic backtesting (look-ahead, slippage, walk-forward) | 2.5h | 2.5h |
| 2. Strategy Finder tab (Optuna default + opt-in Claude) | 3.5h | 6h |
| 3. Top ≤10 strategy library (quality-gated) | 3h | 9h |
| 4. Validate + handoff update | 30min | ~9.5h |

I'll execute one phase per session so each one ships cleanly with tests
and a commit.
