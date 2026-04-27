# Strategy catalogue

All registered strategies live as `BaseModel` subclasses under
`bot/models/builtin/`. Each maps to one `@register_model` class that the
dashboard's model dropdown and the Strategy Finder pick up automatically.

The library was assembled with a quality gate (Phase 3 of the build
plan): each candidate has to clear (a) a peer-reviewed paper or
well-known book chapter as primary source, (b) be implementable on our
6-year (currently ~2-year on disk) daily-bar dataset without
compromise, (c) produce non-zero trades and a positive Sharpe / return
when run on the top-20 most-liquid symbols across the available
history.

Strategies that don't currently clear the gate are still **registered**
so the Strategy Finder can re-tune them — they're just flagged below
as "experimental" with the diagnostic that needs work.

---

## Production library (validated)

| ID | Name | Type | Source | Validation |
|---|---|---|---|---|
| `rsi_macd_v1` | RSI + MACD | Mean-rev / momentum | classical TA | shipped pre-Phase 3 |
| `bollinger_v1` | Bollinger + Sentiment | Mean reversion | Bollinger (1980s) | shipped pre-Phase 3 |
| `sentiment_v1` | Sentiment-driven | Sentiment | internal | shipped pre-Phase 3 |
| `qullamaggie_v1` | Qullamaggie breakout | Pattern | Kullamägi public talks / blog | shipped pre-Phase 3 |
| `vcp_v1` | VCP breakout | Pattern | Minervini, *Trade Like a Stock Market Wizard* (2013) | shipped pre-Phase 3 |
| `golden_cross_v1` | SMA 50/200 Golden Cross | Trend | Murphy, *Technical Analysis of the Financial Markets* (1999) | Sharpe 0.50, +12.76%, 153 trades |
| `connors_rsi2_v1` | Connors RSI(2) | Mean reversion | Connors & Alvarez, *Short-Term Trading Strategies That Work* (2009) | Sharpe 1.23, +84.08%, 67.6% win |
| `ibs_v1` | Internal Bar Strength | Mean reversion | Liew & Roberts (2013) / QuantPedia | Sharpe 0.72, +35.03%, 56.6% win |
| `zscore_reversion_v1` | Z-score reversion | Mean reversion | Avellaneda-Lee (2010) / classical stat-arb | Sharpe 0.64, +31.00%, 64.9% win |

Validation = single-period backtest, top 20 most-liquid symbols, 730d
lookback, conf threshold 0.55, default exits (TP 15% / SL 7% /
time-stop 30d), no slippage. Published in `dashboard/backtests/`.

---

## Experimental / needs tuning

These cleared the source-citation bar but their default parameters
either produce too few trades or negative cumulative return on our
universe. They're registered so the Strategy Finder can search for
better params — open the Finder tab, pick the strategy, click Run.

| ID | Source | Symptom on default params | Likely fix |
|---|---|---|---|
| `donchian_v1` | Faith, *Way of the Turtle* (2007) | 207 trades, Sharpe −0.71, −193% return | Period too short for equities; try `breakout_period` 40-60 |
| `adx_trend_v1` | Wilder, *New Concepts in Technical Trading* (1978) | 128 trades, Sharpe 0.68, −98% return | Tighten ADX gate to 30+ via Finder, add a wider stop |
| `keltner_breakout_v1` | Keltner (1960) / Linda Raschke | 166 trades, Sharpe −0.47, −66% return | ATR multiplier 2.5-3.0 typically beats 2.0 in equities |
| `obv_momentum_v1` | Granville (1963) | 324 trades, Sharpe 0.69 but −101% return | Tighten min 5d return; pair with a volatility filter |

---

## Implementation notes

* Every strategy is fully self-contained: `predict_batch` lazily
  appends any indicator columns it needs via `bot.indicators`, so the
  base feature pipeline doesn't have to be re-run when a new strategy
  is added.
* All strategies emit confidence in the 0.50–0.95 band, so the engine's
  `conf_threshold` works consistently across them. Lowering the
  threshold = trading lower-conviction signals.
* Sell signals are *suggestions* — the engine's exit rules
  (`take_profit_pct`, `stop_loss_pct`, `atr_stop_mult`,
  `time_stop_days`, plus `use_signal_exit` for the model's own sell)
  ultimately decide when a position closes.
* Param spaces for the Strategy Finder live in
  `bot/strategy_finder.py::PARAM_SPACES`. Tunables are kept narrow and
  centred on each strategy's published default — we're searching the
  neighbourhood of well-known settings, not from scratch.

## Skipped strategies and why

| Strategy | Reason |
|---|---|
| Jegadeesh-Titman 12-1 momentum | Cross-sectional ranking; doesn't fit per-symbol `BaseModel` interface. Future addition once the engine grows multi-symbol joint backtests. |
| Antonacci dual momentum | Same — cross-sectional + needs longer-than-6yr history. |
| PEAD (Post-Earnings Drift) | Needs earnings dates; not in our data store yet. |
| News-shock fade / event-driven | Needs intraday news timestamps + intraday bars. |
| ORB (Opening Range Breakout) | Intraday bars required. |
| Chandelier exit / SuperTrend | These are exit rules, not entry strategies — already exposed as `atr_stop_mult` in the engine. |
