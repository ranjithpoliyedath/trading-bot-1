"""
bot/config.py
--------------
Central configuration for the trading bot.
Edit this file to change universe size, lookback, and other settings.
"""

from pathlib import Path


# ── Universe ──────────────────────────────────────────────────────────────────

# How many top symbols to keep from the eligible universe
# (sorted by 14-day avg volume — highest liquidity first)
INITIAL_UNIVERSE_SIZE  = 100   # Phase 1 — fetch data for these now
TARGET_UNIVERSE_SIZE   = 700   # Total target — remaining ~600 fetched on schedule


# ── Lookback periods ──────────────────────────────────────────────────────────

# Years of OHLCV history to fetch for each symbol
DATA_LOOKBACK_YEARS = 6
DATA_LOOKBACK_DAYS  = DATA_LOOKBACK_YEARS * 365

# Days of news history to fetch from Alpaca News API
NEWS_LOOKBACK_DAYS = 60

# Trading-day close in UTC.  Articles / posts published *after* this
# hour get bucketed into the *next* trading day's sentiment window —
# an article released at 6pm ET on Monday wasn't visible to a
# Monday-close trade.  20:00 UTC ≈ 4pm ET (US equity close).
# Set to None to revert to the legacy "any time of day = today" bucketing.
SENTIMENT_CUTOFF_HOUR_UTC = 20


# ── Feature engineering ───────────────────────────────────────────────────────

SIGNAL_CONFIDENCE_THRESHOLD = 0.65
SENTIMENT_MOMENTUM_WINDOW   = 3


# ── Data paths ────────────────────────────────────────────────────────────────

ROOT_DIR        = Path(__file__).parent.parent
DATA_DIR        = ROOT_DIR / "data" / "processed"
RAW_DIR         = ROOT_DIR / "data" / "raw"
UNIVERSE_FILE   = ROOT_DIR / "data" / "universe.parquet"
MODELS_DIR      = ROOT_DIR / "models" / "saved"
BACKTEST_DIR    = ROOT_DIR / "dashboard" / "backtests"
LOG_DIR         = ROOT_DIR / "logs"

for _dir in [DATA_DIR, RAW_DIR, MODELS_DIR, BACKTEST_DIR, LOG_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ── Logging ───────────────────────────────────────────────────────────────────

LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


# ── Alpaca ────────────────────────────────────────────────────────────────────

BAR_TIMEFRAME = "1Day"

# Free tier must use 'iex'; paid tier can use 'sip' for full market coverage
ALPACA_FEED = "iex"

# Rate limiting — Alpaca free tier: 200 req/min
ALPACA_BATCH_SIZE        = 100
ALPACA_BATCH_DELAY_SECS  = 0.5


# ── Risk management ───────────────────────────────────────────────────────────

MAX_POSITION_PCT  = 0.10    # 10% max per trade
STOP_LOSS_PCT     = 0.05    # 5%
TAKE_PROFIT_PCT   = 0.15    # 15%


# ── Backwards compatibility ───────────────────────────────────────────────────
# Existing code uses bot.config.SYMBOLS — provide it dynamically from universe

def _load_symbols():
    """Dynamically load symbols from universe file."""
    try:
        from bot.universe import get_top_n_by_volume
        symbols = get_top_n_by_volume(n=INITIAL_UNIVERSE_SIZE, eligible_only=True)
        if symbols:
            return symbols
    except Exception:
        pass
    # Fallback for first run when universe doesn't exist yet
    return ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "META", "AMZN",
            "SPY", "QQQ", "IWM", "DIA", "VTI"]


SYMBOLS = _load_symbols()

# ── Index & Sector ETFs (always included in pipeline regardless of universe) ──
INDEX_ETFS  = ["SPY", "QQQ", "DIA", "IWM", "VTI"]
SECTOR_ETFS = ["XLK", "XLV", "XLF", "XLE", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC"]

