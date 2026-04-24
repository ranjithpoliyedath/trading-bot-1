"""
bot/config.py
--------------
Central configuration for the trading bot.
Edit this file to change symbols, lookback period, and other settings.
All pipeline modules read from here — change once, applies everywhere.
"""

# ── Symbols ────────────────────────────────────────────────────────────────────

TECH_SYMBOLS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "NVDA",   # Nvidia
    "TSLA",   # Tesla
    "GOOGL",  # Alphabet
    "META",   # Meta
    "AMZN",   # Amazon
]

ETF_SYMBOLS = [
    "SPY",    # S&P 500
    "QQQ",    # Nasdaq 100
    "IWM",    # Russell 2000
    "DIA",    # Dow Jones
    "VTI",    # Total US Market
]

FINANCE_SYMBOLS = [
    # Uncomment to enable
    # "JPM", "GS", "BAC", "V", "MA",
]

# Active symbols used by all pipelines
SYMBOLS = TECH_SYMBOLS + ETF_SYMBOLS + FINANCE_SYMBOLS


# ── Lookback periods ───────────────────────────────────────────────────────────

# Days of OHLCV + technical feature history to fetch and store
DATA_LOOKBACK_DAYS = 730          # 2 years

# Days of news history to fetch from Alpaca News API
NEWS_LOOKBACK_DAYS = 60           # 2 months (API limits older free-tier history)

# StockTwits always returns latest 30 posts per symbol (API limitation)


# ── Feature engineering ────────────────────────────────────────────────────────

# Minimum confidence threshold for ML signals (0.0 - 1.0)
SIGNAL_CONFIDENCE_THRESHOLD = 0.65

# Sentiment rolling window for momentum features (trading days)
SENTIMENT_MOMENTUM_WINDOW = 3


# ── Data paths ─────────────────────────────────────────────────────────────────

from pathlib import Path

ROOT_DIR      = Path(__file__).parent.parent
DATA_DIR      = ROOT_DIR / "data" / "processed"
MODELS_DIR    = ROOT_DIR / "models" / "saved"
BACKTEST_DIR  = ROOT_DIR / "dashboard" / "backtests"
LOG_DIR       = ROOT_DIR / "logs"

# Create directories if they don't exist
for _dir in [DATA_DIR, MODELS_DIR, BACKTEST_DIR, LOG_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ── Logging ────────────────────────────────────────────────────────────────────

LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


# ── Alpaca ─────────────────────────────────────────────────────────────────────

# Timeframe for OHLCV bars ("1Day", "1Hour", "15Min")
BAR_TIMEFRAME = "1Day"

# Alpaca data feed — free tier must use iex, paid tier can use sip
ALPACA_FEED = "iex"


# ── Risk management ────────────────────────────────────────────────────────────

# Max % of portfolio to allocate per trade
MAX_POSITION_PCT  = 0.10    # 10%

# Stop loss % below entry price
STOP_LOSS_PCT     = 0.05    # 5%

# Take profit % above entry price
TAKE_PROFIT_PCT   = 0.15    # 15%
