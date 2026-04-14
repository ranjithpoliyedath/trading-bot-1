# CLAUDE.md — Trading Bot Project Guidelines

> This file is automatically read by Claude when working on this repository.
> It defines coding standards, folder structure, security rules, and contribution guidelines.

---

## Project Overview

- **Name**: trading-bot-1
- **Language**: Python 3.10+
- **Exchange**: Alpaca Markets (REST API + WebSocket)
- **Strategy**: ML / AI-based signal generation
- **Repo**: https://github.com/ranjithpoliyedath/trading-bot-1

---

## Folder Structure

```
trading-bot-1/
├── bot/
│   ├── __init__.py
│   ├── main.py              # Entry point — starts the bot
│   ├── trader.py            # Order execution logic (Alpaca API calls)
│   ├── strategy.py          # ML/AI signal generation
│   ├── data_fetcher.py      # Market data retrieval (Alpaca, yfinance, etc.)
│   └── risk_manager.py      # Position sizing, stop-loss, drawdown limits
│
├── models/
│   ├── train.py             # Model training scripts
│   ├── evaluate.py          # Backtesting and evaluation
│   └── saved/               # Serialized model files (.pkl, .h5, .pt)
│
├── config/
│   ├── settings.py          # App-wide config (loaded from .env)
│   └── logging.yaml         # Logging configuration
│
├── tests/
│   ├── test_trader.py
│   ├── test_strategy.py
│   └── test_data_fetcher.py
│
├── notebooks/               # Jupyter notebooks for research/exploration
│   └── strategy_research.ipynb
│
├── logs/                    # Runtime logs (gitignored)
├── .env.example             # Template for environment variables (NO real keys)
├── .env                     # Real secrets — NEVER commit this
├── .gitignore
├── requirements.txt
├── README.md
└── CLAUDE.md                # This file
```

---

## Coding Standards

### General
- Use **Python 3.10+** syntax and type hints everywhere
- Follow **PEP 8** style guidelines
- Max line length: **100 characters**
- Use **f-strings** for string formatting (not `.format()` or `%`)
- Prefer **explicit** over implicit — no magic numbers, always use named constants

### Functions & Classes
- Every function must have a **docstring** explaining purpose, args, and return value
- Keep functions small and single-purpose (< 40 lines ideally)
- Use **dataclasses** or **Pydantic models** for structured data
- Prefer **composition over inheritance**

### Error Handling
- Always wrap Alpaca API calls in `try/except` blocks
- Log all exceptions with full traceback using the `logging` module
- Never use bare `except:` — always catch specific exception types
- On critical errors (e.g. failed order), alert via log and halt gracefully

### Logging
- Use Python's built-in `logging` module (configured via `config/logging.yaml`)
- Log levels: `DEBUG` for data details, `INFO` for trades, `WARNING` for anomalies, `ERROR` for failures
- Never use `print()` in production code — always use the logger

---

## ML / AI Strategy Rules

- All model training code lives in `models/train.py`
- Trained models are saved to `models/saved/` as versioned files (e.g. `model_v1.pkl`)
- Never hardcode feature lists — define them as constants in `strategy.py`
- Always split data into train/validation/test sets — no data leakage
- Log model version, accuracy metrics, and feature importance on each training run
- Include a `predict()` method that returns both signal direction and confidence score
- Backtest every new model version before deploying — use `models/evaluate.py`

---

## Alpaca API Rules

- All Alpaca interactions go through `bot/trader.py` — no direct API calls elsewhere
- Use **paper trading** endpoints for all testing (`ALPACA_BASE_URL=https://paper-api.alpaca.markets`)
- Switch to live trading only when explicitly configured via `.env`
- Always check account buying power before placing any order
- Respect **rate limits** — add delays between rapid API calls
- Log every order placed, filled, or rejected with timestamp and full details

---

## Security Rules (CRITICAL)

- **NEVER commit `.env`** — it contains real API keys
- **NEVER hardcode API keys, secrets, or tokens** anywhere in the code
- All secrets must be loaded via `python-dotenv` from `.env`
- `.env.example` must always be kept up to date with all required variable names (no values)
- Do not log API keys, account IDs, or secret tokens even at DEBUG level
- If Claude creates new config variables, add them to `.env.example` with placeholder values

### Required .env variables
```
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

---

## Testing Requirements

- Every new function in `bot/` must have a corresponding unit test in `tests/`
- Use `pytest` as the test framework
- Mock all Alpaca API calls in tests — never hit the real API in tests
- Tests must pass before Claude opens a PR
- Run tests with: `pytest tests/ -v`

---

## Git & PR Rules

- Branch naming: `feature/description`, `fix/description`, `model/description`
- Commit messages must be descriptive:
  - ✅ `feat: add RSI fallback signal in strategy.py`
  - ❌ `update stuff`
- Claude must never merge its own PRs — always open for human review
- Each PR must include a description of what changed and why
- Do not modify `.env` or `models/saved/` in any PR

---

## Commands Claude Can Use

```bash
# Install dependencies
pip install -r requirements.txt

# Run the bot (paper trading)
python bot/main.py

# Train the model
python models/train.py

# Run backtesting
python models/evaluate.py

# Run all tests
pytest tests/ -v

# Check code style
flake8 bot/ models/ tests/ --max-line-length=100
```

---

## What Claude Should NOT Do

- Never place real trades or switch to live trading mode
- Never delete or overwrite files in `models/saved/` without asking
- Never commit `.env` or any file containing real credentials
- Never skip writing tests for new code
- Never modify `requirements.txt` without listing the reason in the PR description
- Never use `os.system()` — use `subprocess` with explicit args if shell calls are needed
