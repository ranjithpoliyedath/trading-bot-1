#!/usr/bin/env bash
#
# scripts/setup_cron.sh
# ----------------------
# Sets up daily cron jobs for the trading bot:
#   - 6:00 AM daily — refresh universe (S&P constituents + filters)
#   - 6:30 AM daily — incremental data pipeline (top 100 symbols)
#   - 11:00 PM weekdays — fetch next batch of 100 symbols (background rollout)
#
# Run once to install:
#     bash scripts/setup_cron.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="$PROJECT_DIR/venv/bin/python"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"

# Build the cron entries
CRON_UNIVERSE="0 6 * * * cd $PROJECT_DIR && $PYTHON_BIN -m bot.universe >> $LOG_DIR/cron_universe.log 2>&1"
CRON_PIPELINE="30 6 * * * cd $PROJECT_DIR && $PYTHON_BIN -m bot.pipeline >> $LOG_DIR/cron_pipeline.log 2>&1"
CRON_BATCH="0 23 * * 1-5 cd $PROJECT_DIR && $PYTHON_BIN scripts/rollout_next_batch.py >> $LOG_DIR/cron_rollout.log 2>&1"

# Install — preserving existing crontab
( crontab -l 2>/dev/null | grep -v "bot.universe\|bot.pipeline\|rollout_next_batch" ; \
  echo "$CRON_UNIVERSE" ; \
  echo "$CRON_PIPELINE" ; \
  echo "$CRON_BATCH" ) | crontab -

echo "Cron jobs installed:"
echo "  06:00 daily      — universe refresh"
echo "  06:30 daily      — incremental data pipeline"
echo "  23:00 weekdays   — next batch rollout"
echo
echo "View with:  crontab -l"
echo "Logs in:    $LOG_DIR/"
