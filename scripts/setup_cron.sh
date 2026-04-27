#!/usr/bin/env bash
#
# scripts/setup_cron.sh
# ----------------------
# Daily cron jobs for the trading bot.  Run once to install, then
# `crontab -l` to verify.  Re-runnable: replaces the bot's own entries
# but leaves any other crontab lines you have alone.
#
#   06:00 daily    — refresh S&P universe
#   06:30 daily    — incremental OHLCV pipeline (existing symbols)
#   07:00 daily    — sentiment pipeline (news + StockTwits)
#   07:30 daily    — refresh seed leaderboard for the dashboard
#   23:00 weekdays — fetch next batch of 100 symbols (background rollout)
#
# Run:
#     bash scripts/setup_cron.sh             # install
#     bash scripts/setup_cron.sh --uninstall # remove
#     bash scripts/setup_cron.sh --print     # print without installing

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/venv/bin/python}"
LOG_DIR="$PROJECT_DIR/logs"
TAG="trading-bot-1"

mkdir -p "$LOG_DIR"

# Skip the venv check on --print / --uninstall — those don't run python.
case "${1:-install}" in
    --print|-p|--uninstall|-u) ;;
    *)
        if [ ! -x "$PYTHON_BIN" ]; then
            echo "❌ venv python not found at $PYTHON_BIN"
            echo "   Activate the venv (or set PYTHON_BIN) and pip install -r requirements.txt first."
            exit 1
        fi
        ;;
esac

# Each line tagged so we can grep them out cleanly on uninstall / re-install.
build_cron() {
    cat <<EOF
# >>> $TAG cron jobs (managed by scripts/setup_cron.sh) >>>
0  6 * * *   cd $PROJECT_DIR && $PYTHON_BIN -m bot.universe                                    >> $LOG_DIR/cron_universe.log  2>&1  # $TAG
30 6 * * *   cd $PROJECT_DIR && $PYTHON_BIN -m bot.pipeline                                    >> $LOG_DIR/cron_pipeline.log  2>&1  # $TAG
0  7 * * *   cd $PROJECT_DIR && $PYTHON_BIN -m bot.sentiment.sentiment_pipeline                >> $LOG_DIR/cron_sentiment.log 2>&1  # $TAG
30 7 * * *   cd $PROJECT_DIR && $PYTHON_BIN scripts/rank_strategies.py --scope top_100         >> $LOG_DIR/cron_rank.log      2>&1  # $TAG
0 23 * * 1-5 cd $PROJECT_DIR && $PYTHON_BIN scripts/rollout_next_batch.py                      >> $LOG_DIR/cron_rollout.log   2>&1  # $TAG
# <<< $TAG cron jobs <<<
EOF
}

current_without_tag() {
    crontab -l 2>/dev/null | grep -v "# $TAG" | grep -vE "^# (>|<){3} $TAG" || true
}

case "${1:-install}" in
    --print|-p)
        build_cron
        ;;
    --uninstall|-u)
        echo "Removing $TAG cron entries…"
        current_without_tag | crontab -
        echo "Done.  Remaining crontab:"
        crontab -l 2>/dev/null || echo "  (empty)"
        ;;
    install|"")
        echo "Installing $TAG cron jobs…"
        ( current_without_tag ; build_cron ) | crontab -
        echo
        echo "Installed:"
        echo "  06:00 daily      — universe refresh           → $LOG_DIR/cron_universe.log"
        echo "  06:30 daily      — incremental OHLCV pipeline → $LOG_DIR/cron_pipeline.log"
        echo "  07:00 daily      — sentiment pipeline         → $LOG_DIR/cron_sentiment.log"
        echo "  07:30 daily      — refresh seed leaderboard   → $LOG_DIR/cron_rank.log"
        echo "  23:00 weekdays   — next batch rollout         → $LOG_DIR/cron_rollout.log"
        echo
        echo "Verify:    crontab -l"
        echo "Uninstall: bash scripts/setup_cron.sh --uninstall"
        ;;
    *)
        echo "Unknown flag: $1"
        echo "Usage: bash scripts/setup_cron.sh [install|--print|--uninstall]"
        exit 2
        ;;
esac
