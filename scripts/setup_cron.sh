#!/usr/bin/env bash
#
# scripts/setup_cron.sh
# ----------------------
# Daily cron jobs for the trading bot.  Run once to install, then
# `crontab -l` to verify.  Re-runnable: replaces the bot's own entries
# but leaves any other crontab lines you have alone.
#
#   06:00 daily      — refresh S&P universe (constituent + volume changes)
#   06:30 weekdays   — incremental OHLCV pipeline (yfinance, full universe ~1500)
#   07:00 daily      — sentiment pipeline (news + StockTwits)
#   07:30 daily      — refresh seed leaderboard for the dashboard
#   23:00 weekdays   — fetch next batch of 100 symbols (background rollout)
#
# Notes:
#   • OHLCV runs weekdays only (markets closed Sat/Sun → nothing to fetch).
#   • OHLCV uses the yfinance source so we get the full ~10-year depth on
#     all ~1,500 universe symbols.  Free Alpaca IEX caps at ~5.8 years.
#   • Mode is INCREMENTAL by default — only fetches new bars since last
#     stored timestamp.  Pass --full to refetch everything (rarely needed).
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
0  6 * * *     cd $PROJECT_DIR && $PYTHON_BIN -m bot.universe                                                       >> $LOG_DIR/cron_universe.log  2>&1  # $TAG
30 6 * * 1-5   cd $PROJECT_DIR && $PYTHON_BIN -m bot.pipeline --full-universe --source=yfinance                     >> $LOG_DIR/cron_pipeline.log  2>&1  # $TAG
0  7 * * *     cd $PROJECT_DIR && $PYTHON_BIN -m bot.sentiment.sentiment_pipeline                                   >> $LOG_DIR/cron_sentiment.log 2>&1  # $TAG
30 7 * * *     cd $PROJECT_DIR && $PYTHON_BIN scripts/rank_strategies.py --scope top_100                            >> $LOG_DIR/cron_rank.log      2>&1  # $TAG
0 23 * * 1-5   cd $PROJECT_DIR && $PYTHON_BIN scripts/rollout_next_batch.py                                         >> $LOG_DIR/cron_rollout.log   2>&1  # $TAG
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
        echo "  06:00 daily      — universe refresh                 → $LOG_DIR/cron_universe.log"
        echo "  06:30 weekdays   — OHLCV pipeline (yfinance, ~1500) → $LOG_DIR/cron_pipeline.log"
        echo "  07:00 daily      — sentiment pipeline               → $LOG_DIR/cron_sentiment.log"
        echo "  07:30 daily      — refresh seed leaderboard         → $LOG_DIR/cron_rank.log"
        echo "  23:00 weekdays   — next batch rollout               → $LOG_DIR/cron_rollout.log"
        echo
        echo "Note: OHLCV runs incrementally — appends new bars only.  To"
        echo "      refetch everything pass --full at the CLI manually."
        echo
        echo "Verify:    crontab -l"
        echo "Logs:      ls -la $LOG_DIR/"
        echo "Uninstall: bash scripts/setup_cron.sh --uninstall"
        echo
        if [[ "$PROJECT_DIR" == "$HOME/Documents/"* || \
              "$PROJECT_DIR" == "$HOME/Desktop/"*  || \
              "$PROJECT_DIR" == "$HOME/Downloads/"* ]]; then
            echo "⚠ macOS PRIVACY NOTE — project lives in a TCC-protected folder"
            echo "  ($PROJECT_DIR)"
            echo
            echo "  By default, /usr/sbin/cron runs with restricted permissions"
            echo "  and CANNOT read files in ~/Documents, ~/Desktop, or"
            echo "  ~/Downloads.  Without granting access, the cron line"
            echo "  fires but Python fails with 'Operation not permitted'."
            echo
            echo "  Fix (one-time):"
            echo "    1. Open System Settings → Privacy & Security → Full Disk Access"
            echo "    2. Click + and add:  /usr/sbin/cron"
            echo "       (you may need ⇧⌘. to show hidden /usr in the picker)"
            echo "    3. Logs will start appearing in $LOG_DIR/cron_*.log"
            echo
            echo "  Alternative: bash scripts/setup_launchd.sh — uses launchd"
            echo "  instead, which is more macOS-native and survives sleep."
        fi
        ;;
    *)
        echo "Unknown flag: $1"
        echo "Usage: bash scripts/setup_cron.sh [install|--print|--uninstall]"
        exit 2
        ;;
esac
