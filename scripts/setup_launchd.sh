#!/usr/bin/env bash
#
# scripts/setup_launchd.sh
# ------------------------
# macOS-native alternative to cron.  Installs the daily OHLCV refresh
# as a launchd User Agent that fires at 06:30 every weekday.  Use this
# instead of (or alongside) scripts/setup_cron.sh when:
#
#   • Mac cron requires Full Disk Access on macOS 12+ (Privacy ➔
#     Full Disk Access ➔ + ➔ /usr/sbin/cron) and you don't want to
#     grant it system-wide.
#   • You want missed-run catch-up: launchd runs the job on next
#     wake if the Mac was asleep at the scheduled time.  Pure cron
#     silently skips it.
#
# Usage:
#   bash scripts/setup_launchd.sh             # install
#   bash scripts/setup_launchd.sh --uninstall # remove
#   bash scripts/setup_launchd.sh --status    # show load state + last exit
#   bash scripts/setup_launchd.sh --run-now   # trigger immediately
#
# The plist is written to ~/Library/LaunchAgents/ and managed via
# launchctl.  Logs land in $PROJECT_DIR/logs/cron_pipeline.log so they
# share the location used by setup_cron.sh — easy to combine.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/venv/bin/python}"
LOG_DIR="$PROJECT_DIR/logs"
LABEL="com.trading-bot-1.pipeline"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

mkdir -p "$LOG_DIR" "$(dirname "$PLIST")"

write_plist() {
    cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
   "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>

    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON_BIN</string>
        <string>-m</string>
        <string>bot.pipeline</string>
        <string>--full-universe</string>
        <string>--source=yfinance</string>
    </array>

    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>

    <!-- Fire at 06:30 every weekday (Mon=1 … Fri=5).  launchd will
         run the job on next wake if the Mac was asleep at the
         scheduled time — unlike cron which silently skips it. -->
    <key>StartCalendarInterval</key>
    <array>
        <dict>
            <key>Weekday</key><integer>1</integer>
            <key>Hour</key><integer>6</integer>
            <key>Minute</key><integer>30</integer>
        </dict>
        <dict>
            <key>Weekday</key><integer>2</integer>
            <key>Hour</key><integer>6</integer>
            <key>Minute</key><integer>30</integer>
        </dict>
        <dict>
            <key>Weekday</key><integer>3</integer>
            <key>Hour</key><integer>6</integer>
            <key>Minute</key><integer>30</integer>
        </dict>
        <dict>
            <key>Weekday</key><integer>4</integer>
            <key>Hour</key><integer>6</integer>
            <key>Minute</key><integer>30</integer>
        </dict>
        <dict>
            <key>Weekday</key><integer>5</integer>
            <key>Hour</key><integer>6</integer>
            <key>Minute</key><integer>30</integer>
        </dict>
    </array>

    <key>StandardOutPath</key>
    <string>$LOG_DIR/cron_pipeline.log</string>
    <key>StandardErrorPath</key>
    <string>$LOG_DIR/cron_pipeline.log</string>

    <!-- Don't relaunch on exit; this is a once-a-day batch job. -->
    <key>KeepAlive</key>
    <false/>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
EOF
}

case "${1:-install}" in
    --uninstall|-u)
        echo "Removing $LABEL launchd agent…"
        launchctl unload "$PLIST" 2>/dev/null || true
        rm -f "$PLIST"
        echo "Done."
        ;;

    --status|-s)
        if [ -f "$PLIST" ]; then
            echo "Plist installed at: $PLIST"
        else
            echo "Plist not installed."
            exit 0
        fi
        echo
        echo "── launchctl print ──"
        launchctl print "gui/$(id -u)/$LABEL" 2>&1 | head -25 || \
            echo "  (not loaded — run 'bash scripts/setup_launchd.sh' to install)"
        echo
        echo "── recent log lines ──"
        tail -20 "$LOG_DIR/cron_pipeline.log" 2>/dev/null || echo "  (no log file yet — agent hasn't fired)"
        ;;

    --run-now|-r)
        if [ ! -f "$PLIST" ]; then
            echo "❌ Plist not installed.  Run: bash scripts/setup_launchd.sh"
            exit 1
        fi
        echo "Triggering $LABEL now…"
        launchctl kickstart -k "gui/$(id -u)/$LABEL"
        echo "Triggered.  Tail the log: tail -f $LOG_DIR/cron_pipeline.log"
        ;;

    install|"")
        if [ ! -x "$PYTHON_BIN" ]; then
            echo "❌ venv python not found at $PYTHON_BIN"
            echo "   Activate the venv (or set PYTHON_BIN) and pip install -r requirements.txt first."
            exit 1
        fi

        echo "Installing $LABEL launchd agent…"
        # Unload first so re-installs apply cleanly
        launchctl unload "$PLIST" 2>/dev/null || true

        write_plist
        launchctl load "$PLIST"

        echo
        echo "Installed.  Schedule: 06:30 every weekday (Mon–Fri)."
        echo "  Plist : $PLIST"
        echo "  Log   : $LOG_DIR/cron_pipeline.log"
        echo
        echo "Verify : bash scripts/setup_launchd.sh --status"
        echo "Test   : bash scripts/setup_launchd.sh --run-now"
        echo "Remove : bash scripts/setup_launchd.sh --uninstall"
        echo
        if [[ "$PROJECT_DIR" == "$HOME/Documents/"* || \
              "$PROJECT_DIR" == "$HOME/Desktop/"*  || \
              "$PROJECT_DIR" == "$HOME/Downloads/"* ]]; then
            echo "⚠ macOS PRIVACY NOTE — project lives in a TCC-protected folder"
            echo "  ($PROJECT_DIR)"
            echo
            echo "  By default, scheduled tasks (cron + launchd) run with"
            echo "  restricted permissions and CANNOT read files in ~/Documents,"
            echo "  ~/Desktop, or ~/Downloads.  Without granting access, the"
            echo "  agent will fire at 06:30 but Python will fail with"
            echo "  'Operation not permitted: pyvenv.cfg'."
            echo
            echo "  Fix (one-time):"
            echo "    1. Open System Settings → Privacy & Security → Full Disk Access"
            echo "    2. Click + and add:  $PYTHON_BIN"
            echo "    3. Run: bash scripts/setup_launchd.sh --run-now"
            echo
            echo "  Alternative: move the project to a non-TCC path like"
            echo "    /Users/$USER/code/trading-bot-1   or"
            echo "    /opt/trading-bot-1"
        fi
        ;;

    *)
        echo "Unknown flag: $1"
        echo "Usage: bash scripts/setup_launchd.sh [install|--status|--run-now|--uninstall]"
        exit 2
        ;;
esac
