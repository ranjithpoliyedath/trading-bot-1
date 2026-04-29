"""
dashboard/pages/data_status.py
-------------------------------
Data Status page — read-only snapshot of ``data/processed/`` plus a
manual "Update Now" button that triggers ``bot.pipeline`` in the
background.  The page exists because the user asked to see (a) the
current depth/coverage of the on-disk OHLCV data and (b) trigger a
refresh without dropping to a terminal.

Three cards on the page:

  1. Coverage      — total parquets, depth distribution, deepest /
                     shallowest symbol, total disk MB, freshness
                     (newest bar date + last fetch write time).

  2. Schedule      — next run hint from launchd / cron, plus state of
                     the most recent pipeline run.

  3. Update Now    — button + live status that polls every 2 seconds
                     while a refresh is running.

Layout/styling mirrors the other pages (CARD constant, dbc.Row / Col
with md sizing) for consistency.
"""
from __future__ import annotations

from dash import html, dcc
import dash_bootstrap_components as dbc

from dashboard.services.data_status import (
    get_coverage_summary, get_schedule_info, get_pipeline_status,
)


CARD = {
    "background": "white", "borderRadius": "12px",
    "border":     "1px solid #eee", "padding": "16px",
}

METRIC_TILE = {
    "background": "#F8F8F7", "borderRadius": "8px",
    "padding":    "10px 14px",
}


def _section_label(text: str):
    return html.Div(text, style={"fontSize": "13px", "fontWeight": "600",
                                  "color": "#1f2937", "marginBottom": "10px"})


def _metric_tile(label: str, value: str, sub: str = ""):
    return html.Div([
        html.Div(label, style={"fontSize": "11px", "color": "#666",
                                "fontWeight": "500", "marginBottom": "4px"}),
        html.Div(value, style={"fontSize": "20px", "fontWeight": "700",
                                "color": "#1f2937"}),
        html.Div(sub, style={"fontSize": "11px", "color": "#888",
                              "marginTop": "2px"}) if sub else html.Div(),
    ], style=METRIC_TILE)


def _coverage_card(summary: dict):
    """Top card — counts, depth distribution, freshness."""
    if summary["total_features"] == 0:
        return html.Div([
            html.Div("⚠ No data on disk", style={
                "fontSize": "16px", "fontWeight": "600",
                "color": "#A32D2D", "marginBottom": "8px"}),
            html.P("Run the pipeline to populate `data/processed/`:",
                   style={"fontSize": "13px", "color": "#444"}),
            html.Pre("python -m bot.pipeline --full-universe --source=yfinance",
                     style={"fontFamily": "monospace", "fontSize": "12px",
                            "background": "#F8F8F7", "padding": "8px 12px",
                            "borderRadius": "6px"}),
        ], style=CARD)

    buckets = summary["depth_buckets"]
    total   = summary["total_features"]

    # Stacked breakdown of depth distribution
    bucket_rows = []
    for label, count in buckets.items():
        if count == 0: continue
        pct = (count / total) * 100 if total else 0
        bar_color = {
            "<2y (recent IPO)": "#A32D2D",
            "2-6y":             "#D97706",
            "6-8y":             "#1D9E75",
            "8y+ (deep)":       "#27500A",
        }.get(label, "#666")
        bucket_rows.append(html.Div([
            html.Div([
                html.Span(label,  style={"fontSize": "12px", "color": "#444"}),
                html.Span(f"{count:,}  ·  {pct:.0f}%",
                          style={"fontSize": "12px", "color": "#666",
                                 "marginLeft": "auto",
                                 "fontFamily": "monospace"}),
            ], style={"display": "flex", "alignItems": "center",
                      "marginBottom": "3px"}),
            html.Div(style={"height": "6px", "background": "#F1F1F1",
                             "borderRadius": "3px", "overflow": "hidden"},
                     children=html.Div(style={
                         "width": f"{pct}%", "height": "100%",
                         "background": bar_color, "transition": "width 0.3s"})),
        ], style={"marginBottom": "8px"}))

    return html.Div([
        _section_label("Data coverage"),
        dbc.Row([
            dbc.Col(_metric_tile("Symbols on disk",
                                  f"{summary['total_features']:,}",
                                  f"+{summary['total_raw']:,} raw bar parquets"),
                    md=3),
            dbc.Col(_metric_tile("Earliest bar",
                                  summary['earliest_data'] or "—",
                                  "across all symbols"),
                    md=3),
            dbc.Col(_metric_tile("Latest bar",
                                  summary['latest_bar_date'] or "—",
                                  "newest data point"),
                    md=3),
            dbc.Col(_metric_tile("Disk used",
                                  f"{summary['size_mb']:,.0f} MB",
                                  "raw + features"),
                    md=3),
        ], className="g-2 mb-3"),

        html.Div("Depth distribution",
                 style={"fontSize": "12px", "fontWeight": "600",
                        "color": "#666", "marginBottom": "8px"}),
        html.Div(bucket_rows),

        html.Div([
            html.Div([
                html.Span("Deepest:    ", style={"color": "#888"}),
                html.Span(summary['deepest_symbol'],
                          style={"fontFamily": "monospace"}),
            ], style={"fontSize": "12px"}),
            html.Div([
                html.Span("Shallowest: ", style={"color": "#888"}),
                html.Span(summary['shallowest'],
                          style={"fontFamily": "monospace"}),
            ], style={"fontSize": "12px"}),
            html.Div([
                html.Span("Last write: ", style={"color": "#888"}),
                html.Span(summary['latest_mtime'] or "—",
                          style={"fontFamily": "monospace"}),
            ], style={"fontSize": "12px"}),
        ], style={"marginTop": "12px",
                  "padding":   "10px 12px",
                  "background": "#F8F8F7",
                  "borderRadius": "6px",
                  "lineHeight": "1.7"}),
    ], style=CARD)


def _schedule_card(sched: dict):
    """Schedule + last automatic run state."""
    rows = []
    if sched.get("launchd_loaded"):
        rows.append(("✓ launchd agent installed",
                      "Daily refresh fires at 06:30 weekdays.  Survives sleep.",
                      "#27500A"))
    elif sched.get("cron_installed"):
        rows.append(("✓ cron installed",
                      "Daily refresh fires at 06:30 weekdays "
                      "(skipped if Mac is asleep at that time).",
                      "#27500A"))
    else:
        rows.append(("⚠ Nothing scheduled",
                      "No cron or launchd entry found.  Install one with "
                      "scripts/setup_launchd.sh (recommended on macOS).",
                      "#A32D2D"))

    if sched.get("next_run_hint"):
        rows.append(("Next scheduled run",
                      sched["next_run_hint"],
                      "#1f2937"))

    items = [
        html.Div([
            html.Div(title, style={"fontSize": "12px", "fontWeight": "600",
                                    "color": colour, "marginBottom": "2px"}),
            html.Div(detail, style={"fontSize": "12px", "color": "#666"}),
        ], style={"marginBottom": "10px"})
        for title, detail, colour in rows
    ]

    return html.Div([
        _section_label("Schedule"),
        html.Div(items),
    ], style=CARD)


def _update_now_card(status: dict):
    """Bottom card — Update Now button + live status panel.

    The status panel updates via an Interval-driven callback so the user
    sees progress while the background pipeline runs.  Polled every 2s
    while running, then once after completion to render the result.
    """
    # Button label/state vary by status
    if status["status"] == "running":
        btn_text  = "Refresh in progress…"
        btn_props = {"disabled": True}
        accent    = "#D97706"
    else:
        btn_text  = "Update stock data now"
        btn_props = {}
        accent    = "#3C3489"

    return html.Div([
        _section_label("Manual refresh"),
        html.P("Pulls the latest daily bars for every universe symbol "
               "from yfinance and re-runs feature engineering.  Typical "
               "duration on an incremental run: ~1 minute.",
               style={"fontSize": "12px", "color": "#666",
                      "marginBottom": "12px"}),

        html.Button(
            btn_text,
            id="btn-update-data",
            **btn_props,
            style={
                "background":    accent if not btn_props.get("disabled") else "#999",
                "color":         "white",
                "border":        "none",
                "padding":       "10px 24px",
                "borderRadius":  "8px",
                "fontSize":      "13px",
                "fontWeight":    "600",
                "cursor":        ("not-allowed" if btn_props.get("disabled")
                                  else "pointer"),
            },
        ),

        html.Div(id="data-update-status",
                  children=_render_status(status),
                  style={"marginTop": "16px"}),

        # Polls every 2s; the callback decides when to stop animating
        # (status flips to done/failed/idle).
        dcc.Interval(id="data-update-poll", interval=2000, disabled=False),
    ], style=CARD)


def _render_status(status: dict):
    """Render the status block underneath the button.  Pure function so
    the polling callback can call it with fresh data."""
    s = status.get("status", "idle")

    if s == "idle":
        return html.Div("No refresh has run in this dashboard session yet.",
                        style={"fontSize": "12px", "color": "#888",
                               "fontStyle": "italic"})

    if s == "running":
        started = status.get("started_at", "")
        return html.Div([
            html.Div([
                html.Span("● ", style={"color": "#D97706",
                                        "fontSize": "16px"}),
                html.Span("Pipeline running…",
                          style={"fontSize": "13px", "fontWeight": "600",
                                 "color": "#D97706"}),
            ]),
            html.Div(f"Started: {_short_iso(started)}",
                     style={"fontSize": "12px", "color": "#666",
                            "marginTop": "4px",
                            "fontFamily": "monospace"}),
            html.Div("Fetching ~1,500 symbols from yfinance "
                     "(refreshes every 2 seconds while running).",
                     style={"fontSize": "11px", "color": "#888",
                            "marginTop": "4px"}),
        ], style={"padding": "12px 14px",
                  "background": "#FFF7ED",
                  "border":     "1px solid #FED7AA",
                  "borderRadius": "8px"})

    if s == "done":
        c = status.get("counts") or {}
        dur = status.get("duration_s") or 0
        finished = status.get("finished_at", "")
        trigger  = status.get("trigger", "manual")
        return html.Div([
            html.Div([
                html.Span("✓ ", style={"color": "#27500A",
                                        "fontSize": "16px"}),
                html.Span(f"Refresh complete  ·  {dur:.1f}s",
                          style={"fontSize": "13px", "fontWeight": "600",
                                 "color": "#27500A"}),
            ]),
            html.Div([
                html.Span(f"processed: {c.get('processed', 0):,}   ",
                          style={"color": "#27500A"}),
                html.Span(f"skipped: {c.get('skipped', 0):,}   ",
                          style={"color": "#666"}),
                html.Span(f"failed: {c.get('failed', 0):,}   ",
                          style={"color": "#A32D2D" if c.get('failed', 0) else "#666"}),
                html.Span(f"new bars: {c.get('new_rows', 0):,}",
                          style={"color": "#27500A"}),
            ], style={"fontSize": "12px", "marginTop": "6px",
                      "fontFamily": "monospace"}),
            html.Div(f"Finished: {_short_iso(finished)}  ·  trigger: {trigger}",
                     style={"fontSize": "11px", "color": "#888",
                            "marginTop": "4px",
                            "fontFamily": "monospace"}),
        ], style={"padding": "12px 14px",
                  "background": "#EAF3DE",
                  "border":     "1px solid #C9E2A8",
                  "borderRadius": "8px"})

    if s == "failed":
        err = status.get("error", "(no error message)")
        return html.Div([
            html.Div([
                html.Span("✕ ", style={"color": "#A32D2D",
                                        "fontSize": "16px"}),
                html.Span("Refresh failed",
                          style={"fontSize": "13px", "fontWeight": "600",
                                 "color": "#A32D2D"}),
            ]),
            html.Div(err,
                     style={"fontSize": "12px",
                            "color": "#A32D2D",
                            "marginTop": "6px",
                            "fontFamily": "monospace",
                            "wordBreak": "break-word"}),
            html.Div("Check logs/cron_pipeline.log and the dashboard "
                     "console for the full traceback.",
                     style={"fontSize": "11px", "color": "#888",
                            "marginTop": "4px"}),
        ], style={"padding": "12px 14px",
                  "background": "#FEF2F2",
                  "border":     "1px solid #FECACA",
                  "borderRadius": "8px"})

    return html.Div(f"Unknown state: {s}",
                    style={"fontSize": "12px", "color": "#888"})


def _short_iso(ts: str) -> str:
    """Trim the ISO timestamp to HH:MM:SS for compactness."""
    if not ts or len(ts) < 19:
        return ts or ""
    return ts[:19].replace("T", " ")


def layout(account: str = "paper", model: str = "", symbol: str = ""):
    """Page entry point — fresh snapshot on every navigation.

    The signature accepts the same (account, model, symbol) tuple as
    other pages so the central router in ``app.render_page`` can call
    it uniformly, even though this page doesn't use any of those.
    """
    summary  = get_coverage_summary()
    schedule = get_schedule_info()
    status   = get_pipeline_status()

    return html.Div([
        dbc.Row([
            dbc.Col(_coverage_card(summary), md=8),
            dbc.Col(_schedule_card(schedule), md=4),
        ], className="g-3 mb-3"),

        dbc.Row([
            dbc.Col(_update_now_card(status), md=12),
        ]),
    ])
