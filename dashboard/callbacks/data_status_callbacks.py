"""
dashboard/callbacks/data_status_callbacks.py
---------------------------------------------
Two callbacks back the Data Status page:

  1. ``trigger_data_update`` fires when the user clicks "Update Now".
     Hands off to the async runner in ``services/data_status``;
     returns the current state for immediate UI feedback (the page
     flips to the running banner before the first poll completes).

  2. ``poll_data_update`` fires every 2 seconds via dcc.Interval while
     a refresh is running.  Reads the runner's state and re-renders
     the status panel.  When the run terminates (done/failed) the
     poll naturally stops animating because the status block stops
     changing.

Defensive: both callbacks use ``allow_duplicate`` because the same
``data-update-status`` div is also rendered server-side by the page
layout — we need to update it client-side via these callbacks too.
"""
from __future__ import annotations

from dash import Input, Output, State, callback, no_update

from dashboard.services.data_status import (
    get_pipeline_status, start_pipeline_async,
)
from dashboard.pages.data_status import _render_status


@callback(
    Output("data-update-status", "children", allow_duplicate=True),
    Input("btn-update-data", "n_clicks"),
    prevent_initial_call=True,
)
def trigger_data_update(n_clicks):
    if not n_clicks:
        return no_update
    state = start_pipeline_async(trigger="manual")
    return _render_status(state)


@callback(
    Output("data-update-status", "children", allow_duplicate=True),
    Input("data-update-poll", "n_intervals"),
    prevent_initial_call=True,
)
def poll_data_update(_n_intervals):
    state = get_pipeline_status()
    return _render_status(state)
