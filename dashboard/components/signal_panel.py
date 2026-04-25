"""
dashboard/components/signal_panel.py
--------------------------------------
Renders live ML signals for a given model and symbol.
Loads the trained model from models/saved/, runs predict on latest features.
Falls back to placeholder signals if model not yet trained.
"""

import logging
import pickle
from pathlib import Path

import pandas as pd
from dash import html

logger = logging.getLogger(__name__)

MODEL_DIR   = Path(__file__).parent.parent.parent / "models" / "saved"
DATA_DIR    = Path(__file__).parent.parent.parent / "data" / "processed"

SYMBOLS = ["AAPL", "TSLA", "MSFT", "NVDA", "SPY"]

PILL_STYLES = {
    "buy":  {"background": "#EAF3DE", "color": "#27500A"},
    "sell": {"background": "#FCEBEB", "color": "#A32D2D"},
    "hold": {"background": "#F1EFE8", "color": "#444441"},
}

CARD = {"background": "white", "borderRadius": "12px", "border": "1px solid #eee", "padding": "14px"}


def render_signals(model_name: str, symbol: str):
    signals = _get_signals(model_name)

    rows = []
    for sym, sig, conf in signals:
        pill = PILL_STYLES.get(sig, PILL_STYLES["hold"])
        rows.append(html.Div([
            html.Span(sym,            style={"fontWeight": "500", "fontSize": "13px", "width": "50px", "display": "inline-block"}),
            html.Span(sig.upper(),    style={"fontSize": "11px", "padding": "2px 8px", "borderRadius": "10px",
                                             "background": pill["background"], "color": pill["color"], "marginRight": "10px"}),
            html.Span(f"Conf: {conf:.0f}%", style={"fontSize": "12px", "color": "#888", "flex": "1"}),
        ], style={"display": "flex", "alignItems": "center", "padding": "6px 0",
                  "borderBottom": "1px solid #f5f5f5"}))

    return html.Div([
        html.Div([
            html.P("Live signals", style={"fontSize": "12px", "fontWeight": "500", "margin": "0"}),
            html.Span(model_name, style={"fontSize": "11px", "color": "#888",
                                          "background": "#EEEDFE", "color": "#3C3489",
                                          "padding": "2px 8px", "borderRadius": "10px"}),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "10px"}),
        *rows,
    ], style=CARD)


def _get_signals(model_name: str) -> list[tuple]:
    """Return list of (symbol, signal, confidence_pct) for all watched symbols."""
    model_path = MODEL_DIR / f"{model_name}.pkl"
    results = []

    for sym in SYMBOLS:
        features_path = DATA_DIR / f"{sym}_features.parquet"
        if not features_path.exists():
            results.append((sym, "hold", 55.0))
            continue
        try:
            df = pd.read_parquet(features_path)
            df.sort_index(inplace=True)
            last_row = df.iloc[[-1]]

            if model_path.exists():
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                feature_cols = [c for c in df.columns if c not in ["open","high","low","close","volume","vwap"]]
                X = last_row[feature_cols].dropna(axis=1)
                signal = model.predict(X)[0]
                conf   = model.predict_proba(X).max() * 100
            else:
                signal, conf = _rule_signal(last_row)

            results.append((sym, str(signal), round(float(conf), 1)))
        except Exception as exc:
            logger.warning("Signal error for %s: %s", sym, exc)
            results.append((sym, "hold", 50.0))

    return results


def _rule_signal(row: pd.DataFrame) -> tuple:
    """Simple fallback rule when model not yet trained."""
    try:
        rsi  = float(row["rsi_14"].iloc[0])
        macd = float(row["macd_hist"].iloc[0])
        if rsi < 40 and macd > 0:
            return "buy",  72.0
        if rsi > 60 and macd < 0:
            return "sell", 68.0
    except Exception:
        pass
    return "hold", 55.0
