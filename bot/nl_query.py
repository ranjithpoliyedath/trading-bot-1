"""
bot/nl_query.py
----------------
Natural-language → structured backtest configuration.

The user types something like:

    "for all SPY-500 stocks with RSI under 30 and volume > 1.5x average
     last year, backtest the rsi_macd model"

We send it to Anthropic's API with a tool definition that captures the
structured shape we need.  The tool call response is the parsed config.

The system prompt is wrapped in ``cache_control`` so repeated queries
hit the prompt cache and stay cheap.

Required env var:  ANTHROPIC_API_KEY
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

from bot.screener import SCREENER_FIELDS

load_dotenv()
logger = logging.getLogger(__name__)


MODEL_NAME = "claude-sonnet-4-5"

VALID_OPS = [">", ">=", "<", "<=", "==", "!="]


# ── Tool definition the model must populate ────────────────────────────────

def _tool_schema() -> dict:
    field_keys = list(SCREENER_FIELDS.keys())
    return {
        "name": "configure_backtest",
        "description": (
            "Translate a user's natural-language request into a structured "
            "backtest configuration.  Always call this tool exactly once."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": (
                        "ID of the model to backtest.  Pick one from the "
                        "available built-in or custom models supplied in "
                        "the system prompt.  When unsure, default to "
                        "rsi_macd_v1."
                    ),
                },
                "filters": {
                    "type": "array",
                    "description": (
                        "Per-bar filters that must all be true to allow a buy. "
                        "Empty list means trade every signal the model emits."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string", "enum": field_keys},
                            "op":    {"type": "string", "enum": VALID_OPS},
                            "value": {"type": "number"},
                        },
                        "required": ["field", "op", "value"],
                    },
                },
                "period_days": {
                    "type": "integer",
                    "description": "Lookback window for the backtest, in days.",
                    "minimum": 30,
                    "maximum": 2000,
                },
                "symbols": {
                    "type": "array",
                    "description": (
                        "Optional explicit symbol list. Leave empty to scan "
                        "the full eligible universe."
                    ),
                    "items": {"type": "string"},
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum model confidence to act on (0-1).",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "rationale": {
                    "type": "string",
                    "description": (
                        "One short sentence echoing back how you mapped the "
                        "user's request to the structured fields."
                    ),
                },
            },
            "required": ["model_id", "filters", "period_days", "rationale"],
        },
    }


# ── System prompt (cacheable) ──────────────────────────────────────────────

def _build_system_prompt() -> str:
    from bot.models.registry import list_models

    fields = "\n".join(
        f"  - {key:24s} ({meta['group']:9s}): {meta['label']}"
        for key, meta in SCREENER_FIELDS.items()
    )
    try:
        models = "\n".join(
            f"  - {m.id:20s} [{m.type}]  {m.name}: {m.description}"
            for m in list_models()
        )
    except Exception:
        models = "  - rsi_macd_v1 [rule]"

    return f"""You convert plain-English trading-bot queries into structured backtest configurations.

You ALWAYS respond by calling the `configure_backtest` tool exactly once.
Never reply with prose only; always call the tool.

Available filter fields (use these exact identifiers):
{fields}

Available models (use the id):
{models}

Operator vocabulary: {", ".join(VALID_OPS)}

Heuristics:
- "Qullamaggie" / "stage-2 breakout" → model_id = qullamaggie_v1
- "VCP" / "volatility contraction" → model_id = vcp_v1
- "RSI / MACD oversold" → rsi_macd_v1
- "sentiment-driven" → sentiment_v1
- "Bollinger" → bollinger_v1
- "last year" → 365 days; "last 6 months" → 180; "last 3 months" → 90
- If the user names additional indicator thresholds, encode each as a separate filter.
- Default min_confidence to 0.6 unless the user specifies otherwise.
- Pass an empty filters list when the user gives no specific criteria — the model alone will gate signals.
- Map sector names like "tech" / "energy" to their sector ETF if mentioned for filtering, but otherwise prefer per-stock filters.
"""


# ── Public API ─────────────────────────────────────────────────────────────

@dataclass
class ParsedQuery:
    model_id:        str
    filters:         list[dict]    = field(default_factory=list)
    period_days:     int           = 365
    symbols:         list[str]     = field(default_factory=list)
    min_confidence:  float         = 0.6
    rationale:       str           = ""

    def to_dict(self) -> dict:
        return {
            "model_id":       self.model_id,
            "filters":        self.filters,
            "period_days":    self.period_days,
            "symbols":        self.symbols,
            "min_confidence": self.min_confidence,
            "rationale":      self.rationale,
        }


def parse_query(text: str, *, model: str = MODEL_NAME) -> ParsedQuery:
    """
    Send a NL query to Anthropic and return a structured ParsedQuery.

    Raises:
        RuntimeError if ANTHROPIC_API_KEY is missing or the model declines
        to call the tool.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set in .env — required for NL queries."
        )

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    system = [{
        "type": "text",
        "text": _build_system_prompt(),
        "cache_control": {"type": "ephemeral"},
    }]

    response = client.messages.create(
        model       = model,
        max_tokens  = 1024,
        system      = system,
        tools       = [_tool_schema()],
        tool_choice = {"type": "tool", "name": "configure_backtest"},
        messages    = [{"role": "user", "content": text}],
    )

    # Locate the tool_use block
    tool_block = next(
        (b for b in response.content if getattr(b, "type", None) == "tool_use"),
        None,
    )
    if tool_block is None:
        raise RuntimeError("Claude did not call the configure_backtest tool.")

    payload = tool_block.input or {}
    parsed = ParsedQuery(
        model_id       = payload.get("model_id", "rsi_macd_v1"),
        filters        = payload.get("filters", []) or [],
        period_days    = int(payload.get("period_days", 365)),
        symbols        = payload.get("symbols", []) or [],
        min_confidence = float(payload.get("min_confidence", 0.6)),
        rationale      = payload.get("rationale", ""),
    )

    # Defensive sanity checks
    parsed.filters = [
        f for f in parsed.filters
        if isinstance(f, dict)
        and f.get("field") in SCREENER_FIELDS
        and f.get("op") in VALID_OPS
        and isinstance(f.get("value"), (int, float))
    ]
    return parsed
