"""
bot/models/custom.py
---------------------
A model whose buy/sell rules come from a JSON spec.

This is what powers the dashboard's custom model builder. The user
defines a simple AND-only condition list, saves it to a JSON file,
and CustomRuleModel evaluates it against feature rows.

Spec format:
    {
      "id": "my_strategy",
      "name": "My Oversold Strategy",
      "description": "Buy when RSI is low and volume is high",
      "buy_when":  [
        {"field": "rsi_14", "op": "<", "value": 30},
        {"field": "volume_ratio", "op": ">", "value": 1.5}
      ],
      "sell_when": [
        {"field": "rsi_14", "op": ">", "value": 70}
      ],
      "min_confidence": 0.6
    }

Supported operators: <, <=, >, >=, ==, !=
All conditions are joined with AND (single AND-only logic per the spec).
"""

import operator
from typing import Any

import pandas as pd

from bot.models.base import BaseModel, ModelMetadata, Signal


OPERATORS = {
    "<":   operator.lt,
    "<=":  operator.le,
    ">":   operator.gt,
    ">=":  operator.ge,
    "==":  operator.eq,
    "!=":  operator.ne,
}


class CustomRuleModel(BaseModel):
    """
    A user-defined rule-based model loaded from a JSON spec.

    Supports AND-only condition chaining for buy and sell rules.
    Confidence is fixed at the spec's min_confidence (default 0.65).
    """

    def __init__(self, spec: dict):
        self.spec        = spec
        self.buy_rules   = spec.get("buy_when",  [])
        self.sell_rules  = spec.get("sell_when", [])
        self.confidence  = spec.get("min_confidence", 0.65)

        required = sorted({r["field"] for r in self.buy_rules + self.sell_rules})

        self.metadata = ModelMetadata(
            id                = f"custom:{spec['id']}",
            name              = spec.get("name", spec["id"]),
            description       = spec.get("description", "User-defined rule-based model."),
            type              = "custom",
            required_features = required,
        )

    def predict(self, row: pd.Series) -> tuple[Signal, float]:
        if self.buy_rules and self._all_match(row, self.buy_rules):
            return ("buy", self.confidence)
        if self.sell_rules and self._all_match(row, self.sell_rules):
            return ("sell", self.confidence)
        return ("hold", 0.55)

    def _all_match(self, row: pd.Series, rules: list[dict]) -> bool:
        for rule in rules:
            field, op_str, value = rule["field"], rule["op"], rule["value"]
            cell = row.get(field)
            if pd.isna(cell):
                return False
            op_fn = OPERATORS.get(op_str)
            if op_fn is None:
                return False
            try:
                if not op_fn(cell, value):
                    return False
            except (TypeError, ValueError):
                return False
        return True
