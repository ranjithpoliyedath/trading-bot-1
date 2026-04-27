"""
bot/models/base.py
-------------------
Base interface for all trading models — rule-based, ML-based, custom.

Every model exposes the same shape:
    .id            unique short identifier
    .name          human-readable name
    .description   one-line description
    .type          'rule' | 'ml' | 'custom'
    .predict(row)  takes a feature row, returns (signal, confidence)
    .predict_batch(df)  optional optimisation for many rows at once

This allows the dashboard, screener, and signal generator to treat
every model identically.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import pandas as pd

Signal = Literal["buy", "sell", "hold"]


@dataclass
class ModelMetadata:
    id:          str
    name:        str
    description: str
    type:        Literal["rule", "ml", "custom", "cross_sectional"]
    required_features: list[str]


class BaseModel(ABC):
    """
    Abstract base class for trading models.

    Subclasses must implement:
        - metadata (class attribute or property)
        - predict(row) returning (signal, confidence)
    """

    metadata: ModelMetadata

    @abstractmethod
    def predict(self, row: pd.Series) -> tuple[Signal, float]:
        """
        Generate a trading signal for a single feature row.

        Args:
            row: A pandas Series of features for one symbol on one day.
                 Must contain all columns in self.metadata.required_features.

        Returns:
            (signal, confidence)
            signal: 'buy', 'sell', or 'hold'
            confidence: float in [0.0, 1.0]
        """
        ...

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for a DataFrame of feature rows.

        Default implementation calls predict() row by row. Subclasses
        can override for vectorised performance.

        Returns:
            DataFrame with two columns added: 'signal' and 'confidence'.
        """
        df = df.copy()
        results = df.apply(self.predict, axis=1)
        df["signal"]     = [r[0] for r in results]
        df["confidence"] = [r[1] for r in results]
        return df

    @property
    def id(self) -> str:
        return self.metadata.id

    @property
    def name(self) -> str:
        return self.metadata.name

    def __repr__(self) -> str:
        return f"<{type(self).__name__} id={self.id!r}>"


class CrossSectionalModel(ABC):
    """
    Strategies that need to rank across the universe each bar — they
    can't be implemented as a per-symbol ``predict(row)`` because the
    signal depends on every other symbol's value.

    Examples: Jegadeesh-Titman 12-1 momentum (long the top-decile
    return-rankers, short the bottom), Antonacci dual momentum,
    cross-sectional pairs.

    Subclasses implement ``rank_universe(panel) → ranks_panel`` where
    both ``panel`` and the returned dataframe are wide panels indexed
    by date with one column per symbol.  The orchestrator
    ``dashboard.backtest_engine.run_cross_sectional_backtest`` consumes
    those ranks to build a long-only or long/short portfolio.
    """

    metadata: ModelMetadata

    @abstractmethod
    def rank_universe(self, panel: "pd.DataFrame") -> "pd.DataFrame":
        """
        Args:
            panel: wide DataFrame, index=date, columns=symbols, values=
                   the feature being ranked (typically forward-fill close).

        Returns:
            wide DataFrame same shape, values are ranks in [0,1] where
            1 = best (top of universe), 0 = worst.  NaN where the
            symbol shouldn't trade that day.
        """
        ...

    @property
    def id(self) -> str:   return self.metadata.id

    @property
    def name(self) -> str: return self.metadata.name

    def __repr__(self) -> str:
        return f"<{type(self).__name__} id={self.id!r}>"
