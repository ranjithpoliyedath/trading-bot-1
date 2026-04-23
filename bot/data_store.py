"""
bot/data_store.py
-----------------
Handles saving and loading processed OHLCV + feature DataFrames to/from disk.
Data is stored as compressed Parquet files under data/processed/.
"""

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


class DataStore:
    """
    Persists processed DataFrames to disk as Parquet files.

    Args:
        data_dir: Directory to store files. Defaults to data/processed/.
    """

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("DataStore using directory: %s", self.data_dir)

    def save(self, df: pd.DataFrame, symbol: str, tag: str = "features") -> Path:
        """
        Save a DataFrame to a Parquet file.

        Args:
            df: DataFrame to save.
            symbol: Ticker symbol used in the filename e.g. 'AAPL'.
            tag: Label appended to filename e.g. 'features', 'raw'.

        Returns:
            Path to the saved file.
        """
        filename = self.data_dir / f"{symbol.upper()}_{tag}.parquet"
        df.to_parquet(filename, compression="snappy")
        logger.info("Saved %d rows to %s", len(df), filename)
        return filename

    def load(self, symbol: str, tag: str = "features") -> pd.DataFrame:
        """
        Load a previously saved DataFrame from disk.

        Args:
            symbol: Ticker symbol e.g. 'AAPL'.
            tag: Label used when saving e.g. 'features', 'raw'.

        Returns:
            Loaded DataFrame, or empty DataFrame if file not found.
        """
        filename = self.data_dir / f"{symbol.upper()}_{tag}.parquet"
        if not filename.exists():
            logger.warning("No saved data found for %s (%s).", symbol, tag)
            return pd.DataFrame()
        df = pd.read_parquet(filename)
        logger.info("Loaded %d rows from %s", len(df), filename)
        return df

    def list_saved(self) -> list[str]:
        """Return a list of all saved filenames in the data directory."""
        return [f.name for f in self.data_dir.glob("*.parquet")]
