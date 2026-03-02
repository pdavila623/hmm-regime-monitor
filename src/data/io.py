"""
I/O helpers: unified data loading from cache, CCXT, or CSV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from src.data.ccxt_fetch import fetch_ohlcv, load_csv, load_from_cache


def get_data(
    symbol: str,
    timeframe: str,
    since: Optional[str] = None,
    exchange_id: str = "binance",
    csv_path: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Unified data loader. Priority:
    1. CSV (if provided)
    2. Local cache (if available and use_cache=True)
    3. CCXT live fetch

    Returns cleaned OHLCV DataFrame indexed by UTC timestamp.
    """
    if csv_path:
        logger.info(f"Loading from CSV: {csv_path}")
        return load_csv(csv_path)

    return fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        since=since,
        exchange_id=exchange_id,
        use_cache=use_cache,
    )


def ensure_min_rows(df: pd.DataFrame, min_rows: int = 200) -> pd.DataFrame:
    """Raise if dataframe has too few rows for meaningful HMM training."""
    if len(df) < min_rows:
        raise ValueError(
            f"Not enough data: {len(df)} rows (need at least {min_rows}). "
            "Try a wider date range or shorter timeframe."
        )
    return df
