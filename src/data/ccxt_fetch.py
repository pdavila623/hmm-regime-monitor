"""
Data ingestion via CCXT (Binance default) or CSV.
Caches results as Parquet locally.

Usage:
    python -m src.data.ccxt_fetch --symbol BTC/USDT --tf 1h --since 2022-01-01
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd
import typer
from loguru import logger

from src.config import CACHE_DIR, DEFAULT_EXCHANGE, DEFAULT_SYMBOL, DEFAULT_TIMEFRAME

app = typer.Typer()


def _cache_path(exchange: str, symbol: str, timeframe: str) -> Path:
    safe_sym = symbol.replace("/", "_")
    return CACHE_DIR / f"{exchange}_{safe_sym}_{timeframe}.parquet"


def load_from_cache(exchange: str, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    path = _cache_path(exchange, symbol, timeframe)
    if path.exists():
        logger.info(f"Loading cache from {path}")
        return pd.read_parquet(path)
    return None


def save_to_cache(df: pd.DataFrame, exchange: str, symbol: str, timeframe: str) -> None:
    path = _cache_path(exchange, symbol, timeframe)
    df.to_parquet(path)
    logger.info(f"Saved {len(df)} rows to cache: {path}")


def fetch_ohlcv(
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    since: Optional[str] = None,
    exchange_id: str = DEFAULT_EXCHANGE,
    use_cache: bool = True,
    limit_per_call: int = 1000,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from CCXT exchange with local cache.

    Args:
        symbol: Trading pair, e.g. 'BTC/USDT'
        timeframe: CCXT timeframe string: '5m','15m','1h','4h','1d'
        since: Start date string 'YYYY-MM-DD'
        exchange_id: CCXT exchange id
        use_cache: Load from parquet cache if available
        limit_per_call: Bars per API call (max ~1000 for Binance)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    if use_cache:
        cached = load_from_cache(exchange_id, symbol, timeframe)
        if cached is not None and len(cached) > 0:
            # If since requested is before cache start, still use cache
            # and just filter by date
            if since:
                since_ts = pd.Timestamp(since, tz="UTC")
                filtered = cached[cached.index >= since_ts]
                if len(filtered) > 100:
                    logger.info(f"Using cached data: {len(filtered)} rows")
                    return filtered
            else:
                return cached

    logger.info(f"Fetching {symbol} {timeframe} from {exchange_id}...")

    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
    except AttributeError:
        raise ValueError(f"Unknown exchange: {exchange_id}")

    since_ms = None
    if since:
        dt = datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        since_ms = int(dt.timestamp() * 1000)

    all_ohlcv = []
    current_since = since_ms

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=current_since,
                limit=limit_per_call,
            )
        except ccxt.RateLimitExceeded:
            logger.warning("Rate limit exceeded; sleeping 10s...")
            time.sleep(10)
            continue
        except ccxt.NetworkError as e:
            logger.error(f"Network error: {e}")
            break
        except Exception as e:
            logger.error(f"Unexpected error fetching data: {e}")
            break

        if not ohlcv:
            break

        all_ohlcv.extend(ohlcv)
        logger.info(f"  Fetched {len(ohlcv)} bars, total: {len(all_ohlcv)}")

        # If we got less than requested, we've reached the end
        if len(ohlcv) < limit_per_call:
            break

        # Advance pointer past last bar
        current_since = ohlcv[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)

    if not all_ohlcv:
        raise RuntimeError("No data fetched. Check symbol/timeframe/exchange.")

    df = pd.DataFrame(
        all_ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    save_to_cache(df, exchange_id, symbol, timeframe)
    logger.info(f"Total: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    return df


def load_csv(path: str) -> pd.DataFrame:
    """
    Load OHLCV data from CSV file.
    Expected columns: timestamp (or datetime index), open, high, low, close, volume
    """
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = [c.lower() for c in df.columns]
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df = df.sort_index()
    logger.info(f"Loaded CSV: {len(df)} rows from {df.index[0]} to {df.index[-1]}")
    return df


# ─── CLI ──────────────────────────────────────────────────────────────────────
@app.command()
def main(
    symbol: str = typer.Option(DEFAULT_SYMBOL, help="Trading pair"),
    tf: str = typer.Option(DEFAULT_TIMEFRAME, help="Timeframe: 5m,15m,1h,4h,1d"),
    since: str = typer.Option("2022-01-01", help="Start date YYYY-MM-DD"),
    exchange: str = typer.Option(DEFAULT_EXCHANGE, help="CCXT exchange id"),
    no_cache: bool = typer.Option(False, help="Ignore cache and re-fetch"),
):
    """Fetch OHLCV data and save to local cache."""
    df = fetch_ohlcv(
        symbol=symbol,
        timeframe=tf,
        since=since,
        exchange_id=exchange,
        use_cache=not no_cache,
    )
    typer.echo(f"✓ Data ready: {len(df)} bars | {df.index[0]} → {df.index[-1]}")
    typer.echo(df.tail(3).to_string())


if __name__ == "__main__":
    app()
