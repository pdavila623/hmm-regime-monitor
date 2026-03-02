"""
Feature engineering for HMM market regime detection.

Design principles:
- Use RETURNS and normalized features, never raw price levels.
  (HMM assumes stationary observations; prices are non-stationary.)
- Features capture: direction, volatility, and momentum.
- All features are scaled to similar ranges to avoid one feature
  dominating the HMM Gaussian emissions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import RobustScaler

from src.config import (
    ATR_PERIOD,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    ROLLING_VOL_WINDOW,
    RSI_PERIOD,
    TREND_WINDOW,
    ZSCORE_WINDOW,
)

FEATURE_COLS = [
    "log_return",
    "rolling_vol",
    "trend_strength",
    "rsi_norm",
    "atr_norm",
    "zscore_return",
    "macd_hist_norm",
]


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _linear_slope(series: pd.Series, window: int) -> pd.Series:
    """Rolling OLS slope (normalized by std to be scale-free)."""
    slopes = series.copy() * np.nan
    x = np.arange(window, dtype=float)
    x -= x.mean()
    xTx_inv = 1.0 / (x @ x)
    for i in range(window, len(series) + 1):
        y = series.iloc[i - window:i].values.astype(float)
        if np.isnan(y).any():
            continue
        slopes.iloc[i - 1] = (x @ y) * xTx_inv
    return slopes


def build_features(df: pd.DataFrame, scale: bool = True) -> pd.DataFrame:
    """
    Build feature matrix from OHLCV DataFrame.

    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume
        scale: Apply RobustScaler to features (recommended for HMM)

    Returns:
        DataFrame with feature columns, NaN rows dropped.
    """
    feat = pd.DataFrame(index=df.index)

    # 1. Log return: core signal, stationary
    feat["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # 2. Rolling volatility: std of log returns
    feat["rolling_vol"] = feat["log_return"].rolling(ROLLING_VOL_WINDOW).std()

    # 3. Trend strength: normalized OLS slope on log-close
    #    Using log-close to make slope scale-free across price levels
    log_close = np.log(df["close"])
    feat["trend_strength"] = _linear_slope(log_close, TREND_WINDOW)

    # 4. RSI normalized to [-1, 1]: momentum indicator
    feat["rsi_norm"] = (_rsi(df["close"], RSI_PERIOD) - 50) / 50

    # 5. ATR normalized by close: measures volatility relative to price
    atr = _atr(df["high"], df["low"], df["close"], ATR_PERIOD)
    feat["atr_norm"] = atr / df["close"]

    # 6. Z-score of returns: current return vs recent distribution
    roll_mean = feat["log_return"].rolling(ZSCORE_WINDOW).mean()
    roll_std = feat["log_return"].rolling(ZSCORE_WINDOW).std()
    feat["zscore_return"] = (feat["log_return"] - roll_mean) / (roll_std + 1e-10)

    # 7. MACD histogram normalized: captures momentum crossovers
    ema_fast = df["close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df["close"].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    macd_hist = macd_line - signal_line
    feat["macd_hist_norm"] = macd_hist / (df["close"] + 1e-10)

    # Drop rows with NaN (rolling warmup)
    feat = feat.dropna()
    logger.info(f"Features built: {len(feat)} rows, {feat.shape[1]} features")

    if scale:
        scaler = RobustScaler()  # Robust to outliers (fat tails in crypto)
        scaled_values = scaler.fit_transform(feat[FEATURE_COLS])
        feat_scaled = pd.DataFrame(scaled_values, index=feat.index, columns=FEATURE_COLS)
        feat_scaled._scaler = scaler  # attach for inverse transform if needed
        return feat_scaled

    return feat


def get_feature_matrix(feat_df: pd.DataFrame) -> np.ndarray:
    """Extract numpy array for HMM fitting."""
    cols = [c for c in FEATURE_COLS if c in feat_df.columns]
    return feat_df[cols].values.astype(float)
