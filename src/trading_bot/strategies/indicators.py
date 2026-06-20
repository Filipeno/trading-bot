"""Shared technical-indicator helpers, returned as pandas Series/arrays.

Kept separate so every strategy computes indicators the same way and they are
unit-tested in one place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    # Zero-loss with positive gains → RSI 100; flat (no moves) → neutral 50.
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return out.fillna(50.0)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    """Return (%K, %D) of the stochastic oscillator."""
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    rng = (high_max - low_min).replace(0.0, np.nan)
    percent_k = 100 * (df["close"] - low_min) / rng
    percent_k = percent_k.fillna(50.0)
    percent_d = percent_k.rolling(d_period).mean()
    return percent_k, percent_d


def rolling_vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Rolling volume-weighted average price over `period` bars."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].replace(0.0, np.nan)
    pv = (typical * vol).rolling(period).sum()
    v = vol.rolling(period).sum()
    return pv / v


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """Return the Supertrend direction as a Series of +1 (up) / -1 (down).

    Standard ATR-channel trend filter widely used for crypto. +1 means the
    trend is up (price above the line), -1 means down.
    """
    hl2 = (df["high"] + df["low"]) / 2
    _atr = atr(df, period)
    upper = (hl2 + multiplier * _atr).to_numpy()
    lower = (hl2 - multiplier * _atr).to_numpy()
    close = df["close"].to_numpy()
    n = len(df)

    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    direction = np.ones(n, dtype=int)

    for i in range(n):
        if i == 0:
            final_upper[i] = upper[i]
            final_lower[i] = lower[i]
            direction[i] = 1
            continue

        final_upper[i] = (
            upper[i] if (upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1])
            else final_upper[i - 1]
        )
        final_lower[i] = (
            lower[i] if (lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1])
            else final_lower[i - 1]
        )

        if close[i] > final_upper[i - 1]:
            direction[i] = 1
        elif close[i] < final_lower[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

    return pd.Series(direction, index=df.index)
