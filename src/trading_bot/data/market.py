"""Asset-agnostic market data feeds.

The strategies, backtester, and risk manager all operate on a plain OHLCV
DataFrame, so supporting a new asset class is purely a data-layer concern.

    crypto  → ccxt   (Binance, Kraken, …) — the existing path
    stocks  → yfinance (free, no API key; data is delayed ~15 min)

Pick the asset class in config["market"]["asset_class"]. Everything downstream
is identical.
"""

from __future__ import annotations

import math
from typing import Protocol

import pandas as pd

from .fetcher import fetch_ohlcv, fetch_ohlcv_history, make_exchange


def asset_class(config: dict) -> str:
    return config.get("market", {}).get("asset_class", "crypto").lower().strip()


class MarketFeed(Protocol):
    def recent(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame: ...
    def history(self, symbol: str, timeframe: str, total: int) -> pd.DataFrame: ...


# ──────────────────────────────────────────────────────────────────────────
# Crypto (ccxt) — wraps the existing fetcher functions
# ──────────────────────────────────────────────────────────────────────────

class CryptoFeed:
    def __init__(self, config: dict, api_key: str = "", secret: str = "") -> None:
        self.exchange = make_exchange(config, api_key=api_key, secret=secret)

    def recent(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        return fetch_ohlcv(self.exchange, symbol, timeframe, limit=limit)

    def history(self, symbol: str, timeframe: str, total: int) -> pd.DataFrame:
        return fetch_ohlcv_history(self.exchange, symbol, timeframe, total=total)


# ──────────────────────────────────────────────────────────────────────────
# Stocks (yfinance)
# ──────────────────────────────────────────────────────────────────────────

# yfinance interval names that map cleanly from our timeframes.
_YF_INTERVAL = {
    "1m": "1m", "2m": "2m", "5m": "5m", "15m": "15m",
    "30m": "30m", "1h": "1h", "1d": "1d", "1wk": "1wk",
}
# Max calendar lookback yfinance allows per interval (intraday is limited).
_YF_MAX_DAYS = {
    "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "1h": 730,
}
# Approximate number of bars in one US trading day per interval.
_BARS_PER_DAY = {"1m": 390, "2m": 195, "5m": 78, "15m": 26, "30m": 13, "1h": 7, "1d": 1}


def _yf_period_days(interval: str, total: int) -> int:
    """Calendar days to request so we end up with ~`total` bars (with buffer)."""
    per_day = _BARS_PER_DAY.get(interval, 7)
    # 1.5× for weekends/holidays, + a small pad.
    days = math.ceil(total / per_day * 1.5) + 5
    cap = _YF_MAX_DAYS.get(interval)
    return min(days, cap) if cap else days


class StockFeed:
    def __init__(self, config: dict) -> None:
        try:
            import yfinance  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Stock support needs the 'yfinance' package. Install with:\n"
                "    pip install yfinance"
            ) from exc

    def _download(self, symbol: str, timeframe: str, total: int) -> pd.DataFrame:
        import yfinance as yf

        interval = _YF_INTERVAL.get(timeframe)
        if interval is None:
            raise ValueError(
                f"Timeframe '{timeframe}' is not supported for stocks. "
                f"Use one of: {', '.join(_YF_INTERVAL)}"
            )

        if interval == "1d":
            period = f"{max(total + 10, 30)}d"
        else:
            period = f"{_yf_period_days(interval, total)}d"

        raw = yf.download(
            symbol, period=period, interval=interval,
            progress=False, auto_adjust=False,
        )
        if raw is None or len(raw) == 0:
            raise ValueError(f"No data returned for stock symbol '{symbol}'.")

        # Newer yfinance returns MultiIndex columns ('Close','AAPL') for one ticker.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw.rename(columns=str.lower)
        df = raw[["open", "high", "low", "close", "volume"]].copy()
        df = df.dropna()

        # Normalize to a UTC DatetimeIndex.
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df.index.name = "timestamp"
        return df.tail(total)

    def history(self, symbol: str, timeframe: str, total: int) -> pd.DataFrame:
        return self._download(symbol, timeframe, total)

    def recent(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        # A touch more than `limit` so indicators have warm-up room.
        return self._download(symbol, timeframe, max(limit, limit + 5))


# ──────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────

def make_feed(config: dict, api_key: str = "", secret: str = "") -> MarketFeed:
    if asset_class(config) == "stocks":
        return StockFeed(config)
    return CryptoFeed(config, api_key=api_key, secret=secret)
