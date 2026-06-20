import time
from typing import Optional

import ccxt
import pandas as pd

# Milliseconds per bar, used to page backward through history.
_TIMEFRAME_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    "6h": 21_600_000, "12h": 43_200_000, "1d": 86_400_000,
}


def make_exchange(config: dict, api_key: str = "", secret: str = "") -> ccxt.Exchange:
    exchange_id: str = config["exchange"]["id"]
    cls = getattr(ccxt, exchange_id)
    exchange: ccxt.Exchange = cls(
        {
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
        }
    )
    if config["exchange"].get("testnet"):
        exchange.set_sandbox_mode(True)
    return exchange


def fetch_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    limit: int = 1000,
    since: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch OHLCV candles and return a DataFrame with a UTC DatetimeIndex."""
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def fetch_ohlcv_history(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    total: int = 5000,
    page_limit: int = 1000,
    pause: float = 0.0,
) -> pd.DataFrame:
    """Fetch up to `total` candles by paging backward past the per-call API cap.

    Most exchanges return at most ~1000 candles per request. To evaluate a
    strategy across bull AND bear regimes you need far more than that, so this
    walks backward in time, stitching pages together until it has `total` bars
    (or the exchange runs out of history).

    Returns a chronologically-sorted, de-duplicated DataFrame with a UTC index.
    """
    step_ms = _TIMEFRAME_MS.get(timeframe)
    if step_ms is None:
        # Unknown timeframe — fall back to a single page.
        return fetch_ohlcv(exchange, symbol, timeframe, limit=min(total, page_limit))

    now_ms = exchange.milliseconds()
    # Start one page back from "now" and walk earlier each iteration.
    end_ms = now_ms
    frames: list[pd.DataFrame] = []
    collected = 0

    while collected < total:
        want = min(page_limit, total - collected)
        since = end_ms - want * step_ms
        raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=want)
        if not raw:
            break

        page = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        page["timestamp"] = pd.to_datetime(page["timestamp"], unit="ms", utc=True)
        page.set_index("timestamp", inplace=True)
        frames.append(page)
        collected += len(page)

        earliest_ms = int(raw[0][0])
        if earliest_ms <= since or earliest_ms >= end_ms:
            # No older data available (or no progress) — stop to avoid looping.
            if earliest_ms >= end_ms:
                break
        end_ms = earliest_ms
        if pause:
            time.sleep(pause)
        if len(page) < want:
            break  # exchange returned a short page → history exhausted

    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    if len(df) > total:
        df = df.iloc[-total:]
    return df
