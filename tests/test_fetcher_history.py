"""Tests for the paginating deep-history fetcher (no network — mock exchange)."""

import pandas as pd
import pytest

from trading_bot.data.fetcher import fetch_ohlcv_history

_STEP_MS = 900_000  # 15m


class _FakeExchange:
    """Serves OHLCV from an in-memory history, mimicking a 1000-bar API cap."""

    def __init__(self, n_bars: int, cap: int = 1000) -> None:
        self.cap = cap
        self.calls = 0
        base = 1_700_000_000_000  # arbitrary epoch ms
        self._bars = [
            [base + i * _STEP_MS, 100.0, 101.0, 99.0, 100.0 + i, 10.0]
            for i in range(n_bars)
        ]

    def milliseconds(self) -> int:
        return self._bars[-1][0] + _STEP_MS

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
        self.calls += 1
        limit = min(limit, self.cap)
        rows = self._bars
        if since is not None:
            rows = [r for r in rows if r[0] >= since]
        return rows[:limit]


class TestFetchOhlcvHistory:
    def test_pages_beyond_single_cap(self):
        ex = _FakeExchange(n_bars=3000, cap=1000)
        df = fetch_ohlcv_history(ex, "BTC/USDT", "15m", total=2500, page_limit=1000)
        # Should have stitched multiple pages → more than one call, lots of bars
        assert ex.calls > 1
        assert len(df) >= 2000

    def test_respects_total_cap(self):
        ex = _FakeExchange(n_bars=5000, cap=1000)
        df = fetch_ohlcv_history(ex, "BTC/USDT", "15m", total=1500, page_limit=1000)
        assert len(df) <= 1500

    def test_index_is_sorted_and_unique(self):
        ex = _FakeExchange(n_bars=3000, cap=1000)
        df = fetch_ohlcv_history(ex, "BTC/USDT", "15m", total=2500)
        assert df.index.is_monotonic_increasing
        assert not df.index.has_duplicates

    def test_stops_when_history_exhausted(self):
        # Only 500 bars exist but we ask for 5000 → returns what's available, no hang
        ex = _FakeExchange(n_bars=500, cap=1000)
        df = fetch_ohlcv_history(ex, "BTC/USDT", "15m", total=5000)
        assert len(df) <= 500
        assert len(df) > 0

    def test_has_ohlcv_columns(self):
        ex = _FakeExchange(n_bars=1500, cap=1000)
        df = fetch_ohlcv_history(ex, "BTC/USDT", "15m", total=1200)
        for col in ("open", "high", "low", "close", "volume"):
            assert col in df.columns

    def test_unknown_timeframe_falls_back_to_single_page(self):
        ex = _FakeExchange(n_bars=2000, cap=1000)
        df = fetch_ohlcv_history(ex, "BTC/USDT", "7m", total=2000)
        # Unknown tf → single fetch, capped
        assert len(df) <= 1000
