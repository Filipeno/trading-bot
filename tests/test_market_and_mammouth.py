"""Tests for the asset-agnostic market feed and the OpenAI-compatible LLM client.

All offline — yfinance and requests are monkeypatched, no network calls.
"""

import sys
import types

import pandas as pd
import pytest

from trading_bot.data import market
from trading_bot.data.market import CryptoFeed, StockFeed, asset_class, make_feed


# ── asset_class dispatch ─────────────────────────────────────────────────────

class TestAssetClass:
    def test_defaults_to_crypto(self):
        assert asset_class({}) == "crypto"

    def test_reads_stocks(self):
        assert asset_class({"market": {"asset_class": "stocks"}}) == "stocks"

    def test_case_insensitive(self):
        assert asset_class({"market": {"asset_class": "STOCKS"}}) == "stocks"


# ── make_feed routing ────────────────────────────────────────────────────────

class TestMakeFeed:
    def test_crypto_returns_cryptofeed(self, monkeypatch):
        # Avoid building a real ccxt exchange
        monkeypatch.setattr(market, "make_exchange", lambda *a, **k: object())
        feed = make_feed({"market": {"asset_class": "crypto"},
                          "exchange": {"id": "binance", "symbol": "BTC/USDT"}})
        assert isinstance(feed, CryptoFeed)

    def test_stocks_returns_stockfeed(self, monkeypatch):
        # Provide a fake yfinance module so StockFeed import check passes
        fake_yf = types.ModuleType("yfinance")
        monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
        feed = make_feed({"market": {"asset_class": "stocks"}})
        assert isinstance(feed, StockFeed)


# ── StockFeed via mocked yfinance ────────────────────────────────────────────

def _fake_yf_module(df: pd.DataFrame):
    mod = types.ModuleType("yfinance")
    def download(symbol, period=None, interval=None, progress=False, auto_adjust=False):
        return df
    mod.download = download
    return mod


def _yf_frame(n=100, tz_aware=True):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h",
                        tz="America/New_York" if tz_aware else None)
    return pd.DataFrame({
        "Open": range(n), "High": range(n), "Low": range(n),
        "Close": [100 + i for i in range(n)], "Adj Close": range(n),
        "Volume": [1000] * n,
    }, index=idx)


class TestStockFeed:
    def test_history_normalizes_columns_and_index(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "yfinance", _fake_yf_module(_yf_frame(120)))
        feed = StockFeed({})
        df = feed.history("AAPL", "1h", total=50)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert str(df.index.tz) == "UTC"
        assert len(df) == 50

    def test_history_handles_multiindex_columns(self, monkeypatch):
        raw = _yf_frame(60)
        raw.columns = pd.MultiIndex.from_product([raw.columns, ["AAPL"]])
        monkeypatch.setitem(sys.modules, "yfinance", _fake_yf_module(raw))
        feed = StockFeed({})
        df = feed.history("AAPL", "1h", total=30)
        assert "close" in df.columns

    def test_unsupported_timeframe_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "yfinance", _fake_yf_module(_yf_frame(10)))
        feed = StockFeed({})
        with pytest.raises(ValueError):
            feed.history("AAPL", "4h", total=10)

    def test_empty_response_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "yfinance", _fake_yf_module(pd.DataFrame()))
        feed = StockFeed({})
        with pytest.raises(ValueError):
            feed.history("AAPL", "1h", total=10)

    def test_daily_index_localized(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "yfinance", _fake_yf_module(_yf_frame(40, tz_aware=False)))
        feed = StockFeed({})
        df = feed.history("AAPL", "1d", total=20)
        assert str(df.index.tz) == "UTC"


# ── OpenAI-compatible / Mammouth client (mocked requests) ────────────────────

class TestOpenAICompatibleClient:
    def _client(self, monkeypatch, response_json=None, raise_http=False):
        monkeypatch.setenv("MAMMOUTH_API_KEY", "test-key")
        from trading_bot.strategies.llm_strategy import OpenAICompatibleClient

        captured = {}

        class FakeResp:
            def raise_for_status(self):
                if raise_http:
                    raise RuntimeError("HTTP 401")
            def json(self):
                return response_json or {
                    "choices": [{"message": {"content": '{"signal":"BUY","confidence":0.8}'}}]
                }

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return FakeResp()

        fake_requests = types.ModuleType("requests")
        fake_requests.post = fake_post
        monkeypatch.setitem(sys.modules, "requests", fake_requests)

        return OpenAICompatibleClient(
            base_url="https://api.mammouth.ai/v1", model="gpt-4o-mini",
        ), captured

    def test_requires_key(self, monkeypatch):
        monkeypatch.delenv("MAMMOUTH_API_KEY", raising=False)
        from trading_bot.strategies.llm_strategy import OpenAICompatibleClient
        with pytest.raises(ValueError):
            OpenAICompatibleClient()

    def test_completes_and_parses_content(self, monkeypatch):
        client, _ = self._client(monkeypatch)
        out = client.complete("hello")
        assert "BUY" in out

    def test_sends_bearer_auth_and_model(self, monkeypatch):
        client, captured = self._client(monkeypatch)
        client.complete("hi")
        assert captured["headers"]["Authorization"] == "Bearer test-key"
        assert captured["json"]["model"] == "gpt-4o-mini"
        assert captured["url"].endswith("/chat/completions")

    def test_http_error_propagates(self, monkeypatch):
        client, _ = self._client(monkeypatch, raise_http=True)
        with pytest.raises(RuntimeError):
            client.complete("hi")

    def test_explicit_api_key_works_without_env(self, monkeypatch):
        # BYOK: a per-session key passed directly, no env var present
        monkeypatch.delenv("MAMMOUTH_API_KEY", raising=False)
        from trading_bot.strategies.llm_strategy import OpenAICompatibleClient
        client = OpenAICompatibleClient(api_key="session-key-123")
        assert client._api_key == "session-key-123"

    def test_explicit_api_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("MAMMOUTH_API_KEY", "env-key")
        from trading_bot.strategies.llm_strategy import OpenAICompatibleClient
        client = OpenAICompatibleClient(api_key="byok-key")
        assert client._api_key == "byok-key"


# ── Factory builds Mammouth-backed LLM strategy ──────────────────────────────

class TestFactoryMammouth:
    def test_factory_builds_mammouth_llm(self, monkeypatch):
        monkeypatch.setenv("MAMMOUTH_API_KEY", "test-key")
        from trading_bot.strategies.factory import make_strategy
        from trading_bot.strategies.llm_strategy import LLMStrategy, OpenAICompatibleClient

        strat = make_strategy({"strategy": {
            "name": "llm", "llm_provider": "mammouth",
            "llm_base_url": "https://api.mammouth.ai/v1", "llm_model": "gpt-4o-mini",
        }})
        assert isinstance(strat, LLMStrategy)
        assert isinstance(strat.client, OpenAICompatibleClient)
