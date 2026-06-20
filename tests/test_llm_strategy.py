"""Tests for the LLM strategy — all offline using a fake client (no network, no key)."""

import numpy as np
import pandas as pd
import pytest

from trading_bot.strategies.base import SignalType
from trading_bot.strategies.llm_strategy import (
    LLMClient,
    LLMStrategy,
    _parse_response,
    build_market_summary,
)


# ── Fakes ───────────────────────────────────────────────────────────────────

class FakeClient:
    """Returns a canned response; records the prompt it was given."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.last_prompt: str | None = None

    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response


class RaisingClient:
    def complete(self, prompt: str) -> str:
        raise RuntimeError("simulated API outage")


def _df(n: int = 60, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prices = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    prices = np.clip(prices, 1.0, None)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"open": prices, "high": prices + 0.5, "low": prices - 0.5,
         "close": prices, "volume": 10.0},
        index=idx,
    )


# ── Response parsing ─────────────────────────────────────────────────────────

class TestParseResponse:
    def test_plain_json(self):
        d = _parse_response('{"signal": "BUY", "confidence": 0.8, "reason": "uptrend"}')
        assert d["signal"] == "BUY"

    def test_json_in_markdown_fence(self):
        text = '```json\n{"signal": "SELL", "confidence": 0.7, "reason": "x"}\n```'
        d = _parse_response(text)
        assert d["signal"] == "SELL"

    def test_json_with_surrounding_prose(self):
        text = 'Here is my decision: {"signal": "HOLD", "confidence": 0.2, "reason": "unclear"} ok?'
        d = _parse_response(text)
        assert d["signal"] == "HOLD"

    def test_unparseable_returns_none(self):
        assert _parse_response("I think you should buy maybe") is None

    def test_empty_returns_none(self):
        assert _parse_response("") is None

    def test_missing_signal_key_returns_none(self):
        assert _parse_response('{"confidence": 0.9}') is None


# ── Market summary ───────────────────────────────────────────────────────────

class TestMarketSummary:
    def test_summary_has_expected_keys(self):
        s = build_market_summary(_df(60), lookback=20)
        for key in ("current_price", "ema12", "ema26", "trend", "rsi14",
                    "position_in_range_pct", "recent_closes"):
            assert key in s

    def test_rsi_in_valid_range(self):
        s = build_market_summary(_df(60))
        assert 0.0 <= s["rsi14"] <= 100.0

    def test_trend_is_up_or_down(self):
        s = build_market_summary(_df(60))
        assert s["trend"] in ("up", "down")


# ── Strategy behavior ────────────────────────────────────────────────────────

class TestLLMStrategy:
    def test_buy_signal_high_confidence(self):
        client = FakeClient('{"signal": "BUY", "confidence": 0.9, "reason": "strong uptrend"}')
        strat = LLMStrategy(client=client, min_confidence=0.55)
        sig = strat.next(_df(60))
        assert sig.type == SignalType.BUY

    def test_sell_signal_high_confidence(self):
        client = FakeClient('{"signal": "SELL", "confidence": 0.8, "reason": "breakdown"}')
        strat = LLMStrategy(client=client, min_confidence=0.55)
        assert strat.next(_df(60)).type == SignalType.SELL

    def test_hold_signal(self):
        client = FakeClient('{"signal": "HOLD", "confidence": 0.9, "reason": "choppy"}')
        strat = LLMStrategy(client=client)
        assert strat.next(_df(60)).type == SignalType.HOLD

    def test_low_confidence_buy_downgraded_to_hold(self):
        client = FakeClient('{"signal": "BUY", "confidence": 0.30, "reason": "weak"}')
        strat = LLMStrategy(client=client, min_confidence=0.55)
        assert strat.next(_df(60)).type == SignalType.HOLD

    def test_api_error_yields_hold(self):
        strat = LLMStrategy(client=RaisingClient())
        sig = strat.next(_df(60))
        assert sig.type == SignalType.HOLD
        assert "error" in sig.reason.lower()

    def test_unparseable_response_yields_hold(self):
        client = FakeClient("sure, buy some bitcoin!")
        strat = LLMStrategy(client=client)
        assert strat.next(_df(60)).type == SignalType.HOLD

    def test_invalid_signal_value_yields_hold(self):
        client = FakeClient('{"signal": "MOON", "confidence": 0.99}')
        strat = LLMStrategy(client=client)
        assert strat.next(_df(60)).type == SignalType.HOLD

    def test_insufficient_data_yields_hold_without_calling_client(self):
        client = FakeClient('{"signal": "BUY", "confidence": 0.9}')
        strat = LLMStrategy(client=client, lookback=30)
        sig = strat.next(_df(10))   # too few bars
        assert sig.type == SignalType.HOLD
        assert client.last_prompt is None   # client never called

    def test_prompt_contains_market_numbers(self):
        client = FakeClient('{"signal": "HOLD", "confidence": 0.5, "reason": "x"}')
        strat = LLMStrategy(client=client)
        strat.next(_df(60))
        assert "current_price" in client.last_prompt
        assert "rsi14" in client.last_prompt

    def test_reset_is_noop(self):
        client = FakeClient('{"signal": "BUY", "confidence": 0.9, "reason": "x"}')
        strat = LLMStrategy(client=client, min_confidence=0.55)
        s1 = strat.next(_df(60))
        strat.reset()
        s2 = strat.next(_df(60))
        assert s1.type == s2.type

    def test_fake_client_satisfies_protocol(self):
        assert isinstance(FakeClient("x"), LLMClient)
