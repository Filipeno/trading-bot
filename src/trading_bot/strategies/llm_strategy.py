"""LLM-driven trading strategy.

An LLM reads a compact, numeric description of the recent chart (the same things
a human sees on a graph: trend, momentum, RSI, where price sits in its range) and
returns a BUY / SELL / HOLD decision with its reasoning.

HONEST CAVEAT — read this:
    An LLM has no real-time market data, no speed advantage, and can be confidently
    wrong. It will NOT reliably beat the market, and every call costs money. This is
    here because it is a legitimate, interesting tool and the architecture supports
    it — not because it prints money. Validate it with the Reality Check
    (`optimize.py`) like any other strategy, and never trade it with real funds until
    it has proven itself on the testnet for a long time. It still passes through the
    RiskManager, so stop-loss / daily-loss / kill-switch protections all apply.

Design:
    - The LLM is reached through an `LLMClient` Protocol, so tests (and offline use)
      can inject a fake client with no network and no API key.
    - Any failure — network, bad key, unparseable response, low confidence — yields
      HOLD. The bot never trades on a broken or uncertain LLM response.
    - The API key is read from the environment (ANTHROPIC_API_KEY), never hardcoded.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .base import Signal, SignalType, Strategy

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Client abstraction (so the strategy is testable without the network)
# ──────────────────────────────────────────────────────────────────────────

@runtime_checkable
class LLMClient(Protocol):
    """Anything that turns a prompt into a text completion."""

    def complete(self, prompt: str) -> str: ...


class AnthropicClient:
    """Calls the Anthropic API. Key comes from the environment, never from code.

    Lazy-imports the `anthropic` SDK so the rest of the project works without it
    installed. Install with:  pip install anthropic
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_tokens: int = 400,
        api_key: str = "",
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        # Explicit api_key (e.g. a per-session BYOK key) wins over the env var.
        self._api_key = api_key or os.getenv(api_key_env, "")
        if not self._api_key:
            raise ValueError(
                f"No API key found in ${api_key_env}. Add it to your .env file. "
                "Get one at https://console.anthropic.com/"
            )
        self._client = None  # built lazily

    def _ensure_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "The 'anthropic' package is required for the LLM strategy. "
                    "Install it with: pip install anthropic"
                ) from exc
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(self, prompt: str) -> str:
        client = self._ensure_client()
        msg = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # Concatenate any text blocks in the response.
        return "".join(getattr(b, "text", "") for b in msg.content)


class OpenAICompatibleClient:
    """Calls any OpenAI-compatible /chat/completions endpoint via plain HTTP.

    Works with Mammouth.ai, OpenAI, OpenRouter, Together, local servers, etc.
    Uses `requests` (already a dependency) so no extra SDK is needed.

    Mammouth.ai note:
        Set `base_url` and `model` to the values shown in YOUR Mammouth account /
        API docs, and put the key in the env var named by `api_key_env`
        (default MAMMOUTH_API_KEY). The default base_url below is a best guess —
        verify it against your dashboard; if requests fail, the base_url or model
        name is the first thing to check.
    """

    def __init__(
        self,
        base_url: str = "https://api.mammouth.ai/v1",
        model: str = "gpt-4o",
        api_key_env: str = "MAMMOUTH_API_KEY",
        max_tokens: int = 400,
        timeout: int = 30,
        api_key: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        # Explicit api_key (e.g. a per-session BYOK key) wins over the env var.
        self._api_key = api_key or os.getenv(api_key_env, "")
        if not self._api_key:
            raise ValueError(
                f"No API key found in ${api_key_env}. Add it to your .env file."
            )

    def complete(self, prompt: str) -> str:
        import requests

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        # Standard OpenAI shape: choices[0].message.content
        return data["choices"][0]["message"]["content"]


# ──────────────────────────────────────────────────────────────────────────
# Indicator helpers (turn the graph into numbers the LLM can read)
# ──────────────────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if pd.notna(val) else 50.0


def _pct_change(close: pd.Series, bars: int) -> float:
    if len(close) <= bars:
        return 0.0
    return float((close.iloc[-1] / close.iloc[-1 - bars] - 1.0) * 100.0)


def build_market_summary(df: pd.DataFrame, lookback: int = 20) -> dict:
    """Produce the numeric 'picture of the chart' the LLM will reason over."""
    close = df["close"]
    price = float(close.iloc[-1])
    ema_fast = float(close.ewm(span=12, adjust=False).mean().iloc[-1])
    ema_slow = float(close.ewm(span=26, adjust=False).mean().iloc[-1])
    window = close.iloc[-lookback:]
    hi, lo = float(window.max()), float(window.min())
    pos_in_range = (price - lo) / (hi - lo) * 100 if hi > lo else 50.0
    recent = [round(float(c), 2) for c in close.iloc[-min(lookback, 20):]]

    return {
        "current_price": round(price, 2),
        "ema12": round(ema_fast, 2),
        "ema26": round(ema_slow, 2),
        "trend": "up" if ema_fast > ema_slow else "down",
        "rsi14": round(_rsi(close), 1),
        "change_1bar_pct": round(_pct_change(close, 1), 2),
        "change_6bar_pct": round(_pct_change(close, 6), 2),
        "change_24bar_pct": round(_pct_change(close, 24), 2),
        f"high_{lookback}bar": round(hi, 2),
        f"low_{lookback}bar": round(lo, 2),
        "position_in_range_pct": round(pos_in_range, 1),
        "recent_closes": recent,
    }


_PROMPT_TEMPLATE = """You are a disciplined crypto trading analyst. Analyze this BTC chart snapshot and decide whether to BUY, SELL, or HOLD a long spot position.

Market snapshot (most recent bar last):
{summary}

Rules:
- BUY only if there is a clear bullish setup you would risk money on.
- SELL only to exit/avoid a clear bearish setup.
- When uncertain, choose HOLD. Most of the time the right answer is HOLD.
- You have no information beyond this snapshot. Do not assume external news.

Respond with ONLY a JSON object, no other text:
{{"signal": "BUY" | "SELL" | "HOLD", "confidence": 0.0-1.0, "reason": "one short sentence"}}"""


def _parse_response(text: str) -> Optional[dict]:
    """Extract the JSON decision from the model's text. Returns None if unparseable."""
    if not text:
        return None
    # Strip markdown fences if present, then grab the first {...} block.
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if "signal" not in data:
        return None
    return data


# ──────────────────────────────────────────────────────────────────────────
# The strategy
# ──────────────────────────────────────────────────────────────────────────

class LLMStrategy(Strategy):
    """Ask an LLM for a trading decision each bar.

    Parameters
    ----------
    client : LLMClient
        Anything with `.complete(prompt) -> str`. Inject AnthropicClient for real
        use, or a fake for tests/offline.
    lookback : int
        How many recent bars to summarize for the model.
    min_confidence : float
        Below this confidence the decision is downgraded to HOLD.
    """

    def __init__(
        self,
        client: LLMClient,
        lookback: int = 30,
        min_confidence: float = 0.55,
    ) -> None:
        self.client = client
        self.lookback = lookback
        self.min_confidence = min_confidence

    def reset(self) -> None:
        pass  # stateless across the session (each bar is an independent query)

    def next(self, df: pd.DataFrame) -> Signal:
        price = float(df["close"].iloc[-1])
        ts = df.index[-1]

        # Need enough history for the indicators to be meaningful.
        if len(df) < max(self.lookback, 30):
            return Signal(SignalType.HOLD, price, ts, "insufficient data for LLM")

        summary = build_market_summary(df, self.lookback)
        prompt = _PROMPT_TEMPLATE.format(summary=json.dumps(summary, indent=2))

        # Any failure → HOLD. We never trade on a broken or missing response.
        try:
            raw = self.client.complete(prompt)
        except Exception as exc:  # network, auth, rate-limit, etc.
            logger.warning("LLM call failed (%s) — holding.", exc)
            return Signal(SignalType.HOLD, price, ts, f"LLM error: {exc}")

        data = _parse_response(raw)
        if data is None:
            logger.warning("LLM response unparseable — holding. Raw: %.120s", raw)
            return Signal(SignalType.HOLD, price, ts, "LLM response unparseable")

        signal_str = str(data.get("signal", "HOLD")).upper().strip()
        confidence = float(data.get("confidence", 0.0) or 0.0)
        reason = str(data.get("reason", ""))[:160]

        if signal_str not in ("BUY", "SELL", "HOLD"):
            return Signal(SignalType.HOLD, price, ts, f"LLM invalid signal '{signal_str}'")

        # Low-confidence non-HOLD calls are downgraded to HOLD.
        if signal_str in ("BUY", "SELL") and confidence < self.min_confidence:
            return Signal(
                SignalType.HOLD, price, ts,
                f"LLM {signal_str} but low confidence {confidence:.2f} < {self.min_confidence}",
            )

        sig_type = SignalType[signal_str]
        return Signal(sig_type, price, ts, f"LLM: {reason} (conf={confidence:.2f})")
