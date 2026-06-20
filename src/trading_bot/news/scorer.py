"""Sentiment scoring for news headlines.

Extension point for LLM-based scoring
--------------------------------------
Implement the SentimentScorer Protocol with an LLM-based class:

    class LLMSentimentScorer:
        def score(self, text: str) -> SentimentScore:
            # Call your LLM API here.
            # IMPORTANT: only call for fresh/relevant headlines — cache by URL or
            # title hash. The keyword scorer is synchronous and called per bar;
            # an LLM scorer should use a background queue with rate limiting.
            ...

Then pass it to NewsSentimentStrategy instead of KeywordSentimentScorer.
No other code needs to change.
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class SentimentScore:
    value: float                          # -1.0 (very bearish) to +1.0 (very bullish)
    confidence: float                     # 0.0 to 1.0
    matched_keywords: list[str] = field(default_factory=list)


@runtime_checkable
class SentimentScorer(Protocol):
    def score(self, text: str) -> SentimentScore: ...


# Crypto-domain lexicons — extend these lists to tune precision/recall

_BULLISH: frozenset[str] = frozenset([
    "surge", "soar", "rally", "bullish", "breakout", "all-time high", "ath",
    "adoption", "institutional", "partnership", "upgrade", "launch", "listing",
    "approval", "etf", "bull", "accumulate", "milestone", "integration",
    "record high", "buy", "long",
])

_BEARISH: frozenset[str] = frozenset([
    "crash", "plunge", "dump", "bearish", "hack", "exploit", "ban",
    "lawsuit", "sec", "fraud", "sell-off", "collapse", "scam",
    "liquidation", "liquidated", "fear", "warning", "suspend",
    "halt", "investigation", "delist", "selloff", "exit scam",
])


class KeywordSentimentScorer:
    """Fast, zero-dependency lexicon-based scorer.

    Scores by counting bullish vs bearish keyword hits and normalising to [-1, 1].
    Confidence rises with the number of keyword matches (saturates at 3+).

    Suitable as a default and as a baseline to compare against an LLM scorer.
    """

    def score(self, text: str) -> SentimentScore:
        lower = text.lower()
        bullish_hits = [kw for kw in _BULLISH if kw in lower]
        bearish_hits = [kw for kw in _BEARISH if kw in lower]

        total = len(bullish_hits) + len(bearish_hits)
        if total == 0:
            return SentimentScore(0.0, 0.0)

        raw = len(bullish_hits) - len(bearish_hits)
        value = max(-1.0, min(1.0, raw / total))
        confidence = min(1.0, total / 3)

        return SentimentScore(value, confidence, bullish_hits + bearish_hits)
