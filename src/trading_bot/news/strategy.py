"""NewsSentimentStrategy — implements the Strategy interface using news sentiment.

Backtest lag
------------
To avoid look-ahead bias, only news published before
    current_bar_timestamp - ingestion_lag_seconds
is considered. This simulates the time between a headline being published and
the bot actually ingesting and processing it. Do not set this to 0 in backtests.
"""

import logging
from datetime import timedelta

import pandas as pd

from ..strategies.base import Signal, SignalType, Strategy
from .scorer import SentimentScorer
from .sources.base import NewsSource

logger = logging.getLogger(__name__)


class NewsSentimentStrategy(Strategy):
    """Generates BUY/SELL/HOLD signals from aggregated recent news sentiment.

    Works as an independent strategy or as a signal source for CombinedStrategy.
    Uses the same next(df) interface as all other strategies — no special wiring needed.
    """

    def __init__(
        self,
        source: NewsSource,
        scorer: SentimentScorer,
        symbol: str,
        ingestion_lag_seconds: int = 60,
        lookback_minutes: int = 60,
        sentiment_threshold: float = 0.3,
        min_confidence: float = 0.3,
    ) -> None:
        self._source = source
        self._scorer = scorer
        self._symbol = symbol
        self._ingestion_lag = timedelta(seconds=ingestion_lag_seconds)
        self._lookback = timedelta(minutes=lookback_minutes)
        self._sentiment_threshold = sentiment_threshold
        self._min_confidence = min_confidence

    def reset(self) -> None:
        pass  # stateless

    def next(self, df: pd.DataFrame) -> Signal:
        price = float(df["close"].iloc[-1])
        current_ts = df.index[-1]

        score = self._aggregate_score(current_ts)
        if score is None:
            return Signal(SignalType.HOLD, price, current_ts, "no relevant/confident news")

        if score > self._sentiment_threshold:
            return Signal(SignalType.BUY, price, current_ts, f"bullish news aggregate={score:.2f}")
        if score < -self._sentiment_threshold:
            return Signal(SignalType.SELL, price, current_ts, f"bearish news aggregate={score:.2f}")

        return Signal(SignalType.HOLD, price, current_ts, f"neutral news aggregate={score:.2f}")

    def aggregate_score(self, df: pd.DataFrame) -> float | None:
        """Raw aggregate sentiment value for the current bar's window.

        Returns None when no confident signals were found in the window.
        Used by CombinedStrategy to filter technical signals without
        making a full BUY/SELL decision.
        """
        return self._aggregate_score(df.index[-1])

    # ------------------------------------------------------------------

    def _aggregate_score(self, current_ts: pd.Timestamp) -> float | None:
        cutoff_ts = current_ts - self._ingestion_lag
        lookback_ts = current_ts - self._lookback

        try:
            items = self._source.fetch(self._symbol)
        except Exception as exc:
            logger.warning("News fetch failed: %s", exc)
            return None

        relevant = [
            item for item in items
            if lookback_ts <= item.published_at <= cutoff_ts
        ]

        if not relevant:
            return None

        scores = [self._scorer.score(item.title) for item in relevant]
        confident = [s for s in scores if s.confidence >= self._min_confidence]

        if not confident:
            return None

        return sum(s.value for s in confident) / len(confident)
