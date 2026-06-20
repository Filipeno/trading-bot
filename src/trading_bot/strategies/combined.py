"""CombinedStrategy — filters technical signals through news sentiment.

Logic:
  - Technical says BUY  + news bearish  → HOLD (news contradicts, skip)
  - Technical says SELL + news bullish  → HOLD (news contradicts, skip)
  - Technical says BUY/SELL + news neutral (no confident signal) → pass through
  - Technical says HOLD → always HOLD regardless of news
  - require_agreement=False → ignore news, behave like the technical strategy alone
"""

import logging

import pandas as pd

from .base import Signal, SignalType, Strategy
from ..news.strategy import NewsSentimentStrategy

logger = logging.getLogger(__name__)


class CombinedStrategy(Strategy):
    def __init__(
        self,
        technical: Strategy,
        news: NewsSentimentStrategy,
        require_agreement: bool = True,
    ) -> None:
        self._technical = technical
        self._news = news
        self._require_agreement = require_agreement

    def reset(self) -> None:
        self._technical.reset()
        self._news.reset()

    def next(self, df: pd.DataFrame) -> Signal:
        tech_signal = self._technical.next(df)

        if tech_signal.type == SignalType.HOLD or not self._require_agreement:
            return tech_signal

        score = self._news.aggregate_score(df)
        if score is None:
            # No confident news in window — trust the technical signal
            return tech_signal

        price = float(df["close"].iloc[-1])
        ts = df.index[-1]

        if tech_signal.type == SignalType.BUY and score < 0:
            logger.info("CombinedStrategy: suppressing BUY — news score=%.2f", score)
            return Signal(
                SignalType.HOLD, price, ts,
                f"technical BUY suppressed by bearish news (score={score:.2f})"
            )

        if tech_signal.type == SignalType.SELL and score > 0:
            logger.info("CombinedStrategy: suppressing SELL — news score=%.2f", score)
            return Signal(
                SignalType.HOLD, price, ts,
                f"technical SELL suppressed by bullish news (score={score:.2f})"
            )

        return tech_signal
