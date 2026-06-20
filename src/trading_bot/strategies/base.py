from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    type: SignalType
    price: float
    timestamp: pd.Timestamp
    reason: str = ""


class Strategy(ABC):
    @abstractmethod
    def next(self, df: pd.DataFrame) -> Signal:
        """Return a Signal given OHLCV history up to and including the current bar.

        df must have columns [open, high, low, close, volume] and a DatetimeIndex.
        The last row is the current (just-closed) bar.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state. Called before each backtest or paper session."""
