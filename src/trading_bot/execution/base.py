from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class OrderResult:
    symbol: str
    side: str        # "buy" | "sell"
    size: float      # base currency units
    fill_price: float
    fee: float
    timestamp: pd.Timestamp


class Executor(ABC):
    """Swappable execution backend. PaperExecutor for testing, LiveExecutor for production."""

    @abstractmethod
    def submit_order(self, symbol: str, side: str, size: float, price: float) -> OrderResult: ...

    @abstractmethod
    def get_balance(self) -> float:
        """Return available quote-currency balance (e.g. USDT)."""

    @abstractmethod
    def get_position(self, symbol: str) -> float:
        """Return current base-currency position size (e.g. BTC units)."""

    def get_equity(self, symbol: str, current_price: float) -> float:
        """Return total portfolio equity at current_price.

        Override in subclasses that use margin/leverage accounting.
        The default implementation (cash + position * price) is correct for
        1x trading but will overstate equity for leveraged positions.
        """
        return self.get_balance() + self.get_position(symbol) * current_price
