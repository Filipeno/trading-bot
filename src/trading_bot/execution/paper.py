import pandas as pd

from .base import Executor, OrderResult


class PaperExecutor(Executor):
    """Simulates order fills locally — no network calls, no real money.

    When leverage > 1 only the margin (notional / leverage) is deducted from
    capital on entry.  P&L on close is computed on the full notional so gains
    and losses are properly amplified.  get_equity() returns the true equity
    (free cash + locked margin + unrealized P&L) rather than the raw balance.
    """

    def __init__(self, initial_capital: float, fee_rate: float, leverage: int = 1) -> None:
        self._capital = initial_capital
        self._fee_rate = fee_rate
        self._leverage = max(1, int(leverage))
        self._positions: dict[str, float] = {}
        self._margin: dict[str, float] = {}      # margin locked per symbol
        self._avg_entry: dict[str, float] = {}   # average entry price per symbol

    # ------------------------------------------------------------------
    # Executor interface
    # ------------------------------------------------------------------

    def submit_order(self, symbol: str, side: str, size: float, price: float) -> OrderResult:
        notional = size * price
        margin = notional / self._leverage   # capital committed as collateral
        fee = notional * self._fee_rate
        ts = pd.Timestamp.now(tz="UTC")

        if side == "buy":
            old_size = self._positions.get(symbol, 0.0)
            old_entry = self._avg_entry.get(symbol, price)
            new_size = old_size + size
            self._avg_entry[symbol] = (
                (old_size * old_entry + size * price) / new_size if new_size > 0 else price
            )
            self._margin[symbol] = self._margin.get(symbol, 0.0) + margin
            self._capital -= margin + fee
            self._positions[symbol] = new_size

        elif side == "sell":
            old_size = self._positions.get(symbol, 0.0)
            entry = self._avg_entry.get(symbol, price)
            # Pro-rata share of locked margin being released
            fraction = (size / old_size) if old_size > 0 else 1.0
            entry_margin = self._margin.get(symbol, 0.0) * fraction
            pnl = size * (price - entry)          # full leveraged P&L
            self._capital += entry_margin + pnl - fee
            self._margin[symbol] = self._margin.get(symbol, 0.0) - entry_margin
            self._positions[symbol] = max(0.0, old_size - size)

        else:
            raise ValueError(f"Unknown side: {side!r}")

        return OrderResult(symbol=symbol, side=side, size=size, fill_price=price, fee=fee, timestamp=ts)

    def get_balance(self) -> float:
        """Free cash (margin locked in positions is excluded)."""
        return self._capital

    def get_position(self, symbol: str) -> float:
        return self._positions.get(symbol, 0.0)

    def get_equity(self, symbol: str, current_price: float) -> float:
        """True equity: free cash + locked margin + unrealized P&L."""
        entry = self._avg_entry.get(symbol, current_price)
        position = self._positions.get(symbol, 0.0)
        unrealized_pnl = position * (current_price - entry)
        return self._capital + self._margin.get(symbol, 0.0) + unrealized_pnl
