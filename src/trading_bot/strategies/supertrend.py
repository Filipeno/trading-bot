import pandas as pd

from .base import Signal, SignalType, Strategy
from .indicators import supertrend


class SupertrendStrategy(Strategy):
    """Supertrend (ATR-channel) trend follower.

    BUY  — Supertrend flips to the up direction (+1).
    SELL — Supertrend flips to the down direction (-1).

    Very popular for crypto because it adapts its stop distance to volatility
    via ATR, so it stays out of the way during calm chop but reacts when a real
    trend develops. Lower `multiplier` = more sensitive (more trades).
    """

    def __init__(self, period: int = 10, multiplier: float = 3.0) -> None:
        self.period = period
        self.multiplier = multiplier

    def reset(self) -> None:
        pass

    def next(self, df: pd.DataFrame) -> Signal:
        price = float(df["close"].iloc[-1])
        ts = df.index[-1]
        if len(df) < self.period + 2:
            return Signal(SignalType.HOLD, price, ts, "insufficient data")

        direction = supertrend(df, self.period, self.multiplier)
        prev, curr = int(direction.iloc[-2]), int(direction.iloc[-1])

        if prev < 0 and curr > 0:
            return Signal(SignalType.BUY, price, ts, "Supertrend flipped UP")
        if prev > 0 and curr < 0:
            return Signal(SignalType.SELL, price, ts, "Supertrend flipped DOWN")
        return Signal(SignalType.HOLD, price, ts,
                      f"Supertrend {'up' if curr > 0 else 'down'}")
