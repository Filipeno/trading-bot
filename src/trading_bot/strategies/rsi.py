import pandas as pd

from .base import Signal, SignalType, Strategy
from .indicators import rsi


class RSIStrategy(Strategy):
    """RSI mean-reversion.

    BUY  — RSI crosses back UP through the oversold level (momentum turning up
           after being beaten down).
    SELL — RSI crosses back DOWN through the overbought level.

    Using threshold *crossings* (not just "RSI < 30") avoids firing the same
    signal on every bar while price stays stretched.
    """

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0) -> None:
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def reset(self) -> None:
        pass

    def next(self, df: pd.DataFrame) -> Signal:
        price = float(df["close"].iloc[-1])
        ts = df.index[-1]
        if len(df) < self.period + 2:
            return Signal(SignalType.HOLD, price, ts, "insufficient data")

        r = rsi(df["close"], self.period)
        prev, curr = float(r.iloc[-2]), float(r.iloc[-1])

        if prev <= self.oversold and curr > self.oversold:
            return Signal(SignalType.BUY, price, ts,
                          f"RSI crossed up out of oversold ({curr:.1f})")
        if prev >= self.overbought and curr < self.overbought:
            return Signal(SignalType.SELL, price, ts,
                          f"RSI crossed down out of overbought ({curr:.1f})")
        return Signal(SignalType.HOLD, price, ts, f"RSI {curr:.1f}")
