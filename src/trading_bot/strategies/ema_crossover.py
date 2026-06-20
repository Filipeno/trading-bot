import pandas as pd

from .base import Signal, SignalType, Strategy


class EMACrossover(Strategy):
    """Long-only EMA crossover: BUY when fast crosses above slow, SELL when it crosses below."""

    def __init__(self, fast: int = 20, slow: int = 50) -> None:
        self.fast = fast
        self.slow = slow

    def reset(self) -> None:
        pass  # stateless — EMA is recomputed from the full history each call

    def next(self, df: pd.DataFrame) -> Signal:
        price = df["close"].iloc[-1]
        ts = df.index[-1]

        if len(df) < self.slow + 1:
            return Signal(SignalType.HOLD, price, ts, "insufficient data")

        ema_fast = df["close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.slow, adjust=False).mean()

        curr_fast, curr_slow = ema_fast.iloc[-1], ema_slow.iloc[-1]
        prev_fast, prev_slow = ema_fast.iloc[-2], ema_slow.iloc[-2]

        if prev_fast <= prev_slow and curr_fast > curr_slow:
            return Signal(SignalType.BUY, price, ts, f"EMA{self.fast} crossed above EMA{self.slow}")
        if prev_fast >= prev_slow and curr_fast < curr_slow:
            return Signal(SignalType.SELL, price, ts, f"EMA{self.fast} crossed below EMA{self.slow}")

        return Signal(SignalType.HOLD, price, ts, "no crossover")
