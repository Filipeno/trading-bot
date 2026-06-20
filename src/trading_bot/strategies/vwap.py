import pandas as pd

from .base import Signal, SignalType, Strategy
from .indicators import rolling_vwap


class VWAPStrategy(Strategy):
    """Rolling VWAP cross.

    BUY  — price crosses ABOVE the volume-weighted average price (demand taking over).
    SELL — price crosses BELOW it.

    VWAP is a favourite intraday reference because it weights by volume — it
    shows the average price people actually traded at, not just the midpoint.
    """

    def __init__(self, period: int = 20) -> None:
        self.period = period

    def reset(self) -> None:
        pass

    def next(self, df: pd.DataFrame) -> Signal:
        price = float(df["close"].iloc[-1])
        ts = df.index[-1]
        if len(df) < self.period + 1:
            return Signal(SignalType.HOLD, price, ts, "insufficient data")

        vwap = rolling_vwap(df, self.period)
        if pd.isna(vwap.iloc[-1]) or pd.isna(vwap.iloc[-2]):
            return Signal(SignalType.HOLD, price, ts, "vwap warming up")

        prev_close = float(df["close"].iloc[-2])
        prev_vwap = float(vwap.iloc[-2])
        curr_vwap = float(vwap.iloc[-1])

        if prev_close <= prev_vwap and price > curr_vwap:
            return Signal(SignalType.BUY, price, ts, f"price crossed above VWAP ({curr_vwap:.2f})")
        if prev_close >= prev_vwap and price < curr_vwap:
            return Signal(SignalType.SELL, price, ts, f"price crossed below VWAP ({curr_vwap:.2f})")
        return Signal(SignalType.HOLD, price, ts, f"VWAP {curr_vwap:.2f}")
