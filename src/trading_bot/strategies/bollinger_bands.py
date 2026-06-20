import pandas as pd

from .base import Signal, SignalType, Strategy


class BollingerBandsStrategy(Strategy):
    """Bollinger Bands mean-reversion strategy.

    BUY  — close below the lower band (oversold / stretched move down).
    SELL — close above the upper band (overbought / stretched move up).

    The RiskManager's stop-loss handles downside protection if the price
    continues through the lower band rather than reverting.

    For day trading on 15m bars, period=20 / std=2.0 is a standard starting
    point.  Tighter bands (std=1.5) generate more signals but more false ones.
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0) -> None:
        self.period = period
        self.std_dev = std_dev

    def reset(self) -> None:
        pass  # stateless

    def next(self, df: pd.DataFrame) -> Signal:
        price = float(df["close"].iloc[-1])
        ts = df.index[-1]

        if len(df) < self.period + 1:
            return Signal(SignalType.HOLD, price, ts, "insufficient data")

        close = df["close"]
        middle = close.rolling(self.period).mean()
        std = close.rolling(self.period).std(ddof=0)
        upper = middle + self.std_dev * std
        lower = middle - self.std_dev * std

        curr_lower = float(lower.iloc[-1])
        curr_upper = float(upper.iloc[-1])
        curr_mid = float(middle.iloc[-1])

        if price < curr_lower:
            pct = (curr_lower - price) / curr_lower * 100
            return Signal(SignalType.BUY, price, ts,
                          f"price {pct:.2f}% below lower band ({curr_lower:.2f})")
        if price > curr_upper:
            pct = (price - curr_upper) / curr_upper * 100
            return Signal(SignalType.SELL, price, ts,
                          f"price {pct:.2f}% above upper band ({curr_upper:.2f})")

        return Signal(SignalType.HOLD, price, ts,
                      f"inside bands [{curr_lower:.2f}, {curr_mid:.2f}, {curr_upper:.2f}]")
