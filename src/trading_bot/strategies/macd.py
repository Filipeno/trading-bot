import pandas as pd

from .base import Signal, SignalType, Strategy


class MACDStrategy(Strategy):
    """MACD crossover: BUY when the MACD histogram turns positive, SELL when it turns negative.

    Uses histogram zero-crosses (MACD - signal) rather than the raw line cross,
    which gives slightly earlier entries and is less prone to whipsaws on short timeframes.

    Default periods (12/26/9) are the standard for daily charts.
    For 15m day trading, shorter periods (6/13/4 or 8/21/5) produce more signals.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def reset(self) -> None:
        pass  # stateless

    def next(self, df: pd.DataFrame) -> Signal:
        price = float(df["close"].iloc[-1])
        ts = df.index[-1]

        min_bars = self.slow + self.signal + 1
        if len(df) < min_bars:
            return Signal(SignalType.HOLD, price, ts, "insufficient data")

        close = df["close"]
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
        histogram = macd_line - signal_line

        prev_h = histogram.iloc[-2]
        curr_h = histogram.iloc[-1]

        if prev_h <= 0 and curr_h > 0:
            return Signal(SignalType.BUY, price, ts,
                          f"MACD histogram crossed positive (h={curr_h:.4f})")
        if prev_h >= 0 and curr_h < 0:
            return Signal(SignalType.SELL, price, ts,
                          f"MACD histogram crossed negative (h={curr_h:.4f})")

        return Signal(SignalType.HOLD, price, ts, f"MACD no cross (h={curr_h:.4f})")
