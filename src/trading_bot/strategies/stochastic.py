import pandas as pd

from .base import Signal, SignalType, Strategy
from .indicators import stochastic


class StochasticStrategy(Strategy):
    """Stochastic oscillator crossover.

    BUY  — %K crosses above %D while in the oversold zone (<= oversold).
    SELL — %K crosses below %D while in the overbought zone (>= overbought).

    A classic momentum-reversal trigger that's responsive on intraday charts.
    """

    def __init__(
        self,
        k_period: int = 14,
        d_period: int = 3,
        oversold: float = 20.0,
        overbought: float = 80.0,
    ) -> None:
        self.k_period = k_period
        self.d_period = d_period
        self.oversold = oversold
        self.overbought = overbought

    def reset(self) -> None:
        pass

    def next(self, df: pd.DataFrame) -> Signal:
        price = float(df["close"].iloc[-1])
        ts = df.index[-1]
        if len(df) < self.k_period + self.d_period + 1:
            return Signal(SignalType.HOLD, price, ts, "insufficient data")

        k, d = stochastic(df, self.k_period, self.d_period)
        k_prev, k_curr = float(k.iloc[-2]), float(k.iloc[-1])
        d_prev, d_curr = float(d.iloc[-2]), float(d.iloc[-1])

        crossed_up = k_prev <= d_prev and k_curr > d_curr
        crossed_down = k_prev >= d_prev and k_curr < d_curr

        if crossed_up and k_curr <= self.oversold + 10:
            return Signal(SignalType.BUY, price, ts,
                          f"%K crossed above %D in oversold ({k_curr:.1f})")
        if crossed_down and k_curr >= self.overbought - 10:
            return Signal(SignalType.SELL, price, ts,
                          f"%K crossed below %D in overbought ({k_curr:.1f})")
        return Signal(SignalType.HOLD, price, ts, f"%K={k_curr:.1f} %D={d_curr:.1f}")
