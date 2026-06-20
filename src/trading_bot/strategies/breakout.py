import pandas as pd

from .base import Signal, SignalType, Strategy


class BreakoutStrategy(Strategy):
    """Donchian-channel breakout strategy.

    BUY  — current close breaks above the highest close of the prior N bars.
    SELL — current close breaks below the lowest  close of the prior N bars.

    Uses the *prior* N bars (excludes the current bar) so there is no
    look-ahead bias.

    This is a momentum / trend-following strategy — the opposite of Bollinger
    mean-reversion.  Works well on trending assets like BTC on 15m bars.
    The stop-loss in RiskManager is the primary downside guard; the SELL
    signal is an additional exit when the price breaks down.
    """

    def __init__(self, period: int = 20) -> None:
        self.period = period

    def reset(self) -> None:
        pass  # stateless

    def next(self, df: pd.DataFrame) -> Signal:
        price = float(df["close"].iloc[-1])
        ts = df.index[-1]

        # Need at least period + 1 bars (period for the lookback, 1 current)
        if len(df) < self.period + 1:
            return Signal(SignalType.HOLD, price, ts, "insufficient data")

        prior = df["close"].iloc[-(self.period + 1):-1]
        prev_high = float(prior.max())
        prev_low = float(prior.min())

        if price > prev_high:
            return Signal(SignalType.BUY, price, ts,
                          f"breakout above {self.period}-bar high ({prev_high:.2f})")
        if price < prev_low:
            return Signal(SignalType.SELL, price, ts,
                          f"breakdown below {self.period}-bar low ({prev_low:.2f})")

        return Signal(SignalType.HOLD, price, ts,
                      f"inside channel [{prev_low:.2f}, {prev_high:.2f}]")
