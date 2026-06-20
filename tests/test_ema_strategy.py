import pandas as pd
import pytest

from trading_bot.strategies.base import SignalType
from trading_bot.strategies.ema_crossover import EMACrossover


def _df(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1.0},
        index=idx,
    )


def test_hold_with_insufficient_data():
    strat = EMACrossover(fast=20, slow=50)
    strat.reset()
    df = _df([100.0] * 40)  # 40 bars but need slow+1 = 51
    assert strat.next(df).type == SignalType.HOLD


def test_hold_when_no_crossover():
    strat = EMACrossover(fast=5, slow=20)
    strat.reset()
    # Flat prices — no crossover possible
    df = _df([100.0] * 60)
    assert strat.next(df).type == SignalType.HOLD


def test_buy_signal_on_upward_crossover():
    strat = EMACrossover(fast=5, slow=20)
    strat.reset()
    # Bear phase then sharp spike — fast EMA crosses above slow EMA during the spike
    closes = [100.0] * 30 + [60.0] * 20 + [300.0] * 10
    df = _df(closes)
    signals = [strat.next(df.iloc[: i + 1]) for i in range(len(df))]
    assert any(s.type == SignalType.BUY for s in signals)


def test_sell_signal_on_downward_crossover():
    strat = EMACrossover(fast=5, slow=20)
    strat.reset()
    # Bull phase then sharp crash — fast EMA crosses below slow EMA during the crash
    closes = [100.0] * 30 + [200.0] * 20 + [10.0] * 10
    df = _df(closes)
    signals = [strat.next(df.iloc[: i + 1]) for i in range(len(df))]
    assert any(s.type == SignalType.SELL for s in signals)


def test_reset_is_idempotent():
    strat = EMACrossover()
    strat.reset()
    strat.reset()  # calling twice must not raise


def test_signal_price_matches_last_close():
    strat = EMACrossover(fast=5, slow=20)
    strat.reset()
    closes = [100.0] * 60
    df = _df(closes)
    sig = strat.next(df)
    assert sig.price == closes[-1]


def test_signal_timestamp_matches_last_bar():
    strat = EMACrossover(fast=5, slow=20)
    strat.reset()
    closes = [100.0] * 60
    df = _df(closes)
    sig = strat.next(df)
    assert sig.timestamp == df.index[-1]
