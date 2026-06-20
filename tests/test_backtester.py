import pandas as pd
import pytest

from trading_bot.backtest.engine import BacktestEngine
from trading_bot.strategies.base import Signal, SignalType, Strategy


# ---------------------------------------------------------------------------
# Test strategies
# ---------------------------------------------------------------------------


class _AlwaysHold(Strategy):
    def reset(self) -> None:
        pass

    def next(self, df: pd.DataFrame) -> Signal:
        return Signal(SignalType.HOLD, df["close"].iloc[-1], df.index[-1])


class _BuyThenSell(Strategy):
    """Buys on bar `buy_bar` and sells on bar `sell_bar` (1-indexed count of bars seen)."""

    def __init__(self, buy_bar: int, sell_bar: int) -> None:
        self.buy_bar = buy_bar
        self.sell_bar = sell_bar

    def reset(self) -> None:
        pass

    def next(self, df: pd.DataFrame) -> Signal:
        n = len(df)
        price = df["close"].iloc[-1]
        ts = df.index[-1]
        if n == self.buy_bar:
            return Signal(SignalType.BUY, price, ts)
        if n == self.sell_bar:
            return Signal(SignalType.SELL, price, ts)
        return Signal(SignalType.HOLD, price, ts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _df(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1.0},
        index=idx,
    )


def _engine(fee: float = 0.001, slip: float = 0.0005) -> BacktestEngine:
    return BacktestEngine(initial_capital=10_000.0, fee_rate=fee, slippage_pct=slip)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_equity_curve_length_matches_input():
    result = _engine().run(_df([100.0] * 20), _AlwaysHold())
    assert len(result.equity_curve) == 20


def test_no_trades_when_always_hold():
    result = _engine().run(_df([100.0] * 20), _AlwaysHold())
    assert result.metrics["n_trades"] == 0
    assert result.metrics["total_return_pct"] == 0.0


def test_profitable_trade_increases_equity():
    # Buy at 100, sell at 200 → ~100% gross profit minus fees/slippage
    closes = [100.0] * 5 + [200.0] * 5
    result = _engine(fee=0.0, slip=0.0).run(_df(closes), _BuyThenSell(buy_bar=3, sell_bar=8))
    assert result.metrics["total_return_pct"] > 0
    assert result.metrics["n_trades"] == 1


def test_fees_reduce_returns():
    closes = [100.0] * 5 + [150.0] * 5
    r_no_fee = BacktestEngine(10_000.0, fee_rate=0.0, slippage_pct=0.0).run(
        _df(closes), _BuyThenSell(3, 8)
    )
    r_fee = BacktestEngine(10_000.0, fee_rate=0.01, slippage_pct=0.005).run(
        _df(closes), _BuyThenSell(3, 8)
    )
    assert r_no_fee.metrics["total_return_pct"] > r_fee.metrics["total_return_pct"]


def test_losing_trade_decreases_equity():
    # Buy at 100, sell at 50 → loss
    closes = [100.0] * 5 + [50.0] * 5
    result = _engine(fee=0.0, slip=0.0).run(_df(closes), _BuyThenSell(buy_bar=3, sell_bar=8))
    assert result.metrics["total_return_pct"] < 0


def test_win_rate_one_winning_trade():
    closes = [100.0] * 5 + [200.0] * 5
    result = _engine().run(_df(closes), _BuyThenSell(buy_bar=3, sell_bar=8))
    assert result.metrics["win_rate_pct"] == 100.0


def test_metrics_keys_present():
    result = _engine().run(_df([100.0] * 10), _AlwaysHold())
    for key in ("total_return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate_pct", "n_trades", "final_equity"):
        assert key in result.metrics


def test_strategy_reset_called():
    class _Counting(Strategy):
        def __init__(self):
            self.resets = 0
        def reset(self):
            self.resets += 1
        def next(self, df):
            return Signal(SignalType.HOLD, df["close"].iloc[-1], df.index[-1])

    strat = _Counting()
    _engine().run(_df([100.0] * 5), strat)
    assert strat.resets == 1
