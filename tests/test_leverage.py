"""Tests for leverage support in PaperExecutor, BacktestEngine, and RiskManager."""

import pandas as pd
import pytest

from trading_bot.backtest.engine import BacktestEngine
from trading_bot.execution.paper import PaperExecutor
from trading_bot.strategies.base import Signal, SignalType


# ── Helpers ────────────────────────────────────────────────────────────────

def _df(prices: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="15min", tz="UTC")
    return pd.DataFrame({"close": prices}, index=idx)


# Fixed signal strategy for backtester tests
class _FixedStrategy:
    def __init__(self, signals: list[SignalType]) -> None:
        self._signals = signals
        self._i = 0

    def reset(self) -> None:
        self._i = 0

    def next(self, df: pd.DataFrame) -> Signal:
        t = self._signals[min(self._i, len(self._signals) - 1)]
        self._i += 1
        price = float(df["close"].iloc[-1])
        return Signal(t, price, df.index[-1], "fixed")


# ══════════════════════════════════════════════════════════════════════════════
# PaperExecutor with leverage
# ══════════════════════════════════════════════════════════════════════════════

class TestPaperExecutorLeverage:
    def test_1x_buy_deducts_full_notional(self):
        ex = PaperExecutor(initial_capital=1000.0, fee_rate=0.0, leverage=1)
        ex.submit_order("BTC/USDT", "buy", 0.01, 50_000.0)
        # 0.01 BTC * 50_000 = 500 deducted; margin = 500 / 1 = 500
        assert ex.get_balance() == pytest.approx(500.0)

    def test_5x_buy_deducts_only_margin(self):
        ex = PaperExecutor(initial_capital=1000.0, fee_rate=0.0, leverage=5)
        # With 5x leverage: margin = notional / 5
        # size = 0.01, notional = 500, margin = 100
        ex.submit_order("BTC/USDT", "buy", 0.01, 50_000.0)
        assert ex.get_balance() == pytest.approx(900.0)   # 1000 - 100

    def test_5x_sell_amplifies_profit(self):
        ex = PaperExecutor(initial_capital=1000.0, fee_rate=0.0, leverage=5)
        # Buy 0.1 BTC at 50_000 (notional=5000, margin=1000)
        ex.submit_order("BTC/USDT", "buy", 0.1, 50_000.0)
        assert ex.get_balance() == pytest.approx(0.0)   # all capital as margin
        # Sell at 55_000: pnl = 0.1 * 5000 = 500
        ex.submit_order("BTC/USDT", "sell", 0.1, 55_000.0)
        # capital = margin(1000) + pnl(500) = 1500
        assert ex.get_balance() == pytest.approx(1500.0)

    def test_5x_sell_amplifies_loss(self):
        ex = PaperExecutor(initial_capital=1000.0, fee_rate=0.0, leverage=5)
        ex.submit_order("BTC/USDT", "buy", 0.1, 50_000.0)
        # Sell at 48_000: pnl = 0.1 * -2000 = -200
        ex.submit_order("BTC/USDT", "sell", 0.1, 48_000.0)
        assert ex.get_balance() == pytest.approx(800.0)  # 1000 - 200

    def test_get_equity_reflects_unrealized_pnl(self):
        ex = PaperExecutor(initial_capital=1000.0, fee_rate=0.0, leverage=5)
        ex.submit_order("BTC/USDT", "buy", 0.1, 50_000.0)
        # Mark-to-market at 55_000: pnl = 500
        eq = ex.get_equity("BTC/USDT", 55_000.0)
        # equity = 0 (free cash) + 1000 (margin) + 500 (unrealized) = 1500
        assert eq == pytest.approx(1500.0)

    def test_get_equity_at_entry_price_equals_initial(self):
        ex = PaperExecutor(initial_capital=1000.0, fee_rate=0.0, leverage=5)
        ex.submit_order("BTC/USDT", "buy", 0.1, 50_000.0)
        eq = ex.get_equity("BTC/USDT", 50_000.0)
        assert eq == pytest.approx(1000.0)   # no unrealized P&L yet

    def test_position_tracks_correctly(self):
        ex = PaperExecutor(initial_capital=1000.0, fee_rate=0.0, leverage=1)
        ex.submit_order("BTC/USDT", "buy", 0.02, 50_000.0)
        assert ex.get_position("BTC/USDT") == pytest.approx(0.02)
        ex.submit_order("BTC/USDT", "sell", 0.02, 52_000.0)
        assert ex.get_position("BTC/USDT") == pytest.approx(0.0)

    def test_fee_deducted_on_notional(self):
        ex = PaperExecutor(initial_capital=1000.0, fee_rate=0.001, leverage=5)
        # 0.01 BTC * 50_000 = 500 notional; fee = 0.5; margin = 100
        result = ex.submit_order("BTC/USDT", "buy", 0.01, 50_000.0)
        assert result.fee == pytest.approx(0.5)          # fee on notional
        assert ex.get_balance() == pytest.approx(899.5)  # 1000 - 100 - 0.5

    def test_1x_equity_matches_position_mark(self):
        # For 1x, get_equity should equal get_balance + position * price
        ex = PaperExecutor(initial_capital=1000.0, fee_rate=0.0, leverage=1)
        ex.submit_order("BTC/USDT", "buy", 0.01, 50_000.0)
        price = 55_000.0
        expected = ex.get_balance() + ex.get_position("BTC/USDT") * price
        assert ex.get_equity("BTC/USDT", price) == pytest.approx(expected)


# ══════════════════════════════════════════════════════════════════════════════
# BacktestEngine with leverage
# ══════════════════════════════════════════════════════════════════════════════

class TestBacktestEngineLeverage:
    def _engine(self, leverage: int = 1):
        return BacktestEngine(
            initial_capital=1000.0,
            fee_rate=0.0,
            slippage_pct=0.0,
            leverage=leverage,
        )

    def test_5x_amplifies_return_vs_1x(self):
        prices = [100.0] * 5 + [110.0]   # 10% price gain
        df = _df(prices)
        # BUY at bar 0, SELL at bar 5
        sigs = [SignalType.BUY] + [SignalType.HOLD] * 4 + [SignalType.SELL]
        strat = _FixedStrategy(sigs)

        result_1x = self._engine(leverage=1).run(df, strat)
        strat.reset()
        result_5x = self._engine(leverage=5).run(df, strat)

        # 5x should produce ~5× larger return
        ret_1x = result_1x.metrics["total_return_pct"]
        ret_5x = result_5x.metrics["total_return_pct"]
        assert ret_5x > ret_1x
        assert ret_5x == pytest.approx(ret_1x * 5, rel=0.05)

    def test_5x_amplifies_loss_vs_1x(self):
        prices = [100.0] * 5 + [90.0]   # 10% price drop
        df = _df(prices)
        sigs = [SignalType.BUY] + [SignalType.HOLD] * 4 + [SignalType.SELL]

        strat = _FixedStrategy(sigs)
        result_1x = self._engine(leverage=1).run(df, strat)
        strat.reset()
        result_5x = self._engine(leverage=5).run(df, strat)

        ret_1x = result_1x.metrics["total_return_pct"]
        ret_5x = result_5x.metrics["total_return_pct"]
        # Both negative; 5x should be ~5× worse
        assert ret_5x < ret_1x
        assert abs(ret_5x) == pytest.approx(abs(ret_1x) * 5, rel=0.05)

    def test_1x_return_matches_no_leverage_formula(self):
        prices = [100.0, 100.0, 110.0]
        df = _df(prices)
        sigs = [SignalType.BUY, SignalType.HOLD, SignalType.SELL]
        strat = _FixedStrategy(sigs)
        result = self._engine(leverage=1).run(df, strat)
        # Return = 10%
        assert result.metrics["total_return_pct"] == pytest.approx(10.0, rel=0.01)

    def test_liquidation_sets_equity_to_zero(self):
        # 5x leverage → liquidation at -20% price move
        prices = [100.0] * 3 + [79.0]   # > 20% drop
        df = _df(prices)
        sigs = [SignalType.BUY, SignalType.HOLD, SignalType.HOLD, SignalType.HOLD]
        strat = _FixedStrategy(sigs)
        result = self._engine(leverage=5).run(df, strat)
        # Equity should hit 0 at or before the last bar
        assert result.equity_curve.min() <= 0.0

    def test_no_trades_produces_flat_equity(self):
        df = _df([100.0] * 10)
        sigs = [SignalType.HOLD] * 10
        strat = _FixedStrategy(sigs)
        result = self._engine(leverage=5).run(df, strat)
        assert all(v == pytest.approx(1000.0) for v in result.equity_curve.values)

    def test_leverage_1_behaves_same_as_no_leverage_arg(self):
        prices = [100.0] * 3 + [110.0]
        df = _df(prices)
        sigs = [SignalType.BUY, SignalType.HOLD, SignalType.HOLD, SignalType.SELL]

        strat = _FixedStrategy(sigs)
        result_default = BacktestEngine(1000.0, 0.0, 0.0).run(df, strat)
        strat.reset()
        result_1x = self._engine(leverage=1).run(df, strat)

        assert result_default.metrics["total_return_pct"] == pytest.approx(
            result_1x.metrics["total_return_pct"]
        )
