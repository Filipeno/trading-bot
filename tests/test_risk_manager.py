"""Unit tests for the risk management layer.

These cover the highest-stakes logic: stop-loss, take-profit, daily loss limit,
kill switch, and position sizing. Bugs here cost real money.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from trading_bot.execution.paper import PaperExecutor
from trading_bot.risk.manager import (
    DailyLossLimitError,
    KillSwitchError,
    RiskManager,
)
from trading_bot.strategies.base import Signal, SignalType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIG = {
    "risk": {
        "max_position_pct": 0.10,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
        "daily_loss_limit_pct": 0.05,
    }
}
SYMBOL = "BTC/USDT"


def _make_rm(capital: float = 10_000.0) -> tuple[RiskManager, PaperExecutor]:
    ex = PaperExecutor(capital, fee_rate=0.001)
    rm = RiskManager(ex, CONFIG, SYMBOL)
    return rm, ex


def _signal(t: SignalType, price: float = 50_000.0) -> Signal:
    return Signal(t, price, pd.Timestamp.now(tz="UTC"))


def buy(price: float = 50_000.0) -> Signal:
    return _signal(SignalType.BUY, price)


def sell(price: float = 50_000.0) -> Signal:
    return _signal(SignalType.SELL, price)


def hold(price: float = 50_000.0) -> Signal:
    return _signal(SignalType.HOLD, price)


# ---------------------------------------------------------------------------
# Stop-loss & take-profit
# ---------------------------------------------------------------------------


class TestStopLoss:
    def test_triggers_when_loss_exceeds_threshold(self):
        rm, ex = _make_rm()
        rm.process(buy(50_000), 50_000)
        # 3% drop > 2% SL
        result = rm.process(hold(48_500), 48_500)
        assert result is not None
        assert result.side == "sell"

    def test_does_not_trigger_within_threshold(self):
        rm, ex = _make_rm()
        rm.process(buy(50_000), 50_000)
        # 1.8% drop < 2% SL
        result = rm.process(hold(49_100), 49_100)
        assert result is None

    def test_position_cleared_after_stop_loss(self):
        rm, ex = _make_rm()
        rm.process(buy(50_000), 50_000)
        rm.process(hold(48_500), 48_500)  # SL fires
        assert ex.get_position(SYMBOL) == pytest.approx(0.0, abs=1e-9)

    def test_take_profit_triggers_sell(self):
        rm, ex = _make_rm()
        rm.process(buy(50_000), 50_000)
        # 4.2% gain > 4% TP
        result = rm.process(hold(52_100), 52_100)
        assert result is not None
        assert result.side == "sell"

    def test_take_profit_within_threshold_does_not_trigger(self):
        rm, ex = _make_rm()
        rm.process(buy(50_000), 50_000)
        # 3.9% gain < 4% TP
        result = rm.process(hold(51_950), 51_950)
        assert result is None


# ---------------------------------------------------------------------------
# Daily loss limit
# ---------------------------------------------------------------------------


class TestDailyLossLimit:
    # Use 50% position size and SL disabled so a price drop can breach the 5% daily limit.
    # With 10% position size the max single-trade loss is 10%, but SL fires at 2% first,
    # capping the loss well below 5% of portfolio — so SL would always win.
    _CONFIG = {
        "risk": {
            "max_position_pct": 0.50,
            "stop_loss_pct": 0.99,       # effectively disabled for this test group
            "take_profit_pct": 0.99,
            "daily_loss_limit_pct": 0.05,
        }
    }

    def _make(self, capital: float = 10_000.0) -> tuple[RiskManager, PaperExecutor]:
        ex = PaperExecutor(capital, fee_rate=0.001)
        rm = RiskManager(ex, self._CONFIG, SYMBOL)
        return rm, ex

    def test_halts_when_limit_breached(self):
        rm, _ = self._make()
        rm._day_start_equity = 10_000.0
        # 50% of 10 000 = 5 000 worth @ 50 000 → 0.1 BTC
        # At 43 000: equity ≈ 5 000 + 0.1 * 43 000 = 9 300 → 7% loss > 5%
        rm.process(buy(50_000), 50_000)
        with pytest.raises(DailyLossLimitError):
            rm.process(hold(43_000), 43_000)

    def test_does_not_halt_before_limit(self):
        rm, _ = self._make()
        rm._day_start_equity = 10_000.0
        rm.process(buy(50_000), 50_000)
        # Tiny drop — well within 5%
        result = rm.process(hold(49_900), 49_900)
        assert result is None

    def test_kill_switch_set_after_daily_loss_breach(self):
        rm, _ = self._make()
        rm._day_start_equity = 10_000.0
        rm.process(buy(50_000), 50_000)
        with pytest.raises(DailyLossLimitError):
            rm.process(hold(43_000), 43_000)
        assert rm._killed is True


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------


class TestKillSwitch:
    def test_pre_set_kill_switch_blocks_all_orders(self):
        rm, _ = _make_rm()
        rm._killed = True
        with pytest.raises(KillSwitchError):
            rm.process(buy(50_000), 50_000)

    def test_unhandled_exception_activates_kill_switch(self):
        rm, _ = _make_rm()
        rm._executor = MagicMock()
        rm._executor.get_balance.side_effect = RuntimeError("exchange API down")
        rm._executor.get_position.return_value = 0.0
        with pytest.raises(KillSwitchError):
            rm.process(buy(50_000), 50_000)
        assert rm._killed is True

    def test_subsequent_call_after_exception_raises_kill_switch(self):
        rm, _ = _make_rm()
        rm._executor = MagicMock()
        rm._executor.get_balance.side_effect = RuntimeError("API down")
        rm._executor.get_position.return_value = 0.0
        with pytest.raises(KillSwitchError):
            rm.process(buy(50_000), 50_000)
        # Second call must also raise, not silently proceed
        with pytest.raises(KillSwitchError):
            rm.process(hold(50_000), 50_000)


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------


class TestPositionSizing:
    def test_size_respects_max_position_pct(self):
        rm, ex = _make_rm(10_000.0)
        rm.process(buy(50_000), 50_000)
        # 10% of 10 000 = 1 000 / 50 000 = 0.02 BTC
        assert ex.get_position(SYMBOL) == pytest.approx(0.02, rel=1e-3)

    def test_no_double_entry(self):
        rm, ex = _make_rm(10_000.0)
        rm.process(buy(50_000), 50_000)
        pos_after_first = ex.get_position(SYMBOL)
        rm.process(buy(51_000), 51_000)  # second BUY while already long — ignore
        assert ex.get_position(SYMBOL) == pos_after_first

    def test_no_sell_without_position(self):
        rm, ex = _make_rm(10_000.0)
        result = rm.process(sell(50_000), 50_000)  # no open position
        assert result is None
