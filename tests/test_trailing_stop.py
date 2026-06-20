"""Tests for the trailing-stop exit in RiskManager."""

import pandas as pd
import pytest

from trading_bot.execution.paper import PaperExecutor
from trading_bot.risk.manager import RiskManager
from trading_bot.strategies.base import Signal, SignalType


def _config(trailing_stop_pct=0.0, stop_loss_pct=0.50, take_profit_pct=0.50):
    return {
        "risk": {
            "max_position_pct": 0.50,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "trailing_stop_pct": trailing_stop_pct,
            "daily_loss_limit_pct": 0.95,
            "leverage": 1,
        },
        "news": {},
    }


def _rm(trailing_stop_pct=0.0, **kw):
    ex = PaperExecutor(initial_capital=10_000.0, fee_rate=0.0)
    rm = RiskManager(ex, _config(trailing_stop_pct=trailing_stop_pct, **kw), "BTC/USDT")
    return rm, ex


def _sig(t, price):
    return Signal(t, price, pd.Timestamp.now(tz="UTC"), "test")


class TestTrailingStop:
    def test_disabled_by_default(self):
        rm, _ = _rm(trailing_stop_pct=0.0)
        rm.process(_sig(SignalType.BUY, 100.0), 100.0)
        # Price rises then dips 5% from peak — should NOT exit (trailing disabled, SL is 50%)
        rm.process(_sig(SignalType.HOLD, 120.0), 120.0)
        order = rm.process(_sig(SignalType.HOLD, 114.0), 114.0)
        assert order is None

    def test_triggers_after_peak_then_drop(self):
        rm, _ = _rm(trailing_stop_pct=0.05)   # 5% trailing
        rm.process(_sig(SignalType.BUY, 100.0), 100.0)
        rm.process(_sig(SignalType.HOLD, 120.0), 120.0)   # peak = 120
        # Drop to 113 = 5.8% below peak → should exit
        order = rm.process(_sig(SignalType.HOLD, 113.0), 113.0)
        assert order is not None
        assert order.side == "sell"

    def test_does_not_trigger_within_threshold(self):
        rm, _ = _rm(trailing_stop_pct=0.05)
        rm.process(_sig(SignalType.BUY, 100.0), 100.0)
        rm.process(_sig(SignalType.HOLD, 120.0), 120.0)   # peak = 120
        # Drop to 116 = 3.3% below peak → still within 5%, hold
        order = rm.process(_sig(SignalType.HOLD, 116.0), 116.0)
        assert order is None

    def test_peak_keeps_rising(self):
        rm, _ = _rm(trailing_stop_pct=0.10)
        rm.process(_sig(SignalType.BUY, 100.0), 100.0)
        rm.process(_sig(SignalType.HOLD, 110.0), 110.0)
        rm.process(_sig(SignalType.HOLD, 130.0), 130.0)   # peak now 130
        # 120 is 9% below new peak 130 → within 10%, hold (even though above earlier prices)
        assert rm.process(_sig(SignalType.HOLD, 120.0), 120.0) is None
        # 116 is 10.7% below peak → exit
        assert rm.process(_sig(SignalType.HOLD, 116.0), 116.0) is not None

    def test_resets_between_positions(self):
        rm, _ = _rm(trailing_stop_pct=0.05)
        # First position: buy, peak 120, trailing-stop exit
        rm.process(_sig(SignalType.BUY, 100.0), 100.0)
        rm.process(_sig(SignalType.HOLD, 120.0), 120.0)
        rm.process(_sig(SignalType.HOLD, 113.0), 113.0)   # exits
        # New position at 100 — peak should reset, a small dip shouldn't exit
        rm.process(_sig(SignalType.BUY, 100.0), 100.0)
        order = rm.process(_sig(SignalType.HOLD, 98.0), 98.0)   # 2% below new peak
        assert order is None

    def test_fixed_stop_loss_still_works_with_trailing_on(self):
        rm, _ = _rm(trailing_stop_pct=0.10, stop_loss_pct=0.03)
        rm.process(_sig(SignalType.BUY, 100.0), 100.0)
        # Immediate 4% drop, no peak above entry → fixed SL (3%) fires
        order = rm.process(_sig(SignalType.HOLD, 96.0), 96.0)
        assert order is not None
        assert order.side == "sell"
