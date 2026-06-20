"""Tests for the news-event halt hook in RiskManager.

The halt is independent of whether the news strategy is enabled —
it's a pure risk guard activated by notify_news_event().
"""

import pandas as pd
import pytest

from trading_bot.execution.paper import PaperExecutor
from trading_bot.risk.manager import KillSwitchError, RiskManager
from trading_bot.strategies.base import Signal, SignalType

SYMBOL = "BTC/USDT"

_CONFIG = {
    "risk": {
        "max_position_pct": 0.10,
        "stop_loss_pct": 0.99,   # disabled so it doesn't interfere
        "take_profit_pct": 0.99,
        "daily_loss_limit_pct": 0.99,
    },
    "news": {
        "event_halt_minutes": 5,
        "high_impact_threshold": 0.70,
    },
}


def _make_rm(capital: float = 10_000.0) -> tuple[RiskManager, PaperExecutor]:
    ex = PaperExecutor(capital, fee_rate=0.001)
    return RiskManager(ex, _CONFIG, SYMBOL), ex


def _buy(ts: pd.Timestamp) -> Signal:
    return Signal(SignalType.BUY, 50_000.0, ts, "test")


def _hold(ts: pd.Timestamp) -> Signal:
    return Signal(SignalType.HOLD, 50_000.0, ts, "test")


class TestNewsHalt:
    def test_high_impact_news_suppresses_new_entry(self):
        rm, ex = _make_rm()
        now = pd.Timestamp.now(tz="UTC")

        # High-impact bearish event
        rm.notify_news_event(-0.9, now)

        # BUY signal 2 minutes later — should be suppressed
        result = rm.process(_buy(now + pd.Timedelta(minutes=2)), 50_000.0)
        assert result is None
        assert ex.get_position(SYMBOL) == 0.0

    def test_halt_expires_after_configured_minutes(self):
        rm, ex = _make_rm()
        now = pd.Timestamp.now(tz="UTC")
        rm.notify_news_event(-0.9, now)

        # BUY signal 6 minutes later — halt has expired (halt_minutes=5)
        result = rm.process(_buy(now + pd.Timedelta(minutes=6)), 50_000.0)
        assert result is not None
        assert result.side == "buy"

    def test_low_impact_news_does_not_halt(self):
        rm, ex = _make_rm()
        now = pd.Timestamp.now(tz="UTC")

        # Score 0.5 is below high_impact_threshold=0.70
        rm.notify_news_event(0.5, now)

        result = rm.process(_buy(now + pd.Timedelta(minutes=2)), 50_000.0)
        assert result is not None  # trade proceeds normally

    def test_halt_does_not_block_exits(self):
        """SL/TP exits must never be suppressed by a news halt."""
        rm, ex = _make_rm()
        now = pd.Timestamp.now(tz="UTC")

        # Open a position first
        rm.process(_buy(now - pd.Timedelta(hours=1)), 50_000.0)

        # High-impact news event right now
        rm.notify_news_event(-0.9, now)

        # Price drops enough to trigger SL (2% configured in risk, but we set 99%
        # for this test config — use a custom config here to test SL interaction)
        # Instead: verify a SELL signal still goes through during halt
        sell_signal = Signal(SignalType.SELL, 50_000.0, now + pd.Timedelta(minutes=2), "strategy sell")
        result = rm.process(sell_signal, 50_000.0)
        # SELL on open position during news halt is allowed (closes risk)
        assert result is not None

    def test_both_positive_and_negative_high_impact_triggers_halt(self):
        for score in [0.9, -0.9]:
            rm, ex = _make_rm()
            now = pd.Timestamp.now(tz="UTC")
            rm.notify_news_event(score, now)
            result = rm.process(_buy(now + pd.Timedelta(minutes=1)), 50_000.0)
            assert result is None, f"Expected halt for score={score}"

    def test_news_halt_independent_of_news_enabled_flag(self):
        """The halt fires from notify_news_event() regardless of config[news][enabled]."""
        config_news_disabled = {
            "risk": _CONFIG["risk"],
            # No [news] section at all — uses defaults
        }
        ex = PaperExecutor(10_000.0, fee_rate=0.001)
        rm = RiskManager(ex, config_news_disabled, SYMBOL)

        now = pd.Timestamp.now(tz="UTC")
        rm.notify_news_event(-0.9, now)

        # halt_minutes defaults to 5
        result = rm.process(_buy(now + pd.Timedelta(minutes=2)), 50_000.0)
        assert result is None
