"""Tests for MACD, Bollinger Bands, and Breakout strategies."""

import numpy as np
import pandas as pd
import pytest

from trading_bot.strategies.base import SignalType
from trading_bot.strategies.bollinger_bands import BollingerBandsStrategy
from trading_bot.strategies.breakout import BreakoutStrategy
from trading_bot.strategies.factory import make_strategy, strategy_names
from trading_bot.strategies.macd import MACDStrategy


# ── Helpers ────────────────────────────────────────────────────────────────

def _df(prices: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="15min", tz="UTC")
    return pd.DataFrame({"close": prices}, index=idx)


def _trending_up(n: int = 80, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    prices = [start + i * step for i in range(n)]
    return _df(prices)


def _trending_down(n: int = 80, start: float = 180.0, step: float = 1.0) -> pd.DataFrame:
    prices = [start - i * step for i in range(n)]
    return _df(prices)


def _sideways(n: int = 50, base: float = 100.0, noise: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    prices = base + noise * rng.standard_normal(n)
    return _df(prices.tolist())


# ══════════════════════════════════════════════════════════════════════════════
# MACD Strategy
# ══════════════════════════════════════════════════════════════════════════════

class TestMACDStrategy:
    def test_hold_when_insufficient_data(self):
        strat = MACDStrategy(fast=12, slow=26, signal=9)
        df = _df([100.0] * 30)   # need 26+9+1=36 bars
        sig = strat.next(df)
        assert sig.type == SignalType.HOLD

    def test_hold_on_flat_prices(self):
        strat = MACDStrategy()
        df = _df([100.0] * 50)
        sig = strat.next(df)
        assert sig.type == SignalType.HOLD

    def test_buy_signal_after_downtrend_reverses_to_uptrend(self):
        # V-shape: MACD histogram goes negative during fall, then crosses zero on rise
        strat = MACDStrategy(fast=6, slow=13, signal=4)
        prices = [100 - i * 1.5 for i in range(40)] + [40 + i * 1.5 for i in range(40)]
        df = _df(prices)
        signals = [strat.next(df.iloc[:i+1]).type for i in range(len(df))]
        assert SignalType.BUY in signals

    def test_sell_signal_after_uptrend_reverses_to_downtrend(self):
        # Inverted V: MACD histogram goes positive during rise, then crosses zero on fall
        strat = MACDStrategy(fast=6, slow=13, signal=4)
        prices = [60 + i * 1.5 for i in range(40)] + [120 - i * 1.5 for i in range(40)]
        df = _df(prices)
        signals = [strat.next(df.iloc[:i+1]).type for i in range(len(df))]
        assert SignalType.SELL in signals

    def test_reset_is_noop(self):
        strat = MACDStrategy()
        df = _trending_up(n=50)
        sig_before = strat.next(df)
        strat.reset()
        sig_after = strat.next(df)
        # Stateless strategy — same output
        assert sig_before.type == sig_after.type

    def test_reason_contains_histogram_value(self):
        strat = MACDStrategy()
        df = _df([100.0] * 50)
        sig = strat.next(df)
        assert "h=" in sig.reason

    def test_macd_buy_then_sell_on_full_cycle(self):
        # Full W-M cycle: down → up → down produces both a BUY and a SELL cross
        strat = MACDStrategy(fast=6, slow=13, signal=4)
        prices = (
            [100 - i for i in range(30)]   # falling → MACD goes negative
            + [70 + i * 2 for i in range(35)]  # rising fast → BUY cross
            + [140 - i for i in range(35)]  # falling → SELL cross
        )
        df = _df(prices)
        signals = [strat.next(df.iloc[:i+1]).type for i in range(len(df))]
        assert SignalType.BUY in signals
        assert SignalType.SELL in signals


# ══════════════════════════════════════════════════════════════════════════════
# Bollinger Bands Strategy
# ══════════════════════════════════════════════════════════════════════════════

class TestBollingerBandsStrategy:
    def test_hold_when_insufficient_data(self):
        strat = BollingerBandsStrategy(period=20)
        df = _df([100.0] * 15)
        assert strat.next(df).type == SignalType.HOLD

    def test_hold_on_flat_prices(self):
        # Flat prices → std = 0, bands = middle → price not outside bands
        strat = BollingerBandsStrategy(period=20)
        df = _df([100.0] * 25)
        assert strat.next(df).type == SignalType.HOLD

    def test_buy_when_price_spikes_below_lower_band(self):
        strat = BollingerBandsStrategy(period=10, std_dev=2.0)
        # Stable prices, then a big drop
        prices = [100.0] * 15 + [70.0]   # 70 is far below the lower band
        df = _df(prices)
        sig = strat.next(df)
        assert sig.type == SignalType.BUY

    def test_sell_when_price_spikes_above_upper_band(self):
        strat = BollingerBandsStrategy(period=10, std_dev=2.0)
        prices = [100.0] * 15 + [130.0]  # 130 is far above the upper band
        df = _df(prices)
        sig = strat.next(df)
        assert sig.type == SignalType.SELL

    def test_reason_contains_band_values(self):
        strat = BollingerBandsStrategy(period=10, std_dev=2.0)
        prices = [100.0] * 15 + [70.0]
        df = _df(prices)
        sig = strat.next(df)
        assert "band" in sig.reason.lower()

    def test_reset_is_noop(self):
        strat = BollingerBandsStrategy(period=10)
        prices = [100.0] * 15 + [70.0]
        df = _df(prices)
        sig1 = strat.next(df)
        strat.reset()
        sig2 = strat.next(df)
        assert sig1.type == sig2.type


# ══════════════════════════════════════════════════════════════════════════════
# Breakout Strategy
# ══════════════════════════════════════════════════════════════════════════════

class TestBreakoutStrategy:
    def test_hold_when_insufficient_data(self):
        strat = BreakoutStrategy(period=20)
        df = _df([100.0] * 15)
        assert strat.next(df).type == SignalType.HOLD

    def test_buy_on_upward_breakout(self):
        strat = BreakoutStrategy(period=5)
        # 5 bars at 100, then a breakout
        prices = [100.0] * 6 + [101.0]
        df = _df(prices)
        sig = strat.next(df)
        assert sig.type == SignalType.BUY

    def test_sell_on_downward_breakdown(self):
        strat = BreakoutStrategy(period=5)
        prices = [100.0] * 6 + [99.0]
        df = _df(prices)
        sig = strat.next(df)
        assert sig.type == SignalType.SELL

    def test_hold_inside_channel(self):
        strat = BreakoutStrategy(period=5)
        # Price stays within the channel
        prices = [99.0, 100.0, 101.0, 100.0, 99.0, 100.0]
        df = _df(prices)
        sig = strat.next(df)
        assert sig.type == SignalType.HOLD

    def test_buy_signal_appears_in_uptrend(self):
        strat = BreakoutStrategy(period=10)
        df = _trending_up(n=60)
        signals = [strat.next(df.iloc[:i+1]).type for i in range(len(df))]
        assert SignalType.BUY in signals

    def test_sell_signal_appears_in_downtrend(self):
        strat = BreakoutStrategy(period=10)
        df = _trending_down(n=60)
        signals = [strat.next(df.iloc[:i+1]).type for i in range(len(df))]
        assert SignalType.SELL in signals

    def test_reason_contains_channel_values(self):
        strat = BreakoutStrategy(period=5)
        prices = [100.0] * 6 + [101.0]
        df = _df(prices)
        sig = strat.next(df)
        assert any(kw in sig.reason for kw in ("breakout", "breakdown", "channel", "high", "low"))

    def test_reset_is_noop(self):
        strat = BreakoutStrategy(period=5)
        prices = [100.0] * 6 + [101.0]
        df = _df(prices)
        sig1 = strat.next(df)
        strat.reset()
        sig2 = strat.next(df)
        assert sig1.type == sig2.type


# ══════════════════════════════════════════════════════════════════════════════
# Strategy Factory
# ══════════════════════════════════════════════════════════════════════════════

class TestStrategyFactory:
    def test_ema_crossover_is_default(self):
        from trading_bot.strategies.ema_crossover import EMACrossover
        strat = make_strategy({"strategy": {}})
        assert isinstance(strat, EMACrossover)

    def test_make_macd(self):
        strat = make_strategy({"strategy": {"name": "macd", "macd_fast": 6, "macd_slow": 13, "macd_signal": 4}})
        assert isinstance(strat, MACDStrategy)
        assert strat.fast == 6

    def test_make_bollinger_bands(self):
        strat = make_strategy({"strategy": {"name": "bollinger_bands", "bb_period": 15}})
        assert isinstance(strat, BollingerBandsStrategy)
        assert strat.period == 15

    def test_make_breakout(self):
        strat = make_strategy({"strategy": {"name": "breakout", "breakout_period": 30}})
        assert isinstance(strat, BreakoutStrategy)
        assert strat.period == 30

    def test_unknown_name_falls_back_to_ema(self):
        from trading_bot.strategies.ema_crossover import EMACrossover
        strat = make_strategy({"strategy": {"name": "xyzzy"}})
        assert isinstance(strat, EMACrossover)

    def test_strategy_names_returns_all(self):
        names = strategy_names()
        assert "ema_crossover" in names
        assert "macd" in names
        assert "bollinger_bands" in names
        assert "breakout" in names
