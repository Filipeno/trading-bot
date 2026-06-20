"""Tests for RSI, Stochastic, VWAP, Supertrend strategies + their indicators."""

import numpy as np
import pandas as pd
import pytest

from trading_bot.strategies.base import SignalType
from trading_bot.strategies.factory import make_strategy, strategy_names
from trading_bot.strategies.indicators import (
    atr,
    rolling_vwap,
    rsi,
    stochastic,
    supertrend,
)
from trading_bot.strategies.rsi import RSIStrategy
from trading_bot.strategies.stochastic import StochasticStrategy
from trading_bot.strategies.supertrend import SupertrendStrategy
from trading_bot.strategies.vwap import VWAPStrategy


def _ohlc(closes: list[float], vol: float = 10.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="15min", tz="UTC")
    c = pd.Series(closes)
    return pd.DataFrame({
        "open": c.shift(1).fillna(c).to_numpy(),
        "high": (c + 0.5).to_numpy(),
        "low": (c - 0.5).to_numpy(),
        "close": c.to_numpy(),
        "volume": vol,
    }, index=idx)


def _signals(strat, df):
    return [strat.next(df.iloc[:i + 1]).type for i in range(len(df))]


# ── Indicators ───────────────────────────────────────────────────────────────

class TestIndicators:
    def test_rsi_bounded(self):
        df = _ohlc([100 + np.sin(i / 3) * 5 for i in range(80)])
        r = rsi(df["close"], 14)
        assert r.between(0, 100).all()

    def test_rsi_high_on_strong_uptrend(self):
        df = _ohlc([100 + i for i in range(40)])
        assert rsi(df["close"], 14).iloc[-1] > 70

    def test_rsi_low_on_strong_downtrend(self):
        df = _ohlc([140 - i for i in range(40)])
        assert rsi(df["close"], 14).iloc[-1] < 30

    def test_atr_positive(self):
        df = _ohlc([100 + np.sin(i) * 3 for i in range(50)])
        assert (atr(df, 14).iloc[20:] > 0).all()

    def test_stochastic_bounded(self):
        df = _ohlc([100 + np.sin(i / 2) * 5 for i in range(60)])
        k, d = stochastic(df, 14, 3)
        assert k.between(0, 100).all()

    def test_vwap_within_price_range(self):
        df = _ohlc([100 + np.sin(i / 4) * 5 for i in range(60)])
        v = rolling_vwap(df, 20).dropna()
        assert (v > 50).all() and (v < 150).all()

    def test_supertrend_direction_values(self):
        df = _ohlc([100 + i for i in range(60)])
        d = supertrend(df, 10, 3.0)
        assert set(d.unique()).issubset({-1, 1})

    def test_supertrend_up_in_uptrend(self):
        df = _ohlc([100 + i * 2 for i in range(60)])
        assert supertrend(df, 10, 3.0).iloc[-1] == 1


# ── RSI strategy ─────────────────────────────────────────────────────────────

class TestRSIStrategy:
    def test_hold_insufficient_data(self):
        assert RSIStrategy().next(_ohlc([100] * 5)).type == SignalType.HOLD

    def test_buy_when_leaving_oversold(self):
        # Sharp drop (RSI < 30) then a bounce → BUY crossing up out of oversold
        prices = [100 - i * 2 for i in range(20)] + [62 + i * 2 for i in range(20)]
        sigs = _signals(RSIStrategy(period=14, oversold=30, overbought=70), _ohlc(prices))
        assert SignalType.BUY in sigs

    def test_sell_when_leaving_overbought(self):
        prices = [100 + i * 2 for i in range(20)] + [138 - i * 2 for i in range(20)]
        sigs = _signals(RSIStrategy(period=14, oversold=30, overbought=70), _ohlc(prices))
        assert SignalType.SELL in sigs


# ── Stochastic strategy ──────────────────────────────────────────────────────

class TestStochasticStrategy:
    def test_hold_insufficient_data(self):
        assert StochasticStrategy().next(_ohlc([100] * 5)).type == SignalType.HOLD

    def test_produces_signals_on_oscillation(self):
        prices = [100 + np.sin(i / 3) * 10 for i in range(120)]
        sigs = _signals(StochasticStrategy(), _ohlc(prices))
        assert SignalType.BUY in sigs or SignalType.SELL in sigs

    def test_reset_noop(self):
        df = _ohlc([100 + np.sin(i / 3) * 10 for i in range(60)])
        s = StochasticStrategy()
        a = s.next(df); s.reset(); b = s.next(df)
        assert a.type == b.type


# ── VWAP strategy ────────────────────────────────────────────────────────────

class TestVWAPStrategy:
    def test_hold_insufficient_data(self):
        assert VWAPStrategy(period=20).next(_ohlc([100] * 5)).type == SignalType.HOLD

    def test_buy_on_cross_above(self):
        # Below average then pushing above it
        prices = [100] * 25 + [108]
        sigs = _signals(VWAPStrategy(period=20), _ohlc(prices))
        assert SignalType.BUY in sigs

    def test_produces_signals_on_oscillation(self):
        prices = [100 + np.sin(i / 5) * 6 for i in range(100)]
        sigs = _signals(VWAPStrategy(period=20), _ohlc(prices))
        assert SignalType.BUY in sigs and SignalType.SELL in sigs


# ── Supertrend strategy ──────────────────────────────────────────────────────

class TestSupertrendStrategy:
    def test_hold_insufficient_data(self):
        assert SupertrendStrategy().next(_ohlc([100] * 5)).type == SignalType.HOLD

    def test_buy_then_sell_on_reversal(self):
        prices = [100 - i for i in range(30)] + [70 + i * 2 for i in range(30)] + [130 - i * 2 for i in range(30)]
        sigs = _signals(SupertrendStrategy(period=10, multiplier=2.0), _ohlc(prices))
        assert SignalType.BUY in sigs
        assert SignalType.SELL in sigs


# ── Factory wiring ───────────────────────────────────────────────────────────

class TestFactoryNewStrategies:
    def test_all_new_names_registered(self):
        for name in ("rsi", "stochastic", "vwap", "supertrend"):
            assert name in strategy_names()

    def test_make_rsi(self):
        s = make_strategy({"strategy": {"name": "rsi", "rsi_period": 7}})
        assert isinstance(s, RSIStrategy) and s.period == 7

    def test_make_stochastic(self):
        s = make_strategy({"strategy": {"name": "stochastic", "stoch_k_period": 9}})
        assert isinstance(s, StochasticStrategy) and s.k_period == 9

    def test_make_vwap(self):
        s = make_strategy({"strategy": {"name": "vwap", "vwap_period": 50}})
        assert isinstance(s, VWAPStrategy) and s.period == 50

    def test_make_supertrend(self):
        s = make_strategy({"strategy": {"name": "supertrend", "supertrend_multiplier": 2.0}})
        assert isinstance(s, SupertrendStrategy) and s.multiplier == 2.0
