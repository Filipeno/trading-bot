"""Tests for the dollar-cost-averaging simulation."""

import numpy as np
import pandas as pd
import pytest

from trading_bot.backtest.dca import simulate_dca


def _df(prices: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="15min", tz="UTC")
    return pd.DataFrame({"close": prices}, index=idx)


class TestDCA:
    def test_buys_at_each_interval(self):
        df = _df([100.0] * 10)
        res = simulate_dca(df, contribution=50.0, interval_bars=2, fee_rate=0.0)
        # bars 0,2,4,6,8 → 5 buys
        assert res.n_buys == 5
        assert res.total_invested == pytest.approx(250.0)

    def test_flat_price_no_fee_breaks_even(self):
        df = _df([100.0] * 20)
        res = simulate_dca(df, contribution=10.0, interval_bars=5, fee_rate=0.0)
        assert res.return_pct == pytest.approx(0.0, abs=1e-9)
        assert res.final_value == pytest.approx(res.total_invested)

    def test_rising_price_is_profitable(self):
        prices = [100.0 + i for i in range(40)]
        res = simulate_dca(_df(prices), contribution=10.0, interval_bars=5, fee_rate=0.0)
        assert res.return_pct > 0
        assert res.final_value > res.total_invested

    def test_falling_price_loses(self):
        prices = [140.0 - i for i in range(40)]
        res = simulate_dca(_df(prices), contribution=10.0, interval_bars=5, fee_rate=0.0)
        assert res.return_pct < 0

    def test_fees_reduce_units(self):
        df = _df([100.0] * 10)
        no_fee = simulate_dca(df, contribution=50.0, interval_bars=2, fee_rate=0.0)
        with_fee = simulate_dca(df, contribution=50.0, interval_bars=2, fee_rate=0.01)
        assert with_fee.units_held < no_fee.units_held

    def test_avg_cost_between_min_and_max_price(self):
        prices = [80.0, 90.0, 100.0, 110.0, 120.0] * 4
        res = simulate_dca(_df(prices), contribution=10.0, interval_bars=3, fee_rate=0.0)
        assert min(prices) <= res.avg_cost <= max(prices)

    def test_curves_have_full_length(self):
        df = _df([100.0] * 30)
        res = simulate_dca(df, contribution=10.0, interval_bars=5, fee_rate=0.0)
        assert len(res.equity_curve) == len(df)
        assert len(res.invested_curve) == len(df)

    def test_invested_curve_is_non_decreasing(self):
        df = _df([100.0] * 30)
        res = simulate_dca(df, contribution=10.0, interval_bars=5, fee_rate=0.0)
        diffs = res.invested_curve.diff().dropna()
        assert (diffs >= 0).all()

    def test_dca_reduces_timing_risk_vs_lump_sum(self):
        # Price dips then recovers — DCA should capture the cheap middle.
        prices = [100.0] * 5 + [50.0] * 5 + [100.0] * 5
        res = simulate_dca(_df(prices), contribution=10.0, interval_bars=1, fee_rate=0.0)
        # Bought through the dip → avg cost below the 100 start/end price
        assert res.avg_cost < 100.0
        assert res.return_pct > 0   # ends at 100 with avg cost < 100
