"""Tests for walk-forward analysis and parameter optimization."""

import numpy as np
import pandas as pd
import pytest

from trading_bot.backtest.optimize import (
    STRATEGY_REGISTRY,
    buy_hold_return_pct,
    build,
    evaluate,
    grid_search,
    iter_param_combos,
    multi_window,
    valid_combos,
    walk_forward,
)
from trading_bot.strategies.macd import MACDStrategy


def _synthetic(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    """Random-walk price series — enough bars for multi-fold tests."""
    rng = np.random.default_rng(seed)
    steps = rng.standard_normal(n) * 0.5
    prices = 100.0 + np.cumsum(steps)
    prices = np.clip(prices, 1.0, None)   # never non-positive
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"close": prices}, index=idx)


# ── Param combination helpers ───────────────────────────────────────────────

class TestParamCombos:
    def test_iter_param_combos_count(self):
        grid = {"a": [1, 2], "b": [3, 4, 5]}
        combos = list(iter_param_combos(grid))
        assert len(combos) == 6
        assert {"a": 1, "b": 3} in combos

    def test_valid_combos_respects_ema_constraint(self):
        combos = valid_combos("ema_crossover")
        # Every combo must have fast < slow
        assert all(c["fast"] < c["slow"] for c in combos)
        assert len(combos) > 0

    def test_valid_combos_macd_constraint(self):
        combos = valid_combos("macd")
        assert all(c["fast"] < c["slow"] for c in combos)

    def test_build_returns_correct_type(self):
        strat = build("macd", {"fast": 6, "slow": 13, "signal": 4})
        assert isinstance(strat, MACDStrategy)
        assert strat.fast == 6

    def test_all_registry_strategies_buildable(self):
        for name in STRATEGY_REGISTRY:
            combos = valid_combos(name)
            strat = build(name, combos[0])
            assert strat is not None


# ── buy & hold helper ───────────────────────────────────────────────────────

class TestBuyHold:
    def test_positive_when_price_rises(self):
        df = pd.DataFrame(
            {"close": [100.0, 110.0]},
            index=pd.date_range("2024-01-01", periods=2, freq="15min", tz="UTC"),
        )
        assert buy_hold_return_pct(df) == pytest.approx(10.0)

    def test_negative_when_price_falls(self):
        df = pd.DataFrame(
            {"close": [100.0, 90.0]},
            index=pd.date_range("2024-01-01", periods=2, freq="15min", tz="UTC"),
        )
        assert buy_hold_return_pct(df) == pytest.approx(-10.0)

    def test_zero_on_single_bar(self):
        df = pd.DataFrame(
            {"close": [100.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="15min", tz="UTC"),
        )
        assert buy_hold_return_pct(df) == 0.0


# ── evaluate / grid search ───────────────────────────────────────────────────

class TestGridSearch:
    def test_evaluate_returns_metrics(self):
        df = _synthetic(300)
        m = evaluate(df, "breakout", {"period": 20})
        assert "total_return_pct" in m
        assert "n_trades" in m

    def test_grid_search_ranks_descending(self):
        df = _synthetic(400)
        ranked = grid_search(df, "breakout", metric="total_return_pct")
        returns = [g.metrics["total_return_pct"] for g in ranked]
        assert returns == sorted(returns, reverse=True)

    def test_grid_search_covers_all_valid_combos(self):
        df = _synthetic(400)
        ranked = grid_search(df, "bollinger_bands")
        assert len(ranked) == len(valid_combos("bollinger_bands"))

    def test_grid_search_best_is_first(self):
        df = _synthetic(400)
        ranked = grid_search(df, "ema_crossover")
        best = ranked[0].metrics["total_return_pct"]
        assert all(best >= g.metrics["total_return_pct"] for g in ranked)


# ── walk-forward ─────────────────────────────────────────────────────────────

class TestWalkForward:
    def test_produces_requested_folds(self):
        df = _synthetic(1200)
        report = walk_forward(df, "breakout", n_folds=5)
        assert len(report.folds) == 5

    def test_train_and_test_are_disjoint_in_time(self):
        df = _synthetic(1200)
        report = walk_forward(df, "breakout", n_folds=4)
        for f in report.folds:
            # test starts after train ends
            assert f.test_span[0] >= f.train_span[1]

    def test_test_windows_are_sequential(self):
        df = _synthetic(1200)
        report = walk_forward(df, "macd", n_folds=4)
        starts = [f.test_span[0] for f in report.folds]
        assert starts == sorted(starts)

    def test_oos_aggregates_are_computed(self):
        df = _synthetic(1200)
        report = walk_forward(df, "ema_crossover", n_folds=5)
        # These should be real numbers, derived from out-of-sample folds
        assert isinstance(report.oos_mean_return, float)
        assert 0.0 <= report.pct_profitable_folds <= 100.0
        assert 0.0 <= report.pct_beat_buy_hold <= 100.0
        assert report.worst_fold_return <= report.best_fold_return

    def test_raises_when_insufficient_data(self):
        df = _synthetic(60)
        with pytest.raises(ValueError):
            walk_forward(df, "macd", n_folds=10)

    def test_best_params_come_from_registry_grid(self):
        df = _synthetic(1200)
        report = walk_forward(df, "bollinger_bands", n_folds=4)
        valid = valid_combos("bollinger_bands")
        for f in report.folds:
            assert f.best_params in valid

    def test_compounded_return_consistent_with_folds(self):
        df = _synthetic(1200)
        report = walk_forward(df, "breakout", n_folds=5)
        # Manually compound and compare
        eq = 1.0
        for r in report.oos_returns:
            eq *= (1 + r / 100)
        assert report.oos_total_compounded_pct == pytest.approx((eq - 1) * 100, rel=1e-6)


# ── multi-window (fixed params) ──────────────────────────────────────────────

class TestMultiWindow:
    def test_produces_requested_windows(self):
        df = _synthetic(600)
        results = multi_window(df, "breakout", {"period": 20}, n_windows=6)
        assert len(results) == 6

    def test_windows_have_buy_hold_baseline(self):
        df = _synthetic(600)
        results = multi_window(df, "ema_crossover", {"fast": 9, "slow": 21}, n_windows=5)
        for w in results:
            assert isinstance(w.buy_hold_pct, float)
            assert "total_return_pct" in w.metrics
