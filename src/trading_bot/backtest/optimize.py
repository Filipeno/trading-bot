"""Walk-forward analysis and parameter optimization.

The single most important idea in this module is **out-of-sample validation**:
we only ever trust a result measured on data the parameters were NOT tuned on.

Why this matters:
    If you grid-search 200 parameter combinations on one slice of history and
    pick the best, you WILL find something that looks amazing — purely by luck.
    That is "curve fitting" (a.k.a. overfitting). It almost never survives
    contact with new data. Walk-forward optimization is the standard defense:
    tune on one window, then measure on the *next* window the tuner never saw.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from ..strategies.bollinger_bands import BollingerBandsStrategy
from ..strategies.breakout import BreakoutStrategy
from ..strategies.ema_crossover import EMACrossover
from ..strategies.macd import MACDStrategy
from ..strategies.rsi import RSIStrategy
from ..strategies.stochastic import StochasticStrategy
from ..strategies.supertrend import SupertrendStrategy
from ..strategies.vwap import VWAPStrategy
from .engine import BacktestEngine

# ── Strategy registry: name → (class, parameter grid) ──────────────────────
# Param keys MUST match each strategy's __init__ kwargs.
STRATEGY_REGISTRY: dict[str, tuple[type, dict[str, list]]] = {
    "ema_crossover": (
        EMACrossover,
        {"fast": [5, 9, 12, 20], "slow": [21, 30, 50, 100]},
    ),
    "macd": (
        MACDStrategy,
        {"fast": [6, 12], "slow": [13, 26], "signal": [4, 9]},
    ),
    "bollinger_bands": (
        BollingerBandsStrategy,
        {"period": [10, 20, 30], "std_dev": [1.5, 2.0, 2.5]},
    ),
    "breakout": (
        BreakoutStrategy,
        {"period": [10, 20, 30, 50]},
    ),
    "rsi": (
        RSIStrategy,
        {"period": [7, 14, 21], "oversold": [25, 30, 35], "overbought": [65, 70, 75]},
    ),
    "stochastic": (
        StochasticStrategy,
        {"k_period": [9, 14], "d_period": [3], "oversold": [20, 25], "overbought": [75, 80]},
    ),
    "vwap": (
        VWAPStrategy,
        {"period": [10, 20, 50]},
    ),
    "supertrend": (
        SupertrendStrategy,
        {"period": [7, 10, 14], "multiplier": [2.0, 3.0]},
    ),
}

# Per-strategy validity constraint (e.g. fast must be < slow)
_CONSTRAINTS = {
    "ema_crossover": lambda p: p["fast"] < p["slow"],
    "macd": lambda p: p["fast"] < p["slow"],
    "rsi": lambda p: p["oversold"] < p["overbought"],
    "stochastic": lambda p: p["oversold"] < p["overbought"],
}


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def iter_param_combos(grid: dict[str, list]) -> Iterator[dict]:
    """Yield every combination of the grid as a dict of kwargs."""
    keys = list(grid.keys())
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def valid_combos(name: str) -> list[dict]:
    _, grid = STRATEGY_REGISTRY[name]
    constraint = _CONSTRAINTS.get(name, lambda p: True)
    return [p for p in iter_param_combos(grid) if constraint(p)]


def build(name: str, params: dict):
    cls, _ = STRATEGY_REGISTRY[name]
    return cls(**params)


def buy_hold_return_pct(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.0
    return (df["close"].iloc[-1] / df["close"].iloc[0] - 1.0) * 100.0


def evaluate(
    df: pd.DataFrame,
    name: str,
    params: dict,
    *,
    fee_rate: float = 0.001,
    slippage_pct: float = 0.0005,
    leverage: int = 1,
    timeframe: str = "1h",
    capital: float = 10_000.0,
) -> dict:
    """Run one strategy+params over df and return its metrics dict."""
    engine = BacktestEngine(
        initial_capital=capital,
        fee_rate=fee_rate,
        slippage_pct=slippage_pct,
        leverage=leverage,
        timeframe=timeframe,
    )
    result = engine.run(df, build(name, params))
    return result.metrics


# ──────────────────────────────────────────────────────────────────────────
# Grid search (single dataset — IN-SAMPLE only, use with caution)
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class GridResult:
    params: dict
    metrics: dict


def grid_search(
    df: pd.DataFrame,
    name: str,
    *,
    metric: str = "total_return_pct",
    **engine_kwargs,
) -> list[GridResult]:
    """Rank every parameter combo on a single dataset by `metric` (descending).

    WARNING: results here are IN-SAMPLE. The top combo is the one that best fit
    THIS data — which is exactly what overfitting looks like. Always confirm with
    walk_forward() before believing any of these numbers.
    """
    out: list[GridResult] = []
    for params in valid_combos(name):
        m = evaluate(df, name, params, **engine_kwargs)
        out.append(GridResult(params=params, metrics=m))
    out.sort(key=lambda g: g.metrics.get(metric, float("-inf")), reverse=True)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Walk-forward optimization (the honest one)
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    fold: int
    train_span: tuple[pd.Timestamp, pd.Timestamp]
    test_span: tuple[pd.Timestamp, pd.Timestamp]
    best_params: dict
    in_sample_metric: float        # best metric on the training fold
    out_of_sample: dict            # full metrics on the (unseen) test fold
    buy_hold_test_pct: float       # buy & hold over the test fold


@dataclass
class WalkForwardReport:
    name: str
    metric: str
    folds: list[FoldResult] = field(default_factory=list)

    # ── Aggregate, OUT-OF-SAMPLE only ──────────────────────────────────────
    @property
    def oos_returns(self) -> list[float]:
        return [f.out_of_sample["total_return_pct"] for f in self.folds]

    @property
    def oos_mean_return(self) -> float:
        return float(np.mean(self.oos_returns)) if self.folds else 0.0

    @property
    def oos_median_return(self) -> float:
        return float(np.median(self.oos_returns)) if self.folds else 0.0

    @property
    def pct_profitable_folds(self) -> float:
        if not self.folds:
            return 0.0
        wins = sum(1 for r in self.oos_returns if r > 0)
        return wins / len(self.folds) * 100.0

    @property
    def pct_beat_buy_hold(self) -> float:
        if not self.folds:
            return 0.0
        wins = sum(1 for f in self.folds
                   if f.out_of_sample["total_return_pct"] > f.buy_hold_test_pct)
        return wins / len(self.folds) * 100.0

    @property
    def worst_fold_return(self) -> float:
        return min(self.oos_returns) if self.folds else 0.0

    @property
    def best_fold_return(self) -> float:
        return max(self.oos_returns) if self.folds else 0.0

    @property
    def oos_total_compounded_pct(self) -> float:
        """Compounded return if you rolled OOS fold to OOS fold."""
        equity = 1.0
        for r in self.oos_returns:
            equity *= (1.0 + r / 100.0)
        return (equity - 1.0) * 100.0


def walk_forward(
    df: pd.DataFrame,
    name: str,
    *,
    n_folds: int = 5,
    metric: str = "total_return_pct",
    **engine_kwargs,
) -> WalkForwardReport:
    """Walk-forward optimization.

    The data is cut into `n_folds + 1` consecutive blocks. For each step we
    optimize parameters on block i (in-sample) and then measure those exact
    parameters on block i+1 (out-of-sample — the optimizer never saw it).

    The aggregate OUT-OF-SAMPLE numbers on the returned report are the only
    honest estimate of how the strategy might do on data it hasn't seen.
    """
    report = WalkForwardReport(name=name, metric=metric)

    n_blocks = n_folds + 1
    if len(df) < n_blocks * 30:   # need a reasonable amount of data per block
        raise ValueError(
            f"Not enough data for {n_folds} folds — need ~{n_blocks * 30} bars, "
            f"have {len(df)}. Increase --limit or reduce --folds."
        )

    bounds = np.linspace(0, len(df), n_blocks + 1, dtype=int)

    for i in range(n_folds):
        train = df.iloc[bounds[i]:bounds[i + 1]]
        test = df.iloc[bounds[i + 1]:bounds[i + 2]]
        if len(train) < 30 or len(test) < 30:
            continue

        # Optimize on the training block
        ranked = grid_search(train, name, metric=metric, **engine_kwargs)
        best = ranked[0]

        # Apply the winning params to the unseen test block
        oos = evaluate(test, name, best.params, **engine_kwargs)

        report.folds.append(FoldResult(
            fold=i + 1,
            train_span=(train.index[0], train.index[-1]),
            test_span=(test.index[0], test.index[-1]),
            best_params=best.params,
            in_sample_metric=best.metrics.get(metric, 0.0),
            out_of_sample=oos,
            buy_hold_test_pct=buy_hold_return_pct(test),
        ))

    return report


# ──────────────────────────────────────────────────────────────────────────
# Fixed-parameter multi-window analysis (consistency check, no tuning)
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class WindowResult:
    window: int
    span: tuple[pd.Timestamp, pd.Timestamp]
    metrics: dict
    buy_hold_pct: float


def multi_window(
    df: pd.DataFrame,
    name: str,
    params: dict,
    *,
    n_windows: int = 6,
    **engine_kwargs,
) -> list[WindowResult]:
    """Run ONE fixed parameter set across N consecutive windows.

    Shows whether a strategy is consistent or just got lucky once.
    No optimization here — params are held fixed.
    """
    bounds = np.linspace(0, len(df), n_windows + 1, dtype=int)
    out: list[WindowResult] = []
    for w in range(n_windows):
        chunk = df.iloc[bounds[w]:bounds[w + 1]]
        if len(chunk) < 30:
            continue
        m = evaluate(chunk, name, params, **engine_kwargs)
        out.append(WindowResult(
            window=w + 1,
            span=(chunk.index[0], chunk.index[-1]),
            metrics=m,
            buy_hold_pct=buy_hold_return_pct(chunk),
        ))
    return out
