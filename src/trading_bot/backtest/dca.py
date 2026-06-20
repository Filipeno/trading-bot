"""Dollar-cost averaging (DCA) simulation.

DCA is not a trading "strategy" in the signal sense — it makes no predictions.
You buy a fixed amount of money's worth of the asset on a fixed schedule and
never sell. It is the approach with the most real-world evidence behind it for
small accounts that add money over time (e.g. from a salary), because it removes
timing risk entirely.

In this project's walk-forward tests, passive approaches consistently beat the
active technical strategies. DCA is included so you can compare honestly against
the thing that actually tends to work.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DCAResult:
    total_invested: float
    final_value: float
    return_pct: float
    n_buys: int
    units_held: float
    avg_cost: float
    equity_curve: pd.Series        # portfolio value over time
    invested_curve: pd.Series      # cumulative cash put in over time


def simulate_dca(
    df: pd.DataFrame,
    contribution: float = 50.0,
    interval_bars: int = 96,       # e.g. once per day on 15m bars (96 × 15m = 24h)
    fee_rate: float = 0.001,
) -> DCAResult:
    """Buy `contribution` worth of the asset every `interval_bars` bars; never sell.

    Returns the portfolio value over time vs the cash invested, so you can see
    both the absolute growth and the return on what you actually put in.
    """
    units = 0.0
    invested = 0.0
    n_buys = 0
    equity_points: list[float] = []
    invested_points: list[float] = []

    for i in range(len(df)):
        price = float(df["close"].iloc[i])

        if i % interval_bars == 0:
            fee = contribution * fee_rate
            bought = (contribution - fee) / price
            units += bought
            invested += contribution
            n_buys += 1

        equity_points.append(units * price)
        invested_points.append(invested)

    final_price = float(df["close"].iloc[-1])
    final_value = units * final_price
    return_pct = ((final_value - invested) / invested * 100.0) if invested > 0 else 0.0
    avg_cost = (invested / units) if units > 0 else 0.0

    return DCAResult(
        total_invested=invested,
        final_value=final_value,
        return_pct=return_pct,
        n_buys=n_buys,
        units_held=units,
        avg_cost=avg_cost,
        equity_curve=pd.Series(equity_points, index=df.index),
        invested_curve=pd.Series(invested_points, index=df.index),
    )
