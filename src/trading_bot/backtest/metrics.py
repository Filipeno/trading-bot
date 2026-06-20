import numpy as np
import pandas as pd


_BARS_PER_YEAR = {
    "1m": 525_600, "3m": 175_200, "5m": 105_120,
    "15m": 35_040, "30m": 17_520, "1h": 8_760,
    "4h": 2_190,   "1d": 365,
}


def bars_per_year(timeframe: str) -> int:
    return _BARS_PER_YEAR.get(timeframe.lower(), 8_760)


def compute_metrics(
    equity: pd.Series,
    trades: list,
    initial_capital: float,
    timeframe: str = "1h",
) -> dict:
    total_return = (equity.iloc[-1] - initial_capital) / initial_capital

    # Annualised Sharpe — annualisation factor depends on bar size
    returns = equity.pct_change().dropna()
    bpy = bars_per_year(timeframe)
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(bpy)
    else:
        sharpe = 0.0

    rolling_max = equity.cummax()
    drawdowns = (equity - rolling_max) / rolling_max
    max_drawdown = float(drawdowns.min())

    n_trades = len(trades)
    wins = [t for t in trades if t.pnl > 0]
    win_rate = len(wins) / n_trades if n_trades > 0 else 0.0

    return {
        "total_return_pct": round(total_return * 100, 2),
        "sharpe_ratio": round(float(sharpe), 3),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "win_rate_pct": round(win_rate * 100, 2),
        "n_trades": n_trades,
        "final_equity": round(float(equity.iloc[-1]), 2),
    }
