from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from ..strategies.base import Signal, SignalType, Strategy
from .metrics import compute_metrics


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    size: float
    side: str
    pnl: float
    pnl_pct: float
    exit_reason: str = "signal"


@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_curve: pd.Series
    metrics: dict


class BacktestEngine:
    """Bar-by-bar backtester that mirrors the paper-trader's execution path.

    leverage > 1: the engine uses all available capital as margin and controls
    leverage × capital in notional. A liquidation event is triggered when the
    open loss equals the entire margin.

    Optional exits (all default to disabled / 0.0 so signal-only behavior is
    unchanged):
        stop_loss_pct    — exit if price falls this fraction below entry
        take_profit_pct  — exit if price rises this fraction above entry
        trailing_stop_pct— exit if price falls this fraction from its peak since entry
    These mirror the RiskManager so a backtest reflects how paper/live would behave.
    """

    def __init__(
        self,
        initial_capital: float,
        fee_rate: float,
        slippage_pct: float,
        leverage: int = 1,
        timeframe: str = "1h",
        stop_loss_pct: float = 0.0,
        take_profit_pct: float = 0.0,
        trailing_stop_pct: float = 0.0,
    ) -> None:
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage_pct = slippage_pct
        self.leverage = max(1, int(leverage))
        self.timeframe = timeframe
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct

    def run(self, df: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        strategy.reset()

        capital: float = self.initial_capital
        position: float = 0.0          # BTC units held
        entry_price: float = 0.0
        entry_time: Optional[pd.Timestamp] = None
        peak_price: float = 0.0        # highest price since entry (trailing stop)
        margin_locked: float = 0.0     # capital committed as collateral
        trades: List[Trade] = []
        equity_points: List[float] = []

        def _close(price: float, ts, reason: str):
            nonlocal capital, position, margin_locked, entry_price, peak_price
            fill = price * (1 - self.slippage_pct)
            pnl = position * (fill - entry_price)
            fee = position * fill * self.fee_rate
            net = pnl - fee
            pct = net / margin_locked if margin_locked > 0 else 0.0
            trades.append(Trade(
                entry_time=entry_time, exit_time=ts,
                entry_price=entry_price, exit_price=fill,
                size=position, side="long", pnl=net, pnl_pct=pct,
                exit_reason=reason,
            ))
            capital = max(0.0, margin_locked + net)
            margin_locked = 0.0
            position = 0.0
            peak_price = 0.0

        for i in range(len(df)):
            hist = df.iloc[: i + 1]
            signal = strategy.next(hist)
            current_price = float(df["close"].iloc[i])
            ts = df.index[i]

            forced_exit = False

            if position > 0.0:
                if current_price > peak_price:
                    peak_price = current_price

                # ── Liquidation (leverage > 1) ──────────────────────────
                if self.leverage > 1:
                    unrealized = position * (current_price - entry_price)
                    if margin_locked + unrealized <= 0.0:
                        fee = position * current_price * self.fee_rate
                        trades.append(Trade(
                            entry_time=entry_time, exit_time=ts,
                            entry_price=entry_price, exit_price=current_price,
                            size=position, side="long",
                            pnl=-margin_locked - fee, pnl_pct=-1.0,
                            exit_reason="liquidation",
                        ))
                        capital = 0.0
                        margin_locked = 0.0
                        position = 0.0
                        peak_price = 0.0
                        equity_points.append(0.0)
                        continue

                # ── SL / trailing / TP exits ────────────────────────────
                pnl_pct = (current_price - entry_price) / entry_price
                if self.stop_loss_pct > 0 and pnl_pct <= -self.stop_loss_pct:
                    _close(current_price, ts, "stop_loss")
                    forced_exit = True
                elif self.trailing_stop_pct > 0 and peak_price > 0 and \
                        (peak_price - current_price) / peak_price >= self.trailing_stop_pct:
                    _close(current_price, ts, "trailing_stop")
                    forced_exit = True
                elif self.take_profit_pct > 0 and pnl_pct >= self.take_profit_pct:
                    _close(current_price, ts, "take_profit")
                    forced_exit = True

            # ── Strategy signal processing ──────────────────────────────
            if not forced_exit:
                if signal.type == SignalType.BUY and position == 0.0 and capital > 0.0:
                    fill_price = current_price * (1 + self.slippage_pct)
                    notional = capital * self.leverage
                    size = notional / fill_price
                    fee = notional * self.fee_rate
                    margin_locked = max(0.0, capital - fee)
                    capital = 0.0
                    position = size
                    entry_price = fill_price
                    entry_time = ts
                    peak_price = current_price

                elif signal.type == SignalType.SELL and position > 0.0:
                    _close(current_price, ts, "signal")

            # ── Mark-to-market ──────────────────────────────────────────
            if position > 0.0:
                unrealized_pnl = position * (current_price - entry_price)
                mark = margin_locked + unrealized_pnl
            else:
                mark = capital
            equity_points.append(mark)

        equity_curve = pd.Series(equity_points, index=df.index)
        metrics = compute_metrics(equity_curve, trades, self.initial_capital, self.timeframe)
        return BacktestResult(trades=trades, equity_curve=equity_curve, metrics=metrics)
