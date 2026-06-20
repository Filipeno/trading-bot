import logging
from datetime import timedelta
from typing import Optional

import pandas as pd

from ..execution.base import Executor, OrderResult
from ..strategies.base import Signal, SignalType

logger = logging.getLogger(__name__)


class KillSwitchError(Exception):
    """Trading is halted. Do not catch this silently — let it propagate to the main loop."""


class DailyLossLimitError(Exception):
    """Daily loss limit breached. Trading halted for the rest of the day."""


class RiskManager:
    """Single choke-point for all risk guardrails.

    All parameters come from config["risk"] and config["news"]; nothing is hardcoded.
    Every outbound order passes through process() — no strategy or paper-trader
    code should call the executor directly.

    Leverage:
        Set config["risk"]["leverage"] > 1 for leveraged positions.
        Position size = (balance × max_position_pct × leverage) / price.
        A liquidation check is added that fires before the SL when the price
        would wipe out the entire margin (price <= entry × (1 - 1/leverage)).
    """

    def __init__(self, executor: Executor, config: dict, symbol: str) -> None:
        risk = config["risk"]
        self._executor = executor
        self._symbol = symbol
        self._max_position_pct: float = risk["max_position_pct"]
        self._stop_loss_pct: float = risk["stop_loss_pct"]
        self._take_profit_pct: float = risk["take_profit_pct"]
        self._daily_loss_limit_pct: float = risk["daily_loss_limit_pct"]
        self._leverage: int = max(1, int(risk.get("leverage", 1)))
        # Trailing stop: exit if price falls this % below the peak reached since
        # entry. 0 (default) disables it. Works alongside the fixed stop-loss —
        # whichever triggers first wins.
        self._trailing_stop_pct: float = float(risk.get("trailing_stop_pct", 0.0))

        self._entry_price: Optional[float] = None
        self._position_size: float = 0.0
        self._peak_price: Optional[float] = None   # highest price seen since entry
        self._day_start_equity: float = executor.get_balance()
        self._killed: bool = False

        # News-event halt (independent of whether the news strategy is enabled)
        news_cfg = config.get("news", {})
        self._news_halt_minutes: int = news_cfg.get("event_halt_minutes", 5)
        self._news_high_impact_threshold: float = news_cfg.get("high_impact_threshold", 0.7)
        self._news_halt_until: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_day(self, current_price: float) -> None:
        """Call once per UTC day before processing the first signal."""
        self._day_start_equity = self._current_equity(current_price)
        logger.info("Daily equity reset to %.2f", self._day_start_equity)

    def notify_news_event(self, score: float, ts: pd.Timestamp) -> None:
        """Notify the risk manager that a news item was processed.

        If |score| exceeds the high-impact threshold, a temporary trading halt
        is activated for `event_halt_minutes`. This protects against wide
        spreads and front-running immediately after major headlines — even if
        the news sentiment strategy itself is disabled.
        """
        if abs(score) >= self._news_high_impact_threshold:
            self._news_halt_until = ts + pd.Timedelta(minutes=self._news_halt_minutes)
            logger.warning(
                "NEWS HALT: high-impact event (|score|=%.2f >= %.2f), "
                "pausing new trades until %s",
                abs(score), self._news_high_impact_threshold, self._news_halt_until,
            )

    def process(self, signal: Signal, current_price: float) -> Optional[OrderResult]:
        """Apply risk checks then (maybe) forward the signal to the executor.

        Returns the OrderResult if an order was placed, or None for HOLDs,
        filtered signals, and news-halt windows.
        Raises KillSwitchError / DailyLossLimitError on permanent halt.
        """
        try:
            self._check_kill_switch()
            self._check_daily_loss(current_price)

            # SL/TP/liquidation overrides the incoming signal (exits are never halted by news)
            exit_reason = self._check_sl_tp(current_price)
            if exit_reason:
                return self._force_exit(current_price, exit_reason)

            if signal.type == SignalType.BUY and self._position_size == 0.0:
                # News halt only blocks new entries, never exits
                if self._is_news_halted(signal.timestamp):
                    logger.info(
                        "News halt active until %s — suppressing BUY entry",
                        self._news_halt_until,
                    )
                    return None
                return self._open_long(current_price, signal.reason)

            if signal.type == SignalType.SELL and self._position_size > 0.0:
                return self._close_long(current_price, signal.reason)

            return None

        except (KillSwitchError, DailyLossLimitError):
            raise
        except Exception as exc:
            self._killed = True
            logger.critical(
                "KILL SWITCH — unhandled error in RiskManager.process: %s", exc, exc_info=True
            )
            raise KillSwitchError(f"Unhandled error triggered kill switch: {exc}") from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_equity(self, current_price: float) -> float:
        return self._executor.get_equity(self._symbol, current_price)

    def _check_kill_switch(self) -> None:
        if self._killed:
            raise KillSwitchError("Kill switch is active — all trading halted.")

    def _check_daily_loss(self, current_price: float) -> None:
        equity = self._current_equity(current_price)
        if self._day_start_equity <= 0:
            return
        loss_pct = (self._day_start_equity - equity) / self._day_start_equity
        if loss_pct >= self._daily_loss_limit_pct:
            self._killed = True
            msg = f"Daily loss limit breached: {loss_pct:.2%} >= {self._daily_loss_limit_pct:.2%}"
            logger.critical(msg)
            raise DailyLossLimitError(msg)

    def _check_sl_tp(self, current_price: float) -> Optional[str]:
        if self._entry_price is None or self._position_size <= 0.0:
            return None

        # Track the peak price reached since entry (for the trailing stop).
        if self._peak_price is None or current_price > self._peak_price:
            self._peak_price = current_price

        # Liquidation backstop — fires before SL when leverage amplifies loss to 100% of margin
        if self._leverage > 1:
            liq_price = self._entry_price * (1.0 - 1.0 / self._leverage)
            if current_price <= liq_price:
                return "liquidation"

        pnl_pct = (current_price - self._entry_price) / self._entry_price
        if pnl_pct <= -self._stop_loss_pct:
            return "stop_loss"

        # Trailing stop — exit if price has dropped trailing_stop_pct from the peak.
        if self._trailing_stop_pct > 0.0 and self._peak_price:
            drop_from_peak = (self._peak_price - current_price) / self._peak_price
            if drop_from_peak >= self._trailing_stop_pct:
                return "trailing_stop"

        if pnl_pct >= self._take_profit_pct:
            return "take_profit"
        return None

    def _is_news_halted(self, ts: pd.Timestamp) -> bool:
        return self._news_halt_until is not None and ts < self._news_halt_until

    def _open_long(self, price: float, reason: str) -> OrderResult:
        balance = self._executor.get_balance()
        # margin = max_position_pct of free cash; notional = margin × leverage
        margin = balance * self._max_position_pct
        notional = margin * self._leverage
        size = notional / price
        logger.info(
            "BUY  size=%.6f @ %.2f (leverage=%dx, margin=%.2f, notional=%.2f) | %s",
            size, price, self._leverage, margin, notional, reason,
        )
        result = self._executor.submit_order(self._symbol, "buy", size, price)
        self._entry_price = price
        self._position_size = size
        self._peak_price = price   # start trailing from the entry price
        return result

    def _close_long(self, price: float, reason: str) -> OrderResult:
        logger.info("SELL size=%.6f @ %.2f | %s", self._position_size, price, reason)
        result = self._executor.submit_order(self._symbol, "sell", self._position_size, price)
        self._entry_price = None
        self._position_size = 0.0
        self._peak_price = None
        return result

    def _force_exit(self, price: float, reason: str) -> OrderResult:
        logger.warning("%s triggered at %.2f (entry=%.2f)", reason.upper(), price, self._entry_price)
        return self._close_long(price, reason)
