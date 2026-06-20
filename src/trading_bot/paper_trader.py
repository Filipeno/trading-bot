"""Long-lived paper trading process.

Usage:
    python -m trading_bot.paper_trader

Stop with Ctrl-C or SIGTERM. All signals and simulated trades are appended to
logs/trades.log for later comparison against backtest expectations.
"""

import json
import logging
import os
import signal as _signal
import sys
import time
import tomllib
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from .data.market import make_feed
from .execution.paper import PaperExecutor
from .logging_config import setup_logging
from .risk.manager import DailyLossLimitError, KillSwitchError, RiskManager
from .strategies.factory import make_strategy

load_dotenv()

_ROOT = Path(__file__).parent.parent.parent
_CONFIG_PATH = _ROOT / "config" / "settings.toml"
# Optional overrides written by the UI Paper Trade tab. Lets you choose the
# strategy, params, leverage and stops from the app without editing the TOML.
_OVERRIDES_PATH = _ROOT / "config" / "ui_overrides.json"
logger = logging.getLogger(__name__)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`."""
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _load_config() -> dict:
    with open(_CONFIG_PATH, "rb") as f:
        config = tomllib.load(f)
    if _OVERRIDES_PATH.exists():
        try:
            overrides = json.loads(_OVERRIDES_PATH.read_text(encoding="utf-8"))
            config = _deep_merge(config, overrides)
            logger.info("Applied UI overrides from %s", _OVERRIDES_PATH.name)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Ignoring bad ui_overrides.json: %s", exc)
    return config


def _build_strategy(config: dict):
    technical = make_strategy(config)

    news_cfg = config.get("news", {})
    if not news_cfg.get("enabled", False):
        return technical

    # News is enabled — wrap the technical strategy in a CombinedStrategy
    from .news.scorer import KeywordSentimentScorer
    from .news.strategy import NewsSentimentStrategy
    from .strategies.combined import CombinedStrategy

    news_source_name = news_cfg.get("source", "rss")
    if news_source_name == "cryptopanic":
        from .news.sources.cryptopanic import CryptoPanicSource
        api_key = os.getenv(news_cfg.get("api_key_env", "CRYPTOPANIC_API_KEY"), "")
        source = CryptoPanicSource(api_key=api_key)
    else:
        from .news.sources.rss import RSSNewsSource
        source = RSSNewsSource()

    scorer = KeywordSentimentScorer()
    news_strat = NewsSentimentStrategy(
        source=source,
        scorer=scorer,
        symbol=config["exchange"]["symbol"],
        ingestion_lag_seconds=news_cfg.get("ingestion_lag_seconds", 60),
        lookback_minutes=news_cfg.get("lookback_minutes", 60),
        sentiment_threshold=news_cfg.get("sentiment_threshold", 0.3),
        min_confidence=news_cfg.get("min_confidence", 0.3),
    )
    strategy_name = config.get("strategy", {}).get("name", "ema_crossover")
    logger.info("News sentiment strategy ENABLED (technical=%s, source=%s)", strategy_name, news_source_name)
    return CombinedStrategy(technical=technical, news=news_strat)


def _timeframe_to_seconds(tf: str) -> int:
    return {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}.get(tf, 3600)


def run() -> None:
    setup_logging()
    config = _load_config()

    symbol: str = config["exchange"]["symbol"]
    timeframe: str = config["exchange"]["timeframe"]
    leverage: int = max(1, int(config.get("risk", {}).get("leverage", 1)))

    exc_cfg = config["exchange"]
    feed = make_feed(
        config,
        api_key=os.getenv(exc_cfg.get("api_key_env", "BINANCE_TESTNET_API_KEY"), ""),
        secret=os.getenv(exc_cfg.get("secret_env", "BINANCE_TESTNET_SECRET"), ""),
    )

    strategy = _build_strategy(config)
    strategy.reset()

    executor = PaperExecutor(
        initial_capital=config["backtest"]["initial_capital"],
        fee_rate=config["backtest"]["fee_rate"],
        leverage=leverage,
    )

    risk = RiskManager(executor=executor, config=config, symbol=symbol)

    running = True

    def _handle_stop(sig, frame):  # noqa: ANN001
        nonlocal running
        logger.info("Shutdown signal received — stopping paper trader gracefully.")
        running = False

    _signal.signal(_signal.SIGINT, _handle_stop)
    _signal.signal(_signal.SIGTERM, _handle_stop)

    strategy_name = config.get("strategy", {}).get("name", "ema_crossover")
    asset = config.get("market", {}).get("asset_class", "crypto")
    logger.info(
        "Paper trader started | %s %s (%s) | strategy=%s | leverage=%dx | news=%s",
        symbol, timeframe, asset, strategy_name, leverage,
        config.get("news", {}).get("enabled", False),
    )

    last_day: date | None = None
    interval = _timeframe_to_seconds(timeframe)

    # Warm-up: need enough bars for the slowest indicator
    s = config.get("strategy", {})
    warmup = max(
        s.get("slow_period", 50),
        s.get("macd_slow", 26) + s.get("macd_signal", 9),
        s.get("bb_period", 20),
        s.get("breakout_period", 20),
        s.get("llm_lookback", 30),
    ) + 10

    while running:
        try:
            df = feed.recent(symbol, timeframe, limit=warmup)
            signal = strategy.next(df)
            current_price = float(df["close"].iloc[-1])
            ts = df.index[-1]

            today = ts.date()
            if last_day is None or today != last_day:
                risk.reset_day(current_price)
                last_day = today

            logger.info(
                "[%s] price=%.2f signal=%s | %s",
                ts, current_price, signal.type.value, signal.reason,
            )

            order = risk.process(signal, current_price)
            if order:
                logger.info(
                    "ORDER | side=%s size=%.6f fill=%.2f fee=%.4f | "
                    "equity=%.2f pos=%.6f",
                    order.side, order.size, order.fill_price, order.fee,
                    executor.get_equity(symbol, current_price),
                    executor.get_position(symbol),
                )

            time.sleep(interval)

        except (KillSwitchError, DailyLossLimitError) as exc:
            logger.critical("Trading halted: %s", exc)
            sys.exit(1)
        except Exception as exc:
            logger.critical("Unhandled error in main loop: %s", exc, exc_info=True)
            sys.exit(1)

    logger.info("Paper trader stopped.")


if __name__ == "__main__":
    run()
