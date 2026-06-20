"""Quick CLI to run a backtest and print results.

Usage:
    python run_backtest.py                        # uses config/settings.toml
    python run_backtest.py --limit 500            # fetch last 500 candles
    python run_backtest.py --since 2024-01-01     # fetch from date
"""

import argparse
import sys
import tomllib
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trading_bot.backtest.engine import BacktestEngine
from trading_bot.data.fetcher import fetch_ohlcv, make_exchange
from trading_bot.strategies.ema_crossover import EMACrossover

_CONFIG_PATH = Path(__file__).parent / "config" / "settings.toml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EMA crossover backtest")
    parser.add_argument("--limit", type=int, default=1000, help="Number of candles to fetch")
    parser.add_argument("--since", type=str, default=None, help="Start date YYYY-MM-DD (UTC)")
    args = parser.parse_args()

    with open(_CONFIG_PATH, "rb") as f:
        config = tomllib.load(f)

    symbol: str = config["exchange"]["symbol"]
    timeframe: str = config["exchange"]["timeframe"]

    since_ms = None
    if args.since:
        import pandas as pd
        since_ms = int(pd.Timestamp(args.since, tz="UTC").timestamp() * 1000)

    print(f"Fetching {args.limit} x {timeframe} candles for {symbol} …")
    exchange = make_exchange(config)  # no API key needed for public OHLCV
    df = fetch_ohlcv(exchange, symbol, timeframe, limit=args.limit, since=since_ms)
    print(f"  Got {len(df)} bars: {df.index[0]} → {df.index[-1]}")

    strategy = EMACrossover(
        fast=config["strategy"]["fast_period"],
        slow=config["strategy"]["slow_period"],
    )

    engine = BacktestEngine(
        initial_capital=config["backtest"]["initial_capital"],
        fee_rate=config["backtest"]["fee_rate"],
        slippage_pct=config["backtest"]["slippage_pct"],
    )

    print(f"\nRunning backtest (EMA {strategy.fast}/{strategy.slow}) …")
    result = engine.run(df, strategy)

    print("\n── Backtest Results ──────────────────────────")
    for k, v in result.metrics.items():
        print(f"  {k:<22} {v}")

    if result.trades:
        print(f"\n── Last 5 Trades ─────────────────────────────")
        for t in result.trades[-5:]:
            print(
                f"  {t.entry_time.date()} → {t.exit_time.date()} | "
                f"entry={t.entry_price:.2f} exit={t.exit_price:.2f} | "
                f"pnl={t.pnl:+.2f} ({t.pnl_pct:+.2%})"
            )


if __name__ == "__main__":
    main()
