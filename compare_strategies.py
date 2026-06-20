"""Quick comparison: run every strategy against real BTC data and print a table.

Usage:
    python compare_strategies.py
    python compare_strategies.py --timeframe 15m --limit 1000 --leverage 1
"""

import argparse
import sys
from pathlib import Path

# Windows consoles often default to cp1250 — force UTF-8 so ×, → etc. print.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trading_bot.backtest.engine import BacktestEngine
from trading_bot.data.market import make_feed
from trading_bot.strategies.bollinger_bands import BollingerBandsStrategy
from trading_bot.strategies.breakout import BreakoutStrategy
from trading_bot.strategies.ema_crossover import EMACrossover
from trading_bot.strategies.macd import MACDStrategy


def build_strategies(tf: str) -> dict:
    """Param presets tuned a bit shorter for intraday timeframes."""
    intraday = tf in ("1m", "5m", "15m", "30m")
    return {
        "EMA Crossover":   EMACrossover(fast=9, slow=21) if intraday else EMACrossover(20, 50),
        "MACD":            MACDStrategy(fast=6, slow=13, signal=4) if intraday else MACDStrategy(12, 26, 9),
        "Bollinger Bands": BollingerBandsStrategy(period=20, std_dev=2.0),
        "Breakout":        BreakoutStrategy(period=20),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--asset", default="crypto", choices=["crypto", "stocks"])
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--leverage", type=int, default=1)
    p.add_argument("--capital", type=float, default=10_000.0)
    args = p.parse_args()

    print(f"\nFetching up to {args.limit} × {args.timeframe} candles for {args.symbol} ({args.asset})…")
    config = {
        "market": {"asset_class": args.asset},
        "exchange": {"id": "binance", "testnet": False, "symbol": args.symbol},
    }
    feed = make_feed(config)
    df = feed.history(args.symbol, args.timeframe, total=args.limit)
    print(f"Got {len(df)} candles.")

    span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400
    buy_hold = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    print(f"Period: {df.index[0].date()} → {df.index[-1].date()}  ({span_days:.1f} days)")
    print(f"Buy & Hold return over period: {buy_hold:+.2f}%")
    print(f"Leverage: {args.leverage}×   Starting capital: ${args.capital:,.0f}\n")

    header = f"{'Strategy':<18}{'Return':>10}{'Sharpe':>9}{'MaxDD':>9}{'WinRate':>9}{'Trades':>8}{'Final $':>12}"
    print(header)
    print("-" * len(header))

    rows = []
    for name, strat in build_strategies(args.timeframe).items():
        engine = BacktestEngine(
            initial_capital=args.capital,
            fee_rate=0.001,
            slippage_pct=0.0005,
            leverage=args.leverage,
            timeframe=args.timeframe,
        )
        r = engine.run(df, strat)
        m = r.metrics
        rows.append((name, m))
        print(
            f"{name:<18}"
            f"{m['total_return_pct']:>9.2f}%"
            f"{m['sharpe_ratio']:>9.2f}"
            f"{m['max_drawdown_pct']:>8.1f}%"
            f"{m['win_rate_pct']:>8.0f}%"
            f"{m['n_trades']:>8}"
            f"{m['final_equity']:>12,.0f}"
        )

    # ── Passive benchmarks (the honest comparison) ─────────────────────────
    from trading_bot.backtest.dca import simulate_dca

    # DCA: buy once per day. interval = bars per day for this timeframe.
    bars_per_day = {
        "1m": 1440, "5m": 288, "15m": 96, "30m": 48,
        "1h": 24, "4h": 6, "1d": 1,
    }.get(args.timeframe, 24)
    dca = simulate_dca(df, contribution=50.0, interval_bars=max(1, bars_per_day), fee_rate=0.001)

    print(f"{'Buy & Hold':<18}{buy_hold:>9.2f}%{'—':>9}{'—':>9}{'—':>9}{'—':>8}"
          f"{args.capital * (1 + buy_hold/100):>12,.0f}")
    print(f"{'DCA (daily $50)':<18}{dca.return_pct:>9.2f}%{'—':>9}{'—':>9}{'—':>9}"
          f"{dca.n_buys:>8}{dca.final_value:>12,.0f}")

    print("-" * len(header))
    best = max(rows, key=lambda x: x[1]["total_return_pct"])
    print(f"\nBest active strategy: {best[0]} ({best[1]['total_return_pct']:+.2f}%)")
    passive_best = max(buy_hold, dca.return_pct)
    if passive_best >= best[1]["total_return_pct"]:
        print(f"Passive (Buy&Hold/DCA) beat every active strategy here ({passive_best:+.2f}%).")
        print("This is the usual result — and the honest case for just accumulating, not trading.")
    print("Note: one window is NOT predictive. Use optimize.py for the walk-forward verdict.\n")


if __name__ == "__main__":
    main()
