"""Walk-forward optimizer + honest performance report.

This is the tool that tells you whether a strategy has any real edge — or whether
it just looks good on one lucky slice of history.

Usage:
    python optimize.py                                  # all strategies, 15m, walk-forward
    python optimize.py --strategy macd --timeframe 1h
    python optimize.py --limit 3000 --folds 6
    python optimize.py --leverage 1                     # keep this at 1 until proven

Reading the output:
    The numbers that matter are the OUT-OF-SAMPLE ones. They are measured on data
    the optimizer never tuned on. If out-of-sample is bad, the strategy has no
    edge — no matter how good the in-sample numbers look.
"""

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trading_bot.backtest.optimize import STRATEGY_REGISTRY, walk_forward
from trading_bot.data.market import make_feed


def _fmt_params(p: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in p.items())


def run_strategy(df, name, *, n_folds, leverage, timeframe):
    print(f"\n{'═' * 78}")
    print(f"  {name.upper().replace('_', ' ')}")
    print(f"{'═' * 78}")

    try:
        report = walk_forward(
            df, name,
            n_folds=n_folds,
            metric="total_return_pct",
            leverage=leverage,
            timeframe=timeframe,
        )
    except ValueError as exc:
        print(f"  Skipped: {exc}")
        return None

    # Per-fold detail
    print(f"  {'Fold':<5}{'Test period':<26}{'In-sample':>11}{'Out-sample':>12}"
          f"{'Buy&Hold':>11}{'Beat B&H?':>11}")
    print(f"  {'-' * 74}")
    for f in report.folds:
        oos = f.out_of_sample["total_return_pct"]
        beat = "yes" if oos > f.buy_hold_test_pct else "no"
        span = f"{f.test_span[0].date()}→{f.test_span[1].date()}"
        print(
            f"  {f.fold:<5}{span:<26}"
            f"{f.in_sample_metric:>10.1f}%"
            f"{oos:>11.1f}%"
            f"{f.buy_hold_test_pct:>10.1f}%"
            f"{beat:>11}"
        )

    # Honest aggregate (out-of-sample only)
    print(f"  {'-' * 74}")
    print("  OUT-OF-SAMPLE SUMMARY (the only numbers that matter):")
    print(f"    Mean return per fold ....... {report.oos_mean_return:+.2f}%")
    print(f"    Median return per fold ..... {report.oos_median_return:+.2f}%")
    print(f"    Profitable folds ........... {report.pct_profitable_folds:.0f}%")
    print(f"    Folds that beat buy&hold ... {report.pct_beat_buy_hold:.0f}%")
    print(f"    Best / worst fold .......... {report.best_fold_return:+.1f}% / "
          f"{report.worst_fold_return:+.1f}%")
    print(f"    Compounded across folds .... {report.oos_total_compounded_pct:+.2f}%")
    return report


def verdict(reports: dict) -> None:
    print(f"\n{'═' * 78}")
    print("  VERDICT")
    print(f"{'═' * 78}")

    ranked = sorted(
        [(n, r) for n, r in reports.items() if r is not None],
        key=lambda x: x[1].oos_mean_return,
        reverse=True,
    )
    if not ranked:
        print("  No strategy produced enough data to evaluate.")
        return

    print(f"  {'Strategy':<18}{'OOS mean':>11}{'Profitable':>12}{'Beat B&H':>11}{'Verdict':>16}")
    print(f"  {'-' * 66}")
    for name, r in ranked:
        # An honest bar: positive expectancy AND consistency AND beats holding
        if r.oos_mean_return > 0 and r.pct_profitable_folds >= 60 and r.pct_beat_buy_hold >= 50:
            v = "promising"
        elif r.oos_mean_return > 0:
            v = "weak / unstable"
        else:
            v = "no edge"
        print(
            f"  {name:<18}{r.oos_mean_return:>10.2f}%"
            f"{r.pct_profitable_folds:>11.0f}%{r.pct_beat_buy_hold:>10.0f}%{v:>16}"
        )

    print(f"\n  {'-' * 74}")
    best_name, best = ranked[0]
    if best.oos_mean_return > 0 and best.pct_profitable_folds >= 60:
        print(f"  Best out-of-sample: {best_name} ({best.oos_mean_return:+.2f}% mean per fold).")
        print("  This is worth PAPER trading — not worth real money yet. Run it live on")
        print("  the testnet for weeks and confirm the live results match before risking a cent.")
    else:
        print("  None of these strategies showed a reliable out-of-sample edge on this data.")
        print("  That is the NORMAL result for simple technical strategies. It means: do NOT")
        print("  put real money on them. Keep paper trading and treat this as a learning tool.")
    print("  Reminder: leverage multiplies losses faster than gains. Keep it at 1×.\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Walk-forward strategy optimizer")
    p.add_argument("--asset", default="crypto", choices=["crypto", "stocks"],
                   help="crypto (ccxt) or stocks (yfinance). For stocks use --symbol AAPL")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--limit", type=int, default=5000,
                   help="Number of candles to fetch — pages past the API cap (more = more reliable)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--leverage", type=int, default=1)
    p.add_argument("--strategy", default="all",
                   choices=["all", *STRATEGY_REGISTRY.keys()])
    args = p.parse_args()

    print(f"\nFetching up to {args.limit} × {args.timeframe} candles for {args.symbol} "
          f"({args.asset})…")
    config = {
        "market": {"asset_class": args.asset},
        "exchange": {"id": "binance", "testnet": False, "symbol": args.symbol},
    }
    feed = make_feed(config)
    df = feed.history(args.symbol, args.timeframe, total=args.limit)
    print(f"Got {len(df)} candles.")
    span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400
    print(f"Period: {df.index[0].date()} → {df.index[-1].date()}  ({span_days:.1f} days)")
    print(f"Walk-forward: {args.folds} folds | leverage {args.leverage}× | "
          f"optimizing on past, measuring on unseen future of each fold")

    names = list(STRATEGY_REGISTRY.keys()) if args.strategy == "all" else [args.strategy]
    reports = {}
    for name in names:
        reports[name] = run_strategy(
            df, name,
            n_folds=args.folds,
            leverage=args.leverage,
            timeframe=args.timeframe,
        )

    if len(names) > 1:
        verdict(reports)


if __name__ == "__main__":
    main()
