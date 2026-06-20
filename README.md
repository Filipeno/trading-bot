# Trading Bot (crypto + stocks)

A safety-first trading bot for **learning and honest experimentation** — not a money machine. It backtests, paper-trades, and tells you the truth about whether a strategy actually works before you risk a cent. Works on **crypto** (via ccxt) and **stocks** (via free yfinance data).

> **Read this first.** Across 200+ days of real BTC data — and on stocks like AAPL/MSFT —
> none of the built-in technical strategies showed a reliable edge. Passive accumulation
> (Buy & Hold / DCA) beat all of them. This is the *normal* result for simple TA. Treat
> this project as a way to learn and to avoid losing money — not as a way to get rich.
> Keep leverage at 1× and paper-trade until something proves itself.

---

## What it does

| Capability | Where |
|------------|-------|
| Backtest a strategy on historical data | `run_backtest.py`, Backtest tab |
| Compare all strategies + passive benchmarks | `compare_strategies.py` |
| **Walk-forward reality check** (anti-overfitting) | `optimize.py`, Reality Check tab |
| Paper trade live with simulated fills | `python -m trading_bot.paper_trader`, Paper Trade tab |
| Free news/sentiment scoring (RSS, no API key) | News Feed tab |
| Simple all-in-one UI | `streamlit run ui/app.py` |

Live real-money trading is **intentionally not implemented** — `LiveExecutor` only
raises `NotImplementedError`. This is a deliberate safety choice.

---

## Quick start

```bash
pip install -e ".[dev]"          # core install
pip install -e ".[stocks]"       # optional: stock data (yfinance)
pip install -e ".[llm]"          # optional: Anthropic LLM provider
streamlit run ui/app.py          # launch the UI
```

In the UI, the **Market** selector at the top switches between **crypto** (BTC/USDT…) and
**stocks** (AAPL, TSLA…). Then:
1. **Setup** — get free Binance Testnet keys (crypto). Stocks need no keys.
2. **Backtest** — pick any strategy, tune its parameters, test it on history.
3. **Reality Check** — run the walk-forward test. *This is the most important tab.*
4. **Paper Trade** — choose strategy, leverage, and stops **right in the app**, then run on fake money.

No keys are needed for Backtest or Reality Check (they use public/free market data).

### Markets

| Asset class | Data source | Keys? | Notes |
|-------------|-------------|-------|-------|
| `crypto` | ccxt (Binance/Kraken) | none for data | Live data; testnet keys for paper trading |
| `stocks` | yfinance | none | Free, ~15-min delayed; timeframes 1m/5m/15m/30m/1h/1d (no 4h) |

Set `[market] asset_class` in `config/settings.toml`, or just use the UI selector.

---

## Strategies

All implement the same `Strategy` interface (`next(df) -> Signal`), so the
backtester, paper trader, and optimizer treat them identically.

- **EMA Crossover** — trend-following (fast EMA crosses slow EMA)
- **MACD** — momentum (histogram zero-cross)
- **Bollinger Bands** — mean-reversion (price exits the bands)
- **Breakout** — Donchian channel momentum (N-bar high/low breakout)
- **RSI** — mean-reversion (crossing out of oversold/overbought)
- **Stochastic** — momentum (%K/%D crossover in oversold/overbought zones)
- **VWAP** — intraday (price crossing the volume-weighted average price)
- **Supertrend** — ATR-channel trend follower (great for crypto)
- **DCA** *(benchmark)* — buy a fixed amount on a schedule, never sell. The one
  approach with real evidence behind it for small accounts. See `backtest/dca.py`.
- **LLM** — a Claude model reads a numeric summary of the chart each bar (trend,
  RSI, momentum, position in range) and returns BUY/SELL/HOLD with reasoning.
  See "LLM strategy" below.

Pick the active strategy in `config/settings.toml` under `[strategy] name = ...`.

### LLM strategy

Two providers are supported — pick one in `config/settings.toml` (`llm_provider`):

```toml
# Mammouth.ai (OpenAI-compatible HTTP — uses requests, no SDK needed)
llm_provider = "mammouth"
llm_base_url = "https://api.mammouth.ai/v1"   # VERIFY in your Mammouth dashboard
llm_model    = "gpt-4o-mini"                  # VERIFY available model names
llm_api_key_env = "MAMMOUTH_API_KEY"

# …or Anthropic (needs `pip install anthropic`)
# llm_provider = "anthropic"
# llm_model = "claude-haiku-4-5-20251001"
# llm_api_key_env = "ANTHROPIC_API_KEY"
```

> **Mammouth note:** their API is OpenAI-compatible, but confirm the exact
> `base_url` and model names from your account — those defaults are best-guess.
> Save the key via the UI Setup tab (🤖 expander) or in `.env`.

Then set `name = "llm"`. Each bar, the model is sent a compact snapshot of the
chart and asked for a JSON decision. Safety properties:

- **Fails safe** — any API error, unparseable reply, or low-confidence call → HOLD.
  The bot never trades on a broken or uncertain response.
- **Still risk-managed** — goes through the same `RiskManager` (stop-loss, daily
  limit, kill switch) as every other strategy.
- **Key from env only** — never hardcoded; `.env` is gitignored.
- **Testable offline** — the LLM is behind an `LLMClient` protocol, so the whole
  strategy is unit-tested with a fake client (no network, no key, no cost).

> 💸 **Cost & honesty warning.** One API call per bar. A backtest makes one call
> *per candle* — potentially thousands of paid calls — so use the LLM for **paper
> trading**, not bulk backtests. And like every other strategy here, an LLM will
> **not** reliably beat the market. It has no real-time data and can be confidently
> wrong. Validate with the Reality Check and paper-trade before trusting it.

---

## The Reality Check (why most of this exists)

Any strategy can be made to look brilliant by tuning it on one lucky slice of
history. That's **overfitting**, and it's how people lose money. `optimize.py`
defends against it with **walk-forward optimization**:

1. Split history into consecutive folds.
2. Tune parameters on fold *i* (in-sample).
3. Measure those exact parameters on fold *i+1* — data the tuner never saw (out-of-sample).
4. Only the out-of-sample numbers are reported as trustworthy.

```bash
python optimize.py --timeframe 1h --limit 5000 --folds 6
```

If a strategy looks great in-sample but collapses out-of-sample, it has no edge.
That gap is the single most useful thing this project will show you.

---

## Safety design

- **`RiskManager` is the only path to the executor.** Every order passes through
  stop-loss, take-profit, **trailing-stop**, daily-loss-limit, liquidation, and
  news-halt checks. (Trailing stop: set `trailing_stop_pct` > 0 to lock in gains —
  exits if price falls that % from its peak since entry. Backtest models it too.)
- **Leverage defaults to 1×.** Higher leverage multiplies losses faster than gains;
  the UI warns you with concrete numbers. A 2% stop at 5× = 10% of your margin gone.
- **Kill switch** halts all trading on any unhandled error or daily-loss breach.
- **No secrets in code.** API keys load from environment / `.env` (which is gitignored).
- **Append-only audit log** of every decision in `logs/trades.log`.
- **Live trading is unreachable by default** (`LiveExecutor` raises `NotImplementedError`).

---

## Configuration

Everything lives in `config/settings.toml`: exchange, symbol, timeframe, the active
strategy and its parameters, risk limits, leverage, fees, and the optional news layer.
Switching exchanges (e.g. Binance → Kraken for EU) is a config change only.

---

## Tests

```bash
python -m pytest -q
```

Covers risk management, every strategy, the backtester (incl. leverage + liquidation),
walk-forward optimization, the paginating history fetcher, DCA, and the news layer.

---

## Honest recommendation

1. Keep `leverage = 1`.
2. Use the **Reality Check** before trusting any strategy.
3. **Paper-trade on the testnet for weeks** before considering real money.
4. If you just want exposure to crypto with a small account you're adding to over
   time, the data says **DCA** is the rational choice — no bot required.
