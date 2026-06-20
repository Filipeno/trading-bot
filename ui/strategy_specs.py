"""Declarative parameter specs for every strategy, shared across UI tabs.

Each spec entry is: (config_key, label, kind, min, max, default, step)
where config_key matches exactly what the strategy factory reads from
config["strategy"], so a params dict can be fed straight into make_strategy().
"""

from __future__ import annotations

STRATEGY_LABELS = {
    "ema_crossover": "EMA Crossover (trend-following)",
    "macd": "MACD (momentum)",
    "bollinger_bands": "Bollinger Bands (mean-reversion)",
    "breakout": "Breakout / Donchian (momentum)",
    "rsi": "RSI (mean-reversion)",
    "stochastic": "Stochastic Oscillator (momentum)",
    "vwap": "VWAP cross (intraday)",
    "supertrend": "Supertrend / ATR (trend)",
    "llm": "LLM — Claude reads the chart",
}

# One-line plain-English explanation of how each strategy decides to buy/sell.
STRATEGY_DESC = {
    "ema_crossover": "Buys when a fast moving average crosses above a slow one (uptrend "
                     "starting) and sells when it crosses back below. Classic trend-following.",
    "macd": "Tracks momentum via the MACD histogram. Buys when momentum flips positive, "
            "sells when it flips negative.",
    "bollinger_bands": "Assumes price snaps back to its average. Buys when price dips below the "
                       "lower band (oversold), sells when it pokes above the upper band.",
    "breakout": "Buys when price breaks above its highest high of the last N bars (a breakout) "
                "and sells when it breaks below the recent low.",
    "rsi": "Uses the RSI gauge of overbought/oversold. Buys as RSI climbs out of oversold, "
           "sells as it falls out of overbought.",
    "stochastic": "Momentum oscillator. Buys when the fast line crosses up through the slow line "
                  "in the oversold zone, sells on the opposite cross when overbought.",
    "vwap": "Uses the volume-weighted average price (what traders actually paid). Buys when price "
            "crosses above VWAP, sells when it crosses below. Popular for intraday.",
    "supertrend": "An ATR-based trend filter that adapts to volatility. Buys when the trend flips "
                  "up, sells when it flips down. Well-suited to trending crypto.",
    "llm": "Sends a snapshot of the chart to an AI model each bar and follows its BUY/SELL/HOLD "
           "call. Costs one API call per bar — and has no proven edge.",
}

# Plain-English glossary for the result metrics.
METRIC_GLOSSARY = {
    "Return": "Total % change in your portfolio over the whole test.",
    "Sharpe": "Reward-for-risk. Higher = smoother gains. >1 is decent, <0 means it lost money.",
    "Max Drawdown": "The worst peak-to-trough drop along the way. How much pain you'd have sat through.",
    "Win Rate": "Share of trades that closed in profit. High win rate ≠ profitable if losses are big.",
    "# Trades": "How many round-trip trades the strategy made. Very high = lots of fees.",
}

# Strategies usable in each context.
BACKTESTABLE = ["ema_crossover", "macd", "bollinger_bands", "breakout",
                "rsi", "stochastic", "vwap", "supertrend"]
OPTIMIZABLE = BACKTESTABLE                      # walk-forward needs param grids
PAPER_TRADE = BACKTESTABLE + ["llm"]            # LLM is fine live (1 call/bar)

PARAM_SPECS: dict[str, list[tuple]] = {
    "ema_crossover": [
        ("fast_period", "Fast EMA period", "int", 5, 50, 20, 1),
        ("slow_period", "Slow EMA period", "int", 20, 200, 50, 1),
    ],
    "macd": [
        ("macd_fast", "MACD fast EMA", "int", 3, 30, 12, 1),
        ("macd_slow", "MACD slow EMA", "int", 10, 60, 26, 1),
        ("macd_signal", "Signal period", "int", 3, 20, 9, 1),
    ],
    "bollinger_bands": [
        ("bb_period", "Period", "int", 5, 50, 20, 1),
        ("bb_std_dev", "Std-dev bands", "float", 1.0, 3.5, 2.0, 0.25),
    ],
    "breakout": [
        ("breakout_period", "Lookback period", "int", 5, 100, 20, 1),
    ],
    "rsi": [
        ("rsi_period", "RSI period", "int", 5, 30, 14, 1),
        ("rsi_oversold", "Oversold level", "int", 10, 45, 30, 1),
        ("rsi_overbought", "Overbought level", "int", 55, 90, 70, 1),
    ],
    "stochastic": [
        ("stoch_k_period", "%K period", "int", 5, 30, 14, 1),
        ("stoch_d_period", "%D period", "int", 2, 10, 3, 1),
        ("stoch_oversold", "Oversold level", "int", 10, 40, 20, 1),
        ("stoch_overbought", "Overbought level", "int", 60, 90, 80, 1),
    ],
    "vwap": [
        ("vwap_period", "VWAP period", "int", 5, 100, 20, 1),
    ],
    "supertrend": [
        ("supertrend_period", "ATR period", "int", 5, 30, 10, 1),
        ("supertrend_multiplier", "ATR multiplier", "float", 1.0, 5.0, 3.0, 0.5),
    ],
    "llm": [
        ("llm_lookback", "Bars summarized for the model", "int", 20, 60, 30, 1),
        ("llm_min_confidence", "Min confidence to act", "float", 0.30, 0.90, 0.55, 0.05),
    ],
}

# (ok, message) validators for parameter combinations.
_CONSTRAINTS = {
    "ema_crossover": lambda p: (p["fast_period"] < p["slow_period"], "Fast EMA must be < slow EMA"),
    "macd": lambda p: (p["macd_fast"] < p["macd_slow"], "MACD fast must be < slow"),
    "rsi": lambda p: (p["rsi_oversold"] < p["rsi_overbought"], "Oversold must be < overbought"),
    "stochastic": lambda p: (p["stoch_oversold"] < p["stoch_overbought"], "Oversold must be < overbought"),
}


def render_params(st, name: str, prefix: str) -> dict:
    """Render sliders for `name`'s params and return {config_key: value}."""
    params: dict = {}
    for key, label, kind, lo, hi, default, step in PARAM_SPECS.get(name, []):
        wkey = f"{prefix}_{name}_{key}"
        if kind == "int":
            params[key] = st.slider(label, int(lo), int(hi), int(default), int(step), key=wkey)
        else:
            params[key] = st.slider(label, float(lo), float(hi), float(default), float(step), key=wkey)
    return params


def validate(name: str, params: dict) -> tuple[bool, str]:
    fn = _CONSTRAINTS.get(name)
    return fn(params) if fn else (True, "")


def short_label(name: str, params: dict) -> str:
    """Compact label like 'MACD 12/26/9' for chart titles."""
    vals = "/".join(str(v) for v in params.values())
    base = {
        "ema_crossover": "EMA", "macd": "MACD", "bollinger_bands": "BB",
        "breakout": "Breakout", "rsi": "RSI", "stochastic": "Stoch",
        "vwap": "VWAP", "supertrend": "Supertrend", "llm": "LLM",
    }.get(name, name)
    return f"{base} {vals}".strip()
