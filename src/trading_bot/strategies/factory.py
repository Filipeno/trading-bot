"""Central factory — pick a strategy by name from settings.toml [strategy] section."""

from .base import Strategy
from .bollinger_bands import BollingerBandsStrategy
from .breakout import BreakoutStrategy
from .ema_crossover import EMACrossover
from .macd import MACDStrategy
from .rsi import RSIStrategy
from .stochastic import StochasticStrategy
from .supertrend import SupertrendStrategy
from .vwap import VWAPStrategy

_STRATEGY_NAMES = (
    "ema_crossover", "macd", "bollinger_bands", "breakout",
    "rsi", "stochastic", "vwap", "supertrend", "llm",
)


def make_strategy(config: dict) -> Strategy:
    """Return the technical strategy specified by config["strategy"]["name"].

    Falls back to EMA crossover if the name is missing or unrecognised.
    """
    s = config.get("strategy", {})
    name = s.get("name", "ema_crossover").lower().strip()

    if name == "macd":
        return MACDStrategy(
            fast=s.get("macd_fast", 12),
            slow=s.get("macd_slow", 26),
            signal=s.get("macd_signal", 9),
        )
    if name == "bollinger_bands":
        return BollingerBandsStrategy(
            period=s.get("bb_period", 20),
            std_dev=s.get("bb_std_dev", 2.0),
        )
    if name == "breakout":
        return BreakoutStrategy(period=s.get("breakout_period", 20))

    if name == "rsi":
        return RSIStrategy(
            period=s.get("rsi_period", 14),
            oversold=s.get("rsi_oversold", 30.0),
            overbought=s.get("rsi_overbought", 70.0),
        )

    if name == "stochastic":
        return StochasticStrategy(
            k_period=s.get("stoch_k_period", 14),
            d_period=s.get("stoch_d_period", 3),
            oversold=s.get("stoch_oversold", 20.0),
            overbought=s.get("stoch_overbought", 80.0),
        )

    if name == "vwap":
        return VWAPStrategy(period=s.get("vwap_period", 20))

    if name == "supertrend":
        return SupertrendStrategy(
            period=s.get("supertrend_period", 10),
            multiplier=s.get("supertrend_multiplier", 3.0),
        )

    if name == "llm":
        # Built lazily so the rest of the project never needs an LLM SDK/key
        # unless the LLM strategy is actually selected.
        from .llm_strategy import LLMStrategy

        provider = s.get("llm_provider", "anthropic").lower().strip()
        if provider in ("mammouth", "openai", "openai_compatible"):
            from .llm_strategy import OpenAICompatibleClient
            client = OpenAICompatibleClient(
                base_url=s.get("llm_base_url", "https://api.mammouth.ai/v1"),
                model=s.get("llm_model", "gpt-4o"),
                api_key_env=s.get("llm_api_key_env", "MAMMOUTH_API_KEY"),
            )
        else:  # "anthropic"
            from .llm_strategy import AnthropicClient
            client = AnthropicClient(
                model=s.get("llm_model", "claude-haiku-4-5-20251001"),
                api_key_env=s.get("llm_api_key_env", "ANTHROPIC_API_KEY"),
            )

        return LLMStrategy(
            client=client,
            lookback=s.get("llm_lookback", 30),
            min_confidence=s.get("llm_min_confidence", 0.55),
        )

    # default / "ema_crossover"
    return EMACrossover(
        fast=s.get("fast_period", 20),
        slow=s.get("slow_period", 50),
    )


def strategy_names() -> tuple[str, ...]:
    return _STRATEGY_NAMES
