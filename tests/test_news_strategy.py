"""Tests for NewsSentimentStrategy, CombinedStrategy, and FileNewsSource."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from trading_bot.news.scorer import KeywordSentimentScorer, SentimentScore
from trading_bot.news.sources.base import NewsItem, NewsSource
from trading_bot.news.strategy import NewsSentimentStrategy
from trading_bot.strategies.base import SignalType
from trading_bot.strategies.combined import CombinedStrategy
from trading_bot.strategies.ema_crossover import EMACrossover

SYMBOL = "BTC/USDT"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _df(current_ts: datetime, n_bars: int = 60) -> pd.DataFrame:
    """DataFrame ending at current_ts."""
    idx = pd.date_range(end=current_ts, periods=n_bars, freq="1h", tz="UTC")
    closes = [50_000.0] * n_bars
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1.0},
        index=idx,
    )


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _news_item(title: str, minutes_ago: float = 30) -> NewsItem:
    return NewsItem(
        title=title,
        source="test",
        published_at=_now() - timedelta(minutes=minutes_ago),
        url="https://example.com",
        currencies=["BTC"],
    )


class _StaticSource(NewsSource):
    def __init__(self, items: list[NewsItem]) -> None:
        self._items = items

    def fetch(self, symbol: str, limit: int = 50) -> list[NewsItem]:
        return self._items[:limit]


# ---------------------------------------------------------------------------
# NewsSentimentStrategy
# ---------------------------------------------------------------------------


def test_hold_when_no_news():
    source = _StaticSource([])
    strat = NewsSentimentStrategy(
        source=source,
        scorer=KeywordSentimentScorer(),
        symbol=SYMBOL,
    )
    strat.reset()
    sig = strat.next(_df(_now()))
    assert sig.type == SignalType.HOLD


def test_buy_on_bullish_news():
    items = [_news_item("Bitcoin ETF approval surge record adoption", minutes_ago=30)]
    source = _StaticSource(items)
    strat = NewsSentimentStrategy(
        source=source,
        scorer=KeywordSentimentScorer(),
        symbol=SYMBOL,
        ingestion_lag_seconds=60,
        lookback_minutes=120,
        sentiment_threshold=0.1,
        min_confidence=0.1,
    )
    strat.reset()
    sig = strat.next(_df(_now()))
    assert sig.type == SignalType.BUY


def test_sell_on_bearish_news():
    items = [_news_item("Major crypto crash hack exploit liquidation", minutes_ago=30)]
    source = _StaticSource(items)
    strat = NewsSentimentStrategy(
        source=source,
        scorer=KeywordSentimentScorer(),
        symbol=SYMBOL,
        ingestion_lag_seconds=60,
        lookback_minutes=120,
        sentiment_threshold=0.1,
        min_confidence=0.1,
    )
    strat.reset()
    sig = strat.next(_df(_now()))
    assert sig.type == SignalType.SELL


def test_ingestion_lag_excludes_very_recent_news():
    # News published 30 seconds ago should be excluded with a 60s lag
    ts = _now()
    very_recent = NewsItem(
        title="Bitcoin ETF surge approval",
        source="test",
        published_at=ts - timedelta(seconds=30),
        url="",
        currencies=["BTC"],
    )
    source = _StaticSource([very_recent])
    strat = NewsSentimentStrategy(
        source=source,
        scorer=KeywordSentimentScorer(),
        symbol=SYMBOL,
        ingestion_lag_seconds=60,
        lookback_minutes=120,
        sentiment_threshold=0.1,
        min_confidence=0.1,
    )
    sig = strat.next(_df(ts))
    assert sig.type == SignalType.HOLD  # too fresh to act on


def test_lookback_excludes_stale_news():
    # News from 3 hours ago should be outside a 60-minute window
    ts = _now()
    stale = NewsItem(
        title="Bitcoin ETF surge approval",
        source="test",
        published_at=ts - timedelta(hours=3),
        url="",
        currencies=["BTC"],
    )
    source = _StaticSource([stale])
    strat = NewsSentimentStrategy(
        source=source,
        scorer=KeywordSentimentScorer(),
        symbol=SYMBOL,
        ingestion_lag_seconds=60,
        lookback_minutes=60,
        sentiment_threshold=0.1,
        min_confidence=0.0,
    )
    sig = strat.next(_df(ts))
    assert sig.type == SignalType.HOLD


def test_hold_when_news_fetch_fails():
    failing_source = MagicMock(spec=NewsSource)
    failing_source.fetch.side_effect = RuntimeError("API down")
    strat = NewsSentimentStrategy(
        source=failing_source,
        scorer=KeywordSentimentScorer(),
        symbol=SYMBOL,
    )
    sig = strat.next(_df(_now()))
    assert sig.type == SignalType.HOLD  # never propagates exception


def test_reset_is_noop():
    source = _StaticSource([])
    strat = NewsSentimentStrategy(source=source, scorer=KeywordSentimentScorer(), symbol=SYMBOL)
    strat.reset()
    strat.reset()  # must not raise


# ---------------------------------------------------------------------------
# CombinedStrategy
# ---------------------------------------------------------------------------


class _FixedSignal:
    """Returns the same signal type every call."""

    def __init__(self, t: SignalType) -> None:
        from trading_bot.strategies.base import Signal
        self._signal = Signal(t, 50_000.0, pd.Timestamp.now(tz="UTC"), "test")

    def next(self, df: pd.DataFrame):
        return self._signal

    def reset(self): pass


def _news_strat(score: float | None) -> NewsSentimentStrategy:
    """Returns a news strategy whose aggregate_score is always `score`."""
    strat = MagicMock(spec=NewsSentimentStrategy)
    strat.aggregate_score.return_value = score
    strat.reset.return_value = None
    return strat


def test_combined_passes_through_on_neutral_news():
    from trading_bot.strategies.base import Signal, SignalType as ST
    tech = _FixedSignal(ST.BUY)
    news = _news_strat(None)  # no confident signals
    combined = CombinedStrategy(technical=tech, news=news)
    sig = combined.next(_df(_now()))
    assert sig.type == ST.BUY


def test_combined_suppresses_buy_when_news_bearish():
    from trading_bot.strategies.base import SignalType as ST
    tech = _FixedSignal(ST.BUY)
    news = _news_strat(-0.5)  # bearish
    combined = CombinedStrategy(technical=tech, news=news)
    sig = combined.next(_df(_now()))
    assert sig.type == ST.HOLD


def test_combined_suppresses_sell_when_news_bullish():
    from trading_bot.strategies.base import SignalType as ST
    tech = _FixedSignal(ST.SELL)
    news = _news_strat(0.5)  # bullish
    combined = CombinedStrategy(technical=tech, news=news)
    sig = combined.next(_df(_now()))
    assert sig.type == ST.HOLD


def test_combined_passes_buy_when_news_bullish():
    from trading_bot.strategies.base import SignalType as ST
    tech = _FixedSignal(ST.BUY)
    news = _news_strat(0.5)  # bullish agrees with BUY
    combined = CombinedStrategy(technical=tech, news=news)
    sig = combined.next(_df(_now()))
    assert sig.type == ST.BUY


def test_combined_hold_never_consults_news():
    from trading_bot.strategies.base import SignalType as ST
    tech = _FixedSignal(ST.HOLD)
    news = _news_strat(-1.0)
    combined = CombinedStrategy(technical=tech, news=news)
    combined.next(_df(_now()))
    news.aggregate_score.assert_not_called()


def test_combined_require_agreement_false_ignores_news():
    from trading_bot.strategies.base import SignalType as ST
    tech = _FixedSignal(ST.BUY)
    news = _news_strat(-0.9)  # would suppress in default mode
    combined = CombinedStrategy(technical=tech, news=news, require_agreement=False)
    sig = combined.next(_df(_now()))
    assert sig.type == ST.BUY


# ---------------------------------------------------------------------------
# FileNewsSource
# ---------------------------------------------------------------------------


def test_file_news_source(tmp_path):
    import json
    from trading_bot.news.sources.file_source import FileNewsSource

    data = [
        {
            "title": "BTC surges on ETF news",
            "source": "CoinDesk",
            "published_at": "2024-01-15T10:00:00+00:00",
            "url": "https://example.com/1",
            "currencies": ["BTC"],
        },
        {
            "title": "Ethereum upgrade launches",
            "source": "CoinTelegraph",
            "published_at": "2024-01-15T09:00:00+00:00",
            "url": "https://example.com/2",
            "currencies": ["ETH"],
        },
    ]
    f = tmp_path / "news.json"
    f.write_text(json.dumps(data))

    source = FileNewsSource(f)
    items = source.fetch("BTC/USDT")
    assert len(items) == 1
    assert "BTC" in items[0].currencies


def test_file_news_source_filters_by_currency(tmp_path):
    import json
    from trading_bot.news.sources.file_source import FileNewsSource

    data = [
        {"title": "BTC news", "source": "", "published_at": "2024-01-15T10:00:00+00:00",
         "url": "", "currencies": ["BTC"]},
        {"title": "ETH news", "source": "", "published_at": "2024-01-15T10:00:00+00:00",
         "url": "", "currencies": ["ETH"]},
    ]
    f = tmp_path / "news.json"
    f.write_text(json.dumps(data))
    source = FileNewsSource(f)
    assert len(source.fetch("ETH/USDT")) == 1
    assert source.fetch("ETH/USDT")[0].title == "ETH news"
