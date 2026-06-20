"""File-based news source for backtesting with recorded data.

Record live data with:
    python -m trading_bot.news.record_news --symbol BTC/USDT --output news_data.json

File format — a JSON array of objects:
    [
      {
        "title": "...",
        "source": "CoinDesk",
        "published_at": "2024-01-15T10:30:00+00:00",
        "url": "https://...",
        "currencies": ["BTC"],
        "sentiment_label": "bullish"   (optional)
      },
      ...
    ]

Important for backtesting accuracy: published_at must reflect the actual
publication time, not when you fetched it. The NewsSentimentStrategy applies
its own ingestion_lag on top, so double-lagging is not a concern.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from .base import NewsItem, NewsSource

logger = logging.getLogger(__name__)


class FileNewsSource(NewsSource):
    """Loads pre-recorded news from a JSON file. Items are cached after first load."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._items: list[NewsItem] | None = None

    def _load(self) -> list[NewsItem]:
        if self._items is not None:
            return self._items

        with open(self._path) as f:
            data = json.load(f)

        self._items = []
        for entry in data:
            try:
                self._items.append(
                    NewsItem(
                        title=entry["title"],
                        source=entry.get("source", ""),
                        published_at=datetime.fromisoformat(entry["published_at"]),
                        url=entry.get("url", ""),
                        currencies=entry.get("currencies", []),
                        sentiment_label=entry.get("sentiment_label"),
                        raw=entry,
                    )
                )
            except (KeyError, ValueError) as exc:
                logger.debug("Skipping malformed entry in %s: %s", self._path, exc)

        self._items.sort(key=lambda x: x.published_at, reverse=True)
        logger.info("Loaded %d news items from %s", len(self._items), self._path)
        return self._items

    def fetch(self, symbol: str, limit: int = 50) -> list[NewsItem]:
        currency = symbol.split("/")[0]
        items = self._load()
        matching = [
            item for item in items
            if not item.currencies or currency in item.currencies
        ]
        return matching[:limit]
