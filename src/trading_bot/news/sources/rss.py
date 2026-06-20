"""Free RSS-based news source — no API key required.

Uses requests (already a project dependency) to fetch feeds with a timeout,
then feedparser to parse RSS/Atom. Individual feed failures are non-fatal.
"""

import logging
import socket
from datetime import datetime, timezone

import feedparser
import requests

from .base import NewsItem, NewsSource

logger = logging.getLogger(__name__)

# Reliable free RSS feeds from major crypto news outlets.
# All publicly accessible — no login or API key needed.
BUILTIN_FEEDS: dict[str, str] = {
    "CoinDesk":        "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "CoinTelegraph":   "https://cointelegraph.com/rss",
    "Decrypt":         "https://decrypt.co/feed",
    "Bitcoin Magazine":"https://bitcoinmagazine.com/feed",
    "The Block":       "https://www.theblock.co/rss.xml",
}

# Terms that indicate an article is relevant to crypto in general
_CRYPTO_TERMS = {"bitcoin", "btc", "crypto", "blockchain", "defi", "web3", "altcoin", "ethereum"}


class RSSNewsSource(NewsSource):
    """Aggregates news from multiple RSS feeds. No API key. No rate limits.

    Args:
        feed_urls: Override the default list of feed URLs.
        timeout:   Per-feed HTTP timeout in seconds.
    """

    def __init__(
        self,
        feed_urls: list[str] | None = None,
        timeout: int = 8,
    ) -> None:
        self._feed_urls: list[str] = (
            feed_urls if feed_urls is not None else list(BUILTIN_FEEDS.values())
        )
        self._timeout = timeout

    def fetch(self, symbol: str, limit: int = 50) -> list[NewsItem]:
        currency = symbol.split("/")[0].lower()  # "BTC/USDT" → "btc"
        all_items: list[NewsItem] = []

        for url in self._feed_urls:
            try:
                items = self._fetch_feed(url, currency)
                all_items.extend(items)
            except Exception as exc:
                logger.warning("RSS feed skipped (%s): %s", url, exc)

        all_items.sort(key=lambda x: x.published_at, reverse=True)
        return all_items[:limit]

    def _fetch_feed(self, url: str, currency: str) -> list[NewsItem]:
        resp = requests.get(
            url,
            timeout=self._timeout,
            headers={"User-Agent": "trading-bot/1.0 (educational project)"},
        )
        resp.raise_for_status()

        feed = feedparser.parse(resp.content)
        feed_title = feed.feed.get("title", url)

        items: list[NewsItem] = []
        for entry in feed.entries:
            title: str = entry.get("title", "").strip()
            if not title:
                continue

            # Filter: must mention this currency or crypto broadly
            text = (title + " " + entry.get("summary", "")).lower()
            if currency not in text and not _CRYPTO_TERMS.intersection(text.split()):
                continue

            published_at = _parse_date(entry)
            items.append(
                NewsItem(
                    title=title,
                    source=feed_title,
                    published_at=published_at,
                    url=entry.get("link", ""),
                    currencies=[currency.upper()],
                )
            )
        return items


def _parse_date(entry: dict) -> datetime:
    parsed = entry.get("published_parsed")
    if parsed:
        try:
            return datetime(*parsed[:6], tzinfo=timezone.utc)
        except (TypeError, ValueError):
            pass
    return datetime.now(tz=timezone.utc)
