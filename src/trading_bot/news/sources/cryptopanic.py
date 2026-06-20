import logging
from datetime import datetime

import requests

from .base import NewsItem, NewsSource

logger = logging.getLogger(__name__)

_BASE_URL = "https://cryptopanic.com/api/v1/posts/"


class CryptoPanicSource(NewsSource):
    """Fetches news from the CryptoPanic public API.

    Free tier: 100 requests/day. At 1h bars that's ~4 days of continuous
    paper trading — upgrade to Pro for production use.
    API docs: https://cryptopanic.com/developers/api/
    """

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError(
                "CryptoPanic API key is required. Set CRYPTOPANIC_API_KEY in .env"
            )
        self._api_key = api_key

    def fetch(self, symbol: str, limit: int = 50) -> list[NewsItem]:
        currency = symbol.split("/")[0]  # "BTC/USDT" → "BTC"
        params = {
            "auth_token": self._api_key,
            "currencies": currency,
            "public": "true",
            "kind": "news",
        }
        try:
            resp = requests.get(_BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("CryptoPanic request failed: %s", exc)
            raise

        results = resp.json().get("results", [])[:limit]
        items: list[NewsItem] = []
        for post in results:
            try:
                published_at = datetime.fromisoformat(
                    post["published_at"].replace("Z", "+00:00")
                )
                items.append(
                    NewsItem(
                        title=post["title"],
                        source=post.get("source", {}).get("title", ""),
                        published_at=published_at,
                        url=post.get("url", ""),
                        currencies=[c["code"] for c in post.get("currencies", [])],
                        sentiment_label=post.get("votes", {}).get("sentiment"),
                        raw=post,
                    )
                )
            except (KeyError, ValueError) as exc:
                logger.debug("Skipping malformed news item: %s", exc)

        return items
