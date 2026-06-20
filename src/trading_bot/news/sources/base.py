from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class NewsItem:
    title: str
    source: str
    published_at: datetime   # must be timezone-aware
    url: str
    currencies: list[str] = field(default_factory=list)
    sentiment_label: str | None = None  # pre-tagged by source (if available)
    raw: dict = field(default_factory=dict)


class NewsSource(ABC):
    """Pluggable news data provider.

    Add new sources (exchange announcement feeds, general news APIs) by
    subclassing this and implementing fetch(). The rest of the system only
    talks to this interface.
    """

    @abstractmethod
    def fetch(self, symbol: str, limit: int = 50) -> list[NewsItem]:
        """Return recent news items for the given trading symbol (e.g. 'BTC/USDT').

        Must return items in reverse-chronological order (newest first).
        Implementations are responsible for their own rate limiting.
        """
