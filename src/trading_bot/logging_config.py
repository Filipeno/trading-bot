import logging
import logging.handlers
from pathlib import Path

_LOG_DIR = Path("logs")
_TRADE_LOG = _LOG_DIR / "trades.log"


def setup_logging(level: int = logging.INFO) -> None:
    _LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    root = logging.getLogger()
    if root.handlers:
        return  # already configured
    root.setLevel(level)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    # Append-only rotating file (10 MB, 5 backups = ~50 MB max)
    file_handler = logging.handlers.RotatingFileHandler(
        _TRADE_LOG, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
