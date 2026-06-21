"""Tests for the paper trader's history recording (used by the UI charts)."""

import importlib

import pandas as pd
import pytest


@pytest.fixture
def pt(tmp_path, monkeypatch):
    """Import paper_trader with history paths redirected to a temp dir."""
    import trading_bot.paper_trader as module
    importlib.reload(module)
    monkeypatch.setattr(module, "_HISTORY_PATH", tmp_path / "paper_history.csv")
    monkeypatch.setattr(module, "_ORDERS_PATH", tmp_path / "paper_orders.csv")
    return module


def test_init_writes_headers_and_meta(pt):
    pt._init_history("BTC/USDT", "15m", "supertrend", 10_000.0, 2)
    lines = pt._HISTORY_PATH.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# meta")
    assert "BTC/USDT" in lines[0] and "supertrend" in lines[0]
    assert lines[1] == "timestamp,price,equity,cash,position"
    # orders file has its header
    assert pt._ORDERS_PATH.read_text(encoding="utf-8").splitlines()[0] == \
        "timestamp,side,price,size,fee"


def test_record_point_appends_rows(pt):
    pt._init_history("BTC/USDT", "15m", "ema_crossover", 10_000.0, 1)
    ts = pd.Timestamp("2024-01-01T00:00:00Z")
    pt._record_point(ts, 50_000.0, 10_050.0, 9_000.0, 0.02)
    df = pd.read_csv(pt._HISTORY_PATH, skiprows=1)
    assert len(df) == 1
    assert df["price"].iloc[0] == pytest.approx(50_000.0)
    assert df["equity"].iloc[0] == pytest.approx(10_050.0)


def test_record_order_appends_rows(pt):
    pt._init_history("BTC/USDT", "15m", "ema_crossover", 10_000.0, 1)
    ts = pd.Timestamp("2024-01-01T00:00:00Z")
    pt._record_order(ts, "buy", 50_000.0, 0.02, 1.0)
    pt._record_order(ts, "sell", 50_500.0, 0.02, 1.0)
    df = pd.read_csv(pt._ORDERS_PATH)
    assert list(df["side"]) == ["buy", "sell"]
    assert df["price"].iloc[1] == pytest.approx(50_500.0)


def test_init_resets_previous_session(pt):
    pt._init_history("BTC/USDT", "15m", "ema_crossover", 10_000.0, 1)
    pt._record_point(pd.Timestamp("2024-01-01T00:00:00Z"), 1.0, 1.0, 1.0, 0.0)
    # Re-init should wipe the old rows
    pt._init_history("ETH/USDT", "1h", "macd", 5_000.0, 1)
    df = pd.read_csv(pt._HISTORY_PATH, skiprows=1)
    assert len(df) == 0
