"""
Crypto Trading Bot — Streamlit UI

Launch with:
    streamlit run ui/app.py
"""

import os
import platform
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import psutil
import streamlit as st
from dotenv import dotenv_values, load_dotenv

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
CONFIG_PATH = ROOT / "config" / "settings.toml"
ENV_PATH = ROOT / ".env"
PID_FILE = ROOT / ".trader.pid"
LOG_FILE = ROOT / "logs" / "trades.log"

sys.path.insert(0, str(SRC))

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Trading Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stMetric { background: #1e1e2e; padding: 12px; border-radius: 8px; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Utilities ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def _load_config() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def _flag(name: str) -> bool:
    if os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    try:
        return str(st.secrets.get(name, "")).strip().lower() in ("1", "true", "yes", "on")
    except Exception:
        return False


# Hosted/shared mode: each visitor brings their OWN keys, held only in their
# browser session (never written to disk, never shared). Live paper trading —
# which needs a server-side subprocess + on-disk keys — is disabled here.
HOSTED = _flag("TRADING_BOT_HOSTED") or _flag("TRADING_BOT_DEMO")


def _session_keys() -> dict:
    return st.session_state.setdefault("byok", {})


def _get_key(name: str) -> str:
    """Resolve an API key: session (BYOK) first, then local .env (local use only)."""
    sk = _session_keys().get(name)
    if sk:
        return sk
    if HOSTED:
        return ""   # never read shared/server keys in hosted mode
    return _load_env().get(name, "") or os.getenv(name, "")


def _set_session_key(name: str, value: str) -> None:
    _session_keys()[name] = value


def _load_env() -> dict:
    return dotenv_values(ENV_PATH) if ENV_PATH.exists() else {}


def _save_env(values: dict) -> None:
    existing = _load_env()
    existing.update({k: v for k, v in values.items() if v})
    ENV_PATH.write_text(
        "\n".join(f'{k}="{v}"' for k, v in existing.items()) + "\n"
    )
    load_dotenv(ENV_PATH, override=True)


def _trader_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text().strip())
    except ValueError:
        return None


def _trader_is_running() -> bool:
    pid = _trader_pid()
    if pid is None:
        return False
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        PID_FILE.unlink(missing_ok=True)
        return False


def _start_trader() -> None:
    load_dotenv(ENV_PATH, override=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "trading_bot.paper_trader"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
    )
    PID_FILE.write_text(str(proc.pid))


def _stop_trader() -> None:
    pid = _trader_pid()
    if pid is None:
        return
    try:
        proc = psutil.Process(pid)
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        pass
    PID_FILE.unlink(missing_ok=True)


def _tail_log(n: int = 50) -> list[str]:
    if not LOG_FILE.exists():
        return []
    return LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]


OVERRIDES_PATH = ROOT / "config" / "ui_overrides.json"
PAPER_HISTORY = ROOT / "logs" / "paper_history.csv"
PAPER_ORDERS = ROOT / "logs" / "paper_orders.csv"


def _read_paper_history():
    """Return (history_df, orders_df, meta) the running paper trader recorded."""
    import csv as _csv
    if not PAPER_HISTORY.exists():
        return None, None, {}
    meta = {}
    try:
        with open(PAPER_HISTORY, encoding="utf-8") as f:
            first = f.readline()
        if first.startswith("# meta"):
            parts = list(_csv.reader([first]))[0]
            # ["# meta", symbol, timeframe, strategy, initial_capital, leverage]
            if len(parts) >= 6:
                meta = {"symbol": parts[1], "timeframe": parts[2], "strategy": parts[3],
                        "initial_capital": float(parts[4]), "leverage": int(parts[5])}
        hist = pd.read_csv(PAPER_HISTORY, skiprows=1)
        if not hist.empty:
            hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True, errors="coerce")
            hist = hist.dropna(subset=["timestamp"])
        orders = None
        if PAPER_ORDERS.exists():
            orders = pd.read_csv(PAPER_ORDERS)
            if not orders.empty:
                orders["timestamp"] = pd.to_datetime(orders["timestamp"], utc=True, errors="coerce")
                orders = orders.dropna(subset=["timestamp"])
        return hist, orders, meta
    except Exception:
        return None, None, meta


def _write_overrides(overrides: dict) -> None:
    """Write the UI's chosen strategy/risk settings for the paper trader to read."""
    import json
    OVERRIDES_PATH.write_text(json.dumps(overrides, indent=2), encoding="utf-8")


def _read_overrides() -> dict:
    import json
    if OVERRIDES_PATH.exists():
        try:
            return json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _leverage_warning(leverage: int, stop_loss_pct: float) -> None:
    if leverage > 1:
        capital_at_risk = stop_loss_pct * leverage * 100
        liq_pct = 100 / leverage
        st.warning(
            f"⚠️ **Leverage {leverage}×** — a {stop_loss_pct*100:.0f}% price move triggers your stop-loss, "
            f"costing **{capital_at_risk:.0f}% of your margin**. "
            f"Liquidation occurs at a **{liq_pct:.0f}% adverse move**. "
            f"Only use leverage if you fully understand these risks."
        )


# Allowed timeframes per asset class (stocks have no 4h on yfinance).
_CRYPTO_TFS = ["1m", "5m", "15m", "1h", "4h", "1d"]
_STOCK_TFS = ["1m", "5m", "15m", "30m", "1h", "1d"]


def _timeframes_for(asset: str) -> list[str]:
    return _STOCK_TFS if asset == "stocks" else _CRYPTO_TFS


def _current_asset() -> str:
    return st.session_state.get("asset_class", "crypto")


def _current_symbol() -> str:
    return st.session_state.get("active_symbol") or "BTC/USDT"


def _effective_config(base: dict) -> dict:
    """Overlay the UI's asset-class + symbol choices onto the loaded config."""
    cfg = dict(base)
    cfg["market"] = {**base.get("market", {}), "asset_class": _current_asset()}
    cfg["exchange"] = {**base.get("exchange", {}), "symbol": _current_symbol()}
    return cfg


# ── Symbol & model lookups (cached) ─────────────────────────────────────────

_STOCK_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX", "AMD", "INTC",
    "JPM", "V", "WMT", "DIS", "KO", "PEP", "BA", "XOM", "PLTR", "COIN",
    "SPY", "QQQ", "DIA", "IWM",
]
_MAMMOUTH_MODELS_URL = "https://api.mammouth.ai/public/models"
_MAMMOUTH_FALLBACK = [
    "gpt-4o", "gpt-5.4-mini", "claude-haiku-4-5", "claude-sonnet-4-6",
    "gemini-2.5-flash", "mistral-large-3",
]
_ANTHROPIC_MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-8"]


@st.cache_data(ttl=3600, show_spinner=False)
def _crypto_symbols() -> list[str]:
    """All tradable pairs from the configured exchange (searchable in the dropdown)."""
    try:
        from trading_bot.data.fetcher import make_exchange
        ex = make_exchange(_load_config())
        markets = ex.load_markets()
        syms = sorted(markets.keys())
        common = [s for s in ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
                              "XRP/USDT", "ADA/USDT", "DOGE/USDT"] if s in markets]
        return common + [s for s in syms if s not in common]
    except Exception:
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]


@st.cache_data(ttl=3600, show_spinner=False)
def _mammouth_models() -> list[str]:
    """Live chat-model list from Mammouth's public endpoint (embeddings/image filtered)."""
    try:
        import requests
        r = requests.get(_MAMMOUTH_MODELS_URL, timeout=10)
        r.raise_for_status()
        ids = [m["id"] for m in r.json().get("data", []) if isinstance(m, dict) and "id" in m]
        ids = [i for i in ids if "embedding" not in i and "image" not in i]
        return sorted(ids) if ids else _MAMMOUTH_FALLBACK
    except Exception:
        return _MAMMOUTH_FALLBACK


# ── Header ─────────────────────────────────────────────────────────────────

col_title, col_status = st.columns([4, 1])
with col_title:
    st.title("🤖 Trading Bot")
with col_status:
    running = False if HOSTED else _trader_is_running()
    if HOSTED:
        st.info("🌐 Hosted")
    elif running:
        st.success("🟢 Running")
    else:
        st.info("⚫ Stopped")

if HOSTED:
    st.info(
        "🌐 **Shared demo.** Backtest, Reality Check, and News work with **no keys**. "
        "To try the **LLM**, paste your own key in the Setup tab — it stays in *your* "
        "browser session only, is never saved on the server, and disappears when you "
        "close the tab. Live paper trading is local-only (clone the repo to run it).",
        icon="🌐",
    )

# ── Global market selector (applies to every tab) ───────────────────────────
_cfg0 = _load_config()
_mk1, _mk2, _mk3 = st.columns([1, 1, 2])
with _mk1:
    st.radio(
        "Market", ["crypto", "stocks"],
        horizontal=True,
        index=0 if _cfg0.get("market", {}).get("asset_class", "crypto") == "crypto" else 1,
        key="asset_class",
        help="Crypto uses ccxt (Binance/Kraken). Stocks use free yfinance data.",
    )
with _mk2:
    _asset_now = _current_asset()
    if _asset_now == "stocks":
        _sym_options = _STOCK_SYMBOLS + ["✏️ Custom…"]
        _default_sym = "AAPL"
    else:
        _sym_options = _crypto_symbols() + ["✏️ Custom…"]
        _default_sym = _cfg0.get("exchange", {}).get("symbol", "BTC/USDT")
    _idx = _sym_options.index(_default_sym) if _default_sym in _sym_options else 0
    _pick = st.selectbox(
        "Symbol", _sym_options, index=_idx, key=f"sympick_{_asset_now}",
        help="Type to search. Pick ✏️ Custom to enter any symbol manually.",
    )
    if _pick == "✏️ Custom…":
        _custom = st.text_input(
            "Custom symbol", key=f"symcustom_{_asset_now}",
            placeholder="AAPL" if _asset_now == "stocks" else "BTC/USDT",
        )
        _sym = (_custom or "").strip()
        if _asset_now == "stocks":
            _sym = _sym.upper()
        st.session_state["active_symbol"] = _sym or _default_sym
    else:
        st.session_state["active_symbol"] = _pick
with _mk3:
    if _current_asset() == "stocks":
        st.caption("📈 **Stocks mode** — free delayed data via yfinance "
                   "(`pip install yfinance`). Backtest & Reality Check work fully; "
                   "paper trading uses ~15-min-delayed quotes.")
    else:
        st.caption("🪙 **Crypto mode** — live data via ccxt.")

tab_setup, tab_backtest, tab_reality, tab_paper, tab_news = st.tabs([
    "⚙️  Setup",
    "📊  Backtest",
    "🔬  Reality Check",
    "🤖  Paper Trade",
    "📰  News Feed",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SETUP
# ══════════════════════════════════════════════════════════════════════════════
with tab_setup:
    st.header("Setup")

    cfg = _load_config()
    exc_cfg = cfg["exchange"]
    key_env_name    = exc_cfg.get("api_key_env", "BINANCE_TESTNET_API_KEY")
    secret_env_name = exc_cfg.get("secret_env",  "BINANCE_TESTNET_SECRET")
    exchange_id     = exc_cfg.get("id", "binance")
    is_testnet      = exc_cfg.get("testnet", True)

    env = _load_env()
    has_key    = bool(env.get(key_env_name))
    has_secret = bool(env.get(secret_env_name))

    # ── Exchange instructions ─────────────────────────────────────────────
    if exchange_id == "binance" and is_testnet:
        guide_title = "**Step 1 — Get free Binance Testnet keys** (click to expand)"
        guide_body = """
Binance Testnet is completely free — you get fake money to trade with and can't lose anything real.

**How to get your keys in 2 minutes:**
1. Go to **[testnet.binance.vision](https://testnet.binance.vision)**
2. Click **"Log In with GitHub"**
3. Click **"Generate HMAC_SHA256 Key"**
4. Copy both the **API Key** and **Secret Key**

> ⚠️ The secret key is only shown once. Copy it immediately.
"""
    elif exchange_id == "kraken":
        guide_title = "**Step 1 — Get Kraken API keys** (click to expand)"
        guide_body = """
**How to get your Kraken API keys:**
1. Log in to **[kraken.com](https://www.kraken.com)**
2. Go to **Settings → API → Create API key**
3. Name it (e.g. "trading-bot"), set permissions to **Query + Trade** only
4. Copy both the **API Key** and **Private Key**

> ⚠️ Only grant the minimum permissions needed — Query Funds + Create & Modify Orders.
> Never enable withdrawals on a bot API key.
"""
    else:
        guide_title = f"**Step 1 — Get {exchange_id.title()} API keys** (click to expand)"
        guide_body = f"""
Create an API key on **{exchange_id.title()}** with trading permissions and paste them below.
"""

    with st.expander(guide_title, expanded=not has_key):
        st.markdown(guide_body)

    # ── Step 2: Enter keys ────────────────────────────────────────────────
    st.subheader("Step 2 — Enter your API keys")

    if HOSTED:
        st.info("🌐 Exchange keys aren't needed here — crypto data is public and live "
                "trading is local-only. To try the **LLM**, scroll to the LLM section below "
                "and paste your own key (session-only). Or just use **Backtest** / **Reality Check**.")

    col_l, col_r = st.columns(2)
    with col_l:
        api_key_input = st.text_input(
            f"API Key  (`{key_env_name}`)",
            value=env.get(key_env_name, ""),
            type="password",
            placeholder="Paste API key here…",
            disabled=HOSTED,
        )
    with col_r:
        secret_input = st.text_input(
            f"Secret  (`{secret_env_name}`)",
            value=env.get(secret_env_name, ""),
            type="password",
            placeholder="Paste secret here…",
            disabled=HOSTED,
        )

    # ── Step 3: Save & test ───────────────────────────────────────────────
    st.subheader("Step 3 — Save and test")

    col_save, col_test, _ = st.columns([1, 1, 2])
    with col_save:
        if st.button("💾 Save Keys", width="stretch", disabled=HOSTED):
            if not api_key_input or not secret_input:
                st.error("Please enter both the API key and secret.")
            else:
                _save_env({
                    key_env_name: api_key_input,
                    secret_env_name: secret_input,
                })
                st.success("Keys saved to `.env`")
                st.rerun()

    with col_test:
        if st.button("🔌 Test Connection", width="stretch", type="primary"):
            with st.spinner("Connecting…"):
                try:
                    from trading_bot.data.market import make_feed
                    cfg = _effective_config(_load_config())
                    feed = make_feed(
                        cfg,
                        api_key=env.get(key_env_name, ""),
                        secret=env.get(secret_env_name, ""),
                    )
                    sym = _current_symbol()
                    tf = "1h" if _current_asset() == "crypto" else "1d"
                    df = feed.recent(sym, tf, limit=3)
                    price = df["close"].iloc[-1]
                    st.success(f"✅ Connected!  {sym} = **${price:,.2f}**")
                except Exception as exc:
                    st.error(f"❌ Failed: {exc}")

    # ── Current status ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Current status")

    c1, c2, c3 = st.columns(3)
    c1.metric("API Key", "✅ Saved" if has_key else "❌ Missing")
    c2.metric("Secret", "✅ Saved" if has_secret else "❌ Missing")
    c3.metric("Paper Trader", "🟢 Running" if _trader_is_running() else "⚫ Stopped")

    if has_key and has_secret:
        st.success("✅ All set — head to the **Backtest** or **Paper Trade** tabs.")
    else:
        st.warning("Complete Steps 1–3 above before running a backtest or paper trading.")

    # ── Optional: LLM strategy key ────────────────────────────────────────
    st.divider()
    with st.expander("🤖 Optional — LLM strategy (let a model read the chart)"):
        st.markdown(
            "The **LLM strategy** sends the chart numbers to a model each bar and acts on "
            "its BUY/SELL/HOLD call. Pick your provider and paste the matching key.\n\n"
            "> ⚠️ **Costs API usage** — one call per bar. It will *not* reliably beat the "
            "market. Validate with the **Reality Check** and paper-trade only."
        )
        llm_provider_cfg = _cfg0.get("strategy", {}).get("llm_provider", "mammouth")
        provider = st.radio(
            "Provider", ["mammouth", "anthropic"],
            index=0 if llm_provider_cfg != "anthropic" else 1,
            horizontal=True,
        )
        if provider == "mammouth":
            st.caption(
                "Mammouth.ai is OpenAI-compatible. Pick the model to run in the "
                "**Paper Trade** tab — the live list below comes straight from your "
                "Mammouth account's public models endpoint."
            )
            _setup_models = _mammouth_models()
            st.selectbox(
                f"Available models ({len(_setup_models)})",
                _setup_models,
                index=_setup_models.index("gpt-4o") if "gpt-4o" in _setup_models else 0,
                key="setup_model_browse",
                help="Browse only — the model used is chosen in the Paper Trade tab.",
            )
            key_name = "MAMMOUTH_API_KEY"
            placeholder = "your Mammouth API key"
        else:
            st.caption("Get a key at [console.anthropic.com](https://console.anthropic.com/) "
                       "and `pip install anthropic`. Set `llm_provider = \"anthropic\"`.")
            key_name = "ANTHROPIC_API_KEY"
            placeholder = "sk-ant-…"

        if HOSTED:
            st.info(
                "🔒 **Your key stays in this browser session only.** It is never written to "
                "the server's disk and is gone when you close the tab. Other visitors can't "
                "see or use it."
            )

        llm_key_input = st.text_input(
            f"API Key  (`{key_name}`)",
            value=_session_keys().get(key_name, "") if HOSTED else env.get(key_name, ""),
            type="password",
            placeholder=placeholder,
        )
        if st.button("💾 Save LLM Key"):
            if not llm_key_input:
                st.error("Enter a key first.")
            elif HOSTED:
                _set_session_key(key_name, llm_key_input)
                st.success(f"`{key_name}` set for this session. Try it in the **Backtest** tab "
                           "(🤖 LLM read this chart) — uses your key, in-process, no live trading.")
            else:
                _save_env({key_name: llm_key_input})
                st.success(f"Saved `{key_name}`. Select the **LLM** strategy in the Paper Trade tab to use it.")
                st.rerun()
        _have = bool(_session_keys().get(key_name)) if HOSTED else bool(env.get(key_name))
        st.caption(f"Status: {'✅ key set' if _have else '❌ no key yet'}"
                   f"{' (session only)' if HOSTED else ''}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
from strategy_specs import (
    BACKTESTABLE,
    METRIC_GLOSSARY,
    OPTIMIZABLE,
    PAPER_TRADE,
    STRATEGY_DESC,
    STRATEGY_LABELS as _STRATEGY_LABELS,
    render_params,
    short_label,
    validate,
)

_TIMEFRAME_LABELS = {
    "1m": "1 minute", "5m": "5 minutes", "15m": "15 minutes",
    "1h": "1 hour", "4h": "4 hours", "1d": "1 day",
}

with tab_backtest:
    st.header("Backtest")
    st.caption(
        "Uses public Binance OHLCV data — no API key needed. Pick any strategy, "
        "tune its parameters, and test it on one slice of history."
    )

    ctrl_col, chart_col = st.columns([1, 2], gap="large")

    with ctrl_col:
        # ── Strategy selector ─────────────────────────────────────────────
        strategy_key = st.selectbox(
            "Strategy",
            options=BACKTESTABLE,   # LLM excluded — a backtest = one paid call per candle
            format_func=lambda k: _STRATEGY_LABELS[k],
            index=0,
            help="The rule that decides when to buy and sell. Each one is explained below.",
        )
        st.caption(f"ℹ️ {STRATEGY_DESC.get(strategy_key, '')}")

        _bt_tfs = _timeframes_for(_current_asset())
        timeframe = st.selectbox(
            "Timeframe",
            options=_bt_tfs,
            index=_bt_tfs.index("15m") if "15m" in _bt_tfs else 0,
            format_func=lambda tf: _TIMEFRAME_LABELS.get(tf, tf),
            help="How much time each candle covers. Shorter = more trades & noise; "
                 "longer = slower, steadier signals.",
        )

        candle_default = 500 if timeframe in ("1m", "5m", "15m") else 1000
        n_candles = st.slider(
            "History (candles)", 100, 5000, candle_default, 100,
            help=f"How far back to test. Each candle = {_TIMEFRAME_LABELS.get(timeframe, timeframe)}. "
                 "More candles = a longer, more trustworthy test.",
        )

        # ── Starting budget (prominent) ───────────────────────────────────
        capital = st.number_input(
            "💰 Starting budget ($)", min_value=100, max_value=1_000_000,
            value=10_000, step=100,
            help="The pretend cash you begin with. All results scale from this — "
                 "try your real intended amount to see realistic dollar outcomes.",
        )

        # ── Strategy-specific params (dynamic) ────────────────────────────
        st.markdown("**Strategy parameters**")
        st.caption("Fine-tune the strategy. Hover the ⓘ on each for what it does.")
        bt_params = render_params(st, strategy_key, "bt")
        can_run, msg = validate(strategy_key, bt_params)
        if not can_run:
            st.warning(msg)
        strategy_label = short_label(strategy_key, bt_params)

        # ── Leverage ──────────────────────────────────────────────────────
        st.markdown("**Leverage & exits**")
        leverage = st.slider(
            "Leverage", 1, 10, 1, key="bt_lev",
            help="1× = spot (you can't lose more than you put in). Higher multiplies BOTH "
                 "gains and losses — and adds a liquidation risk. Keep at 1 unless you know why.",
        )

        with st.expander("Exit rules & costs (advanced)"):
            sl_pct    = st.number_input(
                "Stop-loss %", 0.5, 20.0, 2.0, 0.5, format="%.1f",
                help="Auto-sell if price falls this far below entry — caps each loss.") / 100
            tp_pct    = st.number_input(
                "Take-profit %", 0.5, 40.0, 4.0, 0.5, format="%.1f",
                help="Auto-sell once price rises this far above entry — locks in a win.") / 100
            trail_pct = st.number_input(
                "Trailing stop % (0 = off)", 0.0, 20.0, 0.0, 0.5, format="%.1f",
                help="Sell if price drops this far from its highest point since you bought. "
                     "Lets winners run while protecting gains.") / 100
            fee       = st.number_input(
                "Fee rate", 0.0, 0.01, 0.001, 0.0001, format="%.4f",
                help="Trading fee per order. 0.001 = 0.1% (typical exchange taker fee).")
            slip      = st.number_input(
                "Slippage", 0.0, 0.005, 0.0005, 0.0001, format="%.4f",
                help="Realistic gap between expected and actual fill price.")

        _leverage_warning(leverage, sl_pct)

        run_btn = st.button(
            "▶ Run Backtest", type="primary", width="stretch",
            disabled=not can_run,
        )

        # ── 🤖 LLM read this chart (in-process, BYOK) ─────────────────────
        with st.expander("🤖 Ask an LLM about this chart"):
            st.caption("One call to a model on the latest bars — uses your own key, "
                       "in-process (no live trading). Great for trying BYOK safely.")
            _llm_prov = st.selectbox("Provider", ["mammouth", "anthropic"], key="bt_llm_prov")
            if _llm_prov == "mammouth":
                _m = _mammouth_models()
                _llm_model = st.selectbox("Model", _m,
                                          index=_m.index("gpt-4o") if "gpt-4o" in _m else 0,
                                          key="bt_llm_model_m")
                _llm_keyname = "MAMMOUTH_API_KEY"
            else:
                _llm_model = st.selectbox("Model", _ANTHROPIC_MODELS, key="bt_llm_model_a")
                _llm_keyname = "ANTHROPIC_API_KEY"
            _llm_key = _get_key(_llm_keyname)
            if not _llm_key:
                st.caption(f"⚠️ No `{_llm_keyname}` — add it in the Setup tab "
                           f"({'session-only' if HOSTED else '.env'}).")
            ask_llm = st.button("🤖 Analyze latest bars", width="stretch",
                                disabled=not _llm_key)

    # ── Results ───────────────────────────────────────────────────────────
    with chart_col:
        if ask_llm and _llm_key:
            with st.spinner(f"Asking {_llm_model} to read the {_current_symbol()} chart…"):
                try:
                    from trading_bot.data.market import make_feed
                    from trading_bot.strategies.llm_strategy import (
                        AnthropicClient, LLMStrategy, OpenAICompatibleClient,
                    )

                    cfg = _effective_config(_load_config())
                    feed = make_feed(cfg)
                    df_llm = feed.history(_current_symbol(), timeframe, total=60)

                    if _llm_prov == "mammouth":
                        client = OpenAICompatibleClient(
                            base_url="https://api.mammouth.ai/v1",
                            model=_llm_model, api_key=_llm_key,
                        )
                    else:
                        client = AnthropicClient(model=_llm_model, api_key=_llm_key)

                    sig = LLMStrategy(client=client, min_confidence=0.0).next(df_llm)
                    _emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(sig.type.value, "⚪")
                    st.markdown(f"### {_emoji} {sig.type.value} — {_current_symbol()}")
                    st.write(sig.reason)
                    st.caption(f"Model: {_llm_model} · latest price {df_llm['close'].iloc[-1]:,.2f} · "
                               "one call, in-process. Not advice — the LLM has no edge.")
                except Exception as exc:
                    st.error(f"LLM call failed: {exc}")
                    st.caption("If on Mammouth, double-check the base URL/model in your dashboard.")

        if run_btn and can_run:
            with st.spinner(f"Fetching {n_candles} × {timeframe} candles and running backtest…"):
                try:
                    from trading_bot.backtest.engine import BacktestEngine
                    from trading_bot.data.market import make_feed
                    from trading_bot.strategies.factory import make_strategy

                    cfg = _effective_config(_load_config())
                    feed = make_feed(cfg)   # crypto: public ccxt | stocks: yfinance
                    df = feed.history(_current_symbol(), timeframe, total=n_candles)

                    # Build strategy from the dynamic params via the factory.
                    strat = make_strategy({"strategy": {"name": strategy_key, **bt_params}})

                    engine = BacktestEngine(
                        initial_capital=capital,
                        fee_rate=fee,
                        slippage_pct=slip,
                        leverage=leverage,
                        timeframe=timeframe,
                        stop_loss_pct=sl_pct,
                        take_profit_pct=tp_pct,
                        trailing_stop_pct=trail_pct,
                    )
                    result = engine.run(df, strat)
                    m = result.metrics

                    # ── Headline result ────────────────────────────────────
                    final_val = m["final_equity"]
                    profit = final_val - capital
                    bh_ret = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
                    st.markdown(
                        f"### {'🟢' if profit >= 0 else '🔴'} ${capital:,.0f} → "
                        f"**${final_val:,.0f}**  ({m['total_return_pct']:+.1f}%)"
                    )
                    st.caption(
                        f"Buy & hold over the same period: {bh_ret:+.1f}% · "
                        f"{'✅ strategy beat holding' if m['total_return_pct'] > bh_ret else '⚠️ just holding did better'}"
                    )

                    # Metrics row + glossary
                    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                    mc1.metric("Return", f"{m['total_return_pct']:+.1f}%")
                    mc2.metric("Sharpe", f"{m['sharpe_ratio']:.2f}")
                    mc3.metric("Max Drawdown", f"{m['max_drawdown_pct']:.1f}%")
                    mc4.metric("Win Rate", f"{m['win_rate_pct']:.0f}%")
                    mc5.metric("# Trades", str(m['n_trades']))
                    with st.expander("❓ What do these numbers mean?"):
                        for _k, _v in METRIC_GLOSSARY.items():
                            st.markdown(f"- **{_k}** — {_v}")

                    lev_label = f" · {leverage}×" if leverage > 1 else ""

                    # ── Price chart with trades marked ─────────────────────
                    price_fig = go.Figure()
                    price_fig.add_trace(go.Candlestick(
                        x=df.index, open=df["open"], high=df["high"],
                        low=df["low"], close=df["close"], name=_current_symbol(),
                        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                        showlegend=False,
                    ))
                    if result.trades:
                        price_fig.add_trace(go.Scatter(
                            x=[t.entry_time for t in result.trades],
                            y=[t.entry_price for t in result.trades],
                            mode="markers", name="Buy",
                            marker=dict(symbol="triangle-up", size=12, color="#22cc88",
                                        line=dict(width=1, color="#0b3")),
                            hovertext=[f"Buy @ {t.entry_price:,.2f}" for t in result.trades],
                        ))
                        _win = lambda t: t.pnl > 0
                        price_fig.add_trace(go.Scatter(
                            x=[t.exit_time for t in result.trades],
                            y=[t.exit_price for t in result.trades],
                            mode="markers", name="Sell",
                            marker=dict(symbol="triangle-down", size=12,
                                        color=["#ffd166" if _win(t) else "#ff6b6b" for t in result.trades],
                                        line=dict(width=1, color="#900")),
                            hovertext=[f"Sell @ {t.exit_price:,.2f} ({t.pnl:+.0f}$)" for t in result.trades],
                        ))
                    price_fig.update_layout(
                        title=f"{_current_symbol()} price & trades — {strategy_label}{lev_label}",
                        height=380, margin=dict(l=0, r=0, t=36, b=0),
                        xaxis_rangeslider_visible=False,
                        legend=dict(orientation="h", y=1.12),
                        hovermode="x unified",
                    )
                    st.plotly_chart(price_fig, width="stretch")
                    st.caption("🔺 green = buy · 🔻 yellow = winning sell · 🔻 red = losing sell")

                    # ── Portfolio value curve ──────────────────────────────
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=result.equity_curve.index,
                        y=result.equity_curve.values,
                        mode="lines",
                        name="Portfolio ($)",
                        line=dict(color="#00d4aa", width=2),
                        fill="tozeroy",
                        fillcolor="rgba(0,212,170,0.07)",
                    ))
                    fig.add_hline(y=capital, line_dash="dot", line_color="#888",
                                  annotation_text="starting budget", annotation_position="bottom right")
                    fig.update_layout(
                        title="Portfolio value over time",
                        yaxis_title="Portfolio ($)",
                        height=300,
                        margin=dict(l=0, r=0, t=36, b=0),
                        hovermode="x unified",
                        showlegend=False,
                    )
                    st.plotly_chart(fig, width="stretch")

                    # Trade table
                    if result.trades:
                        st.caption(f"All {len(result.trades)} trades")
                        _reason_label = {
                            "signal": "📊 signal", "stop_loss": "🛑 stop-loss",
                            "take_profit": "🎯 take-profit", "trailing_stop": "📉 trailing",
                            "liquidation": "💀 liquidated",
                        }
                        trade_rows = [
                            {
                                "Entry": t.entry_time.strftime("%b %d %H:%M"),
                                "Exit": t.exit_time.strftime("%b %d %H:%M"),
                                "Entry $": f"{t.entry_price:,.0f}",
                                "Exit $": f"{t.exit_price:,.0f}",
                                "P&L $": f"{t.pnl:+.2f}",
                                "P&L %": f"{t.pnl_pct:+.2%}",
                                "Closed by": _reason_label.get(getattr(t, "exit_reason", "signal"), "📊 signal"),
                                "Result": "✅ Win" if t.pnl > 0 else "❌ Loss",
                            }
                            for t in result.trades
                        ]
                        st.dataframe(
                            pd.DataFrame(trade_rows),
                            hide_index=True,
                            width="stretch",
                        )
                    else:
                        st.info("No trades were generated in this period.")

                except Exception as exc:
                    st.error(f"Backtest error: {exc}")
                    st.exception(exc)
        else:
            st.markdown("""
<br><br>
<div style="text-align:center; color:#888; padding:60px">
    <div style="font-size:3em">📊</div>
    <div>Configure parameters on the left<br>and click <b>Run Backtest</b></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB — REALITY CHECK (walk-forward optimization)
# ══════════════════════════════════════════════════════════════════════════════
with tab_reality:
    st.header("🔬 Reality Check")
    st.caption(
        "Walk-forward optimization — the honest test. For each fold, the optimizer "
        "tunes parameters on past data, then measures them on the **next** slice it "
        "never saw. Only the out-of-sample numbers are trustworthy."
    )

    st.info(
        "**Why this tab exists:** any strategy can look great if you tune it on one "
        "lucky slice of history (that's *overfitting*). This finds out whether an edge "
        "survives on unseen data. If a strategy fails here, do **not** trade it with real money."
    )

    rc_ctrl, rc_out = st.columns([1, 2], gap="large")

    with rc_ctrl:
        _rc_tfs = [tf for tf in _timeframes_for(_current_asset()) if tf in ("5m", "15m", "30m", "1h", "4h")]
        rc_timeframe = st.selectbox(
            "Timeframe",
            options=_rc_tfs,
            index=_rc_tfs.index("1h") if "1h" in _rc_tfs else 0,
            format_func=lambda tf: _TIMEFRAME_LABELS.get(tf, tf),
            key="rc_tf",
        )
        rc_candles = st.slider(
            "History depth (candles)", 1000, 8000, 5000, 500,
            help="More history = more market regimes = more reliable verdict. "
                 "Pages past the exchange's per-request cap automatically.",
        )
        rc_folds = st.slider(
            "Walk-forward folds", 3, 8, 6,
            help="How many tune-then-test rounds. The history is split into this many "
                 "chunks; each is tuned on the past and tested on the next unseen chunk.")
        rc_leverage = st.slider("Leverage", 1, 10, 1,
                                help="Kept separate from your trading config. 1× strongly recommended.")
        if rc_leverage > 1:
            st.warning("Testing with leverage amplifies both the gains AND the losses you'll see below.")

        rc_selected = st.multiselect(
            "Strategies to test",
            options=OPTIMIZABLE,
            default=OPTIMIZABLE,
            format_func=lambda k: _STRATEGY_LABELS.get(k, k),
            help="Pick which strategies to put through the walk-forward test.",
        )
        with st.expander("ℹ️ How to read this"):
            st.markdown(
                "- **In-sample** = how it did on data it was *tuned on* (always flattering).\n"
                "- **Out-of-sample (OOS)** = how it did on the *next, unseen* data — the only honest number.\n"
                "- A big drop from in-sample to OOS = **overfitting**.\n"
                "- **Beat Buy&Hold** = did active trading beat simply holding? Usually not."
            )
        st.caption("LLM is excluded here — walk-forward would make thousands of paid API calls.")

        rc_run = st.button("🔬 Run Reality Check", type="primary", width="stretch",
                           disabled=not rc_selected)
        st.caption("Takes ~10–30s — it runs hundreds of backtests.")

    with rc_out:
        if rc_run:
            with st.spinner(f"Fetching {rc_candles} × {rc_timeframe} candles and running "
                            f"walk-forward across all strategies…"):
                try:
                    from trading_bot.backtest.optimize import walk_forward
                    from trading_bot.data.market import make_feed

                    cfg = _effective_config(_load_config())
                    feed = make_feed(cfg)   # crypto: public ccxt | stocks: yfinance
                    df = feed.history(_current_symbol(), rc_timeframe, total=rc_candles)
                    span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400
                    bh = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100

                    st.caption(
                        f"Tested on **{len(df)}** candles · "
                        f"{df.index[0].date()} → {df.index[-1].date()} "
                        f"({span_days:.0f} days) · Buy & Hold over period: **{bh:+.1f}%**"
                    )

                    summary_rows = []
                    reports = {}
                    for name in rc_selected:
                        rep = walk_forward(
                            df, name, n_folds=rc_folds,
                            leverage=rc_leverage, timeframe=rc_timeframe,
                        )
                        reports[name] = rep
                        if rep.oos_mean_return > 0 and rep.pct_profitable_folds >= 60 and rep.pct_beat_buy_hold >= 50:
                            v = "✅ promising"
                        elif rep.oos_mean_return > 0:
                            v = "⚠️ weak / unstable"
                        else:
                            v = "❌ no edge"
                        summary_rows.append({
                            "Strategy": _STRATEGY_LABELS.get(name, name).split(" (")[0],
                            "OOS mean/fold": f"{rep.oos_mean_return:+.2f}%",
                            "Profitable folds": f"{rep.pct_profitable_folds:.0f}%",
                            "Beat Buy&Hold": f"{rep.pct_beat_buy_hold:.0f}%",
                            "Worst fold": f"{rep.worst_fold_return:+.1f}%",
                            "Verdict": v,
                        })

                    summary_rows.sort(
                        key=lambda r: float(r["OOS mean/fold"].rstrip("%")), reverse=True
                    )

                    # ── Visual: out-of-sample mean return per strategy ──────
                    _ranked = sorted(reports.items(), key=lambda kv: kv[1].oos_mean_return)
                    _names = [_STRATEGY_LABELS.get(n, n).split(" (")[0] for n, _ in _ranked]
                    _vals = [r.oos_mean_return for _, r in _ranked]
                    bar = go.Figure(go.Bar(
                        x=_vals, y=_names, orientation="h",
                        marker_color=["#26a69a" if v > 0 else "#ef5350" for v in _vals],
                        text=[f"{v:+.2f}%" for v in _vals], textposition="outside",
                    ))
                    bar.update_layout(
                        title="Out-of-sample mean return per fold (the honest score)",
                        height=70 + 38 * len(_names), margin=dict(l=0, r=0, t=36, b=0),
                        xaxis_title="% per fold (higher = better; <0 = lost money)",
                    )
                    bar.add_vline(x=0, line_color="#888")
                    st.plotly_chart(bar, width="stretch")

                    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")

                    # ── Passive benchmarks over the full period ─────────────
                    from trading_bot.backtest.dca import simulate_dca
                    _bpd = {"5m": 288, "15m": 96, "1h": 24, "4h": 6}.get(rc_timeframe, 24)
                    _dca = simulate_dca(df, contribution=50.0, interval_bars=max(1, _bpd), fee_rate=0.001)
                    bench = pd.DataFrame([
                        {"Passive benchmark": "Buy & Hold", "Return over full period": f"{bh:+.1f}%"},
                        {"Passive benchmark": "DCA (daily $50)", "Return over full period": f"{_dca.return_pct:+.1f}%"},
                    ])
                    st.caption("Passive benchmarks — the honest comparison (no trading, no tuning):")
                    st.dataframe(bench, hide_index=True, width="stretch")
                    if max(bh, _dca.return_pct) >= max((reports[n].oos_total_compounded_pct for n in reports), default=0):
                        st.caption("↳ Passive accumulation beat the active strategies here — the usual outcome.")

                    # Overall conclusion
                    best_name = max(reports, key=lambda n: reports[n].oos_mean_return)
                    best = reports[best_name]
                    if best.oos_mean_return > 0 and best.pct_profitable_folds >= 60:
                        st.success(
                            f"**{_STRATEGY_LABELS.get(best_name, best_name).split(' (')[0]}** showed a "
                            f"positive, fairly consistent out-of-sample result "
                            f"({best.oos_mean_return:+.2f}% mean/fold). Worth **paper trading** — "
                            f"not real money yet. Run it live on the testnet for weeks and confirm "
                            f"the live results match before risking a cent."
                        )
                    else:
                        st.error(
                            "**No strategy showed a reliable out-of-sample edge.** This is the *normal* "
                            "result for simple technical strategies — it means: do **not** put real money "
                            "on them. Keep paper trading and treat this as a learning tool. "
                            "Leverage would only make the losses bigger."
                        )

                    # Per-strategy fold detail
                    st.divider()
                    for name, rep in reports.items():
                        label = _STRATEGY_LABELS.get(name, name)
                        with st.expander(f"{label} — fold-by-fold detail"):
                            fold_rows = [
                                {
                                    "Fold": f.fold,
                                    "Test period": f"{f.test_span[0].date()} → {f.test_span[1].date()}",
                                    "In-sample": f"{f.in_sample_metric:+.1f}%",
                                    "Out-of-sample": f"{f.out_of_sample['total_return_pct']:+.1f}%",
                                    "Buy&Hold": f"{f.buy_hold_test_pct:+.1f}%",
                                    "Params": ", ".join(f"{k}={v}" for k, v in f.best_params.items()),
                                }
                                for f in rep.folds
                            ]
                            # Visual: in-sample vs out-of-sample per fold
                            _folds = [f"Fold {f.fold}" for f in rep.folds]
                            ovf = go.Figure()
                            ovf.add_trace(go.Bar(
                                x=_folds, y=[f.in_sample_metric for f in rep.folds],
                                name="In-sample (tuned)", marker_color="#7e57c2"))
                            ovf.add_trace(go.Bar(
                                x=_folds, y=[f.out_of_sample["total_return_pct"] for f in rep.folds],
                                name="Out-of-sample (real)", marker_color="#26a69a"))
                            ovf.update_layout(
                                barmode="group", height=240, margin=dict(l=0, r=0, t=10, b=0),
                                legend=dict(orientation="h", y=1.15), yaxis_title="% return",
                            )
                            ovf.add_hline(y=0, line_color="#888")
                            st.plotly_chart(ovf, width="stretch")
                            st.dataframe(pd.DataFrame(fold_rows), hide_index=True, width="stretch")
                            st.caption(
                                "Notice how the purple **in-sample** bars often tower over the green "
                                "**out-of-sample** ones — that gap is overfitting, and it's exactly what loses real money."
                            )

                except Exception as exc:
                    st.error(f"Reality check error: {exc}")
                    st.exception(exc)
        else:
            st.markdown("""
<br><br>
<div style="text-align:center; color:#888; padding:50px">
    <div style="font-size:3em">🔬</div>
    <div>Set your parameters on the left and click <b>Run Reality Check</b><br>
    to see whether any strategy survives on data it was never tuned on.</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PAPER TRADE
# ══════════════════════════════════════════════════════════════════════════════
with tab_paper:
    st.header("Paper Trading")
    st.caption(
        "Pick a strategy and risk settings right here, then run them on live market "
        "data with simulated fills — no real money. No need to edit any files."
    )

    env = _load_env()
    _cfg = _load_config()
    _pt_exc = _cfg["exchange"]
    _key_env = _pt_exc.get("api_key_env", "BINANCE_TESTNET_API_KEY")
    _sec_env = _pt_exc.get("secret_env",  "BINANCE_TESTNET_SECRET")
    has_keys = bool(env.get(_key_env) and env.get(_sec_env))
    running = _trader_is_running()

    # ── Live config panel (disabled while running) ────────────────────────
    if running:
        ov = _read_overrides()
        active = ov.get("strategy", {}).get("name") or _cfg.get("strategy", {}).get("name", "ema_crossover")
        active_lev = ov.get("risk", {}).get("leverage", _cfg.get("risk", {}).get("leverage", 1))
        active_tf = ov.get("exchange", {}).get("timeframe", _pt_exc["timeframe"])
        st.success("🟢 **Paper trader is running.** Stop it to change the configuration.")
        lev_badge = f" · Leverage **{active_lev}×**" if active_lev and active_lev > 1 else ""
        st.caption(
            f"Strategy: **{_STRATEGY_LABELS.get(active, active)}**{lev_badge}  |  "
            f"Symbol: {_pt_exc['symbol']}  |  Timeframe: {active_tf}"
        )
        if st.button("⏹  Stop Paper Trader", type="primary", width="stretch"):
            _stop_trader()
            st.rerun()
    elif HOSTED:
        st.info(
            "🌐 **Live paper trading is local-only.** It runs a background process on the "
            "server with on-disk keys — fine on your own machine, but not safe to share. "
            "To try a model on live-ish data here, use **🤖 Ask an LLM about this chart** in "
            "the Backtest tab with your own key. For the full live trader, clone the repo and "
            "run `streamlit run ui/app.py` locally."
        )
    else:
        # Crypto needs exchange keys; stocks (yfinance) do not.
        needs_exchange_keys = _current_asset() == "crypto"
        if needs_exchange_keys and not has_keys:
            st.warning("Add exchange API keys in the **Setup** tab first (crypto only).")

        cfg_col, risk_col = st.columns(2, gap="large")

        pt_llm_extra: dict = {}
        llm_key_ok = True

        with cfg_col:
            st.markdown("**Strategy**")
            pt_strategy = st.selectbox(
                "Strategy",
                options=PAPER_TRADE,
                format_func=lambda k: _STRATEGY_LABELS.get(k, k),
                key="pt_strategy",
                label_visibility="collapsed",
            )
            st.caption(f"ℹ️ {STRATEGY_DESC.get(pt_strategy, '')}")
            _pt_tfs = _timeframes_for(_current_asset())
            pt_timeframe = st.selectbox(
                "Timeframe",
                options=_pt_tfs,
                index=_pt_tfs.index("15m") if "15m" in _pt_tfs else 0,
                format_func=lambda tf: _TIMEFRAME_LABELS.get(tf, tf),
                key="pt_tf",
                help="The trader checks for a new signal once per candle of this size.",
            )
            pt_params = render_params(st, pt_strategy, "pt")
            pt_ok, pt_msg = validate(pt_strategy, pt_params)
            if not pt_ok:
                st.warning(pt_msg)

            # ── LLM: provider + model pickers (any provider, not just Anthropic) ──
            if pt_strategy == "llm":
                pt_provider = st.selectbox(
                    "LLM provider", ["mammouth", "anthropic"], key="pt_llm_provider",
                    help="Mammouth.ai (your free sub) or Anthropic.",
                )
                if pt_provider == "mammouth":
                    _models = _mammouth_models()
                    _mi = _models.index("gpt-4o") if "gpt-4o" in _models else 0
                    pt_model = st.selectbox("Model", _models, index=_mi, key="pt_llm_model_m",
                                            help="Live list from Mammouth's public models endpoint.")
                    pt_base = st.text_input("Base URL", value="https://api.mammouth.ai/v1",
                                            key="pt_llm_base")
                    llm_key_ok = bool(env.get("MAMMOUTH_API_KEY"))
                    pt_llm_extra = {
                        "llm_provider": "mammouth", "llm_base_url": pt_base,
                        "llm_model": pt_model, "llm_api_key_env": "MAMMOUTH_API_KEY",
                    }
                    _key_label = "MAMMOUTH_API_KEY"
                else:
                    pt_model = st.selectbox("Model", _ANTHROPIC_MODELS, key="pt_llm_model_a")
                    llm_key_ok = bool(env.get("ANTHROPIC_API_KEY"))
                    pt_llm_extra = {
                        "llm_provider": "anthropic", "llm_model": pt_model,
                        "llm_api_key_env": "ANTHROPIC_API_KEY",
                    }
                    _key_label = "ANTHROPIC_API_KEY"
                if not llm_key_ok:
                    st.warning(f"Add your **{_key_label}** in the Setup tab to use this provider.")
                st.caption("💸 The LLM makes one API call per bar.")

        with risk_col:
            st.markdown("**Risk & leverage**")
            pt_leverage = st.slider(
                "Leverage", 1, 10, 1, key="pt_lev",
                help="1× = no borrowing. Higher multiplies gains AND losses; keep at 1 unless sure.")
            pt_sl = st.number_input(
                "Stop-loss %", 0.5, 20.0, 2.0, 0.5, key="pt_sl",
                help="Auto-close a position once it's down this much — your safety net per trade.") / 100
            pt_tp = st.number_input(
                "Take-profit %", 0.5, 40.0, 4.0, 0.5, key="pt_tp",
                help="Auto-close once a position is up this much — banks the win.") / 100
            pt_trail = st.number_input(
                "Trailing stop % (0 = off)", 0.0, 20.0, 0.0, 0.5, key="pt_trail",
                help="Follows the price up and sells if it falls this far from the peak. Protects profits.") / 100
            pt_maxpos = st.slider(
                "Max position % of capital", 5, 100, 10, 5, key="pt_maxpos",
                help="How much of your balance a single trade can use. Lower = safer.") / 100
            _leverage_warning(pt_leverage, pt_sl)

        _ready = (not needs_exchange_keys or has_keys) and pt_ok and llm_key_ok
        clicked = st.button(
            "▶  Start Paper Trader", type="primary", width="stretch",
            disabled=not _ready,
        )
        if clicked and _ready:
            _write_overrides({
                "market": {"asset_class": _current_asset()},
                "strategy": {"name": pt_strategy, **pt_params, **pt_llm_extra},
                "exchange": {"timeframe": pt_timeframe, "symbol": _current_symbol()},
                "risk": {
                    "leverage": int(pt_leverage),
                    "stop_loss_pct": float(pt_sl),
                    "take_profit_pct": float(pt_tp),
                    "trailing_stop_pct": float(pt_trail),
                    "max_position_pct": float(pt_maxpos),
                },
            })
            _start_trader()
            time.sleep(1.5)
            st.rerun()

    # ── Live performance (real values recorded by the trader) ─────────────
    st.divider()
    st.subheader("📈 Live performance")
    _hist, _orders, _meta = _read_paper_history()

    if _hist is None or len(_hist) == 0:
        st.caption(
            "No data yet. Once the paper trader runs, it records a real portfolio "
            "snapshot every candle here — equity, price, and each simulated trade. "
            "On a 15m timeframe that's one point every 15 minutes, so the charts fill in over time."
        )
    else:
        init_cap = _meta.get("initial_capital", float(_hist["equity"].iloc[0]))
        last = _hist.iloc[-1]
        equity_now = float(last["equity"])
        ret_pct = (equity_now - init_cap) / init_cap * 100 if init_cap else 0.0
        n_orders = 0 if _orders is None else len(_orders)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Portfolio value", f"${equity_now:,.2f}", f"{ret_pct:+.2f}%")
        k2.metric("Cash", f"${float(last['cash']):,.2f}")
        k3.metric("Position", f"{float(last['position']):.6f}")
        k4.metric("Trades", str(n_orders))
        st.caption(
            f"Started from **${init_cap:,.0f}** · {_meta.get('symbol','')} "
            f"{_meta.get('timeframe','')} · strategy **{_STRATEGY_LABELS.get(_meta.get('strategy',''), _meta.get('strategy',''))}**"
            f"{' · ' + str(_meta.get('leverage')) + '×' if _meta.get('leverage', 1) > 1 else ''} · "
            f"{len(_hist)} snapshots recorded"
        )

        # Price + trade markers (real recorded prices & fills)
        pf = go.Figure()
        pf.add_trace(go.Scatter(
            x=_hist["timestamp"], y=_hist["price"], mode="lines",
            name="Price", line=dict(color="#888", width=1.5)))
        if _orders is not None and len(_orders) > 0:
            buys = _orders[_orders["side"] == "buy"]
            sells = _orders[_orders["side"] == "sell"]
            if len(buys):
                pf.add_trace(go.Scatter(
                    x=buys["timestamp"], y=buys["price"], mode="markers", name="Buy",
                    marker=dict(symbol="triangle-up", size=12, color="#22cc88")))
            if len(sells):
                pf.add_trace(go.Scatter(
                    x=sells["timestamp"], y=sells["price"], mode="markers", name="Sell",
                    marker=dict(symbol="triangle-down", size=12, color="#ff6b6b")))
        pf.update_layout(
            title=f"{_meta.get('symbol','')} price & simulated trades (live)",
            height=300, margin=dict(l=0, r=0, t=36, b=0),
            legend=dict(orientation="h", y=1.15), hovermode="x unified")
        st.plotly_chart(pf, width="stretch")

        # Portfolio equity over time, vs starting budget
        ef = go.Figure()
        ef.add_trace(go.Scatter(
            x=_hist["timestamp"], y=_hist["equity"], mode="lines", name="Portfolio ($)",
            line=dict(color="#00d4aa", width=2), fill="tozeroy",
            fillcolor="rgba(0,212,170,0.07)"))
        ef.add_hline(y=init_cap, line_dash="dot", line_color="#888",
                     annotation_text="starting budget", annotation_position="bottom right")
        ef.update_layout(
            title="Portfolio value over time", height=260,
            margin=dict(l=0, r=0, t=36, b=0), hovermode="x unified", showlegend=False)
        st.plotly_chart(ef, width="stretch")

    # ── Log viewer ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Activity Log")

    refresh_col, auto_col, _ = st.columns([1, 1, 4])
    with refresh_col:
        if st.button("🔄 Refresh Log"):
            st.rerun()
    with auto_col:
        auto_refresh = st.checkbox("Auto-refresh (10s)", value=False)

    lines = _tail_log(60)

    if lines:
        signal_lines = [l for l in lines if "signal=" in l]
        order_lines  = [l for l in lines if "ORDER" in l or "BUY" in l or "SELL" in l]
        error_lines  = [l for l in lines if "ERROR" in l or "CRITICAL" in l or "HALT" in l]

        log_l, log_r = st.columns(2)

        with log_l:
            st.caption("Recent signals")
            st.code(
                "\n".join(signal_lines[-15:]) if signal_lines else "(none yet)",
                language="",
            )

        with log_r:
            st.caption("Simulated orders")
            st.code(
                "\n".join(order_lines[-10:]) if order_lines else "(none yet)",
                language="",
            )

        if error_lines:
            st.error("⚠️ Errors / Halts detected in log:")
            st.code("\n".join(error_lines[-5:]), language="")
    else:
        st.caption("No log entries yet. Start the paper trader to see activity here.")

    if auto_refresh and running:
        time.sleep(10)
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — NEWS FEED
# ══════════════════════════════════════════════════════════════════════════════
with tab_news:
    st.header("📰 News Feed")
    st.caption(
        "Live crypto headlines from CoinDesk, CoinTelegraph, Decrypt, and more. "
        "**Free — no API key needed.** "
        "Headlines are scored with the same keyword scorer the bot uses."
    )

    col_refresh, col_filter, _ = st.columns([1, 1, 3])
    with col_refresh:
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
    with col_filter:
        show_all = st.checkbox("Show all (including neutral)", value=False)

    @st.cache_data(ttl=300)
    def _fetch_headlines() -> list[dict]:
        from trading_bot.news.scorer import KeywordSentimentScorer
        from trading_bot.news.sources.rss import RSSNewsSource

        source = RSSNewsSource()
        scorer = KeywordSentimentScorer()
        items = source.fetch("BTC/USDT", limit=40)
        return [
            {
                "title": item.title,
                "source": item.source,
                "time": item.published_at.strftime("%b %d  %H:%M UTC"),
                "score": scorer.score(item.title).value,
                "confidence": scorer.score(item.title).confidence,
                "url": item.url,
            }
            for item in items
        ]

    with st.spinner("Loading headlines…"):
        try:
            headlines = _fetch_headlines()

            if not headlines:
                st.warning("No headlines found — some RSS feeds may be temporarily unavailable.")
            else:
                for h in headlines:
                    score = h["score"]
                    if abs(score) < 0.1 and not show_all:
                        continue

                    if score > 0.2:
                        icon, color = "🟢", "#22cc88"
                    elif score < -0.2:
                        icon, color = "🔴", "#ff6b6b"
                    else:
                        icon, color = "⚪", "#888"

                    with st.container():
                        c1, c2 = st.columns([0.05, 0.95])
                        c1.markdown(icon)
                        with c2:
                            st.markdown(
                                f"**[{h['title']}]({h['url']})**  "
                                f"<span style='color:{color};font-size:0.8em'> score: {score:+.2f}</span>"
                                f"  <span style='color:#888;font-size:0.8em'>— {h['source']} · {h['time']}</span>",
                                unsafe_allow_html=True,
                            )
                        st.divider()

                with st.expander("How are scores calculated?"):
                    st.markdown("""
**Score** ranges from **-1.0** (very bearish) to **+1.0** (very bullish).

The bot counts bullish keywords (e.g. *surge, rally, ETF approval, adoption*)
and bearish keywords (e.g. *crash, hack, ban, liquidation*) in the headline.
The score is `(bullish - bearish) / total` — simple, fast, no API calls.

To enable news sentiment in the bot, set `enabled = true` in `config/settings.toml`
under `[news]`. The `source = "rss"` setting means it uses these same free feeds.
                    """)

        except Exception as exc:
            st.error(f"Failed to load news: {exc}")
