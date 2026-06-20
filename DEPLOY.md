# Deploying the app so friends can have a look

The safe way to share this: deploy in **hosted mode**, where the analysis tabs
(Backtest, Reality Check, News) work with **no keys at all**, and anyone who wants
to try the LLM pastes **their own** key — which stays only in their browser session.

## What hosted mode does (security)

Set the env var / secret `TRADING_BOT_HOSTED=1`. Then:

- ✅ **Backtest, Reality Check, News** — fully working on free public data, no keys.
- ✅ **LLM "Ask an LLM about this chart"** (Backtest tab) — each visitor pastes their
  own key in Setup; it lives **only in their session** (`st.session_state`), is **never
  written to the server's disk**, is **never shared** with other visitors, and is gone
  when they close the tab.
- 🚫 **Saving exchange keys to disk** — disabled (would be shared on one host).
- 🚫 **Live paper trading** — disabled. It needs a server-side background process with
  on-disk keys, which is single-user by nature. That stays a "run locally" feature.

> You never put YOUR keys in the deployment. There are **no secrets to configure** —
> just the `TRADING_BOT_HOSTED` flag.

## Option A — Streamlit Community Cloud (free, easiest)

1. **Put the project on GitHub** (from the `trading_bot/` folder):
   ```bash
   cd "D:/Claude Code/trading_bot"
   git init
   git add .
   git commit -m "Trading bot app"
   gh repo create my-trading-bot --public --source=. --push   # or use the GitHub UI
   ```
   `.env`, `config/ui_overrides.json`, and logs are git-ignored — no secrets get pushed.
   (Double-check: `git status` should NOT list `.env`.)

2. Go to **https://share.streamlit.io** → sign in with GitHub → **New app**.

3. Fill in:
   - **Repository**: your `my-trading-bot`
   - **Branch**: `main`
   - **Main file path**: `ui/app.py`

4. **Advanced settings → Secrets**, paste:
   ```toml
   TRADING_BOT_HOSTED = "1"
   ```

5. Click **Deploy**. You'll get a public URL like
   `https://my-trading-bot.streamlit.app` — share it with your friends.

## Option B — Hugging Face Spaces

1. Create a **Space** → SDK: **Streamlit**.
2. Upload the repo contents (or link the GitHub repo).
3. In **Settings → Variables and secrets**, add `TRADING_BOT_HOSTED = 1`.
4. Set the app file to `ui/app.py` if asked.

## Running the FULL version locally (with live paper trading)

```bash
pip install -e ".[dev,stocks]"
streamlit run ui/app.py          # no TRADING_BOT_HOSTED → full features
```

## Quick checklist before sharing

- [ ] `git status` does not show `.env`
- [ ] `TRADING_BOT_HOSTED = "1"` is set in the host's secrets/env
- [ ] App loads and shows the "🌐 Shared demo" banner
- [ ] Backtest runs without any key
- [ ] Friends can paste their own LLM key in Setup and it says "(session only)"
