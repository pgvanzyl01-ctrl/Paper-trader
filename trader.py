#!/usr/bin/env python3
"""Paper-trading platform: weekly momentum rotation on liquid Binance USDT coins.

PAPER ONLY. No API keys, no real orders. It simulates a portfolio with real market prices,
charges fees and slippage, enforces risk limits, and publishes a dashboard (docs/index.html).

Run once per day (GitHub Actions does this). Each run:
  1. downloads daily candles (Binance, falling back to Binance.US),
  2. marks the portfolio to today's open,
  3. checks the drawdown kill switch,
  4. every `rebalance_days` days, ranks the most liquid coins by past return and holds the
     strongest `top_k` (equal weight, long only, spot). Fills happen at today's open, using
     only data up to yesterday's close, exactly like the backtest.

Usage:
  python trader.py                    daily run (live data)
  python trader.py --simulate 180     180-day offline demo on synthetic data -> demo_out/index.html
Optional config.json overrides any key in CFG below.
"""
import argparse
import html
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

CFG = dict(
    start_cash=10_000.0,
    lookback=28,            # days of past return used to rank coins
    top_k=5,                # coins held at once
    liquid=15,              # only rank the N most liquid coins
    universe_size=30,       # coins considered before the liquidity cut
    min_quote_volume=2_000_000,
    rebalance_days=7,
    btc_filter=True,        # hold cash when BTC is below its moving average
    btc_ma=100,
    fee=0.001,              # 0.1% per side
    slippage=0.0002,        # 0.02% per side
    max_drawdown_halt=0.25, # kill switch: sell everything and stop at -25% from peak
    min_trade_usd=10.0,
    max_gap=0.40,           # skip a coin if today's open is >40% from yesterday's close (bad data)
    history_days=200,
)
STATE_DIR, DOCS_DIR = "state", "docs"
BTC = "BTC/USDT"
STABLES = {"USDT", "USDC", "FDUSD", "TUSD", "DAI", "BUSD", "USDP", "USDD", "EUR", "GBP", "AEUR", "PYUSD", "XUSD", "USD1"}
FALLBACK = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT", "LTC", "ATOM",
            "UNI", "NEAR", "TRX", "BCH", "ETC", "FIL", "AAVE", "ALGO"]


# ----------------------------------------------------------------------------- config / state
def load_cfg():
    cfg = dict(CFG)
    if os.path.exists("config.json"):
        with open("config.json") as f:
            cfg.update(json.load(f))
    return cfg


def new_state(cfg):
    return dict(created=datetime.now(timezone.utc).isoformat(timespec="seconds"), cash=float(cfg["start_cash"]),
                positions={}, peak_equity=float(cfg["start_cash"]), halted=False, halt_reason="",
                last_rebalance=None, last_run=None, equity_history=[], trades=[], log=[], last_info={}, source="")


def load_state(cfg, state_dir=STATE_DIR):
    p = os.path.join(state_dir, "state.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return new_state(cfg)


def save_state(state, state_dir=STATE_DIR):
    os.makedirs(state_dir, exist_ok=True)
    with open(os.path.join(state_dir, "state.json"), "w") as f:
        json.dump(state, f, indent=1)


def make_log(state, quiet=False):
    def log(msg):
        line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M}Z  {msg}"
        if not quiet:
            print(line, flush=True)
        state["log"].append(line)
        del state["log"][:-60]
    return log


# ----------------------------------------------------------------------------- market data
def to_df(rows):
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
    return df.drop_duplicates("date").set_index("date").sort_index()[["open", "high", "low", "close", "volume"]].astype(float)


def universe(ex, cfg):
    rows = []
    for sym, t in ex.fetch_tickers().items():
        base = sym.split("/")[0]
        if not sym.endswith("/USDT") or base in STABLES or base.endswith(("UP", "DOWN", "BULL", "BEAR")):
            continue
        qv = t.get("quoteVolume") or 0
        if qv >= cfg["min_quote_volume"]:
            rows.append((qv, sym))
    syms = [s for _, s in sorted(rows, reverse=True)[:cfg["universe_size"]]]
    if len(syms) < 10:  # thin ticker list (common on Binance.US): top up with major coins
        syms += [f"{b}/USDT" for b in FALLBACK if f"{b}/USDT" not in syms]
        syms = syms[:max(cfg["universe_size"], 20)]
    return syms


def fetch_market(cfg, log):
    import ccxt
    last = "unknown"
    for name in ("binance", "binanceus"):
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True})
            data = {}
            for s in universe(ex, cfg):
                try:
                    df = to_df(ex.fetch_ohlcv(s, "1d", limit=cfg["history_days"]))
                    if len(df) >= cfg["lookback"] + 40:
                        data[s] = df
                except Exception:
                    continue
            if len(data) >= 8:
                log(f"Data source: {name}, {len(data)} coins")
                return data, name
            last = f"only {len(data)} coins"
        except Exception as e:
            last = type(e).__name__
        log(f"{name} unavailable ({last})")
    raise RuntimeError(f"could not download market data ({last})")


# ----------------------------------------------------------------------------- strategy
def pick_targets(data, today, cfg, opens):
    """Target weights using ONLY data before `today`. Returns (weights, info) or (None, reason)."""
    closes = pd.DataFrame({s: d["close"] for s, d in data.items()})
    vol = pd.DataFrame({s: d["close"] * d["volume"] for s, d in data.items()})
    closes, vol = closes[closes.index < today], vol[vol.index < today]
    if closes.empty or closes.index[-1] != today - pd.Timedelta(days=1):
        return None, "stale data: yesterday's candle is missing"
    L = cfg["lookback"]
    if len(closes) <= max(L, cfg["btc_ma"]):
        return None, "not enough history"
    liq = vol.iloc[-30:].mean()
    past = closes.iloc[-1] / closes.iloc[-1 - L] - 1
    gap_ok = pd.Series({s: (s in opens and abs(opens[s] / closes[s].iloc[-1] - 1) <= cfg["max_gap"])
                        for s in closes.columns if pd.notna(closes[s].iloc[-1])})
    ok = liq.notna() & past.notna() & gap_ok.reindex(closes.columns).fillna(False)
    pool = liq[ok].nlargest(cfg["liquid"]).index
    if len(pool) < cfg["top_k"]:
        return None, f"only {len(pool)} eligible coins"
    ranked = past[pool].sort_values(ascending=False)
    picks = ranked.head(cfg["top_k"])
    picks = picks[picks > 0]
    regime_ok = True
    if cfg["btc_filter"] and BTC in closes.columns:
        regime_ok = bool(closes[BTC].iloc[-1] > closes[BTC].iloc[-(cfg["btc_ma"] + 1):].mean())
    if not regime_ok:
        picks = picks.iloc[0:0]
    weights = {s: 1.0 / cfg["top_k"] for s in picks.index}
    info = dict(regime_ok=regime_ok, ranking=[[s, round(float(v), 4)] for s, v in ranked.head(10).items()],
                targets=list(weights))
    return weights, info


# ----------------------------------------------------------------------------- paper broker
def equity_of(state, prices=None):
    prices = prices or {}
    return state["cash"] + sum(p["qty"] * prices.get(s, p["last_price"]) for s, p in state["positions"].items())


def do_sell(state, sym, qty, px, cfg, day, reason):
    pos = state["positions"][sym]
    qty = min(qty, pos["qty"])
    fill = px * (1 - cfg["slippage"])
    usd = qty * fill
    fee = usd * cfg["fee"]
    state["cash"] += usd - fee
    state["trades"].append(dict(date=day, side="SELL", sym=sym, qty=qty, price=fill, usd=usd, fee=fee,
                                reason=reason, pnl_pct=(fill / pos["entry"] - 1) * 100))
    pos["qty"] -= qty
    if pos["qty"] <= 1e-12:
        del state["positions"][sym]


def do_buy(state, sym, usd, px, cfg, day, reason):
    fill = px * (1 + cfg["slippage"])
    usd = min(usd, state["cash"] / (1 + cfg["fee"]))
    if usd < cfg["min_trade_usd"]:
        return
    qty = usd / fill
    fee = usd * cfg["fee"]
    state["cash"] -= usd + fee
    pos = state["positions"].get(sym)
    if pos:
        total = pos["qty"] + qty
        pos["entry"] = (pos["entry"] * pos["qty"] + fill * qty) / total
        pos["qty"], pos["last_price"] = total, px
    else:
        state["positions"][sym] = dict(qty=qty, entry=fill, last_price=px, entry_date=day)
    state["trades"].append(dict(date=day, side="BUY", sym=sym, qty=qty, price=fill, usd=usd, fee=fee,
                                reason=reason, pnl_pct=None))


def rebalance(state, weights, opens, cfg, day):
    equity = equity_of(state, opens)
    target = {s: equity * w for s, w in weights.items()}
    for sym, pos in list(state["positions"].items()):          # sells first, so cash is available
        px = opens.get(sym, pos["last_price"])
        cur, tgt = pos["qty"] * px, target.get(sym, 0.0)
        if tgt == 0.0:
            do_sell(state, sym, pos["qty"], px, cfg, day, "rotate out")
        elif cur - tgt > cfg["min_trade_usd"]:
            do_sell(state, sym, (cur - tgt) / px, px, cfg, day, "trim")
    for sym, tgt in target.items():
        cur = state["positions"].get(sym, {}).get("qty", 0.0) * opens[sym]
        if tgt - cur > cfg["min_trade_usd"]:
            do_buy(state, sym, tgt - cur, opens[sym], cfg, day, "rotate in")


def rebalance_due(state, today, cfg):
    if not state["last_rebalance"]:
        return True
    return (today - pd.Timestamp(state["last_rebalance"])).days >= cfg["rebalance_days"]


def run_once(state, data, today, cfg, log):
    """One daily step. Returns False if today was already processed."""
    day = str(today.date())
    if state["last_run"] == day:
        log("Already ran for today, nothing to do.")
        return False
    opens = {s: float(d.at[today, "open"]) for s, d in data.items()
             if today in d.index and np.isfinite(d.at[today, "open"]) and d.at[today, "open"] > 0}
    for s, p in state["positions"].items():
        if s in opens:
            p["last_price"] = opens[s]
    equity = equity_of(state)

    if not state["halted"] and state["halt_reason"]:        # user set halted=false: resume with fresh peak
        state["peak_equity"], state["halt_reason"] = equity, ""
        log("Resumed after halt; drawdown peak reset to current equity.")
    state["peak_equity"] = max(state["peak_equity"], equity)
    dd = equity / state["peak_equity"] - 1

    if not state["halted"] and dd <= -cfg["max_drawdown_halt"]:
        for s, p in list(state["positions"].items()):
            do_sell(state, s, p["qty"], opens.get(s, p["last_price"]), cfg, day, "kill switch")
        state["halted"] = True
        state["halt_reason"] = f"Drawdown {dd:.1%} reached the {cfg['max_drawdown_halt']:.0%} limit on {day}"
        log("KILL SWITCH: " + state["halt_reason"] + ". Sold everything.")

    if state["halted"]:
        log("Halted: no trading. To resume, set \"halted\": false in state/state.json.")
    elif rebalance_due(state, today, cfg):
        weights, info = pick_targets(data, today, cfg, opens)
        if weights is None:
            log(f"Rebalance skipped: {info}")
        else:
            rebalance(state, weights, opens, cfg, day)
            state["last_rebalance"], state["last_info"] = day, info
            held = ", ".join(s.split("/")[0] for s in weights) or "cash (BTC trend is down or nothing is rising)"
            log(f"Rebalanced. Holding: {held}")
    else:
        log("No rebalance due. Holding.")

    state["last_run"] = day
    state["equity_history"].append(dict(date=day, equity=round(equity_of(state, opens), 2), btc=opens.get(BTC)))
    return True


# ----------------------------------------------------------------------------- dashboard
def esc(x):
    return html.escape(str(x))


def money(x):
    return f"{x:,.2f}"


def pct(x):
    return f"{x:+.2f}%"


def cls(x):
    return "good" if x > 0 else "bad" if x < 0 else ""


def chart_svg(hist, start_cash):
    if len(hist) < 2:
        return '<p class="muted">The chart appears after the second daily run.</p>'
    s1 = [h["equity"] / start_cash * 100 for h in hist]
    btc0 = next((h["btc"] for h in hist if h.get("btc")), None)
    s2 = [(h["btc"] / btc0 * 100) if (btc0 and h.get("btc")) else None for h in hist]
    vals = s1 + [v for v in s2 if v is not None]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08 or 1.0
    lo, hi = lo - pad, hi + pad
    W, H, L, B = 640, 250, 46, 26
    n = len(hist)
    x = lambda i: L + (W - L - 10) * i / (n - 1)
    y = lambda v: 10 + (H - B - 10) * (1 - (v - lo) / (hi - lo))
    grid = ""
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        grid += f'<line x1="{L}" x2="{W-10}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>' \
                f'<text x="{L-6}" y="{y(v)+4:.1f}" text-anchor="end" class="axis">{v:.0f}</text>'
    p1 = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(s1))
    p2 = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(s2) if v is not None)
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Equity curve versus BTC, both rebased to 100">{grid}'
            f'<polyline points="{p2}" class="l2"/><polyline points="{p1}" class="l1"/>'
            f'<text x="{L}" y="{H-6}" class="axis">{esc(hist[0]["date"])}</text>'
            f'<text x="{W-10}" y="{H-6}" text-anchor="end" class="axis">{esc(hist[-1]["date"])}</text></svg>'
            f'<p class="legend"><span class="k1"></span>Paper portfolio <span class="k2"></span>BTC buy &amp; hold (both start at 100)</p>')


CSS = """
:root{--bg:#f4f5f7;--card:#fff;--text:#13161b;--muted:#5b6470;--line:#e2e5ea;--good:#0a7a43;--bad:#c1252d;--c1:#2358d6;--c2:#8b94a1;--warn:#fff4d6;--warnt:#6b4b00}
@media (prefers-color-scheme:dark){:root{--bg:#0d1014;--card:#161a20;--text:#e8ebef;--muted:#9aa3af;--line:#262c35;--good:#4cc38a;--bad:#ff6b73;--c1:#6c9bff;--c2:#7a8390;--warn:#2b2410;--warnt:#f0d28a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:760px;margin:0 auto;padding:16px}h1{font-size:20px;margin:4px 0 2px}h2{font-size:15px;margin:0 0 10px}
.muted{color:var(--muted);font-size:13px}.banner{background:var(--warn);color:var(--warnt);border-radius:10px;padding:10px 12px;margin:10px 0;font-size:13px}
.banner.halt{background:var(--bad);color:#fff;font-weight:600}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin:12px 0}
.stats{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}@media(min-width:560px){.stats{grid-template-columns:repeat(4,1fr)}}
.stat .v{font-size:19px;font-weight:650}.stat .l{font-size:12px;color:var(--muted)}.good{color:var(--good)}.bad{color:var(--bad)}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:right;padding:6px 4px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left;white-space:nowrap}
th{color:var(--muted);font-weight:500}.scroll{overflow-x:auto}svg{width:100%;height:auto;display:block}
.grid{stroke:var(--line);stroke-width:1}.axis{fill:var(--muted);font-size:11px}.l1,.l2{fill:none;stroke-width:2.2;stroke-linejoin:round}.l1{stroke:var(--c1)}.l2{stroke:var(--c2);stroke-dasharray:5 4}
.legend{font-size:12px;color:var(--muted);margin:6px 0 0}.legend span{display:inline-block;width:14px;height:3px;margin:0 6px 2px 10px;background:var(--c1)}.legend span.k2{background:var(--c2)}.legend span:first-child{margin-left:0}
pre{white-space:pre-wrap;font-size:12px;color:var(--muted);margin:0}
"""


def render_dashboard(state, cfg, out_dir):
    hist = state["equity_history"]
    eq = hist[-1]["equity"] if hist else state["cash"]
    start = cfg["start_cash"]
    ret = (eq / start - 1) * 100
    peak = state["peak_equity"]
    dd = (eq / peak - 1) * 100
    series = [h["equity"] for h in hist]
    maxdd = min(((v / max(series[:i + 1]) - 1) * 100 for i, v in enumerate(series)), default=0.0)
    closed = [t for t in state["trades"] if t["side"] == "SELL" and t["pnl_pct"] is not None and t["reason"] != "trim"]
    wins = sum(1 for t in closed if t["pnl_pct"] > 0)
    info = state.get("last_info", {})
    nxt = "due on next run" if not state["last_rebalance"] else \
        str((pd.Timestamp(state["last_rebalance"]) + pd.Timedelta(days=cfg["rebalance_days"])).date())
    status = ('<div class="banner halt">HALTED: ' + esc(state["halt_reason"]) + '. No trading until you set "halted": false in state/state.json.</div>'
              if state["halted"] else "")

    rows = ""
    for s, p in sorted(state["positions"].items(), key=lambda kv: -kv[1]["qty"] * kv[1]["last_price"]):
        val = p["qty"] * p["last_price"]
        pl = (p["last_price"] / p["entry"] - 1) * 100
        rows += (f'<tr><td>{esc(s.split("/")[0])}</td><td>{val / eq * 100:.1f}%</td><td>{p["entry"]:.4g}</td>'
                 f'<td>{p["last_price"]:.4g}</td><td>{money(val)}</td><td class="{cls(pl)}">{pct(pl)}</td></tr>')
    rows += f'<tr><td>Cash</td><td>{state["cash"] / eq * 100:.1f}%</td><td></td><td></td><td>{money(state["cash"])}</td><td></td></tr>'

    trows = ""
    for t in reversed(state["trades"][-15:]):
        pl = "" if t["pnl_pct"] is None else f'<span class="{cls(t["pnl_pct"])}">{pct(t["pnl_pct"])}</span>'
        trows += (f'<tr><td>{esc(t["date"])}</td><td>{esc(t["side"])}</td><td>{esc(t["sym"].split("/")[0])}</td>'
                  f'<td>{t["price"]:.4g}</td><td>{money(t["usd"])}</td><td>{pl}</td></tr>')
    trows = trows or '<tr><td colspan="6" class="muted">No trades yet.</td></tr>'

    held = set(state["positions"])
    rrows = "".join(f'<tr><td>{esc(s.split("/")[0])}</td><td class="{cls(v)}">{pct(v * 100)}</td><td>{"held" if s in held else ""}</td></tr>'
                    for s, v in info.get("ranking", []))
    regime = ("BTC above its trend: buying allowed" if info.get("regime_ok", True) else "BTC below its trend: staying in cash") \
        if cfg["btc_filter"] else "BTC filter off"

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Paper Trader</title><style>{CSS}</style></head><body><main>
<h1>Paper Trader</h1>
<div class="muted">Momentum rotation, top {cfg["top_k"]} of {cfg["liquid"]} liquid coins, {cfg["lookback"]}-day lookback, rebalance every {cfg["rebalance_days"]} days. Updated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC.</div>
<div class="banner">PAPER TRADING: simulated money, real prices. Unproven strategy; backtests were optimistic. Not financial advice.</div>
{status}
<div class="card"><div class="stats">
<div class="stat"><div class="v">{money(eq)}</div><div class="l">Equity (start {money(start)})</div></div>
<div class="stat"><div class="v {cls(ret)}">{pct(ret)}</div><div class="l">Total return</div></div>
<div class="stat"><div class="v {cls(dd)}">{pct(dd)}</div><div class="l">From peak (halt at -{cfg["max_drawdown_halt"] * 100:.0f}%)</div></div>
<div class="stat"><div class="v {cls(maxdd)}">{pct(maxdd)}</div><div class="l">Max drawdown</div></div>
</div></div>
<div class="card"><h2>Equity vs BTC</h2>{chart_svg(hist, start)}</div>
<div class="card"><h2>Holdings</h2><div class="scroll"><table><tr><th>Coin</th><th>Weight</th><th>Entry</th><th>Last</th><th>Value</th><th>P/L</th></tr>{rows}</table></div>
<p class="muted">Next rebalance: {esc(nxt)}. {esc(regime)}.</p></div>
<div class="card"><h2>Recent trades</h2><div class="scroll"><table><tr><th>Date</th><th>Side</th><th>Coin</th><th>Price</th><th>USD</th><th>P/L</th></tr>{trows}</table></div>
<p class="muted">{len(closed)} closed positions, {wins} winners. Fees paid: {money(sum(t["fee"] for t in state["trades"]))}</p></div>
<div class="card"><h2>Latest momentum ranking ({cfg["lookback"]}-day return)</h2><table><tr><th>Coin</th><th>Return</th><th></th></tr>{rrows or '<tr><td colspan="3" class="muted">Not ranked yet.</td></tr>'}</table></div>
<div class="card"><h2>Log</h2><pre>{esc(chr(10).join(state["log"][-12:]))}</pre><p class="muted">Data source: {esc(state.get("source") or "n/a")}</p></div>
</main></body></html>"""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)


# ----------------------------------------------------------------------------- offline simulation
def synthetic_market(n_days, n_coins=20, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n_days, freq="D")
    data = {}
    for i in range(n_coins):
        sym = BTC if i == 0 else f"SYN{i}/USDT"
        mu, sig = rng.normal(0.0006, 0.0008), rng.uniform(0.02, 0.05)
        close = 100 * np.exp(np.cumsum(rng.normal(mu, sig, n_days)))
        open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.002, n_days))
        spread = np.abs(rng.normal(0, sig / 2, n_days)) * close
        data[sym] = pd.DataFrame(dict(open=open_, high=np.maximum(open_, close) + spread,
                                      low=np.minimum(open_, close) - spread, close=close,
                                      volume=rng.lognormal(12 - i * 0.15, 0.3, n_days)), index=idx)
    return data


def simulate(days, cfg, out_dir):
    warm = 130
    data = synthetic_market(warm + days)
    idx = data[BTC].index
    state = new_state(cfg)
    log = make_log(state, quiet=True)
    for i in range(warm, len(idx)):
        view = {s: d.iloc[:i + 1] for s, d in data.items()}
        run_once(state, view, idx[i], cfg, log)
    state["source"] = "synthetic demo data (no real prices)"
    render_dashboard(state, cfg, out_dir)
    print(f"Simulated {days} days. Equity {equity_of(state):,.2f}, {len(state['trades'])} trades. Dashboard: {out_dir}/index.html")
    return state


# ----------------------------------------------------------------------------- entry point
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", type=int, help="offline demo with N days of synthetic data")
    ap.add_argument("--out", default="demo_out")
    args = ap.parse_args()
    cfg = load_cfg()
    if args.simulate:
        simulate(args.simulate, cfg, args.out)
        return 0
    state = load_state(cfg)
    snapshot = json.loads(json.dumps(state))      # roll back if anything fails mid-run
    log = make_log(state)
    ok = True
    try:
        data, src = fetch_market(cfg, log)
        state["source"] = src
        run_once(state, data, pd.Timestamp(datetime.now(timezone.utc).date()), cfg, log)
    except Exception as e:
        ok = False
        state = snapshot
        make_log(state)(f"ERROR: {type(e).__name__}: {e}. No changes were made to the portfolio.")
    save_state(state)
    render_dashboard(state, cfg, DOCS_DIR)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
