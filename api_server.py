"""
api_server.py
AlgoTrader Pro v7.0 — "The Intelligence Pipeline"

"""

from __future__ import annotations

import os
import secrets
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, TYPE_CHECKING

try:
    import ta_compat
except ImportError:
    pass

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import math
import json

class SafeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        def sanitize(obj):
            if isinstance(obj, float):
                return None if (math.isinf(obj) or math.isnan(obj)) else obj
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize(i) for i in obj]
            return obj
        return json.dumps(sanitize(content), ensure_ascii=False).encode("utf-8")
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

class HaltRequest(BaseModel):
    class Config:
        extra = "allow"
        arbitrary_types_allowed = True

class TradingBot(BaseModel):
    class Config:
        extra = "allow"
        arbitrary_types_allowed = True

from constants import COL_EMA9, COL_EMA21, COL_EMA50, COL_RSI, COL_ATR

from profiles.registry import get_coin_profile
from profiles.base_profile import PROFILE_EMOJI

if TYPE_CHECKING:
    from main import TradingBot

log = logging.getLogger("api")
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def _get_api_key_from_env() -> str:
    key = os.getenv("DASHBOARD_API_KEY", "")
    if not key or len(key) < 16:
        raise RuntimeError(
            "DASHBOARD_API_KEY tidak diset atau terlalu pendek (min 16 karakter). "
            "Generate dengan: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    return key

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    try:
        valid_key = _get_api_key_from_env()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    if not api_key or not secrets.compare_digest(api_key, valid_key):
        raise HTTPException(status_code=401, detail="API key tidak valid.")
    return api_key

def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

def _dur(entry_time: Optional[datetime]) -> str:
    if entry_time is None:
        return "00:00:00"
    delta  = _utcnow() - entry_time
    total  = max(int(delta.total_seconds()), 0)
    h, rem = divmod(total, 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _fmt_usd(v) -> str:
    if v is None:
        return "—"
    v = float(v)
    if v >= 1000:
        return f"${v:,.2f}"
    if v >= 1:
        return f"${v:.4f}"
    if v >= 0.01:
        return f"${v:.5f}"
    return f"${v:.8f}"

def _pos_dict(pos) -> dict:
    return {
        "id":                 pos.id,
        "symbol":             pos.symbol,
        "side":               pos.side,
        "entry_time":         _iso(pos.entry_time),
        "entry_price":        pos.entry_price,
        "current_price":      pos.current_price,
        "amount":             pos.amount,
        "unrealized_pnl":     pos.unrealized_pnl,
        "unrealized_pnl_pct": pos.unrealized_pnl_pct,
        "realized_pnl":       pos.realized_pnl,
        "realized_pnl_pct":   pos.realized_pnl_pct,
        "stop_loss_price":    pos.stop_loss_price,
        "take_profit_price":  pos.take_profit_price,
        "atr_at_entry":       pos.atr_at_entry,
        "strategy":           pos.strategy_name,
        "profile":            pos.strategy_profile or "",
        "entry_order_id":     pos.entry_order_id,
        "duration_secs":      int(
            (_utcnow() - pos.entry_time).total_seconds()
            if pos.entry_time else 0
        ),
        "duration_display":   _dur(pos.entry_time),
        "is_open":            pos.is_open,
        "is_closing":         getattr(pos, "is_closing", False),
    }

def _trade_dict(t) -> dict:
    return {
        "id":                t.id,
        "timestamp":         _iso(t.timestamp),
        "symbol":            t.symbol,
        "side":              t.side,
        "order_type":        t.order_type,
        "status":            t.status,
        "requested_price":   t.requested_price,
        "executed_price":    t.executed_price,
        "amount":            t.amount,
        "filled":            t.filled,
        "cost":              t.cost,
        "fee_cost":          t.fee_cost,
        "fee_currency":      t.fee_currency,
        "fee_rate":          t.fee_rate,
        "slippage_pct":      t.slippage_pct,
        "stop_loss_price":   t.stop_loss_price,
        "take_profit_price": t.take_profit_price,
        "realized_pnl":      t.realized_pnl,
        "realized_pnl_pct":  t.realized_pnl_pct,
        "strategy":          t.strategy_name,
        "signal_origin":     t.signal_origin,
        "notes":             t.notes,
    }

def create_app(bot_getter) -> FastAPI:
    app = FastAPI(
        title="AlgoTrader Pro API v7.0",
        version="7.0.0",
        description="Real-time dashboard API for AlgoTrader Pro — Intelligence Pipeline",
        default_response_class=SafeJSONResponse,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000,http://127.0.0.1:3000").split(","),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )

    def bot() -> "TradingBot":
        b = bot_getter()
        if b is None:
            raise HTTPException(status_code=503, detail="Bot not initialised")
        return b

    dashboard_dir = os.path.join(os.path.dirname(__file__), "dashboard")
    if os.path.isdir(dashboard_dir):
        app.mount("/dashboard", StaticFiles(directory=dashboard_dir), name="dashboard")

    @app.get("/", response_class=HTMLResponse)
    async def root():
        index = os.path.join(dashboard_dir, "index.html")
        if os.path.isfile(index):
            with open(index, "r", encoding="utf-8") as f:
                return f.read()

        return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>AlgoTrader Pro API</title>
    <style>
      body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; background:#0b0d12; color:#e2e6f0; margin:0; padding:28px; }
      .card { max-width:720px; margin:0 auto; background:#10141c; border:1px solid #1e2433; border-radius:12px; padding:18px 18px 10px; }
      h1 { margin:0 0 8px; font-size:18px; }
      p { margin:8px 0; color:#a7afc7; line-height:1.45; }
      a { color:#7dd3fc; text-decoration:none; }
      code { background:#0b0d12; border:1px solid #1e2433; padding:2px 6px; border-radius:6px; }
      ul { margin:10px 0 0; padding-left:18px; }
      li { margin:6px 0; }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>AlgoTrader Pro API</h1>
      <p>Dashboard HTML belum tersedia (folder <code>dashboard/</code> kosong atau <code>index.html</code> tidak ada).</p>
      <p>Endpoint yang tersedia:</p>
      <ul>
        <li><a href="/health">/health</a></li>
        <li><a href="/api/status">/api/status</a></li>
        <li><a href="/api/balance">/api/balance</a></li>
        <li><a href="/api/positions">/api/positions</a></li>
        <li><a href="/api/trades">/api/trades</a></li>
        <li><a href="/api/intelligence/scores">/api/intelligence/scores</a></li>
        <li><a href="/api/intelligence/regime">/api/intelligence/regime</a></li>
      </ul>
    </div>
  </body>
</html>
""".strip()

    @app.get("/health")
    async def health():
        return {"status": "ok", "time": _iso(_utcnow())}

    @app.get("/api/status")
    async def get_status():
        b      = bot()
        uptime = int((_utcnow() - b.start_time).total_seconds()) if b.start_time else 0
        halted = b.risk_manager.is_halted if b.risk_manager else False
        halt_reason = b.risk_manager.halt_reason if b.risk_manager else ""
        return {
            "status":          "running" if b.is_running else "stopped",
            "halted":          halted,
            "halt_reason":     halt_reason,
            "exchange":        b.config.get("exchange_id"),
            "testnet":         b.config.get("testnet"),
            "connected":       b.exchange.is_connected if b.exchange else False,
            "strategy":        b.strategy.name if b.strategy else None,
            "strategy_active": b.strategy.is_active if b.strategy else False,
            "universe_watchlist": b.config.get("universe_watchlist", []),
            "watchlist": (lambda: __import__('json').load(open('universe.json')).get('symbols',[]) and [c['symbol'] for c in __import__('json').load(open('universe.json')).get('symbols',[])] if __import__('os').path.exists('universe.json') else b.config.get("universe_watchlist", []))(),
            "timeframe":       b.config.get("timeframe"),
            "uptime_secs":     uptime,
            "uptime_display":  str(timedelta(seconds=uptime)),
            "timestamp":       _iso(_utcnow()),
        }

    @app.get("/api/balance")
    async def get_balance():
        b  = bot()
        ps = b.portfolio_state
        drawdown_pct = (
            b.risk_manager.current_drawdown_pct
            if b.risk_manager
            else 0.0
        )
        return {
            "total_equity":   ps.get("total_equity",   0),
            "free_balance":   ps.get("free_balance",   0),
            "locked_balance": ps.get("locked_balance", 0),
            "open_pnl":       ps.get("open_pnl",       0),
            "daily_pnl":      ps.get("daily_pnl",      0),
            "daily_pnl_pct":  ps.get("daily_pnl_pct",  0),
            "drawdown_pct":   drawdown_pct,
            "currency":       b.config.get("quote_currency", "USDT"),
            "timestamp":      _iso(_utcnow()),
        }

    @app.get("/api/positions")
    async def get_positions():
        b         = bot()
        positions = await b.db.get_open_positions()
        return {
            "positions": [_pos_dict(p) for p in positions],
            "count":     len(positions),
        }

    @app.get("/api/trades")
    async def get_trades(limit: int = 50):
        b      = bot()
        trades = await b.db.get_recent_trades(limit=min(limit, 200))
        return {"trades": [_trade_dict(t) for t in trades], "count": len(trades)}

    @app.get("/api/equity_curve")
    async def get_equity_curve(limit: int = 500):
        b     = bot()
        snaps = await b.db.get_equity_curve(limit=limit)
        return {
            "curve": [
                {
                    "timestamp":     _iso(s.timestamp),
                    "equity":        s.total_equity,
                    "drawdown":      s.drawdown_pct,
                    "daily_pnl":     s.daily_pnl,
                    "daily_pnl_pct": s.daily_pnl_pct,
                }
                for s in snaps
            ]
        }

    @app.get("/api/metrics")
    async def get_metrics():
        b      = bot()
        rm     = b.risk_manager
        trades = await b.db.get_recent_trades(limit=500)
        closed = [t for t in trades if t.realized_pnl is not None]
        pnl_list = [float(t.realized_pnl) for t in closed]

        snaps    = await b.db.get_equity_curve(limit=500)
        eq_curve = [float(s.total_equity) for s in snaps]
        max_dd   = rm.compute_max_drawdown(eq_curve)

        initial       = b.config.get("initial_capital", 1.0)
        last_eq       = eq_curve[-1] if eq_curve else initial
        total_ret_pct = (last_eq - initial) / initial * 100

        attribution_summary  = {}
        indicator_summary    = {}
        if getattr(b, "analytics", None):
            try:
                snap = await b.db.get_latest_snapshot(scope="global", lookback_days=30)
                if snap:
                    attribution_summary = {
                        "best_regime":   snap.get("best_regime"),
                        "worst_regime":  snap.get("worst_regime"),
                        "lookback_days": snap.get("lookback_days"),
                        "computed_at":   _iso(snap.get("computed_at")),
                    }
                indicator_eff = await b.db.get_indicator_effectiveness(lookback_days=30)
                indicator_summary = indicator_eff or {}
            except Exception as e:
                log.warning("Tidak bisa ambil analytics summary: %s", e)

        return {
            "total_trades":         len(closed),
            "win_rate_pct":         round(rm.compute_win_rate(pnl_list),           4),
            "total_pnl":            round(sum(pnl_list),                            6),
            "avg_pnl_per_trade":    round(rm.compute_expectancy(pnl_list),          6),
            "profit_factor":        (lambda pf: 9999.0 if math.isinf(pf) else round(pf, 4))(rm.compute_profit_factor(pnl_list)),
            "expectancy":           round(rm.compute_expectancy(pnl_list),          6),
            "avg_win_loss_ratio":   round(rm.compute_avg_win_loss_ratio(pnl_list),  4),
            "max_drawdown_pct":     round(max_dd,                                   4),
            "current_drawdown_pct": round(rm.current_drawdown_pct,                 4),
            "sharpe_ratio":         round(rm.compute_sharpe_ratio(pnl_list),        4),
            "sortino_ratio":        round(rm.compute_sortino_ratio(pnl_list),       4),
            "calmar_ratio":         round(rm.compute_calmar_ratio(total_ret_pct, max_dd), 4),
            "total_fees":           round(sum(t.fee_cost or 0 for t in trades),     6),
            "open_positions":       len(await b.db.get_open_positions()),
            "daily_loss_pct":       round(rm.daily_loss_pct,                        4),
            "daily_loss_limit_pct": rm.daily_loss_limit_pct,
            "halt_reason":          rm.halt_reason,
            # v7.0 additions
            "attribution_summary":  attribution_summary,
            "indicator_summary":    indicator_summary,
            "timestamp":            _iso(_utcnow()),
        }

    @app.get("/api/logs")
    async def get_logs(limit: int = 100):
        b    = bot()
        logs = await b.db.get_recent_logs(limit=min(limit, 500))
        return {
            "logs": [
                {
                    "timestamp": _iso(l.timestamp),
                    "level":     l.level,
                    "module":    l.module,
                    "message":   l.message,
                }
                for l in logs
            ]
        }

    @app.get("/api/candles/{symbol:path}")
    async def get_candles(symbol: str, timeframe: str = "15m", limit: int = 100):
        b = bot()

        if not b.exchange or not b.exchange.is_connected:
            raise HTTPException(status_code=503, detail="Exchange belum terhubung")

        try:
            raw     = await b.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            candles = [
                {
                    "timestamp":    bar[0],
                    "open":         bar[1],
                    "high":         bar[2],
                    "low":          bar[3],
                    "close":        bar[4],
                    "volume":       bar[5],
                    "quote_volume": bar[6] if len(bar) > 6 else None,
                }
                for bar in raw
            ]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OHLCV error: {e}")

        trades  = await b.db.get_recent_trades(limit=200)
        markers = [
            {
                "timestamp": t.timestamp.timestamp() * 1000 if t.timestamp else None,
                "price":     t.executed_price,
                "side":      t.side,
                "origin":    t.signal_origin,
                "slippage":  t.slippage_pct,
                "fee":       t.fee_cost,
            }
            for t in trades
            if t.symbol == symbol and t.executed_price
        ]

        return {"candles": candles, "markers": markers}

    @app.get("/api/tickers")
    async def get_tickers():
        b = bot()
        return {"tickers": b.ws_feed.live_tickers if b.ws_feed else {}}

    @app.get("/api/market_info/{symbol:path}")
    async def get_market_info(symbol: str):
        b      = bot()
        market = b.exchange.get_market_info(symbol)
        price_data: dict = {}
        if b.ws_feed:
            ticker     = b.ws_feed.live_tickers.get(symbol, {})
            price_data = {
                "last_price":       ticker.get("last"),
                "mid_price":        b.ws_feed.get_mid_price(symbol),
                "bid":              ticker.get("bid"),
                "ask":              ticker.get("ask"),
                "spread_pct":       b.ws_feed.get_spread(symbol),
                "spread_abs":       b.ws_feed.get_spread_absolute(symbol),
                "volume_base_24h":  ticker.get("volume"),
                "volume_quote_24h": ticker.get("quote_volume"),
                "high_24h":         ticker.get("high_24h"),
                "low_24h":          ticker.get("low_24h"),
                "change_pct_24h":   ticker.get("change_pct"),
                "feed_healthy":     b.ws_feed.is_feed_healthy(symbol),
            }
        return {**market, **price_data, "timestamp": _iso(_utcnow())}

    @app.get("/api/system_health")
    async def get_system_health():
        b       = bot()
        health  = (
            b.risk_manager.get_system_health()
            if b.risk_manager
            else {
                "risk_status": "initializing",
                "halted": False,
                "halt_reason": "",
                "drawdown_pct": 0.0,
            }
        )
        avg_lat = await b.db.get_avg_latency_ms()
        metrics = await b.db.get_api_metrics(limit=20)
        feed_st = b.ws_feed.get_feed_status() if b.ws_feed else {}

        return {
            **health,
            "avg_api_latency_ms": round(avg_lat, 2),
            "strategy_active":    b.strategy.is_active if b.strategy else False,
            "strategy_name":      b.strategy.name if b.strategy else None,
            "testnet":            b.config.get("testnet", True),
            "universe_watchlist": b.config.get("universe_watchlist", []),
            "ws_feed_status":     feed_st,
            "recent_latencies": [
                {
                    "ts":         _iso(m.timestamp),
                    "endpoint":   m.endpoint,
                    "latency_ms": m.latency_ms,
                    "success":    m.success,
                    "error":      m.error_msg,
                }
                for m in metrics
            ],
            "timestamp": _iso(_utcnow()),
        }

    @app.get("/api/dashboard_snapshot")
    async def get_dashboard_snapshot():
        b = bot()
        positions = await b.db.get_open_positions()
        trades = await b.db.get_recent_trades(limit=120)
        snaps = await b.db.get_equity_curve(limit=300)
        health = (
            b.risk_manager.get_system_health()
            if b.risk_manager
            else {
                "risk_status": "initializing",
                "halted": False,
                "halt_reason": "",
                "drawdown_pct": 0.0,
            }
        )
        feed_st = b.ws_feed.get_feed_status() if b.ws_feed else {}
        return {
            "status": await get_status(),
            "balance": await get_balance(),
            "metrics": await get_metrics(),
            "system_health": {
                **health,
                "strategy_active": b.strategy.is_active if b.strategy else False,
                "strategy_name": b.strategy.name if b.strategy else None,
                "ws_feed_status": feed_st,
                "timestamp": _iso(_utcnow()),
            },
            "positions": {
                "positions": [_pos_dict(p) for p in positions],
                "count": len(positions),
            },
            "tickers": {"tickers": b.ws_feed.live_tickers if b.ws_feed else {}},
            "logs": await get_logs(limit=20),
            "equity_curve": {
                "curve": [
                    {
                        "timestamp": _iso(s.timestamp),
                        "equity": s.total_equity,
                        "drawdown": s.drawdown_pct,
                        "daily_pnl": s.daily_pnl,
                        "daily_pnl_pct": s.daily_pnl_pct,
                    }
                    for s in snaps
                ]
            },
            "trades": {"trades": [_trade_dict(t) for t in trades], "count": len(trades)},
        }

    @app.get("/api/diagnosa")
    async def get_diagnosa():
        b          = bot()
        universe  = b.config.get("universe_watchlist", [])
        is_testnet = b.config.get("testnet", True)
        results: list[dict] = []

        _TF_FALLBACK = {
            "1d":  ["4h", "1h"],
            "4h":  ["1h"],
            "1h":  ["15m"],
            "15m": [],
        }

        for symbol in universe:
            entry: dict = {"symbol": symbol}
            try:
                tf = (
                    b.strategy.get_symbol_timeframe(symbol)
                    if b.strategy
                    else b.config.get("timeframe", "15m")
                )

                observation = None
                if hasattr(b, "observer") and b.observer:
                    try:
                        observation = await b.observer.get_cached_observation(symbol, tf)
                    except Exception:
                        pass

                if observation:
                    ind = observation.indicator_set
                    entry.update({
                        "profile":        observation.profile,
                        "regime":         observation.regime.value if observation.regime else "undefined",
                        "regime_confidence": observation.regime_confidence,
                        "total_score":    observation.total_score,
                        "trigger_met":    observation.trigger_met,
                        "threshold":      observation.entry_threshold,
                        "breakdown": {
                            "trend":      ind.trend.composite_score    if ind.trend    else None,
                            "momentum":   ind.momentum.composite_score if ind.momentum else None,
                            "strength":   ind.strength.composite_score if ind.strength else None,
                            "volatility": ind.volatility.composite_score if ind.volatility else None,
                            "pattern":    ind.patterns.composite_score  if ind.patterns  else None,
                        },
                        "narrative":      getattr(observation, "narrative", None),
                        "calculation_errors": getattr(observation, "calculation_errors", []),
                        "tf_used":        tf,
                        "last_updated":   _iso(observation.timestamp),
                        "source":         "observer",
                    })

                    try:
                        open_pos = [
                            p for p in await b.db.get_open_positions()
                            if p.symbol == symbol
                        ]
                        if open_pos:
                            pos = open_pos[0]
                            entry["open_position"] = {
                                "entry_score":       getattr(pos, "entry_score", None),
                                "current_score":     observation.total_score,
                                "score_delta":       (
                                    round(observation.total_score - pos.entry_score, 2)
                                    if getattr(pos, "entry_score", None) is not None
                                    else None
                                ),
                                "entry_price":       pos.entry_price,
                                "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                            }
                    except Exception:
                        pass

                else:
                    bars    = None
                    tf_used = tf
                    tf_note = ""

                    for tf_try in [tf] + _TF_FALLBACK.get(tf, []):
                        try:
                            candidate = await b.exchange.fetch_ohlcv(
                                symbol, tf_try, limit=250
                            )
                            if candidate and len(candidate) >= 60:
                                bars    = candidate
                                tf_used = tf_try
                                if tf_try != tf:
                                    tf_note = f" ⚠️fallback:{tf_try}"
                                break
                        except Exception as tf_err:
                            log.debug("Diagnosa TF fallback %s [%s]: %s", symbol, tf_try, tf_err)
                            continue

                    if not bars or len(bars) < 60:
                        note = " (testnet — data terbatas)" if is_testnet else ""
                        entry["error"] = (
                            f"Data tidak cukup ({len(bars) if bars else 0} bar){note}"
                        )
                        results.append(entry)
                        continue

                    cols = ["timestamp", "open", "high", "low", "close", "volume"]
                    df   = pd.DataFrame(bars, columns=cols)
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df.set_index("timestamp", inplace=True)

                    if len(bars[0]) > 6:
                        df["quote_volume"] = [
                            float(r[6]) if len(r) > 6 and r[6] is not None
                            else float(r[4]) * float(r[5])
                            for r in bars
                        ]
                    else:
                        df["quote_volume"] = df["volume"] * df["close"]

                    df.ta.ema(length=9,  append=True)
                    df.ta.ema(length=21, append=True)
                    df.ta.ema(length=50, append=True)
                    df.ta.rsi(length=14, append=True)
                    df.ta.atr(length=14, append=True)
                    df = df.dropna()

                    if len(df) < 5:
                        entry["error"] = f"Indikator tidak cukup ({len(df)} bar)"
                        results.append(entry)
                        continue

                    df["_resistance"] = df["close"].shift(1).rolling(20).max()
                    df["_vol_ma"]     = df["quote_volume"].rolling(20).mean()
                    df = df.dropna()

                    bar_row  = df.iloc[-2]
                    prev_row = df.iloc[-3]

                    close      = float(bar_row["close"])
                    ema9       = float(bar_row[COL_EMA9])
                    ema21      = float(bar_row[COL_EMA21])
                    ema50      = float(bar_row[COL_EMA50])
                    rsi        = float(bar_row[COL_RSI])
                    atr        = float(bar_row[COL_ATR])
                    atr_pct    = (atr / close * 100) if close > 0 else 0

                    resist = (
                        float(bar_row["_resistance"])
                        if pd.notna(bar_row.get("_resistance"))
                        else close
                    )
                    vol_ma_v = (
                        float(bar_row["_vol_ma"])
                        if pd.notna(bar_row.get("_vol_ma")) and float(bar_row["_vol_ma"]) > 0
                        else float(df["quote_volume"].mean())
                    )
                    vol       = float(bar_row["quote_volume"]) if pd.notna(bar_row.get("quote_volume")) else float(bar_row["volume"])
                    vol_ratio = vol / vol_ma_v if vol_ma_v > 0 else 0.0
                    vol_warn  = " ⚠️sandbox" if (is_testnet and vol_ma_v < 1.0) else ""

                    prev_ema9  = float(prev_row[COL_EMA9])
                    prev_ema21 = float(prev_row[COL_EMA21])

                    prof         = get_coin_profile(symbol)
                    min_dist     = close * (prof.min_breakout_pct / 100)
                    brk_dist     = (close - resist) if resist > 0 else 0.0
                    trigger_a    = (brk_dist >= min_dist) and (vol_ratio >= prof.volume_mult)
                    golden_cross = (prev_ema9 <= prev_ema21) and (ema9 > ema21)
                    trigger_b    = golden_cross and (rsi > prof.rsi_gc_min)
                    cond_trend   = ema9 > ema21 > ema50
                    cond_momentum = prof.rsi_min <= rsi <= prof.rsi_max

                    cond_vwap = True
                    if tf not in ("1d", "3d", "1w"):
                        for vwap_col in ("VWAP_D", "VWAP", "vwap"):
                            if vwap_col in bar_row.index and pd.notna(bar_row[vwap_col]):
                                vwap_val = float(bar_row[vwap_col])
                                if vwap_val > 0:
                                    cond_vwap = close > vwap_val
                                    break

                    entry_ok = (
                        (trigger_a or trigger_b)
                        and cond_trend
                        and cond_momentum
                        and cond_vwap
                    )

                    failed_conditions = []
                    if not (trigger_a or trigger_b):
                        failed_conditions.append(
                            f"NoTrig(vol={vol_ratio:.1f}x,gc={'✅' if golden_cross else '❌'})"
                        )
                    if not cond_trend:
                        failed_conditions.append("EMAStack")
                    if not cond_momentum:
                        failed_conditions.append(
                            f"RSI({rsi:.0f} not in [{prof.rsi_min},{prof.rsi_max}])"
                        )
                    if not cond_vwap:
                        failed_conditions.append("BelowVWAP")

                    cond_count = sum([
                        bool(trigger_a or trigger_b),
                        cond_trend,
                        cond_momentum,
                        cond_vwap,
                    ])

                    if atr > 0:
                        sl_val = close - max(atr * prof.atr_sl_mult, close * (prof.quick_sl_pct / 100))
                        tp_val = close + max(atr * prof.atr_tp_mult, close * (prof.quick_tp_pct / 100))
                    else:
                        sl_val = close * (1 - prof.quick_sl_pct / 100)
                        tp_val = close * (1 + prof.quick_tp_pct / 100)

                    exit_mode = (
                        "RIDE_THE_WAVE"
                        if (vol_ratio >= prof.volume_spike or atr_pct >= prof.atr_pct_threshold)
                        else "QUICK_PROFIT"
                    )

                    prof_emoji = PROFILE_EMOJI.get(prof.profile.value, "⚙️")

                    # Baca score & regime dari DB (signal_scores)
                    _db_score  = None
                    _db_regime = "undefined"
                    _db_source = "fallback_v6"
                    _row       = None
                    try:
                        _row = await b.db.get_latest_signal_score(symbol)
                        if _row:
                            _db_score  = _row.total_score
                            _db_regime = _row.regime or "undefined"
                            _db_source = "database"
                    except Exception as _dbe:
                        log.debug("Diagnosa: gagal baca signal_scores untuk %s: %s", symbol, _dbe)

                    # Ambil threshold dari profile atau DB
                    _db_threshold = 70.0
                    try:
                        _db_threshold = (
                            getattr(prof, "entry_threshold", None)
                            or getattr(prof, "min_score", None)
                            or (_row.threshold_used if _row and hasattr(_row, "threshold_used") else None)
                            or 70.0
                        )
                    except Exception:
                        _db_threshold = 70.0

                    entry.update({
                        "profile":         f"{prof_emoji} {prof.profile.value}",
                        "regime":          _db_regime,
                        "total_score":     _db_score,
                        "threshold":       _db_threshold,
                        "trigger_met":     entry_ok,
                        "conditions_met":  cond_count,
                        "conditions_total": 4,
                        "failed_conditions": failed_conditions,
                        "price":           close,
                        "sl":              round(sl_val, 8),
                        "tp":              round(tp_val, 8),
                        "rsi":             round(rsi, 2),
                        "vol_ratio":       round(vol_ratio, 2),
                        "atr_pct":         round(atr_pct, 4),
                        "vol_warn":        bool(vol_warn),
                        "exit_mode":       exit_mode,
                        "tf_used":         tf_used,
                        "tf_note":         tf_note,
                        "source":          _db_source,
                    })

            except Exception as e:
                log.error("Diagnosa error [%s]: %s", symbol, e, exc_info=True)
                entry["error"] = str(e)[:120]

            results.append(entry)

        return {
            "results":         results,
            "universe_count": len(universe),
            "testnet":         is_testnet,
            "timestamp":       _iso(_utcnow()),
        }

    @app.get("/api/intelligence/scores")
    async def get_intelligence_scores():
        b         = bot()
        universe = b.config.get("universe_watchlist", [])

        scores: list[dict] = []
        for symbol in universe:
            try:
                row = await b.db.get_latest_signal_score(symbol)
                if row:
                    scores.append({
                        "symbol":      symbol,
                        "total_score": row.total_score,
                        "breakdown": {
                            "trend":      row.trend_score,
                            "momentum":   row.momentum_score,
                            "strength":   row.strength_score,
                            "volatility": row.volatility_score,
                            "pattern":    row.pattern_score,
                        },
                        "regime":       row.regime,
                        "trigger_met":  row.trigger_met,
                        "action_taken": row.action_taken,
                        "last_updated": _iso(row.timestamp),
                    })
                else:
                    scores.append({
                        "symbol":       symbol,
                        "total_score":  None,
                        "breakdown":    {},
                        "regime":       "undefined",
                        "trigger_met":  False,
                        "action_taken": None,
                        "last_updated": None,
                    })
            except Exception as e:
                log.warning("get_intelligence_scores [%s]: %s", symbol, e)
                scores.append({"symbol": symbol, "error": str(e)})

        scores.sort(key=lambda x: (x.get("total_score") is not None, x.get("total_score") or 0), reverse=True)

        return {
            "scores":    scores,
            "count":     len(scores),
            "timestamp": _iso(_utcnow()),
        }

    @app.get("/api/intelligence/scores/{symbol:path}")
    async def get_intelligence_score_detail(symbol: str):
        b = bot()

        latest = await b.db.get_latest_signal_score(symbol)
        if not latest:
            raise HTTPException(
                status_code=404,
                detail=f"Belum ada score untuk {symbol}. Bot mungkin belum menganalisis coin ini."
            )

        history_rows = await b.db.get_signal_scores(symbol=symbol, limit=96)
        history = [
            {
                "timestamp":   _iso(r.timestamp),
                "total_score": r.total_score,
                "regime":      r.regime,
                "trigger_met": r.trigger_met,
                "action":      r.action_taken,
            }
            for r in history_rows
        ]

        try:
            from profiles.thresholds import get_dynamic_threshold
            _latest_regime = latest.regime if latest and hasattr(latest, "regime") else "undefined"
            entry_threshold = get_dynamic_threshold(symbol.split("/")[0], _latest_regime)
        except Exception:
            entry_threshold = 70.0

        return {
            "symbol":          symbol,
            "total_score":     latest.total_score,
            "entry_threshold": entry_threshold,
            "above_threshold": (
                latest.total_score >= entry_threshold
                if latest.total_score is not None else False
            ),
            "breakdown": {
                "trend":      latest.trend_score,
                "momentum":   latest.momentum_score,
                "strength":   latest.strength_score,
                "volatility": latest.volatility_score,
                "pattern":    latest.pattern_score,
            },
            "regime":            latest.regime,
            "trigger_met":       latest.trigger_met,
            "action_taken":      latest.action_taken,
            "rejection_reason":  latest.rejection_reason,
            "profile":           latest.strategy_profile,
            "narrative":         getattr(latest, "narrative", None),
            "history_24h":       history,
            "last_updated":      _iso(latest.timestamp),
            "timestamp":         _iso(_utcnow()),
        }

    @app.get("/api/intelligence/regime")
    async def get_intelligence_regime():
        b         = bot()
        universe = b.config.get("universe_watchlist", [])

        regimes: list[dict] = []
        for symbol in universe:
            try:
                row = await b.db.get_latest_regime(symbol)
                if row:
                    regimes.append({
                        "symbol":     symbol,
                        "regime":     row.regime,
                        "confidence": round(row.regime_confidence, 4),
                        "adx":        row.adx_value,
                        "atr_pct":    row.atr_pct,
                        "bb_width":   row.bb_width,
                        "last_updated": _iso(row.timestamp),
                    })
                else:
                    regimes.append({
                        "symbol":     symbol,
                        "regime":     "undefined",
                        "confidence": 0.0,
                        "last_updated": None,
                    })
            except Exception as e:
                log.warning("get_intelligence_regime [%s]: %s", symbol, e)
                regimes.append({"symbol": symbol, "regime": "undefined", "error": str(e)})

        regime_counts: dict[str, int] = {}
        for r in regimes:
            regime_counts[r.get("regime", "undefined")] = (
                regime_counts.get(r.get("regime", "undefined"), 0) + 1
            )

        return {
            "regimes":       regimes,
            "summary":       regime_counts,
            "universe_count": len(universe),
            "timestamp":     _iso(_utcnow()),
        }

    @app.get("/api/analytics/attribution")
    async def get_analytics_attribution(
        lookback_days: int = 30,
        profile: Optional[str] = None,
        symbol: Optional[str] = None,
    ):
        b = bot()

        if not hasattr(b, "analytics") or not b.analytics:
            raise HTTPException(
                status_code=503,
                detail="Analytics engine belum diinisialisasi."
            )

        try:
            filters = {}
            if profile:
                filters["profile"] = profile
            if symbol:
                filters["symbol"] = symbol

            report = await b.analytics.compute_attribution(
                lookback_days=lookback_days,
                filters=filters,
            )
        except Exception as e:
            log.error("Attribution computation error: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Attribution error: {e}")

        return {
            "attribution": report,
            "lookback_days": lookback_days,
            "filters":       {"profile": profile, "symbol": symbol},
            "timestamp":     _iso(_utcnow()),
        }

    @app.get("/api/analytics/indicator_effectiveness")
    async def get_indicator_effectiveness(lookback_days: int = 30):
        b = bot()

        if not hasattr(b, "analytics") or not b.analytics:
            raise HTTPException(
                status_code=503,
                detail="Analytics engine belum diinisialisasi."
            )

        try:
            report = await b.analytics.compute_indicator_effectiveness(
                lookback_days=lookback_days
            )
        except Exception as e:
            log.error("Indicator effectiveness error: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Analytics error: {e}")

        return {
            "indicator_effectiveness": report,
            "lookback_days": lookback_days,
            "timestamp":     _iso(_utcnow()),
        }

    @app.get("/api/analytics/regime_performance")
    async def get_regime_performance(lookback_days: int = 30):
        b = bot()

        if not hasattr(b, "analytics") or not b.analytics:
            raise HTTPException(
                status_code=503,
                detail="Analytics engine belum diinisialisasi."
            )

        try:
            report = await b.analytics.compute_attribution(
                lookback_days=lookback_days,
                filters={},
                group_by="regime",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analytics error: {e}")

        return {
            "regime_performance": report,
            "lookback_days":     lookback_days,
            "timestamp":         _iso(_utcnow()),
        }

    @app.post("/api/analytics/refresh")
    async def refresh_analytics(_: str = Depends(verify_api_key)):
        b = bot()

        if not hasattr(b, "analytics") or not b.analytics:
            raise HTTPException(
                status_code=503,
                detail="Analytics engine belum diinisialisasi."
            )

        try:
            await b.analytics.run_full_analysis()
            return {
                "status":    "refreshed",
                "timestamp": _iso(_utcnow()),
            }
        except Exception as e:
            log.error("Analytics refresh error: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Refresh error: {e}")

    @app.get("/api/meta_learner/suggestions")
    async def get_meta_learner_suggestions():
        b = bot()
        try:
            rows = await b.db.get_pending_suggestions(limit=200)
        except Exception as e:
            log.error("get_suggestions error: %s", e)
            rows = []

        suggestions = []
        for row in rows:
            suggestions.append({
                "id":                 row.get("id"),
                "created_at":         _iso(row.get("timestamp")),
                "symbol":             row.get("symbol"),
                "profile":            row.get("profile"),
                "parameter_name":     row.get("parameter_name"),
                "old_value":          row.get("old_value"),
                "new_value":          row.get("new_value"),
                "reason":             row.get("reason"),
                "confidence":         row.get("confidence"),
                "projected_improvement": row.get("projected_improvement"),
                "status":             row.get("status", "pending"),
            })

        return {
            "suggestions": suggestions,
            "count":       len(suggestions),
            "timestamp":   _iso(_utcnow()),
        }

    @app.post("/api/meta_learner/approve/{suggestion_id}")
    async def approve_suggestion(
        suggestion_id: str,
        _: str = Depends(verify_api_key),
    ):
        b = bot()

        if not hasattr(b, "meta_learner") or not b.meta_learner:
            raise HTTPException(
                status_code=503,
                detail="Meta-learner belum diinisialisasi."
            )

        try:
            ok, msg = await b.meta_learner.approve_suggestion(
                suggestion_id=suggestion_id,
                approved_by="manual_api",
            )
            log.info("Suggestion %s approved via API", suggestion_id)
            return {
                "status":        "approved",
                "suggestion_id": suggestion_id,
                "applied":       ok,
                "message":       msg,
                "timestamp":     _iso(_utcnow()),
            }
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            log.error("approve_suggestion error: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Approve error: {e}")

    @app.post("/api/meta_learner/reject/{suggestion_id}")
    async def reject_suggestion(
        suggestion_id: str,
        _: str = Depends(verify_api_key),
    ):
        b = bot()

        if not hasattr(b, "meta_learner") or not b.meta_learner:
            raise HTTPException(
                status_code=503,
                detail="Meta-learner belum diinisialisasi."
            )

        try:
            ok, msg = await b.meta_learner.reject_suggestion(
                suggestion_id=suggestion_id,
                rejected_by="manual_api",
            )
            log.info("Suggestion %s rejected via API", suggestion_id)
            return {
                "status":        "rejected",
                "suggestion_id": suggestion_id,
                "rejected":      ok,
                "message":       msg,
                "timestamp":     _iso(_utcnow()),
            }
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            log.error("reject_suggestion error: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Reject error: {e}")

    @app.get("/api/meta_learner/history")
    async def get_parameter_history(
        symbol: Optional[str] = None,
        profile: Optional[str] = None,
        limit: int = 50,
    ):
        b = bot()
        try:
            rows = await b.db.get_parameter_history(
                symbol=symbol,
                profile=profile,
                limit=min(limit, 200),
            )
        except Exception as e:
            log.error("get_parameter_history error: %s", e)
            rows = []

        history = []
        for row in rows:
            history.append({
                "id":                  row.get("id"),
                "timestamp":           _iso(row.get("timestamp")),
                "symbol":              row.get("symbol"),
                "profile":             row.get("profile"),
                "parameter_name":      row.get("parameter_name"),
                "old_value":           row.get("old_value"),
                "new_value":           row.get("new_value"),
                "reason":              row.get("reason"),
                "approved_by":         row.get("approved_by"),
                "performance_before":  None,
                "performance_after":   None,
                "outcome":             row.get("outcome"),
                "trades_after_apply":  row.get("trades_after_apply"),
            })

        return {
            "history":   history,
            "count":     len(history),
            "timestamp": _iso(_utcnow()),
        }

    class HaltRequest(BaseModel):
        reason: Optional[str] = "Manual halt from dashboard"

    @app.post("/api/bot/halt")
    async def halt_bot(req: HaltRequest, _: str = Depends(verify_api_key)):
        from risk import HaltReason
        bot().risk_manager.halt_trading(HaltReason.MANUAL, req.reason or "")
        return {"status": "halted", "reason": req.reason}

    @app.post("/api/bot/resume")
    async def resume_bot(_: str = Depends(verify_api_key)):
        bot().risk_manager.resume_trading()
        return {"status": "running"}

    @app.post("/api/bot/pause_strategy")
    async def pause_strategy(_: str = Depends(verify_api_key)):
        b = bot()
        if b.strategy:
            b.strategy.pause()
        return {"status": "strategy_paused"}

    @app.post("/api/bot/resume_strategy")
    async def resume_strategy(_: str = Depends(verify_api_key)):
        b = bot()
        if b.strategy:
            b.strategy.resume()
        return {"status": "strategy_running"}

    @app.post("/api/bot/panic")
    async def panic_close_all(_: str = Depends(verify_api_key)):
        from risk import HaltReason

        b = bot()
        log.critical("PANIC BUTTON ACTIVATED — closing all positions!")
        await b.db.save_log(
            "CRITICAL", "api", "PANIC BUTTON: closing all open positions"
        )

        positions    = await b.db.get_open_positions()
        closed_count = 0
        failed: list = []

        for pos in positions:
            try:
                price = await b._get_current_price(pos.symbol)
                if not price:
                    ticker = await b.exchange.fetch_ticker(pos.symbol)
                    price  = (
                        ticker.get("last")
                        or pos.current_price
                        or pos.entry_price
                    )

                await b._close_position_market(
                    pos, float(price), "PANIC BUTTON"
                )
                closed_count += 1
                log.info("Panic close: %s @ %.6f", pos.symbol, price)

            except Exception as e:
                log.error("Panic close FAILED for %s: %s", pos.symbol, e)
                failed.append(pos.symbol)

        b.risk_manager.halt_trading(
            HaltReason.PANIC_BUTTON,
            "Manual emergency close from dashboard",
        )

        return {
            "status":          "panic_executed",
            "positions_found": len(positions),
            "closed_count":    closed_count,
            "failed_symbols":  failed,
            "halted":          True,
            "timestamp":       _iso(_utcnow()),
        }


    @app.get("/api/crosslearn/status")
    async def crosslearn_status(_: str = Depends(verify_api_key)):
        b = bot()
        try:
            from learning.cross_learn import get_cross_learn_reader
            reader = get_cross_learn_reader()
            summary = reader.get_summary() if hasattr(reader, 'get_summary') else {}
            enabled = getattr(reader, 'enabled', False)
            return {
                "enabled": enabled,
                "summary": summary,
                "timestamp": _iso(_utcnow()),
            }
        except Exception as e:
            return {"enabled": False, "error": str(e)}

    @app.get("/api/crosslearn/swap_history")
    async def swap_history(_: str = Depends(verify_api_key)):
        b = bot()
        try:
            if not b._coin_swap:
                return {"swaps": [], "message": "CoinSwapEngine tidak aktif"}
            history = b._coin_swap.get_swap_history()
            return {"swaps": history, "total": len(history), "timestamp": _iso(_utcnow())}
        except Exception as e:
            return {"swaps": [], "error": str(e)}

    @app.post("/api/positions/{symbol}/close")
    async def close_position(symbol: str, _: str = Depends(verify_api_key)):
        import urllib.parse
        symbol = urllib.parse.unquote(symbol)
        b = bot()
        try:
            pos = await b.db.get_open_position_by_symbol(symbol)
            if not pos:
                return {"success": False, "message": f"Posisi {symbol} tidak ditemukan atau sudah closed"}
            price = await b._get_current_price(symbol)
            if not price:
                ticker = await b.exchange.fetch_ticker(symbol)
                price = ticker.get("last") or pos.current_price or pos.entry_price
            await b._close_position_market(pos, float(price), "MANUAL_CLOSE_DASHBOARD")
            return {"success": True, "message": f"Posisi {symbol} berhasil ditutup @ {price}"}
        except Exception as e:
            log.error("Manual close error [%s]: %s", symbol, e)
            return {"success": False, "error": str(e)}

    @app.post("/api/config/update")
    async def update_config(payload: dict, _: str = Depends(verify_api_key)):
        import json
        b = bot()
        try:
            allowed = [
                "universe_watchlist","max_open_positions","max_drawdown_pct",
                "risk_per_trade_pct","daily_loss_limit_pct","max_position_size_pct",
                "stop_loss_pct","take_profit_pct","atr_multiplier_sl","atr_multiplier_tp",
                "trailing_atr_mult","use_trailing_stop","telegram_enabled",
                "telegram_bot_token","telegram_chat_id","exchange_id",
                "api_key","api_secret","api_passphrase","testnet","initial_capital",
                "min_order_value_usdt","max_slippage_pct",
            ]
            updates = {k: v for k, v in payload.items() if k in allowed}
            if not updates:
                return {"success": False, "message": "Tidak ada field valid untuk diupdate"}
            await b.db.set_bot_state("config_update", json.dumps(updates))
            return {"success": True, "message": f"Config akan diupdate dalam 30 detik", "fields": list(updates.keys())}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.get("/api/forecast")
    async def get_forecast(_: str = Depends(verify_api_key)):
        from profiles.thresholds import DYNAMIC_THRESHOLD_MATRIX, ENTRY_THRESHOLDS
        from profiles.weights import LEVEL1_WEIGHTS
        from datetime import timedelta

        b = bot()
        universe = b.config.get("universe_watchlist", [])
        tf_primary = b.config.get("timeframe", "15m")
        forecasts = []

        TF_CONFIRM = {
            "15m": "1h", "30m": "2h", "1h": "4h", "5m": "15m",
        }
        HOLD_MINUTES = {
            "scalp_volatile":   {"trending_bull": 30, "volatile_expansion": 20, "ranging": 45, "undefined": 35},
            "extreme_momentum": {"trending_bull": 25, "volatile_expansion": 15, "ranging": 60, "undefined": 30},
            "breakout_swift":   {"trending_bull": 120, "volatile_expansion": 90, "ranging": 180, "undefined": 150},
            "trend_follow":     {"trending_bull": 480, "volatile_expansion": 300, "ranging": 600, "undefined": 360},
            "mean_revert":      {"trending_bull": 240, "volatile_expansion": 300, "ranging": 120, "undefined": 180},
            "hodl_accumulate":  {"trending_bull": 2880, "volatile_expansion": 4320, "ranging": 1440, "undefined": 2160},
        }

        for symbol in universe:
            try:
                row = await b.db.get_latest_signal_score(symbol)
                if not row or not row.current_price:
                    continue

                price   = row.current_price
                sl      = row.suggested_sl
                tp      = row.suggested_tp
                regime  = row.regime or "undefined"
                profile = row.strategy_profile or "scalp_volatile"
                conf    = row.signal_confidence or 0.5
                score   = row.total_score or 0.0

                potential_profit_pct = round((tp - price) / price * 100, 2) if tp and price else None
                potential_loss_pct   = round((price - sl) / price * 100, 2) if sl and price else None
                rr_ratio = round(potential_profit_pct / potential_loss_pct, 2) if potential_profit_pct and potential_loss_pct and potential_loss_pct > 0 else None

                score_pct      = min(100, max(0, score))
                threshold_used = row.threshold_used or DYNAMIC_THRESHOLD_MATRIX.get(profile, {}).get(regime, ENTRY_THRESHOLDS.get(profile, 65.0))
                probability_up = round((score_pct / 100 * 0.6 + conf * 0.25 + min(1.0, score_pct / max(threshold_used, 1)) * 0.15) * 100, 1)
                signal_quality = "excellent" if score_pct >= 85 else "good" if score_pct >= 70 else "fair" if score_pct >= 50 else "poor"

                score_breakdown = {
                    "trend":      row.trend_score,
                    "momentum":   row.momentum_score,
                    "strength":   row.strength_score,
                    "volatility": row.volatility_score,
                    "pattern":    row.pattern_score,
                    "oscillator": row.oscillator_score,
                    "structure":  row.structure_score,
                    "orderbook":  row.orderbook_score,
                }
                weights = LEVEL1_WEIGHTS.get(profile, {})
                dyn_threshold = DYNAMIC_THRESHOLD_MATRIX.get(profile, {}).get(regime, threshold_used)
                threshold_gap = round(score - dyn_threshold, 2)

                indicators = {}
                conf_tf_data = {}
                try:
                    if hasattr(b, "observer") and b.observer:
                        observation = await b.observer.get_cached_observation(symbol, tf_primary)
                        if observation and observation.primary_tf_indicators:
                            ind = observation.primary_tf_indicators
                            if ind.trend:
                                indicators["ema9"]        = round(ind.trend.ema9, 8) if ind.trend.ema9 else None
                                indicators["ema21"]       = round(ind.trend.ema21, 8) if ind.trend.ema21 else None
                                indicators["ema50"]       = round(ind.trend.ema50, 8) if ind.trend.ema50 else None
                                indicators["ema_stack"]   = ind.trend.ema_stack_score
                                indicators["ema_bullish"] = (ind.trend.ema9 or 0) > (ind.trend.ema21 or 0) > (ind.trend.ema50 or 0)
                            if ind.momentum:
                                indicators["rsi"]         = round(ind.momentum.rsi, 2) if ind.momentum.rsi else None
                                indicators["rsi_slope"]   = round(ind.momentum.rsi_slope, 4) if ind.momentum.rsi_slope else None
                                indicators["rsi_zone"]    = ind.momentum.rsi_zone_exit
                            if ind.strength:
                                indicators["adx"]          = round(ind.strength.adx, 2) if ind.strength.adx else None
                                indicators["volume_ratio"] = round(ind.strength.volume_ratio, 3) if ind.strength.volume_ratio else None
                                indicators["volume_spike"] = ind.strength.volume_spike
                            if ind.volatility:
                                indicators["atr"]            = round(ind.volatility.atr, 8) if ind.volatility.atr else None
                                indicators["atr_pct"]        = round(ind.volatility.atr_pct, 4) if ind.volatility.atr_pct else None
                                indicators["atr_percentile"] = ind.volatility.atr_percentile
                                indicators["atr_trend"]      = ind.volatility.atr_trend
                            if observation.confirmation_tf_indicators:
                                c = observation.confirmation_tf_indicators
                                if c.momentum:
                                    conf_tf_data["rsi"]       = round(c.momentum.rsi, 2) if c.momentum.rsi else None
                                if c.trend:
                                    conf_tf_data["ema_bullish"] = (c.trend.ema9 or 0) > (c.trend.ema21 or 0)
                                    conf_tf_data["ema9"]        = round(c.trend.ema9, 8) if c.trend.ema9 else None
                                    conf_tf_data["ema21"]       = round(c.trend.ema21, 8) if c.trend.ema21 else None
                                if c.strength:
                                    conf_tf_data["adx"] = round(c.strength.adx, 2) if c.strength.adx else None
                except Exception:
                    pass

                try:
                    from profiles.registry import get_coin_profile
                    prof_obj = get_coin_profile(symbol)
                    indicator_thresholds = {
                        "rsi_min":         prof_obj.rsi_min,
                        "rsi_max":         prof_obj.rsi_max,
                        "rsi_gc_min":      prof_obj.rsi_gc_min,
                        "atr_pct_min":     prof_obj.atr_pct_threshold,
                        "volume_mult_min": prof_obj.volume_mult,
                        "atr_sl_mult":     prof_obj.atr_sl_mult,
                        "atr_tp_mult":     prof_obj.atr_tp_mult,
                    }
                except Exception:
                    indicator_thresholds = {}

                rsi_val       = indicators.get("rsi")
                atr_pct_val   = indicators.get("atr_pct")
                vol_ratio_val = indicators.get("volume_ratio")
                adx_val       = indicators.get("adx")
                indicator_status = {}
                if rsi_val and indicator_thresholds:
                    indicator_status["rsi"] = "pass" if indicator_thresholds.get("rsi_min", 0) <= rsi_val <= indicator_thresholds.get("rsi_max", 100) else "fail"
                    indicator_status["rsi_overbought"] = rsi_val > 75
                if atr_pct_val and indicator_thresholds:
                    indicator_status["atr_pct"] = "pass" if atr_pct_val >= indicator_thresholds.get("atr_pct_min", 0) else "fail"
                if vol_ratio_val and indicator_thresholds:
                    indicator_status["volume"] = "pass" if vol_ratio_val >= indicator_thresholds.get("volume_mult_min", 1.0) else "weak"
                if adx_val:
                    indicator_status["adx"] = "strong" if adx_val >= 25 else "moderate" if adx_val >= 20 else "weak"

                now_utc    = _utcnow()
                hold_map   = HOLD_MINUTES.get(profile, {})
                hold_mins  = hold_map.get(regime, 60)
                hold_mins  = int(hold_mins * (0.7 + conf * 0.6))
                hold_mins  = max(10, hold_mins)
                tp_eta_utc = now_utc + timedelta(minutes=hold_mins)
                tp_eta_wib = tp_eta_utc + timedelta(hours=7)
                hold_display = (
                    f"{hold_mins} menit" if hold_mins < 60
                    else f"{hold_mins // 60}j {hold_mins % 60}m" if hold_mins % 60
                    else f"{hold_mins // 60} jam"
                )

                ema_bull  = indicators.get("ema_bullish", False)
                rsi_slope = indicators.get("rsi_slope", 0) or 0
                trend_summary = (
                    "Bullish kuat" if ema_bull and (rsi_val or 0) > 55 else
                    "Bullish lemah" if ema_bull else
                    "Sideways" if abs(rsi_slope) < 0.5 else
                    "Bearish"
                )

                htf_label   = TF_CONFIRM.get(tf_primary, "1h")
                htf_confirm = None
                if conf_tf_data:
                    htf_rsi  = conf_tf_data.get("rsi")
                    htf_bull = conf_tf_data.get("ema_bullish")
                    if htf_rsi is not None and htf_bull is not None:
                        htf_confirm = "bullish" if htf_bull and htf_rsi > 50 else "bearish" if not htf_bull else "neutral"

                forecasts.append({
                    "symbol":               symbol,
                    "strategy_profile":     profile,
                    "timeframe":            tf_primary,
                    "confirm_tf":           htf_label,
                    "current_price":        price,
                    "suggested_sl":         sl,
                    "suggested_tp":         tp,
                    "nearest_support":      row.nearest_support,
                    "nearest_resistance":   row.nearest_resistance,
                    "fib_support":          row.fib_support,
                    "fib_resistance":       row.fib_resistance,
                    "potential_profit_pct": potential_profit_pct,
                    "potential_loss_pct":   potential_loss_pct,
                    "rr_ratio":             rr_ratio,
                    "total_score":          round(score, 2),
                    "threshold_used":       round(dyn_threshold, 1),
                    "threshold_gap":        threshold_gap,
                    "probability_up_pct":   probability_up,
                    "signal_quality":       signal_quality,
                    "signal_confidence":    round(conf, 3),
                    "trigger_met":          row.trigger_met,
                    "score_breakdown":      {k: round(v, 1) if v else None for k, v in score_breakdown.items()},
                    "category_weights":     {k: round(v * 100, 1) for k, v in weights.items()},
                    "regime":               regime,
                    "regime_confidence":    row.regime_confidence,
                    "indicators":           indicators,
                    "indicator_thresholds": indicator_thresholds,
                    "indicator_status":     indicator_status,
                    "confirm_tf_data":      conf_tf_data,
                    "confirm_tf_result":    htf_confirm,
                    "trend_summary":        trend_summary,
                    "hold_minutes":         hold_mins,
                    "hold_display":         hold_display,
                    "tp_eta_wib":           tp_eta_wib.strftime("%H:%M WIB"),
                    "tp_eta_date":          tp_eta_wib.strftime("%d/%m %H:%M WIB"),
                    "last_updated":         _iso(row.timestamp),
                    "probability_note":     "Composite: score(60%) + confidence(25%) + threshold_ratio(15%)",
                })
            except Exception as e:
                log.warning("forecast [%s]: %s", symbol, e)

        forecasts.sort(key=lambda x: x.get("probability_up_pct", 0), reverse=True)
        return {"forecasts": forecasts, "count": len(forecasts), "timestamp": _iso(_utcnow())}



    @app.get("/api/universe/detail")
    async def get_universe_detail(_: str = Depends(verify_api_key)):
        import json as _json
        from profiles.registry import get_coin_profile, select_profile_from_indicators
        b = bot()
        try:
            try:
                with open("universe.json") as f:
                    udata = _json.load(f)
                coins = udata.get("symbols", [])
                scanned_at = udata.get("scanned_at", "")
            except Exception:
                coins = [{"symbol": s, "volume_24h": 0} for s in b.config.get("universe_watchlist", [])]
                scanned_at = ""
            result = []
            for c in coins:
                symbol = c["symbol"]
                vol    = c.get("volume_24h", 0)
                try:
                    row = await b.db.get_latest_regime(symbol)
                    regime     = row.regime if row else "undefined"
                    confidence = round(row.regime_confidence, 4) if row else 0.0
                    adx        = row.adx_value if row else 0.0
                    atr_pct    = row.atr_pct if row else 0.5
                    profile = select_profile_from_indicators(
                        symbol=symbol, adx=adx or 20.0,
                        atr_pct=atr_pct or 0.5, regime=regime,
                    )
                    score_row   = await b.db.get_latest_signal_score(symbol)
                    total_score = score_row.total_score if score_row else None
                    trigger_met = score_row.trigger_met if score_row else False
                except Exception:
                    regime = "undefined"; confidence = 0.0
                    profile = "scalp_volatile"; total_score = None; trigger_met = False
                result.append({
                    "symbol": symbol, "volume_24h": vol,
                    "volume_m": round(vol/1_000_000, 2),
                    "profile": profile, "regime": regime,
                    "confidence": confidence, "total_score": total_score,
                    "trigger_met": trigger_met,
                })
            return {"universe": result, "total": len(result),
                    "scanned_at": scanned_at, "timestamp": _iso(_utcnow())}
        except Exception as e:
            return {"error": str(e)}

    @app.get("/api/config/current")
    async def get_current_config(_: str = Depends(verify_api_key)):
        import json as _json
        b = bot()
        try:
            safe_config = {k: v for k, v in b.config.items() if k not in ["api_key","api_secret","telegram_bot_token","smtp_password"]}
            try:
                with open("universe.json") as f:
                    udata = _json.load(f)
                safe_config["universe_watchlist"] = [c["symbol"] for c in udata.get("symbols", [])]
                safe_config["universe_scanned_at"] = udata.get("scanned_at", "")
                safe_config["universe_total"] = udata.get("total_coins", 0)
            except Exception:
                pass
            return {"config": safe_config, "timestamp": _iso(_utcnow())}
        except Exception as e:
            return {"error": str(e)}

    return app
