"""
intelligence/position_sync.py
AlgoTrader Pro v7.0 — Binance Position Sync & Guardian
Fungsi: Deteksi posisi aktif di Binance yang tidak ada di DB bot,
        analisis Gate3-5, lalu adopt & kawal otomatis.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("intelligence.position_sync")

# ─── Threshold minimum untuk adopt posisi ────────────────────────────────────
MIN_USDT_VALUE     = 1.0    # Abaikan dust < $1
MIN_ADOPT_SCORE    = 45.0   # Score minimum agar posisi layak dikawal
MIN_CANDLE_BARS    = 50     # Minimum candle untuk analisis

async def fetch_binance_spot_positions(exchange) -> List[Dict]:
    """
    Fetch semua coin yang dipegang di Binance (balance > 0, bukan USDT).
    Return list of {symbol, amount, approx_usdt_value}
    """
    try:
        balance = await exchange.fetch_balance()
        total   = balance.get("total", {})
        results = []

        for coin, amount in total.items():
            if coin in ("USDT", "BUSD", "USDC", "TUSD", "DAI"):
                continue
            if not isinstance(amount, (int, float)) or amount <= 0:
                continue

            symbol = f"{coin}/USDT"
            # Estimasi nilai USDT
            try:
                ticker = await exchange._ex.fetch_ticker(symbol)
                price  = ticker.get("last") or ticker.get("close") or 0.0
                usdt_value = amount * price
            except Exception:
                usdt_value = 0.0
                price      = 0.0

            if usdt_value < MIN_USDT_VALUE:
                continue

            results.append({
                "symbol":      symbol,
                "coin":        coin,
                "amount":      amount,
                "price":       price,
                "usdt_value":  usdt_value,
            })

        log.info("Binance spot: %d posisi aktif ditemukan", len(results))
        return results

    except Exception as e:
        log.error("fetch_binance_spot_positions error: %s", e)
        return []


async def find_untracked_positions(exchange, db_manager) -> List[Dict]:
    """
    Bandingkan posisi Binance vs DB bot.
    Return posisi yang ada di Binance tapi TIDAK ada di DB.
    """
    binance_positions = await fetch_binance_spot_positions(exchange)
    if not binance_positions:
        return []

    # Ambil posisi terbuka di DB
    db_open = await db_manager.get_open_positions()
    db_symbols = {p.symbol for p in db_open}

    untracked = []
    for pos in binance_positions:
        if pos["symbol"] not in db_symbols:
            untracked.append(pos)
            log.warning(
                "⚠️  Posisi tidak tertracking: %s | amount=%.4f | ~$%.2f USDT",
                pos["symbol"], pos["amount"], pos["usdt_value"],
            )

    return untracked


async def analyze_position(
    symbol:   str,
    amount:   float,
    price:    float,
    exchange,
    db_manager,
) -> Tuple[bool, float, Optional[float], Optional[float], str]:
    """
    Analisis Gate3-5 untuk posisi yang sudah terbeli.
    Return: (layak_dikawal, score, sl, tp, alasan)
    """
    from intelligence.observer  import observe
    from intelligence.scorer    import score_signal
    from intelligence.classifier import classify_regime
    from profiles.registry      import get_coin_profile

    try:
        # ── Gate 3: Fetch OHLCV & hitung indikator ──
        bars = await exchange.fetch_ohlcv(symbol, timeframe="15m", limit=200)
        if not bars or len(bars) < MIN_CANDLE_BARS:
            return False, 0.0, None, None, f"Data candle tidak cukup ({len(bars) if bars else 0} bars)"

        import pandas as pd
        df = pd.DataFrame(bars, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # ── Gate 4: Observe & Score ──
        profile     = get_coin_profile(symbol)
        observation = await observe(symbol, df, profile)

        if not observation.primary_tf_valid:
            return False, 0.0, None, None, "Indikator primary TF tidak valid"

        regime, regime_conf = classify_regime(symbol, observation.primary_tf_indicators)
        scored = score_signal(observation, regime, regime_conf, db_manager)

        score = scored.total_score
        sl    = scored.suggested_sl
        tp    = scored.suggested_tp

        # Fallback SL/TP jika scorer tidak menghasilkan
        if sl is None:
            sl = round(price * 0.985, 8)   # SL 1.5%
        if tp is None:
            tp = round(price * 1.025, 8)   # TP 2.5%

        # ── Gate 5: Layak dikawal? ──
        if score >= MIN_ADOPT_SCORE:
            alasan = (
                f"Score {score:.1f} >= {MIN_ADOPT_SCORE} | "
                f"regime={regime.value} | conf={regime_conf:.2f}"
            )
            return True, score, sl, tp, alasan
        else:
            alasan = (
                f"Score {score:.1f} < {MIN_ADOPT_SCORE} (terlalu lemah) | "
                f"regime={regime.value}"
            )
            return False, score, sl, tp, alasan

    except Exception as e:
        log.error("analyze_position error [%s]: %s", symbol, e)
        return False, 0.0, None, None, f"Error analisis: {e}"


async def adopt_position(
    symbol:   str,
    amount:   float,
    price:    float,
    score:    float,
    sl:       float,
    tp:       float,
    regime:   str,
    db_manager,
) -> bool:
    """
    Inject posisi ke DB bot agar Trade Guardian bisa mengawal.
    """
    try:
        entry_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        import sqlite3
        conn = sqlite3.connect("/root/algotrader/data/trading_bot.db")
        cur  = conn.cursor()

        cur.execute("""
            INSERT INTO positions (
                symbol, entry_time, entry_price, current_price,
                amount, side, is_open, is_closing,
                stop_loss_price, take_profit_price,
                strategy_name, strategy_profile,
                entry_score, entry_regime, highest_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol, entry_time, price, price,
            amount, "buy", True, False,
            sl, tp,
            "manual_adopt", "scalp_volatile",
            score, regime, price,
        ))

        conn.commit()
        conn.close()

        log.info(
            "✅ ADOPT %s | amount=%.4f | entry=%.6f | SL=%.6f | TP=%.6f | score=%.1f",
            symbol, amount, price, sl, tp, score,
        )
        return True

    except Exception as e:
        log.error("adopt_position error [%s]: %s", symbol, e)
        return False


async def run_position_sync(exchange, db_manager) -> Dict:
    """
    Main entry point — dipanggil periodik dari main loop.
    Deteksi → Analisis → Adopt posisi yang tidak tertracking.
    """
    result = {
        "untracked_found": 0,
        "adopted":         0,
        "rejected":        0,
        "errors":          0,
    }

    try:
        untracked = await find_untracked_positions(exchange, db_manager)
        result["untracked_found"] = len(untracked)

        if not untracked:
            log.info("✅ Semua posisi Binance sudah tertracking di DB")
            return result

        for pos in untracked:
            symbol = pos["symbol"]
            amount = pos["amount"]
            price  = pos["price"]

            log.info("🔍 Analisis posisi tidak tertracking: %s", symbol)

            layak, score, sl, tp, alasan = await analyze_position(
                symbol, amount, price, exchange, db_manager
            )

            if layak:
                adopted = await adopt_position(
                    symbol, amount, price, score,
                    sl, tp, "undefined", db_manager,
                )
                if adopted:
                    result["adopted"] += 1
                    log.info("✅ %s diadopsi | %s", symbol, alasan)
                else:
                    result["errors"] += 1
            else:
                result["rejected"] += 1
                log.warning(
                    "⚠️  %s TIDAK diadopsi | %s | "
                    "Pertimbangkan jual manual!", symbol, alasan
                )

    except Exception as e:
        log.error("run_position_sync error: %s", e)
        result["errors"] += 1

    return result
