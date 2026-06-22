"""
indicators/orderbook.py
AlgoTrader Pro — Orderbook / Whale Detector
"""
from __future__ import annotations
import logging
from typing import List, Optional

log = logging.getLogger(__name__)
from core.models import OrderbookIndicators

def clamp_score(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))

# ── Constants ──────────────────────────────────────────────────────────────────
WHALE_WALL_PCT      = 0.08   # order > 8% total volume = whale wall
IMBALANCE_BULL      = 0.62   # bid ratio > ini = tekanan beli
IMBALANCE_BEAR      = 0.38   # bid ratio < ini = tekanan jual
ABSORPTION_DROP_PCT = 0.40   # wall turun >40% = kemungkinan diserap

# ── Helpers ───────────────────────────────────────────────────────────────────
def _total_volume(levels: list) -> float:
    return sum(float(p) * float(q) for p, q in levels if float(p) > 0 and float(q) > 0)

def _find_whale_wall(levels: list, total_vol: float) -> tuple:
    """Return (price, strength_pct) dari level terbesar, atau (None, None)."""
    if total_vol <= 0 or not levels:
        return None, None
    best_price    = None
    best_strength = 0.0
    for p, q in levels:
        p, q = float(p), float(q)
        if p <= 0 or q <= 0:
            continue
        strength = (p * q) / total_vol
        if strength > best_strength:
            best_strength = strength
            best_price    = p
    if best_strength >= WHALE_WALL_PCT:
        return best_price, round(best_strength * 100, 2)
    return None, None

# ── Main calculator ───────────────────────────────────────────────────────────
def calculate_orderbook(ob: dict) -> dict:
    result = {
        "bid_ask_imbalance":  None,
        "whale_bid_wall":     None,
        "whale_ask_wall":     None,
        "bid_wall_strength":  None,
        "ask_wall_strength":  None,
        "spread_pct":         None,
        "absorbed_bid":       False,
        "absorbed_ask":       False,
    }
    if not ob:
        return result

    bids = ob.get("bids", [])
    asks = ob.get("asks", [])
    if not bids or not asks:
        return result

    bid_vol = _total_volume(bids)
    ask_vol = _total_volume(asks)
    total   = bid_vol + ask_vol

    if total > 0:
        result["bid_ask_imbalance"] = round(bid_vol / total, 4)

    # Spread
    best_bid = float(bids[0][0]) if bids else 0.0
    best_ask = float(asks[0][0]) if asks else 0.0
    if best_bid > 0 and best_ask > 0:
        mid = (best_bid + best_ask) / 2
        result["spread_pct"] = round((best_ask - best_bid) / mid * 100, 4)

    # Whale walls
    wb_price, wb_str = _find_whale_wall(bids, bid_vol)
    wa_price, wa_str = _find_whale_wall(asks, ask_vol)
    result["whale_bid_wall"]    = wb_price
    result["bid_wall_strength"] = wb_str
    result["whale_ask_wall"]    = wa_price
    result["ask_wall_strength"] = wa_str

    return result

def score_orderbook(data: dict) -> float:
    if data.get("bid_ask_imbalance") is None:
        return 50.0

    score = 50.0
    imb   = data["bid_ask_imbalance"]

    # Imbalance — komponen utama
    if imb >= IMBALANCE_BULL:
        t      = (imb - IMBALANCE_BULL) / (1.0 - IMBALANCE_BULL)
        score += 15.0 + t * 15.0          # +15 sampai +30
    elif imb <= IMBALANCE_BEAR:
        t      = (IMBALANCE_BEAR - imb) / IMBALANCE_BEAR
        score -= 15.0 + t * 15.0          # -15 sampai -30
    else:
        # Netral zone: sedikit lean ke arah majority
        score += (imb - 0.5) * 20.0

    # Whale bid wall → support kuat → bonus
    wb_str = data.get("bid_wall_strength")
    if wb_str:
        score += min(wb_str * 0.3, 8.0)   # maks +8

    # Whale ask wall → resistance kuat → penalty
    wa_str = data.get("ask_wall_strength")
    if wa_str:
        score -= min(wa_str * 0.3, 8.0)   # maks -8

    # Absorbed ask wall = breakout signal → bonus besar
    if data.get("absorbed_ask"):
        score += 12.0

    # Absorbed bid wall = breakdown signal → penalty besar
    if data.get("absorbed_bid"):
        score -= 12.0

    # Spread lebar = likuiditas buruk = slight penalty
    spread = data.get("spread_pct")
    if spread and spread > 0.1:
        score -= min(spread * 5, 5.0)

    return clamp_score(score)

# ── Public entry point ────────────────────────────────────────────────────────
def score_orderbook_data(ob: dict, errors=None):
    """
    Dipanggil dari observer.py dengan data live_orderbooks[symbol].
    Return OrderbookIndicators.
    """
    result = OrderbookIndicators()
    if not ob:
        return result
    try:
        data = calculate_orderbook(ob)
        result.bid_ask_imbalance  = data["bid_ask_imbalance"]
        result.whale_bid_wall     = data["whale_bid_wall"]
        result.whale_ask_wall     = data["whale_ask_wall"]
        result.bid_wall_strength  = data["bid_wall_strength"]
        result.ask_wall_strength  = data["ask_wall_strength"]
        result.spread_pct         = data["spread_pct"]
        result.absorbed_bid       = data["absorbed_bid"]
        result.absorbed_ask       = data["absorbed_ask"]
        result.orderbook_score    = score_orderbook(data)
        result.composite_score    = result.orderbook_score
    except Exception as exc:
        if errors is not None:
            errors.append(f"orderbook: {exc}")
        log.exception("Error kalkulasi orderbook: %s", exc)
    return result
