"""
indicators/structure.py
AlgoTrader Pro — Structure Indicators
Ichimoku Cloud, Parabolic SAR, Pivot Points, Fibonacci Retracement
"""

from __future__ import annotations
import logging
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
from core.models import StructureIndicators

def clamp_score(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))

# ══════════════════════════════════════════════════════════════════════════════
# ICHIMOKU CLOUD
# ══════════════════════════════════════════════════════════════════════════════
TENKAN_PERIOD  = 9
KIJUN_PERIOD   = 26
SENKOU_B_PERIOD = 52
DISPLACEMENT   = 26

def _midpoint(series: pd.Series, period: int) -> pd.Series:
    return (series.rolling(period).max() + series.rolling(period).min()) / 2

def calculate_ichimoku(df: pd.DataFrame) -> dict:
    result = {
        "tenkan": None, "kijun": None,
        "senkou_a": None, "senkou_b": None, "chikou": None,
        "cloud_top": None, "cloud_bottom": None,
        "price_vs_cloud": None, "cloud_thickness": None,
        "tk_cross": None,
    }
    if len(df) < SENKOU_B_PERIOD + DISPLACEMENT:
        return result

    high = df["high"]
    low  = df["low"]
    close = df["close"]

    tenkan  = _midpoint(high, TENKAN_PERIOD)
    kijun   = _midpoint(high, KIJUN_PERIOD)
    senkou_a = ((tenkan + kijun) / 2).shift(DISPLACEMENT)
    senkou_b = _midpoint(high, SENKOU_B_PERIOD).shift(DISPLACEMENT)

    result["tenkan"]   = float(tenkan.iloc[-1]) if not pd.isna(tenkan.iloc[-1]) else None
    result["kijun"]    = float(kijun.iloc[-1])  if not pd.isna(kijun.iloc[-1])  else None
    result["chikou"]   = float(close.iloc[-1])  # chikou = close sekarang, plot -26

    # Senkou A & B saat ini (yang sudah di-displace ke depan, kita ambil current)
    sa = float(senkou_a.iloc[-1]) if not pd.isna(senkou_a.iloc[-1]) else None
    sb = float(senkou_b.iloc[-1]) if not pd.isna(senkou_b.iloc[-1]) else None
    result["senkou_a"] = sa
    result["senkou_b"] = sb

    if sa is not None and sb is not None:
        result["cloud_top"]    = max(sa, sb)
        result["cloud_bottom"] = min(sa, sb)
        result["cloud_thickness"] = abs(sa - sb)
        price = float(close.iloc[-1])
        if price > result["cloud_top"]:
            result["price_vs_cloud"] = "above"
        elif price < result["cloud_bottom"]:
            result["price_vs_cloud"] = "below"
        else:
            result["price_vs_cloud"] = "inside"

    # TK Cross (Tenkan cross Kijun)
    if result["tenkan"] is not None and result["kijun"] is not None:
        t_prev = float(tenkan.iloc[-2]) if len(tenkan) >= 2 else None
        k_prev = float(kijun.iloc[-2])  if len(kijun) >= 2  else None
        if t_prev is not None and k_prev is not None:
            was_below = t_prev < k_prev
            is_above  = result["tenkan"] > result["kijun"]
            was_above = t_prev > k_prev
            is_below  = result["tenkan"] < result["kijun"]
            if was_below and is_above:
                result["tk_cross"] = "bullish"
            elif was_above and is_below:
                result["tk_cross"] = "bearish"

    return result

def score_ichimoku(data: dict, current_price: float) -> float:
    score = 50.0
    pvc = data.get("price_vs_cloud")
    if pvc == "above":
        score += 20.0
    elif pvc == "below":
        score -= 20.0
    elif pvc == "inside":
        score += 0.0  # netral, pasar ragu

    tenkan = data.get("tenkan")
    kijun  = data.get("kijun")
    if tenkan and kijun:
        if tenkan > kijun:
            score += 8.0
        else:
            score -= 8.0
        if current_price > tenkan:
            score += 5.0
        if current_price > kijun:
            score += 5.0

    tk_cross = data.get("tk_cross")
    if tk_cross == "bullish":
        score += 10.0
    elif tk_cross == "bearish":
        score -= 10.0

    # Cloud thickness: tebal = support kuat
    ct = data.get("cloud_thickness")
    if ct and pvc == "above":
        score += min(ct / current_price * 500, 5.0)  # maks +5

    return clamp_score(score)

# ══════════════════════════════════════════════════════════════════════════════
# PARABOLIC SAR
# ══════════════════════════════════════════════════════════════════════════════
SAR_AF_INITIAL = 0.02
SAR_AF_MAX     = 0.20
SAR_AF_STEP    = 0.02

def calculate_sar(df: pd.DataFrame,
                  af_init: float = SAR_AF_INITIAL,
                  af_max:  float = SAR_AF_MAX,
                  af_step: float = SAR_AF_STEP) -> Tuple[Optional[float], Optional[str]]:
    if len(df) < 3:
        return None, None

    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    n = len(close)

    # Init: pakai 2 candle pertama buat deteksi awal trend
    bull = close[1] > close[0]
    af   = af_init
    ep   = high[0] if bull else low[0]
    sar  = low[0]  if bull else high[0]

    for i in range(2, n):
        sar_new = sar + af * (ep - sar)

        if bull:
            sar_new = min(sar_new, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < sar_new:
                bull    = False
                sar_new = ep
                ep      = low[i]
                af      = af_init
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            sar_new = max(sar_new, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > sar_new:
                bull    = True
                sar_new = ep
                ep      = high[i]
                af      = af_init
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)

        sar = sar_new

    direction = "up" if bull else "down"
    return float(sar), direction

def score_sar(sar_value: Optional[float], sar_direction: Optional[str],
              current_price: float) -> float:
    if sar_value is None or sar_direction is None:
        return 50.0
    if sar_direction == "up":
        # Titik di bawah harga = uptrend
        gap_pct = (current_price - sar_value) / current_price * 100
        # Makin dekat ke SAR = makin rentan reversal
        if gap_pct > 3.0:
            return clamp_score(72.0 + min(gap_pct - 3.0, 5.0) * 1.5)
        elif gap_pct > 1.0:
            return clamp_score(62.0 + gap_pct * 5.0)
        else:
            return clamp_score(55.0)  # terlalu dekat, waspada
    else:
        # SAR di atas harga = downtrend
        gap_pct = (sar_value - current_price) / current_price * 100
        if gap_pct > 3.0:
            return clamp_score(28.0 - min(gap_pct - 3.0, 5.0) * 1.5)
        else:
            return clamp_score(38.0)

# ══════════════════════════════════════════════════════════════════════════════
# PIVOT POINTS (Classic Daily)
# ══════════════════════════════════════════════════════════════════════════════
def calculate_pivot_points(df: pd.DataFrame) -> dict:
    """Pakai OHLC dari candle sebelumnya (period terakhir lengkap)."""
    result = {
        "pivot": None,
        "r1": None, "r2": None, "r3": None,
        "s1": None, "s2": None, "s3": None,
        "nearest_support": None, "nearest_resistance": None,
        "price_vs_pivot": None,
    }
    if len(df) < 2:
        return result

    # Gunakan candle kedua dari terakhir sebagai "periode sebelumnya"
    prev = df.iloc[-2]
    h = float(prev["high"])
    l = float(prev["low"])
    c = float(prev["close"])

    pivot = (h + l + c) / 3
    r1 = 2 * pivot - l
    r2 = pivot + (h - l)
    r3 = h + 2 * (pivot - l)
    s1 = 2 * pivot - h
    s2 = pivot - (h - l)
    s3 = l - 2 * (h - pivot)

    result.update({
        "pivot": round(pivot, 8),
        "r1": round(r1, 8), "r2": round(r2, 8), "r3": round(r3, 8),
        "s1": round(s1, 8), "s2": round(s2, 8), "s3": round(s3, 8),
    })

    price = float(df["close"].iloc[-1])
    result["price_vs_pivot"] = "above" if price >= pivot else "below"

    # Nearest support & resistance dari harga sekarang
    supports    = [v for v in [s1, s2, s3] if v < price]
    resistances = [v for v in [r1, r2, r3] if v > price]
    result["nearest_support"]    = max(supports)    if supports    else s1
    result["nearest_resistance"] = min(resistances) if resistances else r1

    return result

def score_pivot(data: dict, current_price: float) -> float:
    if data.get("pivot") is None:
        return 50.0
    score = 50.0
    pivot = data["pivot"]
    ns    = data.get("nearest_support")
    nr    = data.get("nearest_resistance")

    if current_price >= pivot:
        score += 10.0
    else:
        score -= 10.0

    # Seberapa dekat ke support (bagus buat entry)
    if ns and current_price > 0:
        dist_support_pct = (current_price - ns) / current_price * 100
        if dist_support_pct < 1.0:
            score += 12.0   # sangat dekat support = zona ideal entry
        elif dist_support_pct < 2.0:
            score += 6.0
        elif dist_support_pct > 5.0:
            score -= 5.0    # terlalu jauh dari support

    # Seberapa dekat ke resistance (kalau terlalu dekat, upside terbatas)
    if nr and current_price > 0:
        dist_resist_pct = (nr - current_price) / current_price * 100
        if dist_resist_pct < 1.0:
            score -= 15.0   # hampir mentok resistance
        elif dist_resist_pct < 2.0:
            score -= 7.0
        elif dist_resist_pct > 4.0:
            score += 5.0    # ruang gerak ke atas masih besar

    return clamp_score(score)

# ══════════════════════════════════════════════════════════════════════════════
# FIBONACCI RETRACEMENT
# ══════════════════════════════════════════════════════════════════════════════
FIB_LEVELS    = [0.236, 0.382, 0.500, 0.618, 0.786]
SWING_LOOKBACK = 50   # candle terakhir buat cari swing high/low

def _find_swing_points(df: pd.DataFrame, lookback: int = SWING_LOOKBACK
                       ) -> Tuple[Optional[float], Optional[float]]:
    """Cari swing high dan swing low dalam N candle terakhir."""
    subset = df.tail(lookback)
    if len(subset) < 5:
        return None, None
    return float(subset["high"].max()), float(subset["low"].min())

def calculate_fibonacci(df: pd.DataFrame) -> dict:
    result = {
        "fib_swing_high": None, "fib_swing_low": None,
        "fib_236": None, "fib_382": None, "fib_500": None,
        "fib_618": None, "fib_786": None,
        "nearest_fib_support": None, "nearest_fib_resistance": None,
    }
    if len(df) < 10:
        return result

    swing_high, swing_low = _find_swing_points(df)
    if swing_high is None or swing_low is None:
        return result
    if swing_high <= swing_low:
        return result

    diff = swing_high - swing_low
    result["fib_swing_high"] = swing_high
    result["fib_swing_low"]  = swing_low

    # Level retracement dari swing low ke swing high
    levels = {
        "fib_236": swing_high - diff * 0.236,
        "fib_382": swing_high - diff * 0.382,
        "fib_500": swing_high - diff * 0.500,
        "fib_618": swing_high - diff * 0.618,
        "fib_786": swing_high - diff * 0.786,
    }
    for k, v in levels.items():
        result[k] = round(v, 8)

    price = float(df["close"].iloc[-1])
    fib_values = list(levels.values())
    supports    = [v for v in fib_values if v < price]
    resistances = [v for v in fib_values if v > price]
    result["nearest_fib_support"]    = max(supports)    if supports    else None
    result["nearest_fib_resistance"] = min(resistances) if resistances else None

    return result

def score_fibonacci(data: dict, current_price: float) -> float:
    if data.get("fib_swing_high") is None:
        return 50.0
    score = 50.0
    ns = data.get("nearest_fib_support")
    nr = data.get("nearest_fib_resistance")

    if ns and current_price > 0:
        dist_pct = (current_price - ns) / current_price * 100
        # 61.8% adalah golden ratio — bonus ekstra
        fib618 = data.get("fib_618")
        if fib618 and abs(current_price - fib618) / current_price < 0.005:
            score += 15.0   # tepat di golden ratio
        elif dist_pct < 0.5:
            score += 12.0
        elif dist_pct < 1.5:
            score += 7.0
        elif dist_pct > 6.0:
            score -= 5.0

    if nr and current_price > 0:
        dist_pct = (nr - current_price) / current_price * 100
        if dist_pct < 1.0:
            score -= 12.0
        elif dist_pct < 2.0:
            score -= 6.0
        elif dist_pct > 5.0:
            score += 5.0

    return clamp_score(score)

# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def score_structure(df: pd.DataFrame, errors: Optional[List[str]] = None):
    """
    Dipanggil dari observer.py → hasilnya masuk ke iset.structure
    """
    result = StructureIndicators()

    if len(df) == 0 or "close" not in df.columns:
        return result

    current_price = float(df["close"].iloc[-1])
    if current_price <= 0:
        return result

    try:
        ich = calculate_ichimoku(df)
        result.tenkan          = ich["tenkan"]
        result.kijun           = ich["kijun"]
        result.senkou_a        = ich["senkou_a"]
        result.senkou_b        = ich["senkou_b"]
        result.chikou          = ich["chikou"]
        result.cloud_top       = ich["cloud_top"]
        result.cloud_bottom    = ich["cloud_bottom"]
        result.price_vs_cloud  = ich["price_vs_cloud"]
        result.cloud_thickness = ich["cloud_thickness"]
        result.tk_cross        = ich["tk_cross"]
        result.ichimoku_score  = score_ichimoku(ich, current_price)
    except Exception as exc:
        if errors is not None:
            errors.append(f"ichimoku: {exc}")
        log.exception("Error kalkulasi ichimoku: %s", exc)

    try:
        sar_val, sar_dir       = calculate_sar(df)
        result.sar_value       = sar_val
        result.sar_direction   = sar_dir
        result.sar_score       = score_sar(sar_val, sar_dir, current_price)
    except Exception as exc:
        if errors is not None:
            errors.append(f"sar: {exc}")
        log.exception("Error kalkulasi SAR: %s", exc)

    try:
        piv = calculate_pivot_points(df)
        result.pivot              = piv["pivot"]
        result.r1                 = piv["r1"]
        result.r2                 = piv["r2"]
        result.r3                 = piv["r3"]
        result.s1                 = piv["s1"]
        result.s2                 = piv["s2"]
        result.s3                 = piv["s3"]
        result.nearest_support    = piv["nearest_support"]
        result.nearest_resistance = piv["nearest_resistance"]
        result.price_vs_pivot     = piv["price_vs_pivot"]
        result.pivot_score        = score_pivot(piv, current_price)
    except Exception as exc:
        if errors is not None:
            errors.append(f"pivot: {exc}")
        log.exception("Error kalkulasi pivot: %s", exc)

    try:
        fib = calculate_fibonacci(df)
        result.fib_swing_high         = fib["fib_swing_high"]
        result.fib_swing_low          = fib["fib_swing_low"]
        result.fib_236                = fib["fib_236"]
        result.fib_382                = fib["fib_382"]
        result.fib_500                = fib["fib_500"]
        result.fib_618                = fib["fib_618"]
        result.fib_786                = fib["fib_786"]
        result.nearest_fib_support    = fib["nearest_fib_support"]
        result.nearest_fib_resistance = fib["nearest_fib_resistance"]
        result.fib_score              = score_fibonacci(fib, current_price)
    except Exception as exc:
        if errors is not None:
            errors.append(f"fibonacci: {exc}")
        log.exception("Error kalkulasi fibonacci: %s", exc)

    # Composite structure score
    # Ichimoku=0.35 (paling komprehensif), SAR=0.25 (trailing), Pivot=0.25, Fib=0.15
    valid_count = sum([
        result.ichimoku_score != 50.0,
        result.sar_score != 50.0,
        result.pivot_score != 50.0,
        result.fib_score != 50.0,
    ])
    if valid_count > 0:
        result.composite_score = clamp_score(
            result.ichimoku_score * 0.35
            + result.sar_score    * 0.25
            + result.pivot_score  * 0.25
            + result.fib_score    * 0.15
        )

    return result
