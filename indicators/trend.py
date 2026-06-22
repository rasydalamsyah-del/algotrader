"""
indicators/trend.py
AlgoTrader Pro v7.0 — "The Intelligence Pipeline"

"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from constants import (
    COL_EMA9, COL_EMA21, COL_EMA50, COL_EMA100, COL_EMA200,
    COL_VWAP, COL_VWAP_UPPER_1, COL_VWAP_LOWER_1,
    COL_VWAP_UPPER_2, COL_VWAP_LOWER_2,
    COL_SUPERTREND, COL_SUPERTREND_DIR,
    EMA_STACK_WEIGHTS, EMA_GAP_BONUS_MAX,
    SUPERTREND_BULL_SCORE, SUPERTREND_BEAR_SCORE,
    SCORE_NEUTRAL, MIN_CANDLES_FOR_INDICATORS,
)
from core.models import TrendIndicators, clamp_score

log = logging.getLogger("indicators.trend")

_EMA_PERIODS = (9, 21, 50, 100, 200)
_EMA_COL_MAP = {
    9:   COL_EMA9,
    21:  COL_EMA21,
    50:  COL_EMA50,
    100: COL_EMA100,
    200: COL_EMA200,
}

_EMA_STACK_PAIRS: Tuple[Tuple[int, int, int], ...] = (
    (9,   21,  0),
    (21,  50,  1),
    (50,  100, 2),
    (100, 200, 3),
)

def _calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calculate_ema_stack(
    df: pd.DataFrame,
    errors: Optional[List[str]] = None,
) -> TrendIndicators:

    if errors is None:
        errors = []

    result = TrendIndicators()
    close = df["close"] if "close" in df.columns else None

    if close is None or len(close) < 2:
        errors.append("ema_stack: kolom 'close' tidak tersedia atau data < 2 bar")
        result.ema_stack_score = SCORE_NEUTRAL
        return result

    n = len(close)

    ema_values: Dict[int, Optional[float]] = {}
    for period in _EMA_PERIODS:
        if n < period:
            errors.append(
                f"ema{period}: data hanya {n} bar, butuh minimal {period} "
                f"(dikembalikan None, skor pair diabaikan)"
            )
            ema_values[period] = None
            continue
        series = _calc_ema(close, period)
        val = series.iloc[-1]
        ema_values[period] = float(val) if pd.notna(val) else None

    result.ema9   = ema_values.get(9)
    result.ema21  = ema_values.get(21)
    result.ema50  = ema_values.get(50)
    result.ema100 = ema_values.get(100)
    result.ema200 = ema_values.get(200)

    stack_score = 0.0
    available_weight = 0.0
    valid_pairs = 0

    for fast_p, slow_p, weight_idx in _EMA_STACK_PAIRS:
        fast_val = ema_values.get(fast_p)
        slow_val = ema_values.get(slow_p)

        if fast_val is None or slow_val is None:
            continue

        weight = EMA_STACK_WEIGHTS[weight_idx]
        available_weight += weight
        if fast_val > slow_val:
            stack_score += weight
        valid_pairs += 1

    if valid_pairs == 0:
        errors.append("ema_stack: tidak ada pair EMA yang bisa dihitung")
        result.ema_stack_score = SCORE_NEUTRAL
        return result

    gap_bonus = 0.0
    if result.ema9 and result.ema21 and result.ema21 > 0:
        gap_pct = (result.ema9 - result.ema21) / result.ema21 * 100
        gap_bonus = clamp_score(min(EMA_GAP_BONUS_MAX, max(0.0, gap_pct * EMA_GAP_BONUS_MAX)))

    if valid_pairs >= 3:
        normalized = (stack_score / available_weight * 100) if available_weight > 0 else 0.0
    else:
        normalized = stack_score
    raw = clamp_score(normalized + gap_bonus)
    result.ema_stack_score = raw
    return result

def _calculate_supertrend_raw(
    df: pd.DataFrame,
    period: int = 7,
    multiplier: float = 3.0,
    errors: Optional[List[str]] = None,
) -> Tuple[Optional[float], Optional[int], float]:

    if errors is None:
        errors = []

    required = period + 1
    if len(df) < required:
        errors.append(
            f"supertrend: data hanya {len(df)} bar, butuh minimal {required}"
        )
        return None, None, SCORE_NEUTRAL

    for col in ("high", "low", "close"):
        if col not in df.columns:
            errors.append(f"supertrend: kolom '{col}' tidak tersedia")
            return None, None, SCORE_NEUTRAL

    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    n     = len(close)

    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i]  - close[i - 1]),
        )

    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    # Adaptive multiplier — otomatis sesuai volatilitas pasar
    atr_clean = atr[atr > 0]
    if len(atr_clean) >= 20:
        current_atr = atr[-1]
        pctile = float(np.sum(atr_clean < current_atr) / len(atr_clean) * 100)
        if pctile < 30:
            multiplier = 2.5
        elif pctile < 70:
            multiplier = 3.0
        elif pctile < 85:
            multiplier = 3.5
        else:
            multiplier = 4.0
    # fallback: pakai multiplier default kalau data belum cukup

    hl2      = (high + low) / 2.0
    basic_ub = hl2 + multiplier * atr
    basic_lb = hl2 - multiplier * atr

    final_ub  = np.zeros(n)
    final_lb  = np.zeros(n)
    direction = np.zeros(n, dtype=int)
    supertrend = np.zeros(n)

    start = period - 1
    final_ub[start]  = basic_ub[start]
    final_lb[start]  = basic_lb[start]
    direction[start] = 1
    supertrend[start] = final_lb[start]

    for i in range(start + 1, n):
        if basic_ub[i] < final_ub[i - 1] or close[i - 1] > final_ub[i - 1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i - 1]

        if basic_lb[i] > final_lb[i - 1] or close[i - 1] < final_lb[i - 1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i - 1]

        if direction[i - 1] == -1:
            direction[i] = 1 if close[i] > final_ub[i] else -1
        else:
            direction[i] = -1 if close[i] < final_lb[i] else 1

        supertrend[i] = final_lb[i] if direction[i] == 1 else final_ub[i]

    last_value     = float(supertrend[-1])
    last_direction = int(direction[-1])
    score = SUPERTREND_BULL_SCORE if last_direction == 1 else SUPERTREND_BEAR_SCORE

    return last_value, last_direction, score

def calculate_supertrend(
    df: pd.DataFrame,
    period: int = 7,
    multiplier: float = 3.0,
    errors: Optional[List[str]] = None,
) -> TrendIndicators:
    st_val, st_dir, st_score = _calculate_supertrend_raw(
        df=df,
        period=period,
        multiplier=multiplier,
        errors=errors,
    )
    out = TrendIndicators()
    out.supertrend_value = st_val
    out.supertrend_direction = st_dir
    out.supertrend_score = clamp_score(st_score)
    out.composite_score = out.supertrend_score
    return out

def calculate_golden_dead_cross(
    df: pd.DataFrame,
    fast_period: int = 9,
    slow_period: int = 21,
    lookback: int = 50,
    errors: Optional[List[str]] = None,
) -> Tuple[Optional[int], Optional[int], float]:
    if errors is None:
        errors = []

    min_bars = max(fast_period, slow_period) + 2
    if len(df) < min_bars:
        errors.append(
            f"golden_dead_cross: data hanya {len(df)} bar, butuh {min_bars}"
        )
        return None, None, SCORE_NEUTRAL

    close = df["close"]
    fast_ema = _calc_ema(close, fast_period)
    slow_ema = _calc_ema(close, slow_period)

    diff      = fast_ema - slow_ema
    diff_prev = diff.shift(1)

    n         = len(diff)
    scan_from = max(1, n - lookback)

    golden_bars_ago: Optional[int] = None
    dead_bars_ago:   Optional[int] = None

    for i in range(n - 1, scan_from - 1, -1):
        cur  = diff.iloc[i]
        prev = diff_prev.iloc[i]
        if pd.isna(cur) or pd.isna(prev):
            continue
        bars_ago = n - 1 - i

        if prev <= 0 and cur > 0 and golden_bars_ago is None:
            golden_bars_ago = bars_ago
        if prev >= 0 and cur < 0 and dead_bars_ago is None:
            dead_bars_ago = bars_ago

        if golden_bars_ago is not None and dead_bars_ago is not None:
            break

    current_diff = float(diff.iloc[-1]) if pd.notna(diff.iloc[-1]) else 0.0
    current_close = float(close.iloc[-1]) if close.iloc[-1] > 0 else 1.0
    gap_pct = (current_diff / current_close) * 100

    if golden_bars_ago is None and dead_bars_ago is None:
        if gap_pct > 0:
            score = clamp_score(55.0 + min(15.0, gap_pct * 5))
        else:
            score = clamp_score(45.0 + max(-15.0, gap_pct * 5))
        return None, None, score

    gc = golden_bars_ago if golden_bars_ago is not None else lookback + 1
    dc = dead_bars_ago   if dead_bars_ago   is not None else lookback + 1

    if gc < dc:
        recency_bonus = max(0.0, 20.0 - gc * 0.5)
        gap_bonus     = clamp_score(min(15.0, max(0.0, gap_pct * 5)))
        score = clamp_score(65.0 + recency_bonus + gap_bonus)
    else:
        recency_bonus = max(0.0, 20.0 - dc * 0.5)
        gap_penalty   = clamp_score(min(15.0, max(0.0, -gap_pct * 5)))
        score = clamp_score(35.0 - recency_bonus - gap_penalty)

    return golden_bars_ago, dead_bars_ago, score

def calculate_vwap_multiday(
    df: pd.DataFrame,
    errors: Optional[List[str]] = None,
) -> Tuple[Optional[float], Dict[str, Optional[float]], float]:
    if errors is None:
        errors = []

    empty_bands: Dict[str, Optional[float]] = {
        "upper_1": None, "lower_1": None,
        "upper_2": None, "lower_2": None,
    }

    for col in ("high", "low", "close", "volume"):
        if col not in df.columns:
            errors.append(f"vwap: kolom '{col}' tidak tersedia")
            return None, empty_bands, SCORE_NEUTRAL

    if len(df) < 2:
        errors.append("vwap: data terlalu sedikit")
        return None, empty_bands, SCORE_NEUTRAL

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    volume  = df["volume"].replace(0, np.nan)

    if isinstance(df.index, pd.DatetimeIndex):
        date_group = df.index.date
        tpv   = typical * volume
        cumtpv = tpv.groupby(date_group).cumsum()
        cumvol = volume.groupby(date_group).cumsum()
    else:
        tpv    = typical * volume
        cumtpv = tpv.cumsum()
        cumvol = volume.cumsum()

    vwap_series = cumtpv / cumvol.replace(0, np.nan)

    sq_dev = (typical - vwap_series) ** 2
    tpvsq  = sq_dev * volume

    if isinstance(df.index, pd.DatetimeIndex):
        cumtpvsq = tpvsq.groupby(date_group).cumsum()
        cumvol2  = volume.groupby(date_group).cumsum()
    else:
        cumtpvsq = tpvsq.cumsum()
        cumvol2  = volume.cumsum()

    variance    = cumtpvsq / cumvol2.replace(0, np.nan)
    std_series  = np.sqrt(variance.clip(lower=0))

    last_vwap = vwap_series.iloc[-1]
    last_std  = std_series.iloc[-1]
    last_close = float(df["close"].iloc[-1])

    if pd.isna(last_vwap) or pd.isna(last_std) or last_vwap <= 0:
        errors.append("vwap: hasil kalkulasi NaN (kemungkinan volume semua 0)")
        return None, empty_bands, SCORE_NEUTRAL

    vwap_val  = float(last_vwap)
    std_val   = float(last_std)
    upper_1   = vwap_val + 1 * std_val
    lower_1   = vwap_val - 1 * std_val
    upper_2   = vwap_val + 2 * std_val
    lower_2   = vwap_val - 2 * std_val

    bands = {
        "upper_1": upper_1,
        "lower_1": lower_1,
        "upper_2": upper_2,
        "lower_2": lower_2,
    }

    if last_close >= upper_2:
        score = 30.0
    elif last_close >= upper_1:
        score = 55.0
    elif last_close >= vwap_val:
        score = 72.0
    elif last_close >= lower_1:
        score = 45.0
    elif last_close >= lower_2:
        score = 35.0
    else:
        score = 65.0

    return vwap_val, bands, clamp_score(score)

def calculate_vwap(
    df: pd.DataFrame,
    errors: Optional[List[str]] = None,
) -> TrendIndicators:
    vwap_val, bands, vwap_score = calculate_vwap_multiday(df, errors=errors)
    out = TrendIndicators()
    out.vwap = vwap_val
    out.vwap_upper_1 = bands.get("upper_1")
    out.vwap_lower_1 = bands.get("lower_1")
    out.vwap_upper_2 = bands.get("upper_2")
    out.vwap_lower_2 = bands.get("lower_2")
    out.vwap_score = vwap_score
    out.composite_score = vwap_score
    return out

def score_trend(
    df: pd.DataFrame,
    errors: Optional[List[str]] = None,
    timeframe: str = "15m",
) -> TrendIndicators:
    if errors is None:
        errors = []

    result = TrendIndicators()

    ema_result = calculate_ema_stack(df, errors)
    result.ema9   = ema_result.ema9
    result.ema21  = ema_result.ema21
    result.ema50  = ema_result.ema50
    result.ema100 = ema_result.ema100
    result.ema200 = ema_result.ema200
    result.ema_stack_score = ema_result.ema_stack_score
    ema_ok = result.ema9 is not None and result.ema21 is not None

    gc_bars, dc_bars, cross_score = calculate_golden_dead_cross(df, errors=errors)
    result.golden_cross_bars_ago = gc_bars
    result.dead_cross_bars_ago   = dc_bars
    result.cross_score           = cross_score
    cross_ok = True

    # Supertrend aktif untuk semua TF termasuk 1D — adaptive multiplier handle perbedaan volatilitas
    st_val, st_dir, st_score = _calculate_supertrend_raw(df, errors=errors)
    result.supertrend_value     = st_val
    result.supertrend_direction = st_dir
    result.supertrend_score     = st_score
    st_ok = st_val is not None

    skip_vwap = timeframe in ("1d", "3d", "1w")
    if skip_vwap:
        result.vwap_score = SCORE_NEUTRAL
        vwap_ok = False
    else:
        vwap_val, bands, vwap_score = calculate_vwap_multiday(df, errors)
        result.vwap        = vwap_val
        result.vwap_upper_1 = bands.get("upper_1")
        result.vwap_lower_1 = bands.get("lower_1")
        result.vwap_upper_2 = bands.get("upper_2")
        result.vwap_lower_2 = bands.get("lower_2")
        result.vwap_score   = vwap_score
        vwap_ok = vwap_val is not None

    raw_weights = {
        "ema":        (40.0, ema_ok),
        "cross":      (20.0, cross_ok),
        "supertrend": (25.0, st_ok),
        "vwap":       (15.0, vwap_ok),
    }
    raw_scores = {
        "ema":        result.ema_stack_score,
        "cross":      result.cross_score,
        "supertrend": result.supertrend_score,
        "vwap":       result.vwap_score,
    }

    total_available_weight = sum(w for w, ok in raw_weights.values() if ok)

    if total_available_weight < 1e-6:
        result.composite_score = SCORE_NEUTRAL
        errors.append("trend: tidak ada sub-indikator yang valid, composite = neutral")
        return result

    composite = 0.0
    for key, (base_w, ok) in raw_weights.items():
        if not ok:
            continue
        adjusted_w = base_w / total_available_weight
        composite += raw_scores[key] * adjusted_w

    result.composite_score = clamp_score(composite)

    log.debug(
        "trend score: ema=%.1f cross=%.1f st=%.1f vwap=%.1f → composite=%.1f",
        result.ema_stack_score, result.cross_score,
        result.supertrend_score, result.vwap_score,
        result.composite_score,
    )

    return result