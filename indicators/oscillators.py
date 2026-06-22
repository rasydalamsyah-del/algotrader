"""
indicators/oscillators.py
AlgoTrader Pro — Oscillator Indicators
CCI, Williams %R, ROC/Momentum
"""

from __future__ import annotations
import logging
from typing import List, Optional
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
from core.models import OscillatorIndicators

# ── Constants ──────────────────────────────────────────────────────────────────
CCI_PERIOD          = 20
CCI_OVERBOUGHT      = 100.0
CCI_OVERSOLD        = -100.0
CCI_EXTREME_OB      = 200.0
CCI_EXTREME_OS      = -200.0

WILLIAMS_PERIOD     = 14
WILLIAMS_OB         = -20.0
WILLIAMS_OS         = -80.0

ROC_FAST_PERIOD     = 9
ROC_SLOW_PERIOD     = 21
ROC_SIGNAL_PERIOD   = 5

def clamp_score(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))

# ── CCI ────────────────────────────────────────────────────────────────────────
def calculate_cci(df: pd.DataFrame, period: int = CCI_PERIOD) -> Optional[float]:
    if len(df) < period:
        return None
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mean_dev = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    denom = 0.015 * mean_dev.iloc[-1]
    if denom == 0:
        return None
    return float((tp.iloc[-1] - sma.iloc[-1]) / denom)

def score_cci(cci: Optional[float]) -> float:
    if cci is None:
        return 50.0
    if cci >= CCI_EXTREME_OB:
        return 20.0
    if cci >= CCI_OVERBOUGHT:
        t = (cci - CCI_OVERBOUGHT) / (CCI_EXTREME_OB - CCI_OVERBOUGHT)
        return clamp_score(65.0 - t * 45.0)
    if cci >= 0:
        t = cci / CCI_OVERBOUGHT
        return clamp_score(50.0 + t * 15.0)
    if cci >= CCI_OVERSOLD:
        t = cci / CCI_OVERSOLD
        return clamp_score(50.0 - t * 15.0)
    if cci >= CCI_EXTREME_OS:
        t = (cci - CCI_OVERSOLD) / (CCI_EXTREME_OS - CCI_OVERSOLD)
        return clamp_score(35.0 + t * 45.0)
    return 80.0  # extreme oversold = strong buy candidate

# ── Williams %R ────────────────────────────────────────────────────────────────
def calculate_williams_r(df: pd.DataFrame, period: int = WILLIAMS_PERIOD) -> Optional[float]:
    if len(df) < period:
        return None
    hh = df["high"].rolling(period).max().iloc[-1]
    ll = df["low"].rolling(period).min().iloc[-1]
    close = df["close"].iloc[-1]
    denom = hh - ll
    if denom == 0:
        return None
    return float(((hh - close) / denom) * -100)

def score_williams_r(wr: Optional[float]) -> float:
    if wr is None:
        return 50.0
    # -0 to -20 = overbought, -80 to -100 = oversold
    if wr >= -20:
        # overbought zone
        t = (wr - (-20)) / (0 - (-20))
        return clamp_score(30.0 - t * 10.0)
    if wr >= -50:
        t = (wr - (-50)) / (-20 - (-50))
        return clamp_score(50.0 - t * 20.0)
    if wr >= -80:
        t = (wr - (-80)) / (-50 - (-80))
        return clamp_score(70.0 - t * 20.0)
    # oversold zone
    t = (wr - (-100)) / (-80 - (-100))
    return clamp_score(85.0 - t * 15.0)

# ── ROC ────────────────────────────────────────────────────────────────────────
def calculate_roc(df: pd.DataFrame, period: int = ROC_FAST_PERIOD) -> Optional[float]:
    if len(df) < period + 1:
        return None
    close = df["close"]
    prev = close.iloc[-(period + 1)]
    if prev == 0:
        return None
    return float(((close.iloc[-1] - prev) / prev) * 100)

def calculate_roc_slope(df: pd.DataFrame,
                         fast: int = ROC_FAST_PERIOD,
                         signal: int = ROC_SIGNAL_PERIOD) -> Optional[float]:
    """Slope of ROC — positive = momentum accelerating, negative = decelerating."""
    needed = fast + signal + 1
    if len(df) < needed:
        return None
    close = df["close"]
    roc_series = []
    for i in range(signal):
        idx = -(signal - i)
        prev_idx = idx - fast
        try:
            p = float(close.iloc[prev_idx])
            c = float(close.iloc[idx])
        except IndexError:
            return None
        if p == 0:
            return None
        roc_series.append(((c - p) / p) * 100)
    if len(roc_series) < 2:
        return None
    return float(roc_series[-1] - roc_series[0])

def score_roc(roc: Optional[float], roc_slope: Optional[float] = None) -> float:
    if roc is None:
        return 50.0
    # Base score dari ROC value
    if roc > 5.0:
        base = clamp_score(70.0 + min(roc - 5.0, 10.0) * 2.0)
    elif roc > 2.0:
        base = clamp_score(60.0 + (roc - 2.0) * (10.0 / 3.0))
    elif roc > 0:
        base = clamp_score(50.0 + roc * (10.0 / 2.0))
    elif roc > -2.0:
        base = clamp_score(50.0 + roc * (10.0 / 2.0))
    elif roc > -5.0:
        base = clamp_score(40.0 + (roc + 5.0) * (10.0 / 3.0))
    else:
        base = clamp_score(30.0 - min(abs(roc) - 5.0, 10.0) * 2.0)

    # Slope modifier: early warning kalau momentum melambat
    if roc_slope is not None:
        if roc > 0 and roc_slope < -1.0:
            base -= 8.0   # momentum positive tapi melambat → warning
        elif roc > 0 and roc_slope > 1.0:
            base += 5.0   # momentum positive dan akselerasi → strong
        elif roc < 0 and roc_slope > 1.0:
            base += 5.0   # momentum negatif tapi membaik → recovery signal

    return clamp_score(base)

# ── Public entry point ─────────────────────────────────────────────────────────
def score_oscillators(df: pd.DataFrame, errors: Optional[List[str]] = None):
    """
    Returns a dict dengan semua nilai oscillator.
    Dipanggil dari observer.py dan hasilnya dimasukkan ke OscillatorIndicators.
    """
    result = OscillatorIndicators()
    try:
        result.cci       = calculate_cci(df)
        result.cci_score = score_cci(result.cci)

        result.williams_r       = calculate_williams_r(df)
        result.williams_r_score = score_williams_r(result.williams_r)

        result.roc       = calculate_roc(df)
        result.roc_slope = calculate_roc_slope(df)
        result.roc_score = score_roc(result.roc, result.roc_slope)

        # Composite: rata-rata tertimbang
        # CCI=0.35, Williams=0.25, ROC=0.40 (ROC paling useful buat early warning)
        result.composite_score = clamp_score(
            result.cci_score       * 0.35
            + result.williams_r_score * 0.25
            + result.roc_score        * 0.40
        )
    except Exception as exc:
        if errors is not None:
            errors.append(f"oscillators: {exc}")
        log.exception("Error kalkulasi oscillators: %s", exc)
    return result
