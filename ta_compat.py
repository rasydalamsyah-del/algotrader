"""
ta_compat.py — Pengganti pandas_ta untuk Termux (tanpa numba/numba)
Drop-in replacement untuk df.ta.ema(), df.ta.rsi(), df.ta.atr(), df.ta.vwap()
"""
import pandas as pd
import numpy as np


class _TAAccessor:
    def __init__(self, df: pd.DataFrame):
        # --- MULAI EDIT DI SINI ---
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # --- SELESAI EDIT ---
        self._df = df

    def ema(self, length: int = 14, append: bool = True, **kwargs) -> pd.Series:
        col = f"EMA_{length}"
        result = self._df["close"].ewm(span=length, adjust=False).mean()
        if append:
            self._df[col] = result
        return result

    def rsi(self, length: int = 14, append: bool = True, **kwargs) -> pd.Series:
        col = f"RSI_{length}"
        delta = self._df["close"].diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=length - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=length - 1, adjust=False).mean()
        rs     = avg_gain / avg_loss.replace(0, np.nan)
        result = 100 - (100 / (1 + rs))
        result = result.fillna(50)
        if append:
            self._df[col] = result
        return result

    def atr(self, length: int = 14, append: bool = True, **kwargs) -> pd.Series:
        col = f"ATRr_{length}"
        high  = self._df["high"]
        low   = self._df["low"]
        close = self._df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        result = tr.ewm(com=length - 1, adjust=False).mean()
        if append:
            self._df[col] = result
        return result

    def vwap(self, anchor: str = "D", append: bool = True, **kwargs) -> pd.Series:
        col = "VWAP_D"
        df  = self._df
        if "volume" not in df.columns:
            return pd.Series(dtype=float)
        typical = (df["high"] + df["low"] + df["close"]) / 3
        tpv     = typical * df["volume"]

        # Group by day if index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, utc=True)
            except Exception:
                pass
        if isinstance(df.index, pd.DatetimeIndex):
            # Warn kalau data tidak mulai dari 00:00 UTC — VWAP bar pertama mungkin inakurat
            first_ts = df.index[0]
            if first_ts.hour != 0 or first_ts.minute != 0:
                import logging as _log
                _log.getLogger("ta_compat").debug(
                    "VWAP: data mulai %s (bukan 00:00 UTC) — bar pertama mungkin inakurat",
                    first_ts
                )
            cumtpv = tpv.groupby(df.index.date).cumsum()
            cumvol = df["volume"].groupby(df.index.date).cumsum()
        else:
            cumtpv = tpv.cumsum()
            cumvol = df["volume"].cumsum()

        result = cumtpv / cumvol.replace(0, np.nan)
        if append:
            self._df[col] = result
        return result


# Patch pandas DataFrame dengan accessor .ta
try:
    pd.api.extensions.register_dataframe_accessor("ta")(_TAAccessor)
    _PATCHED = True
except Exception:
    _PATCHED = False


def patch():
    """Panggil sekali di awal program untuk aktifkan df.ta.*"""
    pass  # import modul ini sudah cukup


if __name__ == "__main__":
    df = pd.DataFrame({
        "open":   [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        "high":   [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
        "low":    [0.5,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
        "close":  [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        "volume": [100]*15,
    })
    df.ta.ema(length=9,  append=True)
    df.ta.ema(length=21, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    print(df.tail())
    print("ta_compat OK — semua indikator berjalan tanpa pandas_ta/numba")
