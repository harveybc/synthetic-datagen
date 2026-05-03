"""TechStat feature engine — recomputes the full Project-3 feature
matrix from raw OHLCV.

This implements the same column set that `predictor`/`feature-eng`
exports as ``ethusdt_4h_tech_stat_full_model_ready.csv``.  It is meant
to be invoked AFTER synthetic OHLCV has been concatenated with real
OHLCV, so that the deterministic feature columns are recomputed
consistently across the entire series (Phase 4 §6: "deterministic
indicators must be recomputed, never independently generated").

The implementation uses only pandas + numpy.  Outputs are byte-stable
for any given input; rolling-window NaNs are forward-filled at the end
to match the ``model_ready`` semantics.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=1).mean()


def _wilder(s: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing (RMA): EMA with alpha = 1/period."""
    return s.ewm(alpha=1.0 / period, adjust=False, min_periods=1).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    rs_up = _wilder(up, period)
    rs_dn = _wilder(dn, period)
    rs = rs_up / rs_dn.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def _bbands(close: pd.Series, period: int, k: float = 2.0):
    mid = close.rolling(period, min_periods=1).mean()
    std = close.rolling(period, min_periods=1).std().fillna(0.0)
    upper = mid + k * std
    lower = mid - k * std
    width = (upper - lower) / mid.replace(0.0, np.nan)
    pct_b = (close - lower) / (upper - lower).replace(0.0, np.nan)
    return upper, mid, lower, pct_b.fillna(0.5), width.fillna(0.0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev = close.shift(1).fillna(close)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return _wilder(tr, period)


def _stoch(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, d: int = 3):
    ll = low.rolling(period, min_periods=1).min()
    hh = high.rolling(period, min_periods=1).max()
    k = 100.0 * (close - ll) / (hh - ll).replace(0.0, np.nan)
    k = k.fillna(50.0)
    return k, k.rolling(d, min_periods=1).mean()


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    hh = high.rolling(period, min_periods=1).max()
    ll = low.rolling(period, min_periods=1).min()
    wr = -100.0 * (hh - close) / (hh - ll).replace(0.0, np.nan)
    return wr.fillna(-50.0)


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma = tp.rolling(period, min_periods=1).mean()
    mad = (tp - sma).abs().rolling(period, min_periods=1).mean()
    return ((tp - sma) / (0.015 * mad.replace(0.0, np.nan))).fillna(0.0)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    macd = _ema(close, fast) - _ema(close, slow)
    signal = _ema(macd, sig)
    return macd, signal, macd - signal


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff().fillna(0.0))
    return (sign * volume).cumsum()


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series, period: int = 14) -> pd.Series:
    tp = (high + low + close) / 3.0
    mf = tp * volume
    direction = np.sign(tp.diff().fillna(0.0))
    pos = mf.where(direction > 0, 0.0).rolling(period, min_periods=1).sum()
    neg = mf.where(direction < 0, 0.0).rolling(period, min_periods=1).sum()
    ratio = pos / neg.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + ratio)).fillna(50.0)


def _trend_slope(close: pd.Series, period: int = 50):
    """OLS slope of close over a rolling window, normalized by mean."""
    x = np.arange(period, dtype=np.float64)
    x_mean = x.mean()
    x_dev = x - x_mean
    x_var = (x_dev ** 2).sum()

    def _slope(arr):
        if len(arr) < 2 or np.allclose(arr, arr[0]):
            return 0.0
        y = np.asarray(arr, dtype=np.float64)
        return float(((x_dev[: len(y)] * (y - y.mean())).sum()) / x_var) if x_var > 0 else 0.0

    slope = close.rolling(period, min_periods=2).apply(_slope, raw=True).fillna(0.0)
    strength = slope / close.rolling(period, min_periods=1).mean().replace(0.0, np.nan)
    return slope, strength.fillna(0.0)


def _hurst_proxy(close: pd.Series, period: int = 200) -> pd.Series:
    """Cheap Hurst proxy: log(std)/log(period) of rolling log-returns."""
    lr = np.log(close).diff().fillna(0.0)
    std = lr.rolling(period, min_periods=2).std().fillna(0.0)
    val = np.where(std > 0, np.log(std + 1e-12) / np.log(period), 0.5)
    return pd.Series(val, index=close.index)


def _zscore(close: pd.Series, period: int = 100) -> pd.Series:
    m = close.rolling(period, min_periods=1).mean()
    s = close.rolling(period, min_periods=1).std().replace(0.0, np.nan)
    return ((close - m) / s).fillna(0.0)


def _rolling_autocorr(s: pd.Series, lag: int, period: int) -> pd.Series:
    def _ac(arr):
        if len(arr) <= lag:
            return 0.0
        a = np.asarray(arr, dtype=np.float64)
        a0 = a[:-lag]; a1 = a[lag:]
        if a0.std() == 0 or a1.std() == 0:
            return 0.0
        return float(np.corrcoef(a0, a1)[0, 1])
    return s.rolling(period, min_periods=lag + 2).apply(_ac, raw=True).fillna(0.0)


class TechStatFeatureEngine:
    """Recomputes the full Project-3 ``tech_stat`` feature matrix."""

    plugin_params: Dict[str, Any] = {
        "datetime_column": "DATE_TIME",
        "open_col": "OPEN",
        "high_col": "HIGH",
        "low_col": "LOW",
        "close_col": "CLOSE",
        "volume_col": "VOLUME",
        "fillna": "ffill_then_zero",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.params: Dict[str, Any] = dict(self.plugin_params)
        if config:
            for k, v in config.items():
                if k in self.plugin_params:
                    self.params[k] = v

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.plugin_params:
                self.params[k] = v

    # ------------------------------------------------------------------
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        o = df[p["open_col"]].astype(np.float64)
        h = df[p["high_col"]].astype(np.float64)
        l = df[p["low_col"]].astype(np.float64)
        c = df[p["close_col"]].astype(np.float64)
        v = df[p["volume_col"]].astype(np.float64)

        out = pd.DataFrame(index=df.index)
        if p["datetime_column"] in df.columns:
            out[p["datetime_column"]] = df[p["datetime_column"]].values
        out["typical_price"] = (h + l + c) / 3.0
        out[p["open_col"]] = o
        out[p["high_col"]] = h
        out[p["low_col"]] = l
        out[p["close_col"]] = c
        out[p["volume_col"]] = v

        # Returns
        for n in (1, 5, 10, 20, 60):
            out[f"return_{n}"] = c.pct_change(n).fillna(0.0)
            out[f"log_return_{n}"] = np.log(c / c.shift(n)).fillna(0.0)

        # SMA / EMA / ratios
        for n in (10, 20, 50, 100, 200):
            sma = c.rolling(n, min_periods=1).mean()
            ema = _ema(c, n)
            out[f"sma_{n}"] = sma
            out[f"ema_{n}"] = ema
            out[f"close_sma_ratio_{n}"] = (c / sma.replace(0.0, np.nan) - 1.0).fillna(0.0)

        # MACD
        macd, signal, hist = _macd(c)
        out["macd"] = macd; out["macd_signal"] = signal; out["macd_hist"] = hist

        # RSI
        for n in (7, 14, 21):
            out[f"rsi_{n}"] = _rsi(c, n)

        # Stochastic / Williams / CCI
        sk, sd = _stoch(h, l, c, 14, 3)
        out["stoch_k"] = sk; out["stoch_d"] = sd
        out["williams_r_14"] = _williams_r(h, l, c, 14)
        out["cci_14"] = _cci(h, l, c, 14)

        # ROC / MOM
        for n in (10, 20, 60):
            out[f"roc_{n}"] = c.pct_change(n).fillna(0.0)
        for n in (10, 20):
            out[f"mom_{n}"] = (c - c.shift(n)).fillna(0.0)

        # Bollinger
        upper, mid, lower, pct_b, width = _bbands(c, 20, 2.0)
        out["bb_upper"] = upper; out["bb_middle"] = mid; out["bb_lower"] = lower
        out["bb_pct_b"] = pct_b; out["bb_width"] = width

        # ATR / NATR
        atr = _atr(h, l, c, 14)
        out["atr_14"] = atr
        out["natr_14"] = (atr / c.replace(0.0, np.nan)).fillna(0.0) * 100.0

        # Historical volatility (log-returns std × sqrt(n))
        lr = np.log(c / c.shift(1)).fillna(0.0)
        for n in (10, 20, 60):
            out[f"hist_vol_{n}"] = lr.rolling(n, min_periods=1).std().fillna(0.0) * np.sqrt(n)

        # EMA crosses (binary)
        out["ema_cross_10_50"] = (out["ema_10"] > out["ema_50"]).astype(int)
        out["ema_cross_20_100"] = (out["ema_20"] > out["ema_100"]).astype(int)

        # Trend slope / strength
        slope, strength = _trend_slope(c, 50)
        out["trend_slope_50"] = slope; out["trend_strength_50"] = strength

        # OBV / OBV delta
        obv = _obv(c, v)
        out["obv"] = obv
        out["obv_delta_20"] = obv.diff(20).fillna(0.0)

        # Volume SMAs / ratio / VWAP / MFI
        vsma10 = v.rolling(10, min_periods=1).mean()
        vsma20 = v.rolling(20, min_periods=1).mean()
        out["volume_sma_10"] = vsma10; out["volume_sma_20"] = vsma20
        out["volume_ratio_20"] = (v / vsma20.replace(0.0, np.nan)).fillna(1.0)
        tp = out["typical_price"]
        vwap_num = (tp * v).rolling(60, min_periods=1).sum()
        vwap_den = v.rolling(60, min_periods=1).sum().replace(0.0, np.nan)
        out["vwap_60"] = (vwap_num / vwap_den).fillna(c)
        out["mfi_14"] = _mfi(h, l, c, v, 14)

        # Statistical
        log_ret_1 = np.log(c / c.shift(1)).fillna(0.0)
        out["statistical__log_return_1"] = log_ret_1

        # Rolling moments of log-returns
        for n in (20, 60, 252):
            r = log_ret_1
            out[f"roll_mean_ret_{n}"] = r.rolling(n, min_periods=1).mean().fillna(0.0)
            out[f"roll_std_ret_{n}"] = r.rolling(n, min_periods=1).std().fillna(0.0)
            out[f"roll_skew_ret_{n}"] = r.rolling(n, min_periods=4).skew().fillna(0.0)
            out[f"roll_kurt_ret_{n}"] = r.rolling(n, min_periods=4).kurt().fillna(0.0)

        # Realized variance
        for n in (12, 48):
            out[f"realized_var_{n}"] = (log_ret_1 ** 2).rolling(n, min_periods=1).sum().fillna(0.0)

        # Autocorrelation features
        out["autocorr_lag1_100"] = _rolling_autocorr(log_ret_1, 1, 100)
        out["autocorr_lag5_100"] = _rolling_autocorr(log_ret_1, 5, 100)
        out["sqret_autocorr_lag1_100"] = _rolling_autocorr(log_ret_1 ** 2, 1, 100)

        # Vol regime indicators (binary based on rolling-quantile of vol)
        vol = out["hist_vol_20"]
        q25 = vol.rolling(252, min_periods=20).quantile(0.25).fillna(vol)
        q75 = vol.rolling(252, min_periods=20).quantile(0.75).fillna(vol)
        out["vol_regime_high"] = (vol > q75).astype(int)
        out["vol_regime_low"] = (vol < q25).astype(int)

        # Hurst proxy + zscore
        out["hurst_proxy_200"] = _hurst_proxy(c, 200)
        out["zscore_close_100"] = _zscore(c, 100)

        # Final NaN/inf cleanup
        out = out.replace([np.inf, -np.inf], np.nan)
        if p["fillna"] == "ffill_then_zero":
            out = out.ffill().fillna(0.0)
        return out

    # alias for plugin-loader compatibility
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.compute(df)


__all__ = ["TechStatFeatureEngine"]
