"""Stylized-facts evaluator for synthetic financial OHLCV.

Computes a compact set of summary statistics traditionally used to
judge synthetic financial paths:

* return moments (mean / std / skew / kurtosis)
* tail quantiles (1%, 5%, 95%, 99%)
* volatility clustering proxy: ACF of squared returns at lag 1, 5, 20
* return autocorrelation at lag 1 (should be near 0 in efficient mkts)
* drawdown distribution: max drawdown, mean drawdown
* volume distribution: mean / std / quantiles
* volume-volatility Pearson correlation

If a real reference dataframe is supplied, also reports per-metric
absolute differences (synthetic - real).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _autocorr(x: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= len(x):
        return float("nan")
    a = x[:-lag] - x[:-lag].mean()
    b = x[lag:] - x[lag:].mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    if den == 0:
        return 0.0
    return float((a * b).sum() / den)


def _drawdown(c: np.ndarray) -> tuple[float, float]:
    peak = np.maximum.accumulate(c)
    dd = (c - peak) / peak
    return float(dd.min()), float(dd.mean())


def _stats(df: pd.DataFrame, close_col: str, vol_col: str) -> Dict[str, float]:
    c = df[close_col].to_numpy(dtype=np.float64)
    v = df[vol_col].to_numpy(dtype=np.float64)
    log_ret = np.diff(np.log(c))
    sq = log_ret ** 2
    s = pd.Series(log_ret)
    max_dd, mean_dd = _drawdown(c)
    vol_window = pd.Series(sq).rolling(20, min_periods=1).sum().to_numpy()
    n = min(len(vol_window), len(v[1:]))
    vv_corr = (
        float(np.corrcoef(vol_window[:n], v[1:1 + n])[0, 1]) if n > 1 else 0.0
    )
    return {
        "ret_mean": float(s.mean()),
        "ret_std": float(s.std()),
        "ret_skew": float(s.skew()) if len(s) > 2 else 0.0,
        "ret_kurt": float(s.kurt()) if len(s) > 3 else 0.0,
        "ret_q01": float(s.quantile(0.01)) if len(s) else 0.0,
        "ret_q05": float(s.quantile(0.05)) if len(s) else 0.0,
        "ret_q95": float(s.quantile(0.95)) if len(s) else 0.0,
        "ret_q99": float(s.quantile(0.99)) if len(s) else 0.0,
        "acf_ret_lag1": _autocorr(log_ret, 1),
        "acf_sqret_lag1": _autocorr(sq, 1),
        "acf_sqret_lag5": _autocorr(sq, 5),
        "acf_sqret_lag20": _autocorr(sq, 20),
        "max_drawdown": max_dd,
        "mean_drawdown": mean_dd,
        "vol_mean": float(np.mean(v)),
        "vol_std": float(np.std(v)),
        "vol_q05": float(np.quantile(v, 0.05)),
        "vol_q95": float(np.quantile(v, 0.95)),
        "vol_volatility_corr": vv_corr,
    }


class FinancialStylizedFactsEvaluator:
    plugin_params: Dict[str, Any] = {
        "synthetic_data": None,
        "real_data": None,
        "close_col": "CLOSE",
        "volume_col": "VOLUME",
        "metrics_file": None,
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

    def _read(self, path: str) -> pd.DataFrame:
        if str(path).lower().endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def evaluate(self) -> Dict[str, Any]:
        p = self.params
        if not p["synthetic_data"]:
            raise ValueError("synthetic_data is required")
        syn = self._read(p["synthetic_data"])
        report: Dict[str, Any] = {
            "synthetic": _stats(syn, p["close_col"], p["volume_col"]),
        }
        if p["real_data"]:
            real = self._read(p["real_data"])
            report["real"] = _stats(real, p["close_col"], p["volume_col"])
            report["abs_diff"] = {
                k: abs(report["synthetic"][k] - report["real"][k])
                for k in report["synthetic"]
            }
        if p["metrics_file"]:
            os.makedirs(os.path.dirname(os.path.abspath(p["metrics_file"])) or ".", exist_ok=True)
            with open(p["metrics_file"], "w") as f:
                json.dump(report, f, indent=2)
        return report


__all__ = ["FinancialStylizedFactsEvaluator"]
