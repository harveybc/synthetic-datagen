"""Distributional-distance evaluator for synthetic financial OHLCV.

Compares synthetic vs. real on:

* KS distance + p-value on log-returns
* 1-Wasserstein distance on log-returns
* ACF distance on returns and squared returns (lags 1..max_lag)
* Drawdown-distribution KS distance
* (optional) MMD over fixed-length windows with an RBF kernel

All metrics are computed on log-returns of CLOSE.  Tests are tolerant
to small samples; metrics that cannot be computed return ``None``.

Pass/fail thresholds (configurable; defaults follow Phase 4 §4.2):

* ``ks_pvalue_min``     = 0.01   (returns)
* ``wass_ratio_max``    = 1.5    (Wasserstein-1(returns) <= 1.5x inter-decile of real)
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:  # SciPy is already a hard dep of the repo.
    from scipy import stats as _stats
except Exception:  # pragma: no cover
    _stats = None


def _log_returns(c: np.ndarray) -> np.ndarray:
    return np.diff(np.log(c))


def _autocorr(x: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= len(x):
        return float("nan")
    a = x[:-lag] - x[:-lag].mean()
    b = x[lag:] - x[lag:].mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if den == 0:
        return 0.0
    return float((a * b).sum() / den)


def _drawdowns(c: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(c)
    return (c - peak) / peak


def _wasserstein1(a: np.ndarray, b: np.ndarray) -> float:
    """1-Wasserstein on 1D samples; SciPy if available else manual."""
    if _stats is not None and hasattr(_stats, "wasserstein_distance"):
        return float(_stats.wasserstein_distance(a, b))
    a = np.sort(a); b = np.sort(b)
    n = max(len(a), len(b))
    qs = np.linspace(0.0, 1.0, n, endpoint=False) + 0.5 / n
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def _ks(a: np.ndarray, b: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    if _stats is None or len(a) < 2 or len(b) < 2:
        return None, None
    res = _stats.ks_2samp(a, b)
    return float(res.statistic), float(res.pvalue)


def _mmd_rbf(a: np.ndarray, b: np.ndarray, sigma: float) -> float:
    """Squared MMD with an RBF kernel; ``a``,``b`` are (n, d) float."""
    def k(x, y):
        x2 = (x * x).sum(1)[:, None]
        y2 = (y * y).sum(1)[None, :]
        d2 = x2 + y2 - 2.0 * x @ y.T
        return np.exp(-d2 / (2.0 * sigma * sigma))
    Kxx = k(a, a); Kyy = k(b, b); Kxy = k(a, b)
    n = len(a); m = len(b)
    s = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1) + 1e-9)
    t = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1) + 1e-9)
    u = Kxy.mean()
    return float(s + t - 2.0 * u)


class FinancialDistributionEvaluator:
    plugin_params: Dict[str, Any] = {
        "synthetic_data": None,
        "real_data": None,
        "close_col": "CLOSE",
        "metrics_file": None,
        "max_acf_lag": 20,
        "mmd_window": 32,
        "mmd_max_windows": 200,
        "mmd_seed": 0,
        # Pass/fail gate thresholds (Phase 4 §4.2):
        "ks_pvalue_min": 0.01,
        "wass_ratio_max": 1.5,
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
        if not p["synthetic_data"] or not p["real_data"]:
            raise ValueError("synthetic_data and real_data are both required")
        syn = self._read(p["synthetic_data"])[p["close_col"]].to_numpy(dtype=np.float64)
        real = self._read(p["real_data"])[p["close_col"]].to_numpy(dtype=np.float64)
        return self.evaluate_arrays(real, syn)

    def evaluate_arrays(self, real_close: np.ndarray, syn_close: np.ndarray) -> Dict[str, Any]:
        p = self.params
        r_real = _log_returns(real_close)
        r_syn = _log_returns(syn_close)

        ks_stat, ks_p = _ks(r_real, r_syn)
        wass = _wasserstein1(r_real, r_syn)
        # Inter-decile range of REAL returns is the gate scale (Phase 4 §4.2).
        real_idr = float(np.quantile(r_real, 0.9) - np.quantile(r_real, 0.1)) or 1e-12
        wass_ratio = float(wass / real_idr)

        acf_diffs_ret = []
        acf_diffs_sq = []
        max_lag = int(p["max_acf_lag"])
        sq_real = r_real ** 2
        sq_syn = r_syn ** 2
        for lag in range(1, max_lag + 1):
            acf_diffs_ret.append(abs(_autocorr(r_real, lag) - _autocorr(r_syn, lag)))
            acf_diffs_sq.append(abs(_autocorr(sq_real, lag) - _autocorr(sq_syn, lag)))
        acf_l1_ret = float(np.mean(acf_diffs_ret))
        acf_l1_sq = float(np.mean(acf_diffs_sq))

        dd_real = _drawdowns(real_close)
        dd_syn = _drawdowns(syn_close)
        dd_ks_stat, dd_ks_p = _ks(dd_real, dd_syn)

        # MMD on fixed-length return windows (subsampled if needed).
        mmd_val: Optional[float] = None
        w = int(p["mmd_window"])
        if w > 0 and len(r_real) >= w and len(r_syn) >= w:
            rng = np.random.default_rng(int(p["mmd_seed"]))
            n_real = len(r_real) - w + 1
            n_syn = len(r_syn) - w + 1
            n_take = int(min(p["mmd_max_windows"], n_real, n_syn))
            i_r = rng.choice(n_real, size=n_take, replace=False)
            i_s = rng.choice(n_syn, size=n_take, replace=False)
            A = np.stack([r_real[i:i + w] for i in i_r])
            B = np.stack([r_syn[i:i + w] for i in i_s])
            # Median heuristic for sigma on combined sample.
            cat = np.concatenate([A, B])[:200]
            d2 = ((cat[:, None] - cat[None, :]) ** 2).sum(-1)
            sigma = float(np.sqrt(np.median(d2[d2 > 0]) / 2.0)) or 1.0
            mmd_val = _mmd_rbf(A, B, sigma)

        gates = {
            "ks_returns_pass": (ks_p is not None and ks_p > p["ks_pvalue_min"]),
            "wasserstein_pass": wass_ratio <= p["wass_ratio_max"],
        }
        gates["all_pass"] = all(gates.values())

        report = {
            "n_real": int(len(real_close)),
            "n_synthetic": int(len(syn_close)),
            "ks_returns": {"statistic": ks_stat, "pvalue": ks_p},
            "wasserstein_returns": wass,
            "wasserstein_returns_ratio": wass_ratio,
            "acf_l1_returns": acf_l1_ret,
            "acf_l1_sq_returns": acf_l1_sq,
            "drawdown_ks": {"statistic": dd_ks_stat, "pvalue": dd_ks_p},
            "mmd_window_rbf": mmd_val,
            "thresholds": {
                "ks_pvalue_min": p["ks_pvalue_min"],
                "wass_ratio_max": p["wass_ratio_max"],
            },
            "gates": gates,
        }
        if p["metrics_file"]:
            os.makedirs(os.path.dirname(os.path.abspath(p["metrics_file"])) or ".", exist_ok=True)
            with open(p["metrics_file"], "w") as f:
                json.dump(report, f, indent=2)
        return report


__all__ = ["FinancialDistributionEvaluator"]
