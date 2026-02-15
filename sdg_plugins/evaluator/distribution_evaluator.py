"""
Distribution evaluator plugin.

Compares synthetic vs real typical_price timeseries using:
- KL divergence (on returns distribution)
- Wasserstein distance
- Autocorrelation preservation
- ADF stationarity test on returns
- Mean / std of returns
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from app.data_processor import load_csv, prices_to_returns

log = logging.getLogger(__name__)


def _histogram_kl(a: np.ndarray, b: np.ndarray, bins: int = 100) -> float:
    """Symmetric KL divergence via histogram approximation."""
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    ha, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    ha = ha + 1e-10
    hb = hb + 1e-10
    ha = ha / ha.sum()
    hb = hb / hb.sum()
    return float(jensenshannon(ha, hb) ** 2)  # JSD² ≈ symmetric KL


def _wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    return float(stats.wasserstein_distance(a, b))


def _autocorrelation(x: np.ndarray, lag: int = 1) -> float:
    if len(x) <= lag:
        return 0.0
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def _adf_pvalue(x: np.ndarray) -> float:
    """Augmented Dickey-Fuller p-value (lower = more stationary)."""
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(x, maxlag=20, autolag="AIC")
        return float(result[1])
    except ImportError:
        log.warning("statsmodels not installed — skipping ADF test")
        return -1.0
    except Exception:
        return -1.0


class DistributionEvaluator:
    """Plugin: evaluates synthetic vs real data quality."""

    plugin_params: Dict[str, Any] = {}

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def set_params(self, **kw):
        self.cfg.update(kw)

    def evaluate(self) -> Dict[str, Any]:
        cfg = self.cfg
        df_syn = load_csv(cfg["synthetic_data"])
        df_real = load_csv(cfg["real_data"])

        p_syn = df_syn["typical_price"].values
        p_real = df_real["typical_price"].values

        r_syn = prices_to_returns(p_syn)
        r_real = prices_to_returns(p_real)

        metrics: Dict[str, Any] = {}

        # Distribution metrics
        metrics["kl_divergence"] = _histogram_kl(r_real, r_syn)
        metrics["wasserstein_distance"] = _wasserstein(r_real, r_syn)

        # Moment matching
        metrics["real_return_mean"] = float(np.mean(r_real))
        metrics["synthetic_return_mean"] = float(np.mean(r_syn))
        metrics["real_return_std"] = float(np.std(r_real))
        metrics["synthetic_return_std"] = float(np.std(r_syn))

        # Autocorrelation
        for lag in [1, 5, 10]:
            metrics[f"real_autocorr_lag{lag}"] = _autocorrelation(r_real, lag)
            metrics[f"synthetic_autocorr_lag{lag}"] = _autocorrelation(r_syn, lag)

        # Stationarity
        metrics["real_adf_pvalue"] = _adf_pvalue(r_real)
        metrics["synthetic_adf_pvalue"] = _adf_pvalue(r_syn)

        # Summary score (lower = better)
        metrics["quality_score"] = (
            metrics["kl_divergence"]
            + 0.1 * metrics["wasserstein_distance"]
            + abs(metrics["real_return_std"] - metrics["synthetic_return_std"])
        )

        for k, v in metrics.items():
            log.info(f"  {k}: {v}")

        return metrics
