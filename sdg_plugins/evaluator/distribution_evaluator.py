"""
Distribution evaluator plugin.

Programmatic API:

    from sdg_plugins.evaluator.distribution_evaluator import DistributionEvaluator
    ev = DistributionEvaluator()
    metrics = ev.evaluate(synthetic_df, real_df)

Or with raw price arrays:

    metrics = ev.evaluate_arrays(synthetic_prices, real_prices)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from app.data_processor import load_csv, prices_to_returns

log = logging.getLogger(__name__)


def _histogram_kl(a: np.ndarray, b: np.ndarray, bins: int = 100) -> float:
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    ha, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    ha = ha + 1e-10
    hb = hb + 1e-10
    ha /= ha.sum()
    hb /= hb.sum()
    return float(jensenshannon(ha, hb) ** 2)


def _wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    return float(stats.wasserstein_distance(a, b))


def _autocorrelation(x: np.ndarray, lag: int = 1) -> float:
    if len(x) <= lag:
        return 0.0
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def _adf_pvalue(x: np.ndarray) -> float:
    try:
        from statsmodels.tsa.stattools import adfuller
        return float(adfuller(x, maxlag=20, autolag="AIC")[1])
    except (ImportError, Exception):
        return -1.0


class DistributionEvaluator:
    """Plugin: evaluates synthetic vs real data quality."""

    plugin_params: Dict[str, Any] = {}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg: Dict[str, Any] = {}
        if config:
            self.cfg.update(config)

    def configure(self, config: Dict[str, Any]) -> None:
        self.cfg.update(config)

    def set_params(self, **kw) -> None:
        self.cfg.update(kw)

    def evaluate(
        self,
        synthetic: pd.DataFrame | str | None = None,
        real: pd.DataFrame | str | None = None,
    ) -> Dict[str, Any]:
        """
        Compare synthetic vs real timeseries.

        Args can be DataFrames or CSV paths. Falls back to self.cfg paths.
        """
        syn = self._resolve(synthetic, "synthetic_data")
        rea = self._resolve(real, "real_data")
        return self.evaluate_arrays(
            syn["typical_price"].values,
            rea["typical_price"].values,
        )

    def evaluate_arrays(
        self,
        synthetic_prices: np.ndarray,
        real_prices: np.ndarray,
    ) -> Dict[str, Any]:
        """Evaluate from raw price arrays."""
        r_syn = prices_to_returns(synthetic_prices)
        r_real = prices_to_returns(real_prices)

        m: Dict[str, Any] = {}
        m["kl_divergence"] = _histogram_kl(r_real, r_syn)
        m["wasserstein_distance"] = _wasserstein(r_real, r_syn)
        m["real_return_mean"] = float(np.mean(r_real))
        m["synthetic_return_mean"] = float(np.mean(r_syn))
        m["real_return_std"] = float(np.std(r_real))
        m["synthetic_return_std"] = float(np.std(r_syn))

        for lag in [1, 5, 10]:
            m[f"real_autocorr_lag{lag}"] = _autocorrelation(r_real, lag)
            m[f"synthetic_autocorr_lag{lag}"] = _autocorrelation(r_syn, lag)

        m["real_adf_pvalue"] = _adf_pvalue(r_real)
        m["synthetic_adf_pvalue"] = _adf_pvalue(r_syn)

        m["quality_score"] = (
            m["kl_divergence"]
            + 0.1 * m["wasserstein_distance"]
            + abs(m["real_return_std"] - m["synthetic_return_std"])
        )

        for k, v in m.items():
            log.info(f"  {k}: {v}")
        return m

    def _resolve(self, arg, cfg_key: str) -> pd.DataFrame:
        if arg is None:
            arg = self.cfg.get(cfg_key)
        if isinstance(arg, str):
            return load_csv(arg)
        if isinstance(arg, pd.DataFrame):
            return arg
        raise ValueError(f"Provide a DataFrame, CSV path, or set cfg['{cfg_key}']")
