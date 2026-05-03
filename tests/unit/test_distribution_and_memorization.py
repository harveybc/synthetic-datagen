"""Tests for FinancialDistributionEvaluator + MemorizationEvaluator."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sdg_plugins.evaluator.financial_distribution_evaluator import (
    FinancialDistributionEvaluator,
)
from sdg_plugins.evaluator.memorization_evaluator import MemorizationEvaluator


def _gbm_close(n: int, seed: int, mu: float = 0.0, sigma: float = 0.001) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = rng.normal(mu, sigma, size=n)
    return 100.0 * np.exp(np.cumsum(r))


# --- distribution evaluator -------------------------------------------------


def test_distribution_evaluator_passes_when_distributions_match():
    real = _gbm_close(2000, seed=0)
    syn = _gbm_close(2000, seed=1)
    ev = FinancialDistributionEvaluator()
    rep = ev.evaluate_arrays(real, syn)
    assert rep["gates"]["wasserstein_pass"]
    assert rep["gates"]["ks_returns_pass"]


def test_distribution_evaluator_fails_when_scale_is_wildly_different():
    real = _gbm_close(2000, seed=0, sigma=0.001)
    syn = _gbm_close(2000, seed=1, sigma=0.01)   # 10x larger volatility
    ev = FinancialDistributionEvaluator()
    rep = ev.evaluate_arrays(real, syn)
    assert not rep["gates"]["all_pass"]


# --- memorization evaluator -------------------------------------------------


def test_memorization_evaluator_flags_verbatim_copy():
    real = _gbm_close(1500, seed=42)
    syn = real.copy()                             # blatant memorization
    ev = MemorizationEvaluator()
    ev.set_params(window=16, max_windows=200)
    rep = ev.evaluate_arrays(real, syn)
    assert rep["valid"]
    # Verbatim copy must trip at least one gate.
    assert not rep["gates"]["all_pass"]
    assert rep["copied_subseq_ratio"] > 0.5 or rep["duplicate_window_rate"] > 1e-3


def test_memorization_evaluator_passes_for_independent_samples():
    real = _gbm_close(1500, seed=0)
    syn = _gbm_close(1500, seed=99)
    ev = MemorizationEvaluator()
    ev.set_params(window=16, max_windows=200)
    rep = ev.evaluate_arrays(real, syn)
    assert rep["valid"]
    assert rep["gates"]["copied_subseq_pass"]
    assert rep["gates"]["nn_overlap_pass"]
