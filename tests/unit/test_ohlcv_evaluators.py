"""Tests for the OHLCV evaluators (algebraic + stylized facts)."""
import numpy as np
import pandas as pd

from sdg_plugins.evaluator.ohlcv_algebraic_evaluator import OhlcvAlgebraicEvaluator
from sdg_plugins.evaluator.financial_stylized_facts_evaluator import (
    FinancialStylizedFactsEvaluator,
)


def _good(n=200, seed=0):
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0, 0.01, n)
    c = 100.0 * np.exp(np.cumsum(log_ret))
    prev = np.concatenate([[100.0], c[:-1]])
    o = prev * np.exp(rng.normal(0, 0.002, n))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.003, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.003, n)))
    v = np.abs(rng.normal(1000, 100, n))
    return pd.DataFrame({"OPEN": o, "HIGH": h, "LOW": l, "CLOSE": c, "VOLUME": v})


def test_algebraic_passes_on_valid():
    df = _good()
    rep = OhlcvAlgebraicEvaluator().evaluate_df(df)
    assert rep["valid"] is True
    assert all(v == 0 for v in rep["violations"].values())


def test_algebraic_catches_high_violation():
    df = _good()
    df.loc[10, "HIGH"] = df.loc[10, "OPEN"] - 1.0  # high < open
    rep = OhlcvAlgebraicEvaluator().evaluate_df(df)
    assert rep["valid"] is False
    assert rep["violations"]["high_below_max_oc"] == 1


def test_algebraic_catches_negative_volume():
    df = _good()
    df.loc[5, "VOLUME"] = -1.0
    rep = OhlcvAlgebraicEvaluator().evaluate_df(df)
    assert rep["violations"]["negative_volume"] == 1


def test_stylized_facts_returns_expected_keys(tmp_path):
    df = _good(n=400, seed=2)
    p = tmp_path / "syn.csv"
    df.to_csv(p, index=False)
    ev = FinancialStylizedFactsEvaluator({"synthetic_data": str(p)})
    ev.set_params(synthetic_data=str(p))
    rep = ev.evaluate()
    assert "synthetic" in rep
    s = rep["synthetic"]
    for k in ("ret_mean", "ret_std", "ret_skew", "ret_kurt",
              "acf_sqret_lag1", "max_drawdown", "vol_mean",
              "vol_volatility_corr"):
        assert k in s
