"""Tests for the minimal causal feature engine."""
import numpy as np
import pandas as pd

from sdg_plugins.feature_engine.minimal_financial_feature_engine import (
    MinimalFinancialFeatureEngine,
)


def _ohlcv(n=80):
    rng = np.random.default_rng(0)
    c = 100 + np.cumsum(rng.normal(0, 0.5, n))
    o = c + rng.normal(0, 0.1, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.2, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.2, n))
    v = np.abs(rng.normal(100, 10, n))
    return pd.DataFrame({"OPEN": o, "HIGH": h, "LOW": l, "CLOSE": c, "VOLUME": v})


def test_recompute_features_present_and_finite():
    out = MinimalFinancialFeatureEngine().compute(_ohlcv())
    for k in ("TYPICAL_PRICE", "RETURN", "LOG_RETURN", "ROLL_MEAN_14",
              "ROLL_STD_14", "REALIZED_VAR_14", "ATR_14", "VOL_RATIO_14"):
        assert k in out.columns
        assert np.isfinite(out[k]).all()


def test_typical_price_matches_definition():
    df = _ohlcv()
    out = MinimalFinancialFeatureEngine().compute(df)
    np.testing.assert_allclose(
        out["TYPICAL_PRICE"].to_numpy(),
        ((df["HIGH"] + df["LOW"] + df["CLOSE"]) / 3.0).to_numpy(),
    )
