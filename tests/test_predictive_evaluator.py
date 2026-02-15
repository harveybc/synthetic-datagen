"""Tests for the predictive utility evaluator (thesis phase 4 methodology)."""

import numpy as np
import pandas as pd
import pytest


def _make_prices(n=500, seed=0, base=1.3):
    rng = np.random.default_rng(seed)
    return base + np.cumsum(rng.standard_normal(n) * 0.0005)


def _make_df(n=500, seed=0):
    return pd.DataFrame({
        "DATE_TIME": pd.date_range("2020-01-01", periods=n, freq="4h"),
        "typical_price": _make_prices(n, seed),
    })


class TestPredictiveEvaluator:
    def test_basic_evaluation(self):
        """Smoke test: run full evaluate pipeline with DataFrames."""
        from sdg_plugins.evaluator.predictive_evaluator import PredictiveEvaluator

        df_train = _make_df(500, seed=0)
        df_val = _make_df(300, seed=1)
        df_test = _make_df(300, seed=2)
        df_syn = _make_df(200, seed=3)

        ev = PredictiveEvaluator()
        ev.configure({
            "window_size": 20,
            "horizon": 1,
            "eval_epochs": 3,
            "eval_batch_size": 32,
        })

        result = ev.evaluate(
            synthetic=df_syn,
            real_train=df_train,
            real_val=df_val,
            real_test=df_test,
        )

        # Check all expected keys exist
        assert "baseline_mae_val" in result
        assert "baseline_mae_test" in result
        assert "augmented_mae_val" in result
        assert "augmented_mae_test" in result
        assert "mae_delta_val" in result
        assert "mae_delta_test" in result
        assert "synthetic_helps_val" in result
        assert "synthetic_helps_test" in result
        assert "mae_pct_change_val" in result
        assert "mae_pct_change_test" in result
        assert "n_real_train" in result
        assert "n_synthetic" in result

        # Sanity: MAEs should be positive
        assert result["baseline_mae_val"] > 0
        assert result["baseline_mae_test"] > 0
        assert result["augmented_mae_val"] > 0
        assert result["augmented_mae_test"] > 0

        # Delta should be baseline - augmented sign
        assert abs(
            result["mae_delta_test"]
            - (result["augmented_mae_test"] - result["baseline_mae_test"])
        ) < 1e-8

    def test_includes_distribution_metrics(self):
        """Should also include secondary distribution metrics."""
        from sdg_plugins.evaluator.predictive_evaluator import PredictiveEvaluator

        ev = PredictiveEvaluator()
        ev.configure({
            "window_size": 20,
            "horizon": 1,
            "eval_epochs": 2,
            "eval_batch_size": 32,
        })
        result = ev.evaluate(
            synthetic=_make_df(200, seed=10),
            real_train=_make_df(300, seed=0),
            real_val=_make_df(200, seed=1),
            real_test=_make_df(200, seed=2),
        )
        # Distribution metrics should be prefixed with dist_
        assert "dist_kl_divergence" in result
        assert "dist_wasserstein_distance" in result
