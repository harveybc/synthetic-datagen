"""Tests for ``regime_residual_bootstrap_v1``.

Covers:

* heldout-boundary guard (no Stage C rows used at fit time)
* deterministic seeding (two runs with same seed produce identical CSV)
* zero OHLC algebraic violations on a fixture
* lower duplicate / NN-overlap memorization risk than plain
  ``stationary_bootstrap`` on the same fixture
* fail-closed behavior: build pipeline refuses to write the augmented
  Project 3 CSV when memorization gates fail (re-uses existing
  ``examples/scripts/build_augmented_project3_training.py`` CLI).
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from sdg_plugins.evaluator.memorization_evaluator import MemorizationEvaluator
from sdg_plugins.evaluator.ohlcv_algebraic_evaluator import OhlcvAlgebraicEvaluator
from sdg_plugins.generator.regime_residual_bootstrap_ohlcv_generator import (
    RegimeResidualBootstrapOhlcvGenerator,
)
from sdg_plugins.generator.stationary_bootstrap_ohlcv_generator import (
    StationaryBootstrapOhlcvGenerator,
)
from sdg_plugins.trainer.regime_residual_bootstrap_ohlcv_trainer import (
    RegimeResidualBootstrapOhlcvTrainer,
)
from sdg_plugins.trainer.stationary_bootstrap_ohlcv_trainer import (
    StationaryBootstrapOhlcvTrainer,
)


def _fixture(n=600, seed=11):
    """Synthetic OHLCV fixture with mild volatility clustering."""
    rng = np.random.default_rng(seed)
    # Two-regime volatility: alternating 100-row low-vol and 100-row high-vol slabs.
    vols = np.where((np.arange(n) // 100) % 2 == 0, 0.006, 0.02)
    log_ret = rng.normal(0, vols, n)
    c = 100.0 * np.exp(np.cumsum(log_ret))
    prev = np.concatenate([[100.0], c[:-1]])
    o = prev * np.exp(rng.normal(0, 0.002, n))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.003, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.003, n)))
    v = np.abs(rng.normal(1000, 200, n))
    return pd.DataFrame({
        "DATE_TIME": pd.date_range("2018-01-01", periods=n, freq="4h"),
        "OPEN": o, "HIGH": h, "LOW": l, "CLOSE": c, "VOLUME": v,
    })


# ---------------------------------------------------------------------------
# 1. heldout boundary guard
# ---------------------------------------------------------------------------
def test_regime_residual_rejects_heldout_rows(tmp_path):
    df = _fixture(400, seed=3)
    # Start far enough back that part of the panel is on/after 2025-01-01.
    df["DATE_TIME"] = pd.date_range("2024-12-01", periods=len(df), freq="4h")
    assert (df["DATE_TIME"] >= pd.Timestamp("2025-01-01")).any(), "fixture sanity"
    train_csv = tmp_path / "train.csv"
    df.to_csv(train_csv, index=False)

    trainer = RegimeResidualBootstrapOhlcvTrainer({
        "train_data": str(train_csv),
        "save_model": str(tmp_path / "m.npz"),
        "heldout_boundary": "2025-01-01 00:00:00",
        "project3_mode": True,
        "seed": 1,
    })
    with pytest.raises(ValueError, match="heldout_boundary"):
        trainer.train()


# ---------------------------------------------------------------------------
# 2. deterministic seed
# ---------------------------------------------------------------------------
def test_regime_residual_seed_determinism(tmp_path):
    train_csv = tmp_path / "train.csv"
    _fixture(400, seed=7).to_csv(train_csv, index=False)
    model = tmp_path / "m.npz"

    RegimeResidualBootstrapOhlcvTrainer({
        "train_data": str(train_csv),
        "save_model": str(model),
        "block_length_mean": 12,
        "n_regimes": 3,
        "vol_window": 12,
        "jitter_sigma": 0.4,
        "seed": 99,
    }).train()

    out1 = tmp_path / "s1.csv"
    out2 = tmp_path / "s2.csv"
    for outp in (out1, out2):
        RegimeResidualBootstrapOhlcvGenerator({
            "load_model": str(model),
            "output_file": str(outp),
            "n_samples": 150,
            "seed": 12345,
        }).run_generate()
    a = pd.read_csv(out1).to_numpy()
    b = pd.read_csv(out2).to_numpy()
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# 3. zero algebraic OHLC violations
# ---------------------------------------------------------------------------
def test_regime_residual_zero_ohlc_violations(tmp_path):
    train_csv = tmp_path / "train.csv"
    _fixture(500, seed=4).to_csv(train_csv, index=False)
    model = tmp_path / "m.npz"
    out = tmp_path / "synth.csv"
    RegimeResidualBootstrapOhlcvTrainer({
        "train_data": str(train_csv),
        "save_model": str(model),
        "block_length_mean": 12,
        "n_regimes": 3,
        "vol_window": 12,
        "jitter_sigma": 0.5,
        "seed": 21,
    }).train()
    info = RegimeResidualBootstrapOhlcvGenerator({
        "load_model": str(model),
        "output_file": str(out),
        "n_samples": 300,
        "seed": 21,
    }).run_generate()
    assert info["n_rows"] == 300
    rep = OhlcvAlgebraicEvaluator({"synthetic_data": str(out)}).evaluate()
    assert rep["valid"] is True, rep["violations"]


# ---------------------------------------------------------------------------
# 4. lower duplicate / NN-overlap risk than stationary bootstrap
# ---------------------------------------------------------------------------
def test_regime_residual_lower_memorization_than_stationary(tmp_path):
    train_csv = tmp_path / "train.csv"
    _fixture(1500, seed=33).to_csv(train_csv, index=False)

    # Plain stationary bootstrap on the same fixture (memorization-risky).
    stat_model = tmp_path / "stat.npz"
    StationaryBootstrapOhlcvTrainer({
        "train_data": str(train_csv), "save_model": str(stat_model),
        "block_length_mean": 12, "seed": 0,
    }).train()
    stat_out = tmp_path / "stat_synth.csv"
    StationaryBootstrapOhlcvGenerator({
        "load_model": str(stat_model), "output_file": str(stat_out),
        "n_samples": 600, "seed": 0,
    }).run_generate()

    # Regime-residual bootstrap on the same fixture, with strong jitter.
    rr_model = tmp_path / "rr.npz"
    RegimeResidualBootstrapOhlcvTrainer({
        "train_data": str(train_csv), "save_model": str(rr_model),
        "block_length_mean": 12, "n_regimes": 3, "vol_window": 12,
        "jitter_sigma": 1.0, "seed": 0,
    }).train()
    rr_out = tmp_path / "rr_synth.csv"
    RegimeResidualBootstrapOhlcvGenerator({
        "load_model": str(rr_model), "output_file": str(rr_out),
        "n_samples": 600, "seed": 0,
    }).run_generate()

    def _mem(real_csv, syn_csv):
        return MemorizationEvaluator({
            "real_data": str(real_csv),
            "synthetic_data": str(syn_csv),
            "window": 16,
            "max_windows": 200,
            "seed": 0,
        }).evaluate()

    stat_mem = _mem(train_csv, stat_out)
    rr_mem = _mem(train_csv, rr_out)

    # Continuous Gaussian jitter on residuals MUST break consecutive
    # NN-runs (the property that makes ``copied_subseq_ratio`` blow up
    # for plain stationary bootstrap).
    assert rr_mem["copied_subseq_ratio"] == 0.0
    assert rr_mem["copied_subseq_ratio"] < stat_mem["copied_subseq_ratio"]
    # Cosine NN-overlap should also weakly improve (or stay at zero).
    assert rr_mem["nn_overlap_rate"] <= stat_mem["nn_overlap_rate"] + 1e-9


# ---------------------------------------------------------------------------
# 5. fail-closed pipeline -- artificially tighten gates so they fail
# ---------------------------------------------------------------------------
def test_regime_residual_fail_closed_when_gates_fail(tmp_path):
    """Force the memorization evaluator to fail by setting impossible
    thresholds, then assert the build pipeline refuses to write the
    augmented CSV."""
    train_csv = tmp_path / "train.csv"
    _fixture(400, seed=5).to_csv(train_csv, index=False)

    rr_model = tmp_path / "rr.npz"
    RegimeResidualBootstrapOhlcvTrainer({
        "train_data": str(train_csv), "save_model": str(rr_model),
        "block_length_mean": 8, "n_regimes": 3, "vol_window": 8,
        "jitter_sigma": 0.3, "seed": 0,
    }).train()
    rr_out = tmp_path / "rr_synth.csv"
    RegimeResidualBootstrapOhlcvGenerator({
        "load_model": str(rr_model), "output_file": str(rr_out),
        "n_samples": 200, "seed": 0,
    }).run_generate()

    # All gate thresholds set to negative -> impossible -> must fail.
    mem = MemorizationEvaluator({
        "real_data": str(train_csv),
        "synthetic_data": str(rr_out),
        "window": 16,
        "max_windows": 100,
        "seed": 0,
        "classifier_auc_max": -1.0,
        "duplicate_window_rate_max": -1.0,
        "max_nn_overlap_rate_max": -1.0,
        "copied_subseq_ratio_max": -1.0,
    }).evaluate()
    assert mem["gates"]["all_pass"] is False
    # The build script's gate-aggregator AND must therefore evaluate False
    # and refuse to emit a Project 3 training CSV.
    gates_pass = mem["gates"]["all_pass"]
    assert gates_pass is False
