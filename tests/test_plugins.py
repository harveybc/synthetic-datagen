"""Tests for plugin loading and basic plugin functionality."""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_csv(n=500, seed=0):
    """Create a temp CSV with typical_price data. Returns path."""
    rng = np.random.default_rng(seed)
    prices = 1.3 + np.cumsum(rng.standard_normal(n) * 0.0005)
    df = pd.DataFrame({
        "DATE_TIME": pd.date_range("2020-01-01", periods=n, freq="4h"),
        "typical_price": prices,
    })
    f = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    df.to_csv(f.name, index=False)
    f.close()
    return f.name


# ── Tests ───────────────────────────────────────────────────────────────────

class TestVaeGanTrainer:
    """Quick smoke test (very few epochs)."""

    def test_train_and_generate(self):
        csv_path = _make_csv(600)
        try:
            with tempfile.TemporaryDirectory() as td:
                model_path = os.path.join(td, "model.keras")
                out_path = os.path.join(td, "out.csv")

                # Train
                from sdg_plugins.trainer.vae_gan_trainer import VaeGanTrainer
                cfg = {
                    "train_data": [csv_path],
                    "save_model": model_path,
                    "window_size": 20,
                    "batch_size": 32,
                    "epochs": 3,
                    "learning_rate": 1e-3,
                    "latent_dim": 4,
                    "activation": "tanh",
                    "intermediate_layers": 1,
                    "initial_layer_size": 16,
                    "layer_size_divisor": 2,
                    "kl_weight": 1e-3,
                    "kl_anneal_epochs": 2,
                    "mmd_lambda": 1e-2,
                    "l2_reg": 1e-6,
                    "use_returns": True,
                    "early_patience": 999,
                    "start_from_epoch": 1,
                    "min_delta": 1e-7,
                    "downsample_factor": 1,
                    "disc_layers": [16, 8],
                    "disc_dropout": 0.1,
                    "discriminator_lr": 1e-4,
                    "generator_lr": 1e-4,
                }
                t = VaeGanTrainer(cfg)
                t.train()
                assert os.path.exists(model_path)

                # Generate
                from sdg_plugins.generator.typical_price_generator import TypicalPriceGenerator
                gen_cfg = {
                    "load_model": model_path,
                    "output_file": out_path,
                    "seed": 42,
                    "n_samples": 100,
                    "use_returns": True,
                    "start_datetime": "2021-01-01 00:00:00",
                    "interval_hours": 4,
                    "downsample_factor": 1,
                }
                g = TypicalPriceGenerator(gen_cfg)
                df = g.generate()
                assert len(df) == 100
                assert "typical_price" in df.columns
                assert "DATE_TIME" in df.columns

                # Determinism: same seed → same output
                g2 = TypicalPriceGenerator(gen_cfg)
                df2 = g2.generate()
                np.testing.assert_array_equal(
                    df["typical_price"].values, df2["typical_price"].values
                )

                # Different seed → different output
                gen_cfg2 = gen_cfg.copy()
                gen_cfg2["seed"] = 99
                gen_cfg2["output_file"] = os.path.join(td, "out2.csv")
                g3 = TypicalPriceGenerator(gen_cfg2)
                df3 = g3.generate()
                assert not np.array_equal(
                    df["typical_price"].values, df3["typical_price"].values
                )
        finally:
            os.unlink(csv_path)


class TestEvaluator:
    def test_evaluate(self):
        csv1 = _make_csv(500, seed=0)
        csv2 = _make_csv(500, seed=1)
        try:
            from sdg_plugins.evaluator.distribution_evaluator import DistributionEvaluator
            ev = DistributionEvaluator({
                "synthetic_data": csv2,
                "real_data": csv1,
            })
            m = ev.evaluate()
            assert "kl_divergence" in m
            assert "wasserstein_distance" in m
            assert "quality_score" in m
            assert m["kl_divergence"] >= 0
        finally:
            os.unlink(csv1)
            os.unlink(csv2)
