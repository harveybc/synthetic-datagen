"""Tests for plugin loading and basic plugin functionality."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest


def _make_csv(n=500, seed=0):
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


class TestVaeGanTrainer:
    """Quick smoke test (very few epochs)."""

    def test_train_and_generate(self):
        csv_path = _make_csv(600)
        try:
            with tempfile.TemporaryDirectory() as td:
                model_path = os.path.join(td, "model.keras")
                out_path = os.path.join(td, "out.csv")

                # Train via plugin API
                from sdg_plugins.trainer.vae_gan_trainer import VaeGanTrainer
                trainer = VaeGanTrainer()
                trainer.configure({
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
                    "disc_layers": [16, 8],
                    "disc_dropout": 0.1,
                    "discriminator_lr": 1e-4,
                    "generator_lr": 1e-4,
                })
                trainer.train(train_data=[csv_path], save_model=model_path)
                assert os.path.exists(model_path)

                # Generate via plugin API (programmatic — like DOIN would)
                from sdg_plugins.generator.typical_price_generator import TypicalPriceGenerator
                gen = TypicalPriceGenerator()
                gen.load_model(model_path)
                df = gen.generate(seed=42, n_samples=100)

                assert len(df) == 100
                assert "typical_price" in df.columns
                assert "DATE_TIME" in df.columns

                # Determinism: same seed → same output
                df2 = gen.generate(seed=42, n_samples=100)
                np.testing.assert_array_equal(
                    df["typical_price"].values, df2["typical_price"].values
                )

                # Different seed → different output
                df3 = gen.generate(seed=99, n_samples=100)
                assert not np.array_equal(
                    df["typical_price"].values, df3["typical_price"].values
                )
        finally:
            os.unlink(csv_path)


class TestGeneratorCLI:
    """Test the CLI-style run_generate path."""

    def test_run_generate(self):
        csv_path = _make_csv(600)
        try:
            with tempfile.TemporaryDirectory() as td:
                model_path = os.path.join(td, "model.keras")
                out_path = os.path.join(td, "out.csv")

                from sdg_plugins.trainer.vae_gan_trainer import VaeGanTrainer
                t = VaeGanTrainer()
                t.configure({
                    "window_size": 20, "batch_size": 32, "epochs": 2,
                    "latent_dim": 4, "activation": "tanh",
                    "intermediate_layers": 1, "initial_layer_size": 16,
                    "layer_size_divisor": 2, "kl_weight": 1e-3,
                    "kl_anneal_epochs": 2, "mmd_lambda": 1e-2, "l2_reg": 1e-6,
                    "use_returns": True, "early_patience": 999,
                    "start_from_epoch": 1, "disc_layers": [8], "disc_dropout": 0.1,
                    "discriminator_lr": 1e-4, "generator_lr": 1e-4,
                })
                t.train(train_data=[csv_path], save_model=model_path)

                from sdg_plugins.generator.typical_price_generator import TypicalPriceGenerator
                gen = TypicalPriceGenerator()
                gen.configure({
                    "load_model": model_path,
                    "output_file": out_path,
                    "seed": 42,
                    "n_samples": 50,
                    "use_returns": True,
                })
                df = gen.run_generate()
                assert os.path.exists(out_path)
                assert len(df) == 50
        finally:
            os.unlink(csv_path)


class TestEvaluator:
    def test_evaluate_from_paths(self):
        csv1 = _make_csv(500, seed=0)
        csv2 = _make_csv(500, seed=1)
        try:
            from sdg_plugins.evaluator.distribution_evaluator import DistributionEvaluator
            ev = DistributionEvaluator()
            m = ev.evaluate(synthetic=csv2, real=csv1)
            assert "kl_divergence" in m
            assert "quality_score" in m
            assert m["kl_divergence"] >= 0
        finally:
            os.unlink(csv1)
            os.unlink(csv2)

    def test_evaluate_from_dataframes(self):
        rng = np.random.default_rng(0)
        df1 = pd.DataFrame({
            "DATE_TIME": pd.date_range("2020-01-01", periods=200, freq="4h"),
            "typical_price": 1.3 + np.cumsum(rng.standard_normal(200) * 0.0005),
        })
        df2 = pd.DataFrame({
            "DATE_TIME": pd.date_range("2020-01-01", periods=200, freq="4h"),
            "typical_price": 1.3 + np.cumsum(rng.standard_normal(200) * 0.0005),
        })
        from sdg_plugins.evaluator.distribution_evaluator import DistributionEvaluator
        ev = DistributionEvaluator()
        m = ev.evaluate(synthetic=df2, real=df1)
        assert "quality_score" in m

    def test_evaluate_arrays(self):
        rng = np.random.default_rng(0)
        p1 = 1.3 + np.cumsum(rng.standard_normal(200) * 0.0005)
        p2 = 1.3 + np.cumsum(rng.standard_normal(200) * 0.0005)
        from sdg_plugins.evaluator.distribution_evaluator import DistributionEvaluator
        ev = DistributionEvaluator()
        m = ev.evaluate_arrays(p2, p1)
        assert "quality_score" in m
