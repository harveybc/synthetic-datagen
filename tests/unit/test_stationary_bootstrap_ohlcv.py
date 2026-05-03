"""End-to-end test: bootstrap trainer + generator + algebraic evaluator."""
import os
import tempfile

import numpy as np
import pandas as pd

from sdg_plugins.trainer.stationary_bootstrap_ohlcv_trainer import (
    StationaryBootstrapOhlcvTrainer,
)
from sdg_plugins.generator.stationary_bootstrap_ohlcv_generator import (
    StationaryBootstrapOhlcvGenerator,
)
from sdg_plugins.evaluator.ohlcv_algebraic_evaluator import OhlcvAlgebraicEvaluator


def _fixture(n=400, seed=11):
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0, 0.012, n)
    c = 100.0 * np.exp(np.cumsum(log_ret))
    prev = np.concatenate([[100.0], c[:-1]])
    o = prev * np.exp(rng.normal(0, 0.003, n))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.004, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.004, n)))
    v = np.abs(rng.normal(1000, 200, n))
    return pd.DataFrame({
        "DATE_TIME": pd.date_range("2019-01-01", periods=n, freq="1h"),
        "OPEN": o, "HIGH": h, "LOW": l, "CLOSE": c, "VOLUME": v,
    })


def test_bootstrap_train_generate_evaluate(tmp_path):
    train_csv = tmp_path / "train.csv"
    _fixture(400, seed=11).to_csv(train_csv, index=False)
    model = tmp_path / "boot.npz"
    out = tmp_path / "synth.csv"

    trainer = StationaryBootstrapOhlcvTrainer({
        "train_data": str(train_csv),
        "save_model": str(model),
        "block_length_mean": 16,
        "seed": 7,
    })
    trainer.set_params(train_data=str(train_csv), save_model=str(model),
                       block_length_mean=16, seed=7)
    meta = trainer.train()
    assert os.path.exists(meta["model_file"])
    assert os.path.exists(meta["transformer_file"])

    gen = StationaryBootstrapOhlcvGenerator({
        "load_model": str(model),
        "output_file": str(out),
        "n_samples": 250,
        "seed": 7,
    })
    gen.set_params(load_model=str(model), output_file=str(out),
                   n_samples=250, seed=7)
    info = gen.run_generate()
    assert info["n_rows"] == 250
    assert os.path.exists(out)

    ev = OhlcvAlgebraicEvaluator({"synthetic_data": str(out)})
    ev.set_params(synthetic_data=str(out))
    rep = ev.evaluate()
    assert rep["valid"] is True, rep["violations"]
    assert rep["n_rows"] == 250


def test_bootstrap_seed_determinism(tmp_path):
    train_csv = tmp_path / "train.csv"
    _fixture(300, seed=3).to_csv(train_csv, index=False)
    model = tmp_path / "boot.npz"
    trainer = StationaryBootstrapOhlcvTrainer({
        "train_data": str(train_csv), "save_model": str(model),
        "block_length_mean": 8, "seed": 99,
    })
    trainer.set_params(train_data=str(train_csv), save_model=str(model),
                       block_length_mean=8, seed=99)
    trainer.train()

    out1 = tmp_path / "s1.csv"
    out2 = tmp_path / "s2.csv"
    for outp in (out1, out2):
        g = StationaryBootstrapOhlcvGenerator({
            "load_model": str(model), "output_file": str(outp),
            "n_samples": 80, "seed": 12345,
        })
        g.set_params(load_model=str(model), output_file=str(outp),
                     n_samples=80, seed=12345)
        g.run_generate()
    a = pd.read_csv(out1).to_numpy()
    b = pd.read_csv(out2).to_numpy()
    np.testing.assert_array_equal(a, b)
