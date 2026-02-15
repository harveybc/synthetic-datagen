"""
Predictive utility evaluator — THE metric from Harvey's MDSc thesis (phase 4).

Tests whether synthetic data actually helps prediction:

1. Train a predictor on real training data → measure MAE on val/test
2. Generate synthetic data → prepend to real training data
3. Train the SAME predictor on augmented data → measure MAE on val/test
4. Report delta: if MAE decreases → synthetic data is useful

Programmatic API:

    from sdg_plugins.evaluator.predictive_evaluator import PredictiveEvaluator
    ev = PredictiveEvaluator()
    ev.configure({
        "real_train": "examples/data/d4.csv",
        "real_val": "examples/data/d5.csv",
        "real_test": "examples/data/d6.csv",
        "synthetic_data": "synthetic.csv",   # or pass DataFrame
    })
    result = ev.evaluate()
    # result["mae_delta_val"]  < 0 means synthetic helped
    # result["mae_delta_test"] < 0 means synthetic helped

Also supports using the external predictor repo via subprocess:

    ev.configure({..., "predictor_dir": "/home/openclaw/predictor"})
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from app.data_processor import load_csv

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Built-in lightweight predictor (LSTM)
# ═══════════════════════════════════════════════════════════════════════════

def _create_windows_xy(
    prices: np.ndarray, window_size: int, horizon: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create (X, y) for prediction.

    X[i] = prices[i : i + window_size]  (input window)
    y[i] = prices[i + window_size + horizon - 1]  (target: future price)

    Returns X shape (N, window_size, 1), y shape (N,).
    """
    n = len(prices)
    X, y = [], []
    for i in range(n - window_size - horizon + 1):
        X.append(prices[i : i + window_size])
        y.append(prices[i + window_size + horizon - 1])
    X = np.array(X, dtype=np.float32).reshape(-1, window_size, 1)
    y = np.array(y, dtype=np.float32)
    return X, y


def _build_predictor(window_size: int) -> keras.Model:
    """Build a simple LSTM predictor for evaluation purposes."""
    inp = keras.Input(shape=(window_size, 1))
    x = layers.LSTM(32, return_sequences=False)(inp)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out, name="eval_predictor")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def _train_and_evaluate(
    train_prices: np.ndarray,
    val_prices: np.ndarray,
    test_prices: np.ndarray,
    window_size: int = 144,
    horizon: int = 1,
    epochs: int = 50,
    batch_size: int = 64,
    verbose: int = 0,
) -> Dict[str, float]:
    """
    Train an LSTM predictor on train_prices, evaluate on val/test.

    Returns dict with mae_train, mae_val, mae_test.
    """
    X_train, y_train = _create_windows_xy(train_prices, window_size, horizon)
    X_val, y_val = _create_windows_xy(val_prices, window_size, horizon)
    X_test, y_test = _create_windows_xy(test_prices, window_size, horizon)

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError(
            f"Not enough data for windows. "
            f"train={len(train_prices)}, val={len(val_prices)}, test={len(test_prices)}, "
            f"window_size={window_size}, horizon={horizon}"
        )

    model = _build_predictor(window_size)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
        ],
    )

    train_mae = float(model.evaluate(X_train, y_train, verbose=0)[1])
    val_mae = float(model.evaluate(X_val, y_val, verbose=0)[1])
    test_mae = float(model.evaluate(X_test, y_test, verbose=0)[1])

    # Clean up to free GPU memory
    del model
    keras.backend.clear_session()

    return {"mae_train": train_mae, "mae_val": val_mae, "mae_test": test_mae}


# ═══════════════════════════════════════════════════════════════════════════
# Plugin
# ═══════════════════════════════════════════════════════════════════════════

class PredictiveEvaluator:
    """
    Plugin: predictive utility evaluation (thesis phase 4 methodology).

    Discoverable via entry_points as ``sdg.evaluator → predictive_evaluator``.
    """

    plugin_params: Dict[str, Any] = {}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg: Dict[str, Any] = {
            "window_size": 144,
            "horizon": 1,
            "eval_epochs": 50,
            "eval_batch_size": 64,
        }
        if config:
            self.cfg.update(config)

    def configure(self, config: Dict[str, Any]) -> None:
        self.cfg.update(config)

    def set_params(self, **kw) -> None:
        self.cfg.update(kw)

    # ── Main evaluation ─────────────────────────────────────────────────

    def evaluate(
        self,
        synthetic: pd.DataFrame | str | None = None,
        real_train: pd.DataFrame | str | None = None,
        real_val: pd.DataFrame | str | None = None,
        real_test: pd.DataFrame | str | None = None,
    ) -> Dict[str, Any]:
        """
        Run the full predictive utility evaluation.

        Steps:
            1. Train predictor on real_train only → baseline MAE on val, test
            2. Prepend synthetic to real_train → train predictor → augmented MAE
            3. Compute deltas (negative = improvement)

        Also runs distribution metrics as secondary.

        Returns a dict with all metrics and the verdict.
        """
        cfg = self.cfg

        # Resolve inputs
        df_syn = self._resolve(synthetic, "synthetic_data")
        df_train = self._resolve(real_train, "real_train")
        df_val = self._resolve(real_val, "real_val")
        df_test = self._resolve(real_test, "real_test")

        p_train = df_train["typical_price"].values.astype(np.float64)
        p_val = df_val["typical_price"].values.astype(np.float64)
        p_test = df_test["typical_price"].values.astype(np.float64)
        p_syn = df_syn["typical_price"].values.astype(np.float64)

        ws = cfg.get("window_size", 144)
        hz = cfg.get("horizon", 1)
        ep = cfg.get("eval_epochs", 50)
        bs = cfg.get("eval_batch_size", 64)

        # Check if we should use external predictor
        predictor_dir = cfg.get("predictor_dir")
        if predictor_dir and os.path.isdir(predictor_dir):
            log.info("Using external predictor for evaluation")
            return self._evaluate_external(
                p_train, p_syn, p_val, p_test, predictor_dir
            )

        # ── Step 1: Baseline (real only) ────────────────────────────────
        log.info("Step 1: Training predictor on real data only …")
        baseline = _train_and_evaluate(p_train, p_val, p_test, ws, hz, ep, bs)
        log.info(
            f"  Baseline MAE — train: {baseline['mae_train']:.6f}, "
            f"val: {baseline['mae_val']:.6f}, test: {baseline['mae_test']:.6f}"
        )

        # ── Step 2: Augmented (synthetic prepended to real) ─────────────
        augmented_train = np.concatenate([p_syn, p_train])
        log.info(
            f"Step 2: Training predictor on augmented data "
            f"({len(p_syn)} synthetic + {len(p_train)} real = {len(augmented_train)}) …"
        )
        augmented = _train_and_evaluate(
            augmented_train, p_val, p_test, ws, hz, ep, bs
        )
        log.info(
            f"  Augmented MAE — train: {augmented['mae_train']:.6f}, "
            f"val: {augmented['mae_val']:.6f}, test: {augmented['mae_test']:.6f}"
        )

        # ── Step 3: Compute deltas ──────────────────────────────────────
        delta_val = augmented["mae_val"] - baseline["mae_val"]
        delta_test = augmented["mae_test"] - baseline["mae_test"]

        result: Dict[str, Any] = {
            # Baseline
            "baseline_mae_train": baseline["mae_train"],
            "baseline_mae_val": baseline["mae_val"],
            "baseline_mae_test": baseline["mae_test"],
            # Augmented
            "augmented_mae_train": augmented["mae_train"],
            "augmented_mae_val": augmented["mae_val"],
            "augmented_mae_test": augmented["mae_test"],
            # Deltas (negative = improvement)
            "mae_delta_val": delta_val,
            "mae_delta_test": delta_test,
            "mae_pct_change_val": delta_val / (baseline["mae_val"] + 1e-12) * 100,
            "mae_pct_change_test": delta_test / (baseline["mae_test"] + 1e-12) * 100,
            # Training details
            "n_real_train": len(p_train),
            "n_synthetic": len(p_syn),
            "n_augmented_train": len(augmented_train),
            "n_val": len(p_val),
            "n_test": len(p_test),
            "window_size": ws,
            "horizon": hz,
            # Verdict
            "synthetic_helps_val": delta_val < 0,
            "synthetic_helps_test": delta_test < 0,
        }

        # Secondary: distribution metrics
        try:
            from sdg_plugins.evaluator.distribution_evaluator import DistributionEvaluator
            dist_ev = DistributionEvaluator()
            dist_metrics = dist_ev.evaluate_arrays(p_syn, p_train)
            for k, v in dist_metrics.items():
                result[f"dist_{k}"] = v
        except Exception as e:
            log.warning(f"Distribution metrics failed: {e}")

        # Log verdict
        log.info("═" * 60)
        log.info(f"  MAE delta (val):  {delta_val:+.6f}  ({result['mae_pct_change_val']:+.2f}%)")
        log.info(f"  MAE delta (test): {delta_test:+.6f}  ({result['mae_pct_change_test']:+.2f}%)")
        log.info(
            f"  VERDICT: synthetic {'HELPS' if delta_test < 0 else 'HURTS'} "
            f"prediction on test set"
        )
        log.info("═" * 60)

        return result

    # ── External predictor (subprocess) ─────────────────────────────────

    def _evaluate_external(
        self,
        p_train: np.ndarray,
        p_syn: np.ndarray,
        p_val: np.ndarray,
        p_test: np.ndarray,
        predictor_dir: str,
    ) -> Dict[str, Any]:
        """
        Run the actual predictor repo as a subprocess for both baseline
        and augmented runs, then compare results.

        Expects the predictor to accept:
            python -m app.main --x_train_file <csv> --x_validation_file <csv>
                --x_test_file <csv> --save_config <json>
        And output metrics in the saved config or results file.
        """
        with tempfile.TemporaryDirectory() as td:
            # Write CSVs
            val_path = os.path.join(td, "val.csv")
            test_path = os.path.join(td, "test.csv")
            train_real_path = os.path.join(td, "train_real.csv")
            train_aug_path = os.path.join(td, "train_augmented.csv")

            self._write_price_csv(p_train, train_real_path, "2012-01-01")
            self._write_price_csv(
                np.concatenate([p_syn, p_train]), train_aug_path, "2010-01-01"
            )
            self._write_price_csv(p_val, val_path, "2018-01-01")
            self._write_price_csv(p_test, test_path, "2019-01-01")

            # Run baseline
            log.info("Running external predictor: baseline …")
            baseline_results = os.path.join(td, "baseline_config.json")
            self._run_predictor(
                predictor_dir, train_real_path, val_path, test_path,
                baseline_results, td, "baseline"
            )

            # Run augmented
            log.info("Running external predictor: augmented …")
            augmented_results = os.path.join(td, "augmented_config.json")
            self._run_predictor(
                predictor_dir, train_aug_path, val_path, test_path,
                augmented_results, td, "augmented"
            )

            # Parse results
            baseline = self._parse_predictor_results(baseline_results)
            augmented = self._parse_predictor_results(augmented_results)

        delta_val = augmented.get("val_mae", 0) - baseline.get("val_mae", 0)
        delta_test = augmented.get("test_mae", 0) - baseline.get("test_mae", 0)

        result = {
            "baseline_mae_val": baseline.get("val_mae"),
            "baseline_mae_test": baseline.get("test_mae"),
            "augmented_mae_val": augmented.get("val_mae"),
            "augmented_mae_test": augmented.get("test_mae"),
            "mae_delta_val": delta_val,
            "mae_delta_test": delta_test,
            "synthetic_helps_val": delta_val < 0,
            "synthetic_helps_test": delta_test < 0,
            "predictor": "external",
            "predictor_dir": predictor_dir,
            "baseline_raw": baseline,
            "augmented_raw": augmented,
        }

        log.info("═" * 60)
        log.info(f"  [EXTERNAL] MAE delta (val):  {delta_val:+.6f}")
        log.info(f"  [EXTERNAL] MAE delta (test): {delta_test:+.6f}")
        log.info(
            f"  VERDICT: synthetic {'HELPS' if delta_test < 0 else 'HURTS'} "
            f"prediction on test set"
        )
        log.info("═" * 60)
        return result

    def _run_predictor(
        self, predictor_dir, train_csv, val_csv, test_csv,
        save_config, workdir, label
    ):
        """Run the predictor subprocess."""
        cmd = [
            "python3", "-m", "app.main",
            "--x_train_file", train_csv,
            "--x_validation_file", val_csv,
            "--x_test_file", test_csv,
            "--save_config", save_config,
            "--save_model", os.path.join(workdir, f"{label}_model.keras"),
            "--output_file", os.path.join(workdir, f"{label}_predictions.csv"),
            "--results_file", os.path.join(workdir, f"{label}_results.csv"),
            "--epochs", str(self.cfg.get("predictor_epochs", 100)),
            "--headers", "True",
            "--plugin", self.cfg.get("predictor_plugin", "lstm"),
        ]
        result = subprocess.run(
            cmd, cwd=predictor_dir, capture_output=True, text=True, timeout=3600
        )
        if result.returncode != 0:
            log.error(f"Predictor [{label}] failed:\n{result.stderr[-2000:]}")
            raise RuntimeError(f"External predictor [{label}] exited with code {result.returncode}")
        log.info(f"Predictor [{label}] completed successfully")

    @staticmethod
    def _parse_predictor_results(config_path: str) -> Dict[str, Any]:
        """Parse metrics from the predictor's saved config/results."""
        if not os.path.exists(config_path):
            log.warning(f"Results file not found: {config_path}")
            return {}
        with open(config_path) as f:
            data = json.load(f)
        # The predictor saves metrics in the config output
        return {
            "val_mae": data.get("validation_mae", data.get("val_mae")),
            "test_mae": data.get("test_mae"),
            "val_r2": data.get("validation_r2", data.get("val_r2")),
            "test_r2": data.get("test_r2"),
            "raw": data,
        }

    @staticmethod
    def _write_price_csv(
        prices: np.ndarray, path: str, start_date: str = "2012-01-01"
    ):
        """Write a typical_price CSV with DATE_TIME and CLOSE columns.

        The predictor expects a 'CLOSE' column, and our datasets have
        typical_price ≈ (H+L+C)/3.  For evaluation we use typical_price
        directly as CLOSE since we only have one feature.
        """
        dates = pd.date_range(start_date, periods=len(prices), freq="4h")
        df = pd.DataFrame({
            "DATE_TIME": dates,
            "typical_price": prices,
            "CLOSE": prices,  # predictor expects CLOSE
        })
        df.to_csv(path, index=False)

    def _resolve(self, arg, cfg_key: str) -> pd.DataFrame:
        if arg is None:
            arg = self.cfg.get(cfg_key)
        if isinstance(arg, str):
            return load_csv(arg)
        if isinstance(arg, pd.DataFrame):
            return arg
        raise ValueError(f"Provide a DataFrame, CSV path, or set cfg['{cfg_key}']")
