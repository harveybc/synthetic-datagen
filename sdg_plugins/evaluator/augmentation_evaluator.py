"""
Augmentation evaluator plugin.

The REAL test for synthetic data quality (from Harvey's MDSc phase 4):

1. Train predictor on real data only (d4) → measure MAE on d5/d6
2. Save baseline results (computed ONCE, cached to file)
3. Generate synthetic data, append to d4
4. Train predictor on augmented data (d4 + synthetic) → measure MAE on d5/d6
5. Compare: improvement = baseline_mae - augmented_mae

If positive → synthetic data helps.  If negative → it hurts.

RULES:
- Generator trained ONLY on d1-d3 (data before prediction period)
- Baseline computed ONCE, saved to JSON, never repeated
- Prediction uses d4 (train), d5 (validation), d6 (test)
- All data is 4h downsampled typical_price from predictor
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from app.data_processor import load_csv

log = logging.getLogger(__name__)


class AugmentationEvaluator:
    """Plugin: evaluates synthetic data by augmentation impact on predictor."""

    plugin_params: Dict[str, Any] = {}

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def set_params(self, **kw):
        self.cfg.update(kw)

    def evaluate(self) -> Dict[str, Any]:
        """Run the full augmentation evaluation pipeline.

        Config keys:
            predictor_root: Path to predictor repo
            load_config: Path to predictor JSON config
            d4_file: Path to real training data (d4)
            d5_file: Path to validation data (d5)
            d6_file: Path to test data (d6)
            synthetic_data: Path to synthetic CSV to append
            baseline_file: Path to save/load baseline results JSON
            predictor_plugin: Name of predictor plugin (e.g. "ann", "mimo")
            epochs: Training epochs (default from predictor config)
            batch_size: Batch size (default from predictor config)
        """
        cfg = self.cfg

        baseline_file = cfg.get("baseline_file", "baseline_results.json")

        # Step 1: Get or compute baseline (real data only)
        baseline = self._get_or_compute_baseline(baseline_file)

        # Step 2: Train with augmented data (d4 + synthetic)
        augmented = self._train_augmented()

        # Step 3: Compare
        metrics = {
            "baseline_val_mae": baseline["val_mae"],
            "baseline_test_mae": baseline["test_mae"],
            "augmented_val_mae": augmented["val_mae"],
            "augmented_test_mae": augmented["test_mae"],
            "val_improvement": baseline["val_mae"] - augmented["val_mae"],
            "test_improvement": baseline["test_mae"] - augmented["test_mae"],
            "val_improvement_pct": (
                (baseline["val_mae"] - augmented["val_mae"]) / baseline["val_mae"] * 100
                if baseline["val_mae"] > 0 else 0.0
            ),
            "test_improvement_pct": (
                (baseline["test_mae"] - augmented["test_mae"]) / baseline["test_mae"] * 100
                if baseline["test_mae"] > 0 else 0.0
            ),
        }

        verdict = "GOOD" if metrics["val_improvement"] > 0 else "BAD"
        metrics["verdict"] = verdict

        log.info(f"=== Augmentation Evaluation ===")
        log.info(f"Baseline val MAE:  {baseline['val_mae']:.6f}")
        log.info(f"Augmented val MAE: {augmented['val_mae']:.6f}")
        log.info(f"Val improvement:   {metrics['val_improvement']:.6f} ({metrics['val_improvement_pct']:.2f}%)")
        log.info(f"Baseline test MAE:  {baseline['test_mae']:.6f}")
        log.info(f"Augmented test MAE: {augmented['test_mae']:.6f}")
        log.info(f"Test improvement:   {metrics['test_improvement']:.6f} ({metrics['test_improvement_pct']:.2f}%)")
        log.info(f"Verdict: {verdict}")

        # Save results
        metrics_file = cfg.get("metrics_file", "augmentation_metrics.json")
        os.makedirs(os.path.dirname(metrics_file) or ".", exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def _get_or_compute_baseline(self, baseline_file: str) -> Dict[str, float]:
        """Load cached baseline or compute it once."""
        if os.path.exists(baseline_file):
            log.info(f"Loading cached baseline from {baseline_file}")
            with open(baseline_file) as f:
                return json.load(f)

        log.info("Computing baseline (real data only) — this runs ONCE")
        baseline = self._train_predictor(augment_data=None)

        os.makedirs(os.path.dirname(baseline_file) or ".", exist_ok=True)
        with open(baseline_file, "w") as f:
            json.dump(baseline, f, indent=2)
        log.info(f"Baseline saved to {baseline_file}")

        return baseline

    def _train_augmented(self) -> Dict[str, float]:
        """Train predictor with d4 + synthetic data."""
        synthetic_path = self.cfg.get("synthetic_data")
        if not synthetic_path:
            raise ValueError("synthetic_data path required for augmentation evaluation")

        synthetic_df = load_csv(synthetic_path)
        return self._train_predictor(augment_data=synthetic_df)

    def _train_predictor(self, augment_data: pd.DataFrame | None) -> Dict[str, float]:
        """Train predictor and return MAE metrics.

        Args:
            augment_data: If provided, append to training data (d4).
        """
        import gc
        import builtins

        # Keep reference to real print for our own output
        _orig_print = builtins.print

        # Suppress noisy predictor prints when quiet mode is active
        if os.environ.get('PREDICTOR_QUIET', '0') == '1':
            def _quiet_print(*args, **kwargs):
                if args:
                    msg = str(args[0]).upper()
                    if any(k in msg for k in ['ERROR', 'WARN', 'EXCEPTION', 'FATAL']):
                        _orig_print(*args, **kwargs)
            builtins.print = _quiet_print

        cfg = self.cfg
        predictor_root = Path(cfg.get("predictor_root", "/home/openclaw/predictor")).resolve()

        # Force predictor's packages to load from predictor_root, not stale site-packages
        root_str = str(predictor_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        # Pre-import predictor plugin packages so entry_points find the right ones
        import importlib
        for pkg_name in ["preprocessor_plugins", "predictor_plugins", "target_plugins",
                         "pipeline_plugins", "optimizer_plugins"]:
            if pkg_name in sys.modules:
                del sys.modules[pkg_name]
                # Also remove submodules
                for key in list(sys.modules.keys()):
                    if key.startswith(pkg_name + "."):
                        del sys.modules[key]

        try:
            import tensorflow as tf
        except ImportError:
            tf = None

        # Clean slate
        if tf:
            tf.keras.backend.clear_session()
        gc.collect()

        # Load predictor config from its installed package
        import importlib.util as _ilu
        _cfg_spec = _ilu.spec_from_file_location(
            "predictor_config",
            str(predictor_root / "app" / "config.py"),
        )
        _cfg_mod = _ilu.module_from_spec(_cfg_spec)
        _cfg_spec.loader.exec_module(_cfg_mod)
        pred_config = _cfg_mod.DEFAULT_VALUES.copy()

        config_file = cfg.get("predictor_config") or cfg.get("load_config")
        if config_file:
            config_path = Path(config_file)
            if not config_path.is_absolute():
                config_path = predictor_root / config_path
            if config_path.exists():
                with open(config_path) as f:
                    pred_config.update(json.load(f))

        # Override with any sdg-level settings
        for key in ["predictor_plugin", "epochs", "batch_size", "window_size"]:
            if key in cfg:
                pred_config[key] = cfg[key]

        pred_config["disable_postfit_uncertainty"] = True
        pred_config["mc_samples"] = 1
        pred_config["quiet"] = True

        # Resolve relative file paths in pred_config against predictor_root
        # so the preprocessor can find the data files and normalization JSON
        _file_keys = [
            "x_train_file", "y_train_file",
            "x_validation_file", "y_validation_file",
            "x_test_file", "y_test_file",
            "use_normalization_json",
        ]
        for fk in _file_keys:
            val = pred_config.get(fk)
            if val and isinstance(val, str) and not os.path.isabs(val):
                resolved = str(predictor_root / val)
                if os.path.exists(resolved):
                    pred_config[fk] = resolved
                    log.debug(f"Resolved {fk}: {val} -> {resolved}")

        # If augmenting, we need to modify the training data files.
        # Strategy: load the NORMALIZED training CSV from the predictor config,
        # normalize the synthetic data with the same z-score params, append,
        # and point the config at the augmented file.
        if augment_data is not None:
            train_path = pred_config.get("x_train_file")
            if not train_path or not os.path.exists(train_path):
                raise ValueError(
                    f"x_train_file required for augmentation evaluation "
                    f"(got: {train_path})"
                )

            # Load real normalized training data (as preprocessor would)
            real_train_df = pd.read_csv(train_path, parse_dates=True, index_col=0)
            log.info(
                f"Loaded real training data: {real_train_df.shape} from {train_path}"
            )

            # Normalize synthetic data if a normalization JSON is available
            norm_json_path = pred_config.get("use_normalization_json")
            if norm_json_path and os.path.exists(norm_json_path):
                import json as _json
                with open(norm_json_path) as _nf:
                    norm_params = _json.load(_nf)
                # Normalize each column that exists in both synthetic and norm config
                synth_normalized = augment_data.copy()
                for col in synth_normalized.columns:
                    if col == "DATE_TIME":
                        continue
                    if col in norm_params:
                        mean = norm_params[col]["mean"]
                        std = norm_params[col]["std"]
                        synth_normalized[col] = (
                            (synth_normalized[col] - mean) / std
                        )
                        log.info(
                            f"Normalized synthetic '{col}': "
                            f"mean={mean:.6f}, std={std:.6f}"
                        )
                    else:
                        log.warning(
                            f"Column '{col}' not in normalization config — "
                            f"left as-is"
                        )
            else:
                log.warning(
                    "No normalization JSON found — appending synthetic data "
                    "as-is (assuming already normalized or no normalization needed)"
                )
                synth_normalized = augment_data.copy()

            # Convert synthetic to same format as real (DATE_TIME as index)
            if "DATE_TIME" in synth_normalized.columns:
                synth_normalized = synth_normalized.set_index("DATE_TIME")

            # Append and sort
            augmented_df = pd.concat(
                [real_train_df, synth_normalized], ignore_index=False
            )
            augmented_df.sort_index(inplace=True)
            augmented_df = augmented_df[~augmented_df.index.duplicated(keep="first")]

            # Save to temp file (same format as normalized CSVs: index=DATE_TIME)
            tmp_path = os.path.join(
                os.path.dirname(cfg.get("metrics_file", ".")) or ".",
                "_augmented_train.csv"
            )
            augmented_df.to_csv(tmp_path)  # index=True → DATE_TIME as first col

            # Point predictor at the augmented file and expand max_steps_train
            pred_config["x_train_file"] = tmp_path
            pred_config["y_train_file"] = tmp_path
            pred_config["max_steps_train"] = len(augmented_df)
            log.info(
                f"Augmented training data: {len(real_train_df)} real + "
                f"{len(synth_normalized)} synthetic = "
                f"{len(augmented_df)} total (after dedup)"
            )
            log.info(
                f"max_steps_train bumped to {len(augmented_df)} "
                f"to include synthetic data"
            )

        # Load predictor plugins via predictor's own plugin_loader
        _pl_spec = _ilu.spec_from_file_location(
            "predictor_plugin_loader",
            str(predictor_root / "app" / "plugin_loader.py"),
        )
        _pl_mod = _ilu.module_from_spec(_pl_spec)
        _pl_spec.loader.exec_module(_pl_mod)
        pred_load_plugin = _pl_mod.load_plugin

        # Ensure required predictor config keys
        if not pred_config.get("predicted_horizons"):
            pred_config["predicted_horizons"] = [1]
        pred_config["plotted_horizon"] = pred_config["predicted_horizons"][0]
        pred_config.setdefault("time_horizon", 1)

        pred_name = pred_config.get("predictor_plugin", "default_predictor")
        pred_cls, _ = pred_load_plugin("predictor.plugins", pred_name)

        pre_name = pred_config.get("preprocessor_plugin", "default_preprocessor")
        pre_cls, _ = pred_load_plugin("preprocessor.plugins", pre_name)

        tgt_name = pred_config.get("target_plugin", "default_target")
        tgt_cls, _ = pred_load_plugin("target.plugins", tgt_name)

        # Merge plugin default params into config before instantiation
        for cls in [pred_cls, pre_cls, tgt_cls]:
            if hasattr(cls, 'plugin_params'):
                for k, v in cls.plugin_params.items():
                    if k not in pred_config:
                        pred_config[k] = v

        predictor = pred_cls(pred_config)
        predictor.set_params(**pred_config)

        preprocessor = pre_cls()
        preprocessor.set_params(**pred_config)

        target = tgt_cls()
        target.set_params(**pred_config)

        # Preprocess
        datasets = preprocessor.run_preprocessing(target, pred_config)
        if isinstance(datasets, tuple):
            datasets = datasets[0]

        x_train = datasets["x_train"]
        y_train_raw = datasets["y_train"]
        x_val = datasets["x_val"]
        y_val_raw = datasets["y_val"]
        x_test = datasets.get("x_test")
        y_test_raw = datasets.get("y_test")
        baseline_val = datasets.get("baseline_val")
        baseline_test = datasets.get("baseline_test")

        # Convert y from dict/list to the format the model expects
        def _y_to_list_or_array(y):
            """Return a list of arrays (one per horizon) for multi-output models,
            or a single array for single-output models."""
            if y is None:
                return None
            if isinstance(y, dict):
                return [np.asarray(v).astype(np.float32).reshape(-1, 1) for v in y.values()]
            if isinstance(y, list):
                return [np.asarray(a).astype(np.float32).reshape(-1, 1) for a in y]
            return np.asarray(y).astype(np.float32)

        y_train = _y_to_list_or_array(y_train_raw)
        y_val = _y_to_list_or_array(y_val_raw)
        y_test = _y_to_list_or_array(y_test_raw)

        # Build model
        window_size = pred_config.get("window_size")
        if len(x_train.shape) == 3:
            input_shape = (window_size, x_train.shape[2])
        else:
            input_shape = (x_train.shape[1],)

        predictor.build_model(input_shape=input_shape, x_train=x_train, config=pred_config)

        # Train directly using model.fit (bypasses base.py dict requirement)
        import tensorflow as tf
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=pred_config.get("early_patience", 10),
                restore_best_weights=True,
            )
        ]
        epochs = pred_config.get("epochs", 10)
        batch_size = pred_config.get("batch_size", 32)

        quiet = pred_config.get("quiet", False)
        if quiet:
            # Minimal epoch-end callback: just val_loss + best
            class _QuietEpochLogger(tf.keras.callbacks.Callback):
                def __init__(self):
                    self.best = float('inf')
                def on_epoch_end(self, epoch, logs=None):
                    vl = logs.get('val_loss', 0)
                    if vl < self.best:
                        self.best = vl
                        tag = ' *'
                    else:
                        tag = ''
                    _orig_print(f"  Epoch {epoch+1}/{epochs}: val_loss={vl:.6f}{tag}")
            callbacks.append(_QuietEpochLogger())

        predictor.model.fit(
            x_train, y_train,
            epochs=epochs, batch_size=batch_size,
            validation_data=(x_val, y_val),
            callbacks=callbacks, verbose=0 if quiet else 1,
        )

        # Predictions — handle multi-output (list) or single-output
        def _flatten_predictions(preds, y_true):
            if isinstance(preds, list):
                preds_flat = np.concatenate([np.asarray(p).flatten() for p in preds])
                y_flat = np.concatenate([np.asarray(y).flatten() for y in y_true])
            else:
                preds_flat = np.asarray(preds).flatten()
                y_flat = np.asarray(y_true).flatten()
            return float(np.mean(np.abs(preds_flat - y_flat)))

        val_preds = predictor.model.predict(x_val, batch_size=batch_size, verbose=0)
        val_mae = _flatten_predictions(val_preds, y_val)

        test_mae = float("inf")
        if x_test is not None and y_test is not None:
            test_preds = predictor.model.predict(x_test, batch_size=batch_size, verbose=0)
            test_mae = _flatten_predictions(test_preds, y_test)

        return {"val_mae": val_mae, "test_mae": test_mae}

    def _compute_mae(self, preds, y_true, baseline, config) -> float:
        """Compute MAE, matching predictor conventions."""
        predicted_horizons = config.get("predicted_horizons", [1])
        max_horizon = max(predicted_horizons) if predicted_horizons else 1
        max_h_idx = predicted_horizons.index(max_horizon) if predicted_horizons else 0

        pred_h = np.asarray(preds[max_h_idx]).flatten()

        if isinstance(y_true, dict):
            true_h = np.asarray(y_true[f"output_horizon_{max_horizon}"]).flatten()
        elif isinstance(y_true, list):
            true_h = np.asarray(y_true[max_h_idx]).flatten()
        else:
            true_h = np.asarray(y_true).flatten()

        n = min(len(pred_h), len(true_h))
        pred_h = pred_h[:n]
        true_h = true_h[:n]

        # Try denormalization
        try:
            from pipeline_plugins.stl_norm import denormalize
            real_pred = denormalize(pred_h, config)
            real_true = denormalize(true_h, config)
            return float(np.mean(np.abs(real_pred - real_true)))
        except ImportError:
            return float(np.mean(np.abs(pred_h - true_h)))

    @staticmethod
    def _to_horizon_dict(y, config):
        """Convert y (list or array) to dict format expected by predictor: {output_horizon_N: array}."""
        if y is None:
            return None
        if isinstance(y, dict):
            return y
        horizons = config.get("predicted_horizons", [1])
        # Preprocessor returns list of arrays, one per horizon
        if isinstance(y, list):
            result = {}
            for i, h in enumerate(horizons):
                arr = np.asarray(y[i]).astype(np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                result[f"output_horizon_{h}"] = arr
            return result
        # Single array
        y_arr = np.asarray(y).astype(np.float32)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        if len(horizons) == 1:
            return {f"output_horizon_{horizons[0]}": y_arr}
        result = {}
        for i, h in enumerate(horizons):
            if y_arr.ndim == 2 and y_arr.shape[1] > i:
                result[f"output_horizon_{h}"] = y_arr[:, i:i+1]
            else:
                result[f"output_horizon_{h}"] = y_arr
        return result

    @staticmethod
    def _ensure_2d(y):
        if y is None:
            return None
        if isinstance(y, dict):
            return {k: np.asarray(v).reshape(-1, 1).astype(np.float32) for k, v in y.items()}
        return y
