#!/usr/bin/env python3
"""Measure verification tolerance for DOIN predictor domain — v2.

CORRECTED APPROACH: Evaluators train on real data (d4) only, then
evaluate the trained model on different synthetic test sets.

Same model weights + different synthetic test data = small fitness variance.
This is what DOIN evaluators actually do: verify that the optimizer's model
generalizes to unseen synthetic data.

Usage:
    PREDICTOR_QUIET=1 python measure_tolerance_v2.py --n_seeds 8
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Quiet mode
os.environ["PREDICTOR_QUIET"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import builtins
_orig_print = builtins.print
def _quiet_print(*args, **kwargs):
    msg = str(args[0]) if args else ""
    upper = msg.upper()
    if any(k in upper for k in ["ERROR", "FATAL", "TOLERANCE", "SEED", "RESULT", "RECOMMEND", "TRAIN", "EVAL"]):
        _orig_print(*args, **kwargs)
builtins.print = _quiet_print

PREDICTOR_ROOT = Path("/home/openclaw/predictor")
SDG_ROOT = Path("/home/openclaw/.openclaw/workspace/synthetic-datagen")
DOIN_CORE = Path("/home/openclaw/.openclaw/workspace/doin-core/src")
DOIN_PLUGINS = Path("/home/openclaw/.openclaw/workspace/doin-plugins/src")

for p in [str(PREDICTOR_ROOT), str(SDG_ROOT), str(DOIN_CORE), str(DOIN_PLUGINS)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# GPU setup
NVIDIA_LIBS = Path.home() / ".local/lib/python3.12/site-packages/nvidia"
if NVIDIA_LIBS.exists():
    lib_dirs = [str(p / "lib") for p in NVIDIA_LIBS.iterdir() if (p / "lib").is_dir()]
    os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

os.chdir(str(PREDICTOR_ROOT))


def load_config(config_path):
    from app.config import DEFAULT_VALUES
    config = DEFAULT_VALUES.copy()
    with open(config_path) as f:
        config.update(json.load(f))
    config["disable_postfit_uncertainty"] = True
    config["mc_samples"] = 1
    config["quiet"] = True
    return config


def preprocess(config):
    from app.plugin_loader import load_plugin
    pre_cls, _ = load_plugin("preprocessor.plugins", config.get("preprocessor_plugin", "default_preprocessor"))
    preprocessor = pre_cls()
    preprocessor.set_params(**config)
    tgt_cls, _ = load_plugin("target.plugins", config.get("target_plugin", "default_target"))
    target = tgt_cls()
    target.set_params(**config)
    datasets = preprocessor.run_preprocessing(target, config)
    if isinstance(datasets, tuple):
        datasets = datasets[0]
    return datasets


def generate_synthetic_test(seed, n_samples, block_size, config):
    """Generate synthetic test windows from block bootstrap data."""
    from doin_plugins.predictor.synthetic import PredictorSyntheticData

    synth = PredictorSyntheticData()
    synth.configure({
        "predictor_root": str(PREDICTOR_ROOT),
        "n_samples": n_samples,
        "block_size": block_size,
    })
    result = synth.generate(seed=seed)
    synthetic_prices = result["synthetic_df"]["typical_price"].values

    # Normalize with same params as real data
    norm_path = config.get("use_normalization_json")
    if norm_path and not Path(norm_path).is_absolute():
        norm_path = str(PREDICTOR_ROOT / norm_path)
    with open(norm_path) as f:
        norm_config = json.load(f)
    mean_val = norm_config["typical_price"]["mean"]
    std_val = norm_config["typical_price"]["std"]
    norm_synthetic = (synthetic_prices - mean_val) / std_val

    # Build windows
    window_size = config.get("window_size", 48)
    predicted_horizons = config.get("predicted_horizons", [1])
    max_horizon = max(predicted_horizons)

    x_windows = []
    y_windows = {f"output_horizon_{h}": [] for h in predicted_horizons}

    for i in range(len(norm_synthetic) - window_size - max_horizon + 1):
        x_windows.append(norm_synthetic[i:i + window_size].reshape(-1, 1))
        for h in predicted_horizons:
            y_windows[f"output_horizon_{h}"].append(
                norm_synthetic[i + window_size + h - 1].reshape(1)
            )

    x_test = np.array(x_windows, dtype=np.float32)
    y_test = {k: np.array(v, dtype=np.float32) for k, v in y_windows.items()}

    return x_test, y_test


def train_model(datasets, config):
    """Train model ONCE on real data, return the trained predictor."""
    import gc
    import tensorflow as tf
    tf.keras.backend.clear_session()
    gc.collect()

    from app.plugin_loader import load_plugin
    pred_cls, _ = load_plugin("predictor.plugins", config.get("predictor_plugin", "default_predictor"))
    predictor = pred_cls(config)
    predictor.set_params(**config)

    x_train = datasets["x_train"]
    y_train = datasets["y_train"]
    x_val = datasets["x_val"]
    y_val = datasets["y_val"]

    if isinstance(y_train, dict):
        y_train = {k: np.asarray(v).reshape(-1, 1).astype(np.float32) for k, v in y_train.items()}
        y_val = {k: np.asarray(v).reshape(-1, 1).astype(np.float32) for k, v in y_val.items()}

    window_size = config.get("window_size")
    input_shape = (window_size, x_train.shape[2]) if len(x_train.shape) == 3 else (x_train.shape[1],)

    predictor.build_model(input_shape=input_shape, x_train=x_train, config=config)
    predictor.train(
        x_train, y_train,
        epochs=config.get("epochs", 2000),
        batch_size=config.get("batch_size", 32),
        threshold_error=config.get("threshold_error", 0.001),
        x_val=x_val, y_val=y_val, config=config,
    )

    return predictor


def evaluate_on_synthetic(predictor, x_test, y_test, config):
    """Evaluate trained model on synthetic test set. Returns MAE."""
    from pipeline_plugins.stl_norm import denormalize

    predicted_horizons = config.get("predicted_horizons", [1])
    max_horizon = max(predicted_horizons)
    max_h_idx = predicted_horizons.index(max_horizon)

    # Predict via keras model
    predictions = predictor.model.predict(x_test, verbose=0)
    if isinstance(predictions, (list, tuple)):
        val_preds_h = np.asarray(predictions[max_h_idx]).flatten()
    else:
        val_preds_h = np.asarray(predictions).flatten()

    # Get true values
    y_key = f"output_horizon_{max_horizon}"
    if isinstance(y_test, dict):
        y_true = np.asarray(y_test[y_key]).flatten()
    else:
        y_true = np.asarray(y_test).flatten()

    n = min(len(val_preds_h), len(y_true))
    val_preds_h = val_preds_h[:n]
    y_true = y_true[:n]

    # Denormalize
    real_pred = denormalize(val_preds_h, config)
    real_true = denormalize(y_true, config)
    mae = float(np.mean(np.abs(real_pred - real_true)))

    return mae


def main():
    parser = argparse.ArgumentParser(description="Measure DOIN verification tolerance v2")
    parser.add_argument("--config", default="examples/config/phase_1_daily/optimization/phase_1_mimo_1d_optimization_config.json")
    parser.add_argument("--n_seeds", type=int, default=8)
    parser.add_argument("--seed_start", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=1560)
    parser.add_argument("--block_size", type=int, default=30)
    parser.add_argument("--output", default="examples/results/tolerance_v2.json")
    args = parser.parse_args()

    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = str(PREDICTOR_ROOT / config_path)

    _orig_print(f"TOLERANCE v2: Loading config from {config_path}")
    config = load_config(config_path)

    _orig_print(f"TOLERANCE v2: Preprocessing data...")
    datasets = preprocess(config)

    _orig_print(f"TOLERANCE v2: Training model ONCE on real data...")
    t0 = time.time()
    predictor = train_model(datasets, config)
    train_time = time.time() - t0
    _orig_print(f"TRAINING complete in {train_time:.0f}s")

    # Also evaluate on real validation set for baseline
    from pipeline_plugins.stl_norm import denormalize
    predicted_horizons = config.get("predicted_horizons", [1])
    max_horizon = max(predicted_horizons)
    max_h_idx = predicted_horizons.index(max_horizon)

    real_preds = predictor.model.predict(datasets["x_val"], verbose=0)
    if isinstance(real_preds, (list, tuple)):
        real_preds_h = np.asarray(real_preds[max_h_idx]).flatten()
    else:
        real_preds_h = np.asarray(real_preds).flatten()

    if isinstance(datasets["y_val"], dict):
        real_y = np.asarray(datasets["y_val"][f"output_horizon_{max_horizon}"]).flatten()
    else:
        real_y = np.asarray(datasets["y_val"]).flatten()

    n = min(len(real_preds_h), len(real_y))
    real_mae = float(np.mean(np.abs(
        denormalize(real_preds_h[:n], config) - denormalize(real_y[:n], config)
    )))
    _orig_print(f"EVAL on real val: MAE={real_mae:.6f}")

    # Now evaluate on N different synthetic test sets
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    _orig_print(f"EVAL on {len(seeds)} synthetic test sets (seeds {seeds[0]}-{seeds[-1]})...")

    results = []
    for i, seed in enumerate(seeds):
        t1 = time.time()
        x_syn, y_syn = generate_synthetic_test(seed, args.n_samples, args.block_size, config)
        syn_mae = evaluate_on_synthetic(predictor, x_syn, y_syn, config)
        elapsed = time.time() - t1
        results.append({"seed": seed, "syn_mae": syn_mae, "elapsed_s": round(elapsed, 1)})
        _orig_print(f"SEED {seed} ({i+1}/{len(seeds)}): syn_MAE={syn_mae:.6f} ({elapsed:.1f}s)")

    # Analysis
    syn_maes = [r["syn_mae"] for r in results]
    mean_m = np.mean(syn_maes)
    std_m = np.std(syn_maes)
    max_dev = max(abs(m - mean_m) / mean_m for m in syn_maes) if mean_m > 0 else 0

    # Compare to real MAE
    real_vs_syn_gap = abs(real_mae - mean_m) / real_mae if real_mae > 0 else 0

    recommended = min(max_dev * 2.0, 0.50)

    summary = {
        "approach": "eval_on_synthetic (train on real only)",
        "n_seeds": args.n_seeds,
        "n_samples": args.n_samples,
        "block_size": args.block_size,
        "train_time_s": round(train_time, 1),
        "real_val_mae": round(real_mae, 8),
        "syn_mae_mean": round(mean_m, 8),
        "syn_mae_std": round(std_m, 8),
        "syn_mae_min": round(min(syn_maes), 8),
        "syn_mae_max": round(max(syn_maes), 8),
        "max_relative_deviation_between_seeds": round(max_dev, 6),
        "real_vs_syn_gap": round(real_vs_syn_gap, 6),
        "recommended_tolerance_margin": round(recommended, 4),
        "individual_results": results,
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = SDG_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    _orig_print(f"\n{'='*60}")
    _orig_print(f"TOLERANCE v2 RESULTS ({args.n_seeds} seeds)")
    _orig_print(f"{'='*60}")
    _orig_print(f"  Real val MAE:    {real_mae:.6f}")
    _orig_print(f"  Synthetic MAEs:  mean={mean_m:.6f}, std={std_m:.6f}")
    _orig_print(f"  Synthetic range: {min(syn_maes):.6f} - {max(syn_maes):.6f}")
    _orig_print(f"  Max deviation between seeds: {max_dev:.4%}")
    _orig_print(f"  Real vs synthetic gap: {real_vs_syn_gap:.4%}")
    _orig_print(f"  RECOMMENDED tolerance_margin: {recommended:.4f} ({recommended:.1%})")
    _orig_print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
