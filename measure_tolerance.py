#!/usr/bin/env python3
"""Measure verification tolerance for DOIN predictor domain.

Trains the SAME model params N times with DIFFERENT synthetic data seeds
to measure how much fitness varies due to synthetic data alone.

This tells us what tolerance_margin to set in IncentiveConfig so that
honest evaluators using different synthetic seeds can still reach consensus.

Usage:
    PREDICTOR_QUIET=1 python measure_tolerance.py --n_seeds 10

Output:
    - Mean, std, min, max of fitness across seeds
    - Recommended tolerance_margin (as fraction of |mean fitness|)
    - JSON results file
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

# Suppress prints
import builtins
_orig_print = builtins.print
def _quiet_print(*args, **kwargs):
    msg = str(args[0]) if args else ""
    upper = msg.upper()
    if any(k in upper for k in ["ERROR", "FATAL", "EXCEPTION", "TOLERANCE", "SEED", "RESULT", "RECOMMEND"]):
        _orig_print(*args, **kwargs)
builtins.print = _quiet_print

# Setup paths
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


def load_config(config_path: str) -> dict:
    """Load predictor config."""
    from app.config import DEFAULT_VALUES
    config = DEFAULT_VALUES.copy()
    with open(config_path) as f:
        config.update(json.load(f))
    config["disable_postfit_uncertainty"] = True
    config["mc_samples"] = 1
    config["quiet"] = True
    return config


def preprocess(config: dict) -> dict:
    """Run predictor preprocessing to get train/val/test data."""
    from app.plugin_loader import load_plugin

    pre_name = config.get("preprocessor_plugin", "default_preprocessor")
    pre_cls, _ = load_plugin("preprocessor.plugins", pre_name)
    preprocessor = pre_cls()
    preprocessor.set_params(**config)

    tgt_name = config.get("target_plugin", "default_target")
    tgt_cls, _ = load_plugin("target.plugins", tgt_name)
    target = tgt_cls()
    target.set_params(**config)

    datasets = preprocessor.run_preprocessing(target, config)
    if isinstance(datasets, tuple):
        datasets = datasets[0]
    return datasets


def generate_synthetic(seed: int, n_samples: int = 1560, block_size: int = 30) -> np.ndarray:
    """Generate synthetic data using block bootstrap."""
    from doin_plugins.predictor.synthetic import PredictorSyntheticData

    synth = PredictorSyntheticData()
    synth.configure({
        "predictor_root": str(PREDICTOR_ROOT),
        "n_samples": n_samples,
        "block_size": block_size,
    })
    result = synth.generate(seed=seed)
    return result["synthetic_df"]["typical_price"].values


def augment_training_data(datasets: dict, synthetic_prices: np.ndarray, config: dict) -> dict:
    """Append synthetic data to training set (normalized)."""
    import copy

    aug = copy.deepcopy(datasets)
    x_train = aug["x_train"]
    y_train = aug["y_train"]

    # Load normalization params
    norm_path = config.get("use_normalization_json")
    if norm_path and not Path(norm_path).is_absolute():
        norm_path = str(PREDICTOR_ROOT / norm_path)

    with open(norm_path) as f:
        norm_config = json.load(f)

    mean_val = norm_config["typical_price"]["mean"]
    std_val = norm_config["typical_price"]["std"]

    # Normalize synthetic prices
    norm_synthetic = (synthetic_prices - mean_val) / std_val

    # Build windows from synthetic data
    window_size = config.get("window_size", 48)
    predicted_horizons = config.get("predicted_horizons", [1])
    max_horizon = max(predicted_horizons)

    syn_windows_x = []
    syn_windows_y = {f"output_horizon_{h}": [] for h in predicted_horizons}

    for i in range(len(norm_synthetic) - window_size - max_horizon + 1):
        syn_windows_x.append(norm_synthetic[i:i + window_size].reshape(-1, 1))
        for h in predicted_horizons:
            syn_windows_y[f"output_horizon_{h}"].append(
                norm_synthetic[i + window_size + h - 1].reshape(1)
            )

    if not syn_windows_x:
        return aug

    syn_x = np.array(syn_windows_x, dtype=np.float32)

    # Append to training data
    aug["x_train"] = np.concatenate([x_train, syn_x], axis=0)

    if isinstance(y_train, dict):
        new_y = {}
        for k, v in y_train.items():
            syn_y = np.array(syn_windows_y[k], dtype=np.float32)
            # Match dimensions
            if v.ndim == 1:
                syn_y = syn_y.flatten()
            elif v.ndim == 2 and syn_y.ndim == 1:
                syn_y = syn_y.reshape(-1, 1)
            new_y[k] = np.concatenate([v, syn_y], axis=0)
        aug["y_train"] = new_y
    else:
        # Single output
        syn_y = np.array([norm_synthetic[i + window_size] for i in range(len(syn_windows_x))],
                         dtype=np.float32).reshape(-1, 1)
        aug["y_train"] = np.concatenate([y_train, syn_y], axis=0)

    return aug


def train_and_evaluate(datasets: dict, config: dict) -> float:
    """Train model and return fitness (negated MAE delta from naive)."""
    import gc
    import tensorflow as tf

    tf.keras.backend.clear_session()
    gc.collect()

    from app.plugin_loader import load_plugin

    pred_name = config.get("predictor_plugin", "default_predictor")
    pred_cls, _ = load_plugin("predictor.plugins", pred_name)
    predictor = pred_cls(config)
    predictor.set_params(**config)

    x_train = datasets["x_train"]
    y_train = datasets["y_train"]
    x_val = datasets["x_val"]
    y_val = datasets["y_val"]
    baseline_val = datasets.get("baseline_val")

    # Ensure 2D targets
    if isinstance(y_train, dict):
        y_train = {k: np.asarray(v).reshape(-1, 1).astype(np.float32) for k, v in y_train.items()}
        y_val_2d = {k: np.asarray(v).reshape(-1, 1).astype(np.float32) for k, v in y_val.items()}
    else:
        y_val_2d = y_val

    # Build model
    window_size = config.get("window_size")
    input_shape = (window_size, x_train.shape[2]) if len(x_train.shape) == 3 else (x_train.shape[1],)

    predictor.build_model(input_shape=input_shape, x_train=x_train, config=config)

    # Train
    history, train_preds, _, val_preds, _ = predictor.train(
        x_train, y_train,
        epochs=config.get("epochs", 2000),
        batch_size=config.get("batch_size", 32),
        threshold_error=config.get("threshold_error", 0.001),
        x_val=x_val, y_val=y_val_2d, config=config,
    )

    # Compute fitness: denormalized MAE
    from pipeline_plugins.stl_norm import denormalize

    predicted_horizons = config.get("predicted_horizons", [1])
    max_horizon = max(predicted_horizons)
    max_h_idx = predicted_horizons.index(max_horizon)

    val_preds_h = np.asarray(val_preds[max_h_idx]).flatten()

    if isinstance(y_val, dict):
        y_true = np.asarray(y_val[f"output_horizon_{max_horizon}"]).flatten()
    else:
        y_true = np.asarray(y_val).flatten()

    n = min(len(val_preds_h), len(y_true))
    if baseline_val is not None:
        n = min(n, len(np.asarray(baseline_val).flatten()))

    val_preds_h = val_preds_h[:n]
    y_true = y_true[:n]

    real_pred = denormalize(val_preds_h, config)
    real_true = denormalize(y_true, config)
    val_mae = float(np.mean(np.abs(real_pred - real_true)))

    # Naive MAE
    if baseline_val is not None:
        baseline_h = np.asarray(baseline_val).flatten()[:n]
        real_baseline = denormalize(baseline_h, config)
        naive_mae = float(np.mean(np.abs(real_baseline - real_true)))
        fitness = val_mae - naive_mae
    else:
        fitness = val_mae

    # Clean up
    del predictor
    tf.keras.backend.clear_session()
    gc.collect()

    return fitness, val_mae


def main():
    parser = argparse.ArgumentParser(description="Measure DOIN verification tolerance")
    parser.add_argument("--config", default="examples/config/phase_1_daily/optimization/phase_1_mimo_1d_optimization_config.json")
    parser.add_argument("--n_seeds", type=int, default=10, help="Number of different synthetic seeds to test")
    parser.add_argument("--seed_start", type=int, default=100, help="First seed value")
    parser.add_argument("--n_samples", type=int, default=1560, help="Synthetic samples per seed (1560 = 1 year)")
    parser.add_argument("--block_size", type=int, default=30)
    parser.add_argument("--output", default="examples/results/tolerance_measurement.json")
    args = parser.parse_args()

    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = str(PREDICTOR_ROOT / config_path)

    _orig_print(f"TOLERANCE: Loading config from {config_path}")
    config = load_config(config_path)

    # Run from predictor root so relative paths resolve
    os.chdir(str(PREDICTOR_ROOT))

    _orig_print(f"TOLERANCE: Preprocessing data...")
    base_datasets = preprocess(config)

    _orig_print(f"TOLERANCE: Will train {args.n_seeds} times with different synthetic seeds")
    _orig_print(f"TOLERANCE: n_samples={args.n_samples}, block_size={args.block_size}")

    results = []
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    for i, seed in enumerate(seeds):
        t0 = time.time()
        _orig_print(f"\nSEED {seed} ({i+1}/{len(seeds)}): Generating synthetic data...")

        synthetic_prices = generate_synthetic(seed, args.n_samples, args.block_size)
        aug_datasets = augment_training_data(base_datasets, synthetic_prices, config)

        _orig_print(f"SEED {seed}: Training (x_train: {aug_datasets['x_train'].shape})...")
        fitness, val_mae = train_and_evaluate(aug_datasets, config)

        elapsed = time.time() - t0
        results.append({
            "seed": seed,
            "fitness": fitness,
            "val_mae": val_mae,
            "elapsed_s": round(elapsed, 1),
        })
        _orig_print(f"SEED {seed}: fitness={fitness:.6f}, val_mae={val_mae:.6f}, time={elapsed:.0f}s")

    # Analysis
    fitnesses = [r["fitness"] for r in results]
    val_maes = [r["val_mae"] for r in results]

    mean_f = np.mean(fitnesses)
    std_f = np.std(fitnesses)
    min_f = np.min(fitnesses)
    max_f = np.max(fitnesses)
    range_f = max_f - min_f

    # Tolerance = max relative deviation from mean
    if abs(mean_f) > 1e-10:
        max_relative_dev = max(abs(f - mean_f) / abs(mean_f) for f in fitnesses)
        # Add safety margin (2x observed max deviation)
        recommended_tolerance = min(max_relative_dev * 2.0, 0.50)
    else:
        max_relative_dev = range_f
        recommended_tolerance = 0.20  # Default fallback

    summary = {
        "n_seeds": args.n_seeds,
        "n_samples": args.n_samples,
        "block_size": args.block_size,
        "fitness_mean": round(mean_f, 8),
        "fitness_std": round(std_f, 8),
        "fitness_min": round(min_f, 8),
        "fitness_max": round(max_f, 8),
        "fitness_range": round(range_f, 8),
        "val_mae_mean": round(np.mean(val_maes), 8),
        "val_mae_std": round(np.std(val_maes), 8),
        "max_relative_deviation": round(max_relative_dev, 6),
        "recommended_tolerance_margin": round(recommended_tolerance, 4),
        "individual_results": results,
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = SDG_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    _orig_print(f"\n{'='*60}")
    _orig_print(f"TOLERANCE RESULTS ({args.n_seeds} seeds)")
    _orig_print(f"{'='*60}")
    _orig_print(f"  Fitness: mean={mean_f:.6f}, std={std_f:.6f}")
    _orig_print(f"  Fitness: min={min_f:.6f}, max={max_f:.6f}, range={range_f:.6f}")
    _orig_print(f"  Val MAE: mean={np.mean(val_maes):.6f}, std={np.std(val_maes):.6f}")
    _orig_print(f"  Max relative deviation: {max_relative_dev:.4%}")
    _orig_print(f"  RECOMMENDED tolerance_margin: {recommended_tolerance:.4f} ({recommended_tolerance:.1%})")
    _orig_print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
