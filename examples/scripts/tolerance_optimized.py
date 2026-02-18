#!/usr/bin/env python3
"""Tolerance eval: optimized vs default hybrid generator.
Uses AugmentationEvaluator._train_predictor() for GPU-compatible MIMO training.
Generator fitted on d1-d4, synthetic replaces d5/d6 for evaluation.
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
os.environ["PREDICTOR_QUIET"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# GPU setup
from pathlib import Path
NVIDIA_LIBS = Path.home() / ".local/lib/python3.12/site-packages/nvidia"
if NVIDIA_LIBS.exists():
    lib_dirs = [str(p / "lib") for p in NVIDIA_LIBS.iterdir() if (p / "lib").is_dir()]
    os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--n_seeds", type=int, default=5)
parser.add_argument("--predictor_root", type=str, default="/home/openclaw/predictor")
args = parser.parse_args()

PREDICTOR_ROOT = args.predictor_root
DATA = Path(PREDICTOR_ROOT) / "examples/data_downsampled/phase_1"
OUT = Path(__file__).parent.parent / "results/tolerance_optimized"
OUT.mkdir(parents=True, exist_ok=True)

from sdg_plugins.evaluator.augmentation_evaluator import AugmentationEvaluator
from sdg_plugins.generator import regime_bootstrap_hybrid as hybrid

BASE_CFG = {
    "predictor_root": PREDICTOR_ROOT,
    "predictor_config": "examples/config/phase_1_daily/optimization/phase_1_mimo_1d_optimization_config.json",
    "predictor_plugin": "mimo", "preprocessor_plugin": "stl_preprocessor",
    "target_plugin": "default_target", "pipeline_plugin": "stl_pipeline",
}

# Best config from optimization sweep
BEST_CONFIG = {
    "n_regimes": 5, "block_size": 30, "smooth_weight": 0.5,
    "min_block_length": 2, "covariance_type": "diag",
    "vol_short_window": 6, "vol_mid_window": 16, "vol_long_window": 48,
}

# Old default for comparison
OLD_CONFIG = {
    "n_regimes": 4, "block_size": 30, "smooth_weight": 0.3,
    "min_block_length": 3, "covariance_type": "diag",
    "vol_short_window": 6, "vol_mid_window": 24, "vol_long_window": 48,
}


def run_tolerance(gen_config, config_name, n_seeds):
    print(f"\n{'='*60}", flush=True)
    print(f"TOLERANCE EVAL: {config_name}", flush=True)
    print(f"Config: {gen_config}", flush=True)
    print(f"{'='*60}", flush=True)

    # Fit generator on d1-d4
    print("Fitting hybrid generator on d1-d4...", flush=True)
    train_prices = np.concatenate([
        pd.read_csv(DATA / f"base_d{i}.csv")["typical_price"].values for i in [1, 2, 3, 4]
    ])
    d5_real = pd.read_csv(DATA / "base_d5.csv")["typical_price"].values
    d6_real = pd.read_csv(DATA / "base_d6.csv")["typical_price"].values

    model = hybrid.fit(train_prices, quiet=True, **gen_config)

    # Real baseline
    print("Training real baseline...", flush=True)
    evaluator = AugmentationEvaluator(BASE_CFG.copy())
    t0 = time.time()
    real_result = evaluator._train_predictor(augment_data=None)
    print(f"  Real baseline: val_MAE={real_result['val_mae']:.6f}, "
          f"test_MAE={real_result['test_mae']:.6f} ({time.time()-t0:.0f}s)", flush=True)

    # Synthetic evaluations
    synth_results = []
    for seed in range(n_seeds):
        print(f"  [Seed {seed}] Generating synthetic d5/d6...", end="", flush=True)

        synth_d5 = hybrid.generate(model, len(d5_real), seed=seed * 2, initial_price=d5_real[0])
        synth_d6 = hybrid.generate(model, len(d6_real), seed=seed * 2 + 1, initial_price=d6_real[0])

        sd5_path = OUT / f"{config_name}_synth_d5_seed{seed}.csv"
        sd6_path = OUT / f"{config_name}_synth_d6_seed{seed}.csv"
        pd.DataFrame({"typical_price": synth_d5}).to_csv(sd5_path, index=False)
        pd.DataFrame({"typical_price": synth_d6}).to_csv(sd6_path, index=False)

        cfg = BASE_CFG.copy()
        cfg["x_validation_file"] = str(sd5_path)
        cfg["y_validation_file"] = str(sd5_path)
        cfg["x_test_file"] = str(sd6_path)
        cfg["y_test_file"] = str(sd6_path)

        evaluator = AugmentationEvaluator(cfg)
        t0 = time.time()
        res = evaluator._train_predictor(augment_data=None)
        elapsed = time.time() - t0
        print(f" val={res['val_mae']:.6f}, test={res['test_mae']:.6f} ({elapsed:.0f}s)", flush=True)
        synth_results.append({"seed": seed, **res})

    # Analysis
    val_maes = [r["val_mae"] for r in synth_results]
    test_maes = [r["test_mae"] for r in synth_results]
    real_val = real_result["val_mae"]
    real_test = real_result["test_mae"]
    val_devs = [abs(v - real_val) / real_val for v in val_maes]
    test_devs = [abs(t - real_test) / real_test for t in test_maes]

    summary = {
        "config_name": config_name,
        "generator_config": gen_config,
        "real_baseline": real_result,
        "synth_results": synth_results,
        "val_mae_mean": float(np.mean(val_maes)),
        "val_mae_std": float(np.std(val_maes)),
        "test_mae_mean": float(np.mean(test_maes)),
        "test_mae_std": float(np.std(test_maes)),
        "val_deviation_mean": float(np.mean(val_devs)),
        "val_deviation_max": float(np.max(val_devs)),
        "test_deviation_mean": float(np.mean(test_devs)),
        "test_deviation_max": float(np.max(test_devs)),
    }

    print(f"\nSUMMARY — {config_name}:", flush=True)
    print(f"  Real baseline:     val={real_val:.6f}, test={real_test:.6f}", flush=True)
    print(f"  Synth mean±std:    val={np.mean(val_maes):.6f}±{np.std(val_maes):.6f}, "
          f"test={np.mean(test_maes):.6f}±{np.std(test_maes):.6f}", flush=True)
    print(f"  Val deviation:     mean={np.mean(val_devs)*100:.1f}%, max={np.max(val_devs)*100:.1f}%", flush=True)
    print(f"  Test deviation:    mean={np.mean(test_devs)*100:.1f}%, max={np.max(test_devs)*100:.1f}%", flush=True)

    return summary


if __name__ == "__main__":
    results = {}
    results["optimized"] = run_tolerance(BEST_CONFIG, "optimized", args.n_seeds)
    results["default"] = run_tolerance(OLD_CONFIG, "default", args.n_seeds)

    with open(OUT / "tolerance_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print("COMPARISON", flush=True)
    print(f"{'='*60}", flush=True)
    for name, r in results.items():
        print(f"  {name:12s}: val_dev={r['val_deviation_mean']*100:.1f}% (max {r['val_deviation_max']*100:.1f}%), "
              f"test_dev={r['test_deviation_mean']*100:.1f}% (max {r['test_deviation_max']*100:.1f}%)", flush=True)
    print(f"\nResults saved to {OUT}/", flush=True)
