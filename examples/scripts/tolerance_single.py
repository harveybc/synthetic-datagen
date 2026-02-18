#!/usr/bin/env python3
"""Tolerance eval for a single generator config. Can run on any machine.
Usage: python tolerance_single.py --config optimized|default --n_seeds 5 --predictor_root /path
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
os.environ["PREDICTOR_QUIET"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--config", choices=["optimized", "default"], required=True)
parser.add_argument("--n_seeds", type=int, default=5)
parser.add_argument("--predictor_root", type=str, default="/home/openclaw/predictor")
parser.add_argument("--out_dir", type=str, default=None)
args = parser.parse_args()

PREDICTOR_ROOT = args.predictor_root
DATA = Path(PREDICTOR_ROOT) / "examples/data_downsampled/phase_1"

from sdg_plugins.evaluator.augmentation_evaluator import AugmentationEvaluator
from sdg_plugins.generator import regime_bootstrap_hybrid as hybrid

CONFIGS = {
    "optimized": {
        "n_regimes": 5, "block_size": 30, "smooth_weight": 0.5,
        "min_block_length": 2, "covariance_type": "diag",
        "vol_short_window": 6, "vol_mid_window": 16, "vol_long_window": 48,
    },
    "default": {
        "n_regimes": 4, "block_size": 30, "smooth_weight": 0.3,
        "min_block_length": 3, "covariance_type": "diag",
        "vol_short_window": 6, "vol_mid_window": 24, "vol_long_window": 48,
    },
}

gen_config = CONFIGS[args.config]
OUT = Path(args.out_dir) if args.out_dir else Path(__file__).parent.parent / f"results/tolerance_{args.config}"
OUT.mkdir(parents=True, exist_ok=True)

BASE_CFG = {
    "predictor_root": PREDICTOR_ROOT,
    "predictor_config": "examples/config/phase_1_daily/optimization/phase_1_mimo_1d_optimization_config.json",
    "predictor_plugin": "mimo", "preprocessor_plugin": "stl_preprocessor",
    "target_plugin": "default_target", "pipeline_plugin": "stl_pipeline",
}

print(f"Config: {args.config} = {gen_config}", flush=True)
print("Fitting generator on d1-d4...", flush=True)
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
print(f"  Real: val={real_result['val_mae']:.6f}, test={real_result['test_mae']:.6f} ({time.time()-t0:.0f}s)", flush=True)

# Synthetic seeds
synth_results = []
for seed in range(args.n_seeds):
    synth_d5 = hybrid.generate(model, len(d5_real), seed=seed*2, initial_price=d5_real[0])
    synth_d6 = hybrid.generate(model, len(d6_real), seed=seed*2+1, initial_price=d6_real[0])
    sd5_path = OUT / f"synth_d5_seed{seed}.csv"
    sd6_path = OUT / f"synth_d6_seed{seed}.csv"
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
    print(f"  Seed {seed}: val={res['val_mae']:.6f}, test={res['test_mae']:.6f} ({time.time()-t0:.0f}s)", flush=True)
    synth_results.append({"seed": seed, **res})

# Summary
val_maes = [r["val_mae"] for r in synth_results]
test_maes = [r["test_mae"] for r in synth_results]
val_devs = [abs(v - real_result["val_mae"]) / real_result["val_mae"] for v in val_maes]
test_devs = [abs(t - real_result["test_mae"]) / real_result["test_mae"] for t in test_maes]

summary = {
    "config_name": args.config, "generator_config": gen_config,
    "real_baseline": real_result, "synth_results": synth_results,
    "val_mae_mean": float(np.mean(val_maes)), "val_mae_std": float(np.std(val_maes)),
    "test_mae_mean": float(np.mean(test_maes)), "test_mae_std": float(np.std(test_maes)),
    "val_deviation_mean": float(np.mean(val_devs)), "val_deviation_max": float(np.max(val_devs)),
    "test_deviation_mean": float(np.mean(test_devs)), "test_deviation_max": float(np.max(test_devs)),
}

with open(OUT / "results.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSUMMARY ({args.config}):", flush=True)
print(f"  Real:  val={real_result['val_mae']:.6f}, test={real_result['test_mae']:.6f}", flush=True)
print(f"  Synth: val={np.mean(val_maes):.6f}±{np.std(val_maes):.6f}, test={np.mean(test_maes):.6f}±{np.std(test_maes):.6f}", flush=True)
print(f"  Val deviation:  mean={np.mean(val_devs)*100:.1f}%, max={np.max(val_devs)*100:.1f}%", flush=True)
print(f"  Test deviation: mean={np.mean(test_devs)*100:.1f}%, max={np.max(test_devs)*100:.1f}%", flush=True)
