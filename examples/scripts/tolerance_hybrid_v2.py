#!/usr/bin/env python3
"""Tolerance eval with optimized hybrid: n_regimes=5, block_size=30, vol_scale=0.80."""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
os.environ["PREDICTOR_QUIET"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["real", "synth"], required=True)
parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
parser.add_argument("--predictor_root", type=str, default="/home/openclaw/predictor")
parser.add_argument("--out_dir", type=str, default=None)
args = parser.parse_args()

import numpy as np
import pandas as pd
from pathlib import Path

PREDICTOR_ROOT = args.predictor_root
sys.path.insert(0, os.path.join(PREDICTOR_ROOT, "app"))

DATA = Path(PREDICTOR_ROOT) / "examples/data_downsampled/phase_1"
OUT = Path(args.out_dir) if args.out_dir else Path(__file__).parent.parent / "results/tolerance_hybrid_v2"
OUT.mkdir(parents=True, exist_ok=True)

from sdg_plugins.evaluator.augmentation_evaluator import AugmentationEvaluator
from sdg_plugins.generator import regime_bootstrap_hybrid as hybrid

# Optimized config
N_REGIMES = 5
BLOCK_SIZE = 30
VOL_SCALE = 0.80

BASE_CFG = {
    "predictor_root": PREDICTOR_ROOT,
    "predictor_config": "examples/config/phase_1_daily/optimization/phase_1_mimo_1d_optimization_config.json",
    "predictor_plugin": "mimo", "preprocessor_plugin": "stl_preprocessor",
    "target_plugin": "default_target", "pipeline_plugin": "stl_pipeline",
}

def log_returns(p):
    return np.diff(np.log(np.clip(p, 1e-10, None)))

if args.mode == "real":
    print("[Real] Training MIMO on real d4, eval on real d5/d6...")
    evaluator = AugmentationEvaluator(BASE_CFG.copy())
    t0 = time.time()
    res = evaluator._train_predictor(augment_data=None)
    print(f"  val_MAE={res['val_mae']:.6f}, test_MAE={res['test_mae']:.6f} ({time.time()-t0:.0f}s)")
    with open(OUT / "real_baseline.json", "w") as f:
        json.dump(res, f, indent=2)

elif args.mode == "synth":
    print(f"Fitting optimized hybrid (n={N_REGIMES}, bs={BLOCK_SIZE}, vs={VOL_SCALE})...")
    train_prices = np.concatenate([
        pd.read_csv(DATA / f"base_d{i}.csv")["typical_price"].values for i in [1, 2, 3]
    ])
    d5 = pd.read_csv(DATA / "base_d5.csv")["typical_price"].values
    d6 = pd.read_csv(DATA / "base_d6.csv")["typical_price"].values
    model = hybrid.fit(train_prices, n_regimes=N_REGIMES, block_size=BLOCK_SIZE)

    seeds = [int(s) for s in args.seeds.split(",")]
    results = []
    for seed in seeds:
        # Generate with vol scaling
        synth_d5_raw = hybrid.generate(model, len(d5), seed=seed * 2, initial_price=d5[0])
        synth_d6_raw = hybrid.generate(model, len(d6), seed=seed * 2 + 1, initial_price=d6[0])

        # Apply vol_scale
        for synth, real_ref, name in [(synth_d5_raw, d5, "d5"), (synth_d6_raw, d6, "d6")]:
            sr = log_returns(synth)
            sr = sr * VOL_SCALE
            scaled = real_ref[0] * np.exp(np.concatenate([[0], np.cumsum(sr)]))
            scaled = scaled[:len(real_ref)]
            path = OUT / f"synth_{name}_seed{seed}.csv"
            pd.DataFrame({"typical_price": scaled}).to_csv(path, index=False)

        cfg = BASE_CFG.copy()
        cfg["x_validation_file"] = str(OUT / f"synth_d5_seed{seed}.csv")
        cfg["y_validation_file"] = str(OUT / f"synth_d5_seed{seed}.csv")
        cfg["x_test_file"] = str(OUT / f"synth_d6_seed{seed}.csv")
        cfg["y_test_file"] = str(OUT / f"synth_d6_seed{seed}.csv")

        evaluator = AugmentationEvaluator(cfg)
        print(f"[Seed {seed}] Training...", end="", flush=True)
        t0 = time.time()
        res = evaluator._train_predictor(augment_data=None)
        print(f" val={res['val_mae']:.6f}, test={res['test_mae']:.6f} ({time.time()-t0:.0f}s)")
        results.append({"seed": seed, **res})

    with open(OUT / "synth_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results.")
