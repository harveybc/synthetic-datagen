#!/usr/bin/env python3
"""Tolerance eval: train MIMO on real d4, eval on real vs HMM+GARCH synthetic d5/d6.

6 models: 1 real baseline + 5 synthetic seeds. ~7min each on GPU.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, "/home/openclaw/predictor/app")
os.environ["PREDICTOR_QUIET"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from pathlib import Path
from sdg_plugins.generator import regime_hmm_garch as rg

DATA = Path("/home/openclaw/predictor/examples/data_downsampled/phase_1")
OUT = Path(__file__).parent.parent / "results/regime_hmm_garch"
OUT.mkdir(parents=True, exist_ok=True)

# Load data and fit model
print("Loading data & fitting HMM+GARCH...")
train_prices = np.concatenate([
    pd.read_csv(DATA / f"base_d{i}.csv")["typical_price"].values for i in [1, 2, 3]
])
model = rg.fit(train_prices, n_regimes=4)
rg.save_model(model, str(OUT / "model_k4.json"))

d5_base = pd.read_csv(DATA / "base_d5.csv")
d6_base = pd.read_csv(DATA / "base_d6.csv")
norm_cfg = json.load(open(DATA / "normalization_config_b.json"))

# Generate normalized synthetic d5/d6
col = "typical_price"
mean_val = norm_cfg.get(f"{col}_mean", norm_cfg.get("mean", 0))
std_val = norm_cfg.get(f"{col}_std", norm_cfg.get("std", 1))

for seed in range(5):
    for dset, base_df, suffix in [("d5", d5_base, "d5"), ("d6", d6_base, "d6")]:
        synth = rg.generate(model, len(base_df), seed=seed*100+int(suffix[1]),
                           initial_price=base_df[col].iloc[0])
        synth_norm = (synth - mean_val) / std_val
        pd.DataFrame({col: synth_norm}).to_csv(OUT / f"synth_{suffix}_seed{seed}.csv", index=False)

print("Synthetic data generated. Starting predictor evals...\n")

# Use AugmentationEvaluator
from sdg_plugins.evaluator.augmentation_evaluator import AugmentationEvaluator
config = {
    "predictor_root": "/home/openclaw/predictor",
    "predictor_config": "examples/config/phase_1_daily/optimization/phase_1_mimo_1d_optimization_config.json",
    "predictor_plugin": "mimo", "preprocessor_plugin": "stl_preprocessor",
    "target_plugin": "default_target", "pipeline_plugin": "stl_pipeline",
}
evaluator = AugmentationEvaluator(config)

# 1. Real baseline
print("[Real] Training...")
t0 = time.time()
real_res = evaluator._train_predictor(augment_data=None)
print(f"  val={real_res['val_mae']:.6f}, test={real_res['test_mae']:.6f} ({time.time()-t0:.0f}s)")

# 2. Synthetic evals
synth_results = []
for seed in range(5):
    sd5 = str(OUT / f"synth_d5_seed{seed}.csv")
    sd6 = str(OUT / f"synth_d6_seed{seed}.csv")

    orig = evaluator.predictor_config.copy()
    evaluator.predictor_config["x_validation_file"] = sd5
    evaluator.predictor_config["y_validation_file"] = sd5
    evaluator.predictor_config["x_test_file"] = sd6
    evaluator.predictor_config["y_test_file"] = sd6

    print(f"[Seed {seed}] Training...", end="", flush=True)
    t0 = time.time()
    res = evaluator._train_predictor(augment_data=None)
    val_gap = abs(res["val_mae"] - real_res["val_mae"]) / real_res["val_mae"]
    test_gap = abs(res["test_mae"] - real_res["test_mae"]) / real_res["test_mae"]
    print(f" val={res['val_mae']:.6f} ({val_gap:.1%}), test={res['test_mae']:.6f} ({test_gap:.1%}) [{time.time()-t0:.0f}s]")
    synth_results.append({"seed": seed, "val_mae": res["val_mae"], "test_mae": res["test_mae"],
                          "val_gap": val_gap, "test_gap": test_gap})
    evaluator.predictor_config = orig

# Summary
avg_vg = np.mean([r["val_gap"] for r in synth_results])
avg_tg = np.mean([r["test_gap"] for r in synth_results])
max_vg = max(r["val_gap"] for r in synth_results)
max_tg = max(r["test_gap"] for r in synth_results)
rec = max(max_vg, max_tg) * 1.2

print(f"\n=== HMM+GARCH TOLERANCE ===")
print(f"Real: val={real_res['val_mae']:.6f}, test={real_res['test_mae']:.6f}")
print(f"Avg gap: val={avg_vg:.1%}, test={avg_tg:.1%}")
print(f"Max gap: val={max_vg:.1%}, test={max_tg:.1%}")
print(f"Recommended tolerance: {rec:.1%}")

results = {"real": real_res, "synth": synth_results,
           "avg_val_gap": avg_vg, "avg_test_gap": avg_tg,
           "max_val_gap": max_vg, "max_test_gap": max_tg,
           "recommended_tolerance": rec}
with open(OUT / "tolerance_results.json", "w") as f:
    json.dump(results, f, indent=2)
