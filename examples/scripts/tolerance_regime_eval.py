#!/usr/bin/env python3
"""Train MIMO on real d4, eval on real vs regime-synthetic d5/d6. Measures tolerance."""
import sys, os, json, time
sys.path.insert(0, "/home/openclaw/predictor/app")
os.environ["PREDICTOR_QUIET"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from pathlib import Path

DATA = Path("/home/openclaw/predictor/examples/data_downsampled/phase_1")
SYNTH = Path("/home/openclaw/.openclaw/workspace/synthetic-datagen/examples/results/regime_conditional")
OUT = SYNTH

# Use AugmentationEvaluator's _train_predictor for consistency
from sdg_plugins.evaluator.augmentation_evaluator import AugmentationEvaluator

config = {
    "predictor_root": "/home/openclaw/predictor",
    "predictor_config": "examples/config/phase_1_daily/optimization/phase_1_mimo_1d_optimization_config.json",
    "predictor_plugin": "mimo",
    "preprocessor_plugin": "stl_preprocessor",
    "target_plugin": "default_target",
    "pipeline_plugin": "stl_pipeline",
}
evaluator = AugmentationEvaluator(config)

# Train once on real d4, eval on real d5/d6
print("Training MIMO on real d4 (seed 42)...")
t0 = time.time()
real_results = evaluator._train_predictor(augment_data=None)
elapsed = time.time() - t0
real_val_mae = real_results["val_mae"]
real_test_mae = real_results["test_mae"]
print(f"  Real: val_MAE={real_val_mae:.6f}, test_MAE={real_test_mae:.6f} ({elapsed:.0f}s)")

# Now eval same model on synthetic d5/d6
# We need to swap validation/test files and re-evaluate
# Actually, we need to train+eval with synthetic as val/test
# Simpler: train multiple times, compare real vs synth eval

print("\nTraining with synthetic val/test sets...")
synth_results = []
for seed in range(5):
    synth_d5 = str(SYNTH / f"synth_d5_seed{seed}.csv")
    synth_d6 = str(SYNTH / f"synth_d6_seed{seed}.csv")

    # Override val/test files temporarily
    orig_config = evaluator.predictor_config.copy()
    evaluator.predictor_config["x_validation_file"] = synth_d5
    evaluator.predictor_config["y_validation_file"] = synth_d5
    evaluator.predictor_config["x_test_file"] = synth_d6
    evaluator.predictor_config["y_test_file"] = synth_d6

    print(f"  [Seed {seed}] Training with synth val/test...")
    t0 = time.time()
    res = evaluator._train_predictor(augment_data=None)
    elapsed = time.time() - t0
    synth_val = res["val_mae"]
    synth_test = res["test_mae"]
    val_gap = abs(synth_val - real_val_mae) / real_val_mae
    test_gap = abs(synth_test - real_test_mae) / real_test_mae
    print(f"    Synth: val={synth_val:.6f} ({val_gap:.1%} gap), test={synth_test:.6f} ({test_gap:.1%} gap) [{elapsed:.0f}s]")

    synth_results.append({
        "seed": seed, "val_mae": synth_val, "test_mae": synth_test,
        "val_gap": val_gap, "test_gap": test_gap, "elapsed_s": elapsed,
    })

    # Restore
    evaluator.predictor_config = orig_config

# Summary
avg_val_gap = np.mean([r["val_gap"] for r in synth_results])
avg_test_gap = np.mean([r["test_gap"] for r in synth_results])
max_val_gap = max(r["val_gap"] for r in synth_results)
max_test_gap = max(r["test_gap"] for r in synth_results)

print(f"\n=== TOLERANCE RESULTS ===")
print(f"Real baseline: val={real_val_mae:.6f}, test={real_test_mae:.6f}")
print(f"Avg gap: val={avg_val_gap:.1%}, test={avg_test_gap:.1%}")
print(f"Max gap: val={max_val_gap:.1%}, test={max_test_gap:.1%}")
print(f"Recommended tolerance: {max(max_val_gap, max_test_gap) * 1.2:.1%}")

results = {
    "real_val_mae": real_val_mae, "real_test_mae": real_test_mae,
    "synth_results": synth_results,
    "avg_val_gap": avg_val_gap, "avg_test_gap": avg_test_gap,
    "max_val_gap": max_val_gap, "max_test_gap": max_test_gap,
    "recommended_tolerance": max(max_val_gap, max_test_gap) * 1.2,
}
with open(OUT / "tolerance_regime_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUT / 'tolerance_regime_results.json'}")
