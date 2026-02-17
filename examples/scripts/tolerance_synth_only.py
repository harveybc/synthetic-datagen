#!/usr/bin/env python3
"""Train MIMO on real d4, eval on HMM+GARCH synthetic d5/d6. 5 seeds. For Dragon."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, "/home/openclaw/predictor/app")
os.environ["PREDICTOR_QUIET"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from pathlib import Path
from sdg_plugins.evaluator.augmentation_evaluator import AugmentationEvaluator

OUT = Path(__file__).parent.parent / "results/regime_hmm_garch"

BASE_CFG = {
    "predictor_root": "/home/openclaw/predictor",
    "predictor_config": "examples/config/phase_1_daily/optimization/phase_1_mimo_1d_optimization_config.json",
    "predictor_plugin": "mimo", "preprocessor_plugin": "stl_preprocessor",
    "target_plugin": "default_target", "pipeline_plugin": "stl_pipeline",
}

results = []
for seed in range(5):
    cfg = BASE_CFG.copy()
    # Override val/test to synthetic files (absolute paths so evaluator doesn't resolve relative to predictor)
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
print(f"\nDone. Saved {len(results)} results.")
