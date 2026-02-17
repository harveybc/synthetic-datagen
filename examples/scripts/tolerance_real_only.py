#!/usr/bin/env python3
"""Train MIMO on real d4, eval on real d5/d6 only. For running on Omega."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.environ.get("PREDICTOR_ROOT", "/home/openclaw/predictor"), "app"))
os.environ["PREDICTOR_QUIET"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from pathlib import Path
from sdg_plugins.evaluator.augmentation_evaluator import AugmentationEvaluator

PREDICTOR_ROOT = os.environ.get("PREDICTOR_ROOT", "/home/openclaw/predictor")
OUT = Path(__file__).parent.parent / "results/regime_hmm_garch"
OUT.mkdir(parents=True, exist_ok=True)

config = {
    "predictor_root": PREDICTOR_ROOT,
    "predictor_config": "examples/config/phase_1_daily/optimization/phase_1_mimo_1d_optimization_config.json",
    "predictor_plugin": "mimo", "preprocessor_plugin": "stl_preprocessor",
    "target_plugin": "default_target", "pipeline_plugin": "stl_pipeline",
}
evaluator = AugmentationEvaluator(config)

print("[Real] Training MIMO on real d4, eval on real d5/d6...")
t0 = time.time()
res = evaluator._train_predictor(augment_data=None)
print(f"  val_MAE={res['val_mae']:.6f}, test_MAE={res['test_mae']:.6f} ({time.time()-t0:.0f}s)")

with open(OUT / "real_baseline.json", "w") as f:
    json.dump(res, f, indent=2)
print("Saved to real_baseline.json")
