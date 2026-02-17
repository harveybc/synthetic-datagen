#!/usr/bin/env python3
"""Train MIMO on real d4, eval on HMM+GARCH synthetic d5/d6. Configurable seed range."""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
os.environ["PREDICTOR_QUIET"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=str, default="0,1,2,3,4", help="Comma-separated seeds")
parser.add_argument("--predictor_root", type=str, default="/home/openclaw/predictor")
parser.add_argument("--synth_dir", type=str, default=None)
args = parser.parse_args()

seeds = [int(s) for s in args.seeds.split(",")]
PREDICTOR_ROOT = args.predictor_root
sys.path.insert(0, os.path.join(PREDICTOR_ROOT, "app"))

from sdg_plugins.evaluator.augmentation_evaluator import AugmentationEvaluator

OUT = Path(args.synth_dir) if args.synth_dir else Path(__file__).parent.parent / "results/regime_hmm_garch"

results = []
for seed in seeds:
    cfg = {
        "predictor_root": PREDICTOR_ROOT,
        "predictor_config": "examples/config/phase_1_daily/optimization/phase_1_mimo_1d_optimization_config.json",
        "predictor_plugin": "mimo", "preprocessor_plugin": "stl_preprocessor",
        "target_plugin": "default_target", "pipeline_plugin": "stl_pipeline",
        "x_validation_file": str(OUT / f"synth_d5_seed{seed}.csv"),
        "y_validation_file": str(OUT / f"synth_d5_seed{seed}.csv"),
        "x_test_file": str(OUT / f"synth_d6_seed{seed}.csv"),
        "y_test_file": str(OUT / f"synth_d6_seed{seed}.csv"),
    }
    evaluator = AugmentationEvaluator(cfg)
    print(f"[Seed {seed}] Training...", end="", flush=True)
    t0 = time.time()
    res = evaluator._train_predictor(augment_data=None)
    print(f" val={res['val_mae']:.6f}, test={res['test_mae']:.6f} ({time.time()-t0:.0f}s)")
    results.append({"seed": seed, **res})

out_file = OUT / f"synth_results_seeds_{'_'.join(str(s) for s in seeds)}.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {out_file}")
