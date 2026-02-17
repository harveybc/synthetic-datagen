#!/usr/bin/env python3
"""
Tolerance v3: Measure training variance on REAL test sets (d5/d6).
Trains MIMO predictor N times with different seeds, evaluating on same d5/d6.
"""
import os, sys, json, time, numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PREDICTOR_QUIET"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
CONDA = "/home/harveybc/anaconda3/envs/tensorflow/lib/python3.12/site-packages/nvidia"
libs = [f"{CONDA}/{d}/lib" for d in ["cublas","cuda_cupti","cuda_nvcc","cuda_runtime","cudnn","cufft","curand","cusolver","cusparse","nccl","nvjitlink"]]
os.environ["LD_LIBRARY_PATH"] = ":".join(libs)

sys.path.insert(0, "/home/openclaw/predictor")
sys.path.insert(0, "/home/openclaw/.openclaw/workspace/synthetic-datagen")

N_SEEDS = 10
CONFIG_PATH = "/home/openclaw/predictor/examples/config/phase_1_daily/optimization/phase_1_mimo_1d_optimization_config.json"
RESULTS_DIR = "/home/openclaw/.openclaw/workspace/synthetic-datagen/examples/results"

with open(CONFIG_PATH) as f:
    base_config = json.load(f)

# Need a minimal evaluator config that wraps the predictor config
eval_config = {
    "predictor_config": CONFIG_PATH,
    "metrics_file": os.path.join(RESULTS_DIR, "tolerance_v3.json"),
}

from sdg_plugins.evaluator.augmentation_evaluator import AugmentationEvaluator

results = []
print(f"=== Tolerance v3: {N_SEEDS} seeds on REAL d5/d6 ===")
print(f"Config: {os.path.basename(CONFIG_PATH)}")
print()

for i, seed in enumerate(range(42, 42 + N_SEEDS)):
    print(f"[{i+1}/{N_SEEDS}] Seed {seed}...", end=" ", flush=True)
    
    # Create evaluator with this seed's config
    cfg = eval_config.copy()
    cfg["random_seed"] = seed
    cfg["quiet"] = True
    cfg["use_optimizer"] = False  # Train directly, don't run GA
    
    evaluator = AugmentationEvaluator(cfg)
    
    import tensorflow as tf
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    t0 = time.time()
    try:
        metrics = evaluator._train_predictor(augment_data=None)
        elapsed = time.time() - t0
        
        r = {"seed": seed, "val_mae": metrics["val_mae"], "test_mae": metrics["test_mae"], "elapsed_s": round(elapsed, 1)}
        results.append(r)
        print(f"val={metrics['val_mae']:.8f} test={metrics['test_mae']:.8f} ({elapsed:.0f}s)")
        
        tf.keras.backend.clear_session()
    except Exception as e:
        elapsed = time.time() - t0
        print(f"FAILED ({elapsed:.0f}s): {e}")
        import traceback; traceback.print_exc()

# Compute tolerance
val_maes = [r["val_mae"] for r in results]
test_maes = [r["test_mae"] for r in results]

if len(val_maes) >= 2:
    vm, vs = np.mean(val_maes), np.std(val_maes)
    tm, ts = np.mean(test_maes), np.std(test_maes)
    val_cv = vs / vm
    val_max_dev = max(abs(v - vm) / vm for v in val_maes)
    test_max_dev = max(abs(v - tm) / tm for v in test_maes)
    
    tol_3sigma = 3 * vs / vm
    recommended = max(tol_3sigma, val_max_dev) * 1.2

    summary = {
        "approach": "real_test_set_training_variance",
        "n_seeds": len(val_maes),
        "val_mae": {"mean": round(vm,8), "std": round(vs,8), "min": round(min(val_maes),8),
                     "max": round(max(val_maes),8), "cv": round(val_cv,6), "max_dev": round(val_max_dev,6)},
        "test_mae": {"mean": round(tm,8), "std": round(ts,8), "min": round(min(test_maes),8),
                      "max": round(max(test_maes),8), "cv": round(ts/tm,6), "max_dev": round(test_max_dev,6)},
        "tolerance_3sigma": round(tol_3sigma,6),
        "tolerance_max_observed": round(val_max_dev,6),
        "recommended_tolerance": round(recommended,4),
        "individual_results": results
    }
    
    out = os.path.join(RESULTS_DIR, "tolerance_v3.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Val MAE:  {vm:.8f} ± {vs:.8f} (CV={val_cv:.4f})")
    print(f"Val max deviation: {val_max_dev:.4%}")
    print(f"Test MAE: {tm:.8f} ± {ts:.8f}")
    print(f"Test max deviation: {test_max_dev:.4%}")
    print(f"\nRecommended DOIN tolerance: {recommended:.4f} ({recommended:.2%})")
    print(f"Saved: {out}")
