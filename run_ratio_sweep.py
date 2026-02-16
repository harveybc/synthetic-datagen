"""
Sweep augmentation ratios for block bootstrap and TimeGAN.
Tests: 100, 250, 500, 750, 1000, 1575 synthetic samples appended to d4.
Also tests pre-train on synthetic → fine-tune on real (transfer approach).
"""
import sys, os, json, logging, copy
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ratio_sweep")

DATA_DIR = "/home/openclaw/predictor/examples/data_downsampled/phase_1"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "examples", "results")
BASELINE_FILE = os.path.join(RESULTS_DIR, "baseline_results_proper.json")
PREDICTOR_ROOT = "/home/openclaw/predictor"
PREDICTOR_CONFIG = os.path.join(
    PREDICTOR_ROOT, "examples/config/phase_1/optimization/phase_1_mimo_optimization_config.json"
)

TRAIN_DATA = [
    f"{DATA_DIR}/base_d1.csv",
    f"{DATA_DIR}/base_d2.csv",
    f"{DATA_DIR}/base_d3.csv",
]

# Ratios to test (number of synthetic samples to append)
RATIOS = [100, 250, 500, 750, 1000, 1575]

# Also test block sizes
BLOCK_SIZES = [10, 20, 30, 48, 60]


def run_eval_with_synthetic(synthetic_csv, label, baseline):
    """Run augmentation eval with given synthetic CSV."""
    from sdg_plugins.evaluator.augmentation_evaluator import AugmentationEvaluator
    eval_cfg = {
        "predictor_root": PREDICTOR_ROOT,
        "predictor_config": PREDICTOR_CONFIG,
        "synthetic_data": synthetic_csv,
        "baseline_file": BASELINE_FILE,
        "metrics_file": os.path.join(RESULTS_DIR, f"metrics_{label}.json"),
    }
    evaluator = AugmentationEvaluator(eval_cfg)
    metrics = evaluator.evaluate()
    return metrics


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(BASELINE_FILE) as f:
        baseline = json.load(f)
    log.info(f"Baseline: val_mae={baseline['val_mae']:.6f}, test_mae={baseline['test_mae']:.6f}")

    all_results = {}

    # ============================================================
    # PART 1: Block Bootstrap — sweep n_samples
    # ============================================================
    log.info("\n" + "=" * 60)
    log.info("PART 1: Block Bootstrap — Augmentation Ratio Sweep")
    log.info("=" * 60)

    from sdg_plugins.generator.block_bootstrap_generator import BlockBootstrapGenerator

    for n_samples in RATIOS:
        label = f"bb_n{n_samples}"
        log.info(f"\n--- {label}: {n_samples} synthetic samples, block_size=30 ---")

        gen = BlockBootstrapGenerator({
            "train_data": TRAIN_DATA,
            "block_size": 30,
        })
        df = gen.generate(seed=42, n_samples=n_samples)
        csv_path = os.path.join(RESULTS_DIR, f"synthetic_{label}.csv")
        df.to_csv(csv_path, index=False)

        metrics = run_eval_with_synthetic(csv_path, label, baseline)
        all_results[label] = {
            "n_samples": n_samples,
            "block_size": 30,
            "val_pct": metrics["val_improvement_pct"],
            "test_pct": metrics["test_improvement_pct"],
            "verdict": metrics["verdict"],
            "val_mae": metrics["augmented_val_mae"],
            "test_mae": metrics["augmented_test_mae"],
        }
        log.info(f"  val: {metrics['val_improvement_pct']:+.2f}%  test: {metrics['test_improvement_pct']:+.2f}%  [{metrics['verdict']}]")

    # ============================================================
    # PART 2: Block Bootstrap — sweep block_size (at n=500)
    # ============================================================
    log.info("\n" + "=" * 60)
    log.info("PART 2: Block Bootstrap — Block Size Sweep (n=500)")
    log.info("=" * 60)

    for bs in BLOCK_SIZES:
        label = f"bb_bs{bs}"
        log.info(f"\n--- {label}: 500 samples, block_size={bs} ---")

        gen = BlockBootstrapGenerator({
            "train_data": TRAIN_DATA,
            "block_size": bs,
        })
        df = gen.generate(seed=42, n_samples=500)
        csv_path = os.path.join(RESULTS_DIR, f"synthetic_{label}.csv")
        df.to_csv(csv_path, index=False)

        metrics = run_eval_with_synthetic(csv_path, label, baseline)
        all_results[label] = {
            "n_samples": 500,
            "block_size": bs,
            "val_pct": metrics["val_improvement_pct"],
            "test_pct": metrics["test_improvement_pct"],
            "verdict": metrics["verdict"],
            "val_mae": metrics["augmented_val_mae"],
            "test_mae": metrics["augmented_test_mae"],
        }
        log.info(f"  val: {metrics['val_improvement_pct']:+.2f}%  test: {metrics['test_improvement_pct']:+.2f}%  [{metrics['verdict']}]")

    # ============================================================
    # PART 3: TimeGAN — sweep n_samples (subsample existing)
    # ============================================================
    log.info("\n" + "=" * 60)
    log.info("PART 3: TimeGAN — Augmentation Ratio Sweep (subsampled)")
    log.info("=" * 60)

    timegan_full = os.path.join(RESULTS_DIR, "synthetic_timegan.csv")
    if os.path.exists(timegan_full):
        tg_df = pd.read_csv(timegan_full)
        for n_samples in [100, 250, 500]:
            label = f"tg_n{n_samples}"
            log.info(f"\n--- {label}: {n_samples} TimeGAN samples ---")

            # Take first n samples (they're sequential)
            sub_df = tg_df.head(n_samples).copy()
            csv_path = os.path.join(RESULTS_DIR, f"synthetic_{label}.csv")
            sub_df.to_csv(csv_path, index=False)

            metrics = run_eval_with_synthetic(csv_path, label, baseline)
            all_results[label] = {
                "n_samples": n_samples,
                "generator": "timegan",
                "val_pct": metrics["val_improvement_pct"],
                "test_pct": metrics["test_improvement_pct"],
                "verdict": metrics["verdict"],
                "val_mae": metrics["augmented_val_mae"],
                "test_mae": metrics["augmented_test_mae"],
            }
            log.info(f"  val: {metrics['val_improvement_pct']:+.2f}%  test: {metrics['test_improvement_pct']:+.2f}%  [{metrics['verdict']}]")

    # ============================================================
    # SUMMARY
    # ============================================================
    log.info("\n" + "=" * 60)
    log.info("  FULL SWEEP SUMMARY")
    log.info("=" * 60)
    log.info(f"Baseline: val={baseline['val_mae']:.6f}, test={baseline['test_mae']:.6f}")
    log.info(f"{'Label':20s} {'Val Δ':>8s} {'Test Δ':>9s} {'Verdict':>8s}")
    log.info("-" * 50)
    for label, r in sorted(all_results.items()):
        log.info(f"{label:20s} {r['val_pct']:+7.2f}% {r['test_pct']:+8.2f}% {r['verdict']:>8s}")

    # Find best
    best_val = max(all_results.items(), key=lambda x: x[1]["val_pct"])
    best_test = max(all_results.items(), key=lambda x: x[1]["test_pct"])
    log.info(f"\nBest val:  {best_val[0]} ({best_val[1]['val_pct']:+.2f}%)")
    log.info(f"Best test: {best_test[0]} ({best_test[1]['test_pct']:+.2f}%)")

    # Save
    summary_path = os.path.join(RESULTS_DIR, "ratio_sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
