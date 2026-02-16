"""
Option A evaluation: Block Bootstrap vs Grasynda
Both are training-free — just generate and evaluate augmentation impact.
"""
import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("option_a")

DATA_DIR = "/home/openclaw/predictor/examples/data_downsampled/phase_1"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "examples", "results")
BASELINE_FILE = os.path.join(RESULTS_DIR, "baseline_results_proper.json")
PREDICTOR_ROOT = "/home/openclaw/predictor"
PREDICTOR_CONFIG = os.path.join(
    PREDICTOR_ROOT, "examples/config/phase_1/optimization/phase_1_mimo_optimization_config.json"
)
N_SAMPLES = 1575  # 1 forex year

TRAIN_DATA = [
    f"{DATA_DIR}/base_d1.csv",
    f"{DATA_DIR}/base_d2.csv",
    f"{DATA_DIR}/base_d3.csv",
]


def run_generator(gen_name, generator, seed=42):
    """Generate synthetic data and run augmentation eval."""
    log.info(f"\n{'='*60}")
    log.info(f"  {gen_name}")
    log.info(f"{'='*60}")

    # Generate
    df = generator.generate(seed=seed, n_samples=N_SAMPLES)
    out_csv = os.path.join(RESULTS_DIR, f"synthetic_{gen_name}.csv")
    df.to_csv(out_csv, index=False)
    log.info(f"Saved {len(df)} synthetic samples to {out_csv}")

    # Quick stats
    prices = df["typical_price"].values
    log.info(f"Price range: {prices.min():.4f} - {prices.max():.4f}")
    log.info(f"Price mean: {prices.mean():.4f}, std: {prices.std():.4f}")

    # Evaluate
    from sdg_plugins.evaluator.augmentation_evaluator import AugmentationEvaluator
    eval_cfg = {
        "predictor_root": PREDICTOR_ROOT,
        "predictor_config": PREDICTOR_CONFIG,
        "synthetic_data": out_csv,
        "baseline_file": BASELINE_FILE,
        "metrics_file": os.path.join(RESULTS_DIR, f"metrics_{gen_name}.json"),
    }
    evaluator = AugmentationEvaluator(eval_cfg)
    metrics = evaluator.evaluate()

    log.info(f"\n--- {gen_name} Results ---")
    log.info(f"Val  improvement: {metrics['val_improvement_pct']:+.2f}%")
    log.info(f"Test improvement: {metrics['test_improvement_pct']:+.2f}%")
    log.info(f"Verdict: {metrics['verdict']}")
    return metrics


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Verify baseline exists
    if not os.path.exists(BASELINE_FILE):
        log.error(f"Baseline not found: {BASELINE_FILE}")
        log.error("Run baseline computation first!")
        sys.exit(1)

    with open(BASELINE_FILE) as f:
        baseline = json.load(f)
    log.info(f"Baseline: val_mae={baseline['val_mae']:.6f}, test_mae={baseline['test_mae']:.6f}")

    results = {}

    # --- Block Bootstrap ---
    from sdg_plugins.generator.block_bootstrap_generator import BlockBootstrapGenerator
    bb_gen = BlockBootstrapGenerator({
        "train_data": TRAIN_DATA,
        "block_size": 30,  # 1 trading week
    })
    results["block_bootstrap"] = run_generator("block_bootstrap", bb_gen)

    # --- Grasynda ---
    from sdg_plugins.generator.grasynda_generator import GrasyndaGenerator
    gr_gen = GrasyndaGenerator({
        "train_data": TRAIN_DATA,
        "n_bins": 50,
        "use_returns": True,
    })
    results["grasynda"] = run_generator("grasynda", gr_gen)

    # Summary
    log.info(f"\n{'='*60}")
    log.info(f"  SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"Baseline: val_mae={baseline['val_mae']:.6f}, test_mae={baseline['test_mae']:.6f}")
    for name, m in results.items():
        log.info(
            f"{name:20s}: val {m['val_improvement_pct']:+.2f}%  "
            f"test {m['test_improvement_pct']:+.2f}%  [{m['verdict']}]"
        )

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "option_a_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
