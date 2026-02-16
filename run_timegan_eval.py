"""
TimeGAN augmentation evaluation.
Uses pre-generated synthetic_timegan.csv through the augmentation evaluator.
"""
import sys, os, json, logging

sys.path.insert(0, os.path.dirname(__file__))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("timegan_eval")

DATA_DIR = "/home/openclaw/predictor/examples/data_downsampled/phase_1"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "examples", "results")
BASELINE_FILE = os.path.join(RESULTS_DIR, "baseline_results_proper.json")
PREDICTOR_ROOT = "/home/openclaw/predictor"
PREDICTOR_CONFIG = os.path.join(
    PREDICTOR_ROOT, "examples/config/phase_1/optimization/phase_1_mimo_optimization_config.json"
)
SYNTHETIC_CSV = os.path.join(RESULTS_DIR, "synthetic_timegan.csv")


def main():
    if not os.path.exists(BASELINE_FILE):
        log.error(f"Baseline not found: {BASELINE_FILE}")
        sys.exit(1)
    if not os.path.exists(SYNTHETIC_CSV):
        log.error(f"Synthetic data not found: {SYNTHETIC_CSV}")
        sys.exit(1)

    with open(BASELINE_FILE) as f:
        baseline = json.load(f)
    log.info(f"Baseline: val_mae={baseline['val_mae']:.6f}, test_mae={baseline['test_mae']:.6f}")

    from sdg_plugins.evaluator.augmentation_evaluator import AugmentationEvaluator
    eval_cfg = {
        "predictor_root": PREDICTOR_ROOT,
        "predictor_config": PREDICTOR_CONFIG,
        "synthetic_data": SYNTHETIC_CSV,
        "baseline_file": BASELINE_FILE,
        "metrics_file": os.path.join(RESULTS_DIR, "metrics_timegan.json"),
    }
    evaluator = AugmentationEvaluator(eval_cfg)
    metrics = evaluator.evaluate()

    log.info(f"\n{'='*60}")
    log.info(f"  TimeGAN Augmentation Results")
    log.info(f"{'='*60}")
    log.info(f"Baseline:  val_mae={baseline['val_mae']:.6f}, test_mae={baseline['test_mae']:.6f}")
    log.info(f"Augmented: val_mae={metrics['augmented_val_mae']:.6f}, test_mae={metrics['augmented_test_mae']:.6f}")
    log.info(f"Val  improvement: {metrics['val_improvement_pct']:+.2f}%")
    log.info(f"Test improvement: {metrics['test_improvement_pct']:+.2f}%")
    log.info(f"Verdict: {metrics['verdict']}")

    # Save
    summary = {"timegan": metrics}
    with open(os.path.join(RESULTS_DIR, "timegan_eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved to timegan_eval_summary.json")


if __name__ == "__main__":
    main()
