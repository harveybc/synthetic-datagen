"""Train TimeGAN (Option B) and generate synthetic data."""
import sys, os, json, logging
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("timegan_train")

DATA_DIR = "/home/openclaw/predictor/examples/data_downsampled/phase_1"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "examples", "results")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "examples", "models", "timegan_4h")

TRAIN_DATA = [
    f"{DATA_DIR}/base_d1.csv",
    f"{DATA_DIR}/base_d2.csv",
    f"{DATA_DIR}/base_d3.csv",
]

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    from sdg_plugins.generator.timegan_generator import TimeGANGenerator

    gen = TimeGANGenerator({
        "train_data": TRAIN_DATA,
        "window_size": 48,
        "latent_dim": 24,
        "hidden_dim": 24,
        "n_layers": 3,
        "epochs_ae": 300,
        "epochs_sup": 300,
        "epochs_joint": 300,
        "batch_size": 64,
        "learning_rate": 5e-4,
        "use_returns": True,
        "save_model": SAVE_DIR,
    })

    log.info("Starting TimeGAN training...")
    save_path = gen.train(save_dir=SAVE_DIR)
    log.info(f"Training complete. Model saved to: {save_path}")

    # Generate synthetic data
    log.info("Generating 1575 synthetic samples...")
    df = gen.generate(seed=42, n_samples=1575)
    out_csv = os.path.join(RESULTS_DIR, "synthetic_timegan.csv")
    df.to_csv(out_csv, index=False)
    log.info(f"Saved to {out_csv}")

    # Quick stats
    prices = df["typical_price"].values
    log.info(f"Price range: {prices.min():.4f} - {prices.max():.4f}")
    log.info(f"Price mean: {prices.mean():.4f}, std: {prices.std():.4f}")
    log.info("TimeGAN training + generation complete! Run augmentation eval next.")

if __name__ == "__main__":
    main()
