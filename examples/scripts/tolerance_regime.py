#!/usr/bin/env python3
"""Tolerance test: train MIMO on real d4, eval on real d5/d6 vs regime-synthetic d5/d6.

Measures: how different is MAE(real_val) vs MAE(synth_val)?
If close → low tolerance needed → regime-conditional works for DOIN verification.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, "/home/openclaw/predictor/app")
import numpy as np
import pandas as pd
from pathlib import Path

os.environ["PREDICTOR_QUIET"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sdg_plugins.generator.regime_conditional import fit_regime_model, generate_synthetic

DATA = Path("/home/openclaw/predictor/examples/data_downsampled/phase_1")
OUT = Path(__file__).parent.parent / "results/regime_conditional"
OUT.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading data...")
train_prices = []
for d in ["base_d1.csv", "base_d2.csv", "base_d3.csv"]:
    train_prices.append(pd.read_csv(DATA / d)["typical_price"].values)
train_all = np.concatenate(train_prices)

d4_norm = pd.read_csv(DATA / "normalized_d4.csv")
d5_norm = pd.read_csv(DATA / "normalized_d5.csv")
d6_norm = pd.read_csv(DATA / "normalized_d6.csv")
d5_base = pd.read_csv(DATA / "base_d5.csv")
d6_base = pd.read_csv(DATA / "base_d6.csv")
norm_cfg = json.load(open(DATA / "normalization_config_b.json"))

print("Fitting regime model (K=4)...")
model = fit_regime_model(train_all, n_regimes=4)

# Generate synthetic versions of d5 and d6 with multiple seeds
n_seeds = 5
results = []

for seed in range(n_seeds):
    print(f"\n[Seed {seed}] Generating synthetic d5/d6...")
    synth_d5 = generate_synthetic(model, len(d5_base), seed=seed*100, initial_price=d5_base["typical_price"].iloc[0])
    synth_d6 = generate_synthetic(model, len(d6_base), seed=seed*100+1, initial_price=d6_base["typical_price"].iloc[0])

    # Normalize synthetic data using same normalization as real
    col = "typical_price"
    mean_val = norm_cfg.get(f"{col}_mean", norm_cfg.get("mean", 0))
    std_val = norm_cfg.get(f"{col}_std", norm_cfg.get("std", 1))

    synth_d5_norm = (synth_d5 - mean_val) / std_val
    synth_d6_norm = (synth_d6 - mean_val) / std_val

    # Create DataFrames matching expected format
    synth_d5_df = pd.DataFrame({"typical_price": synth_d5_norm})
    synth_d6_df = pd.DataFrame({"typical_price": synth_d6_norm})

    # Save for predictor eval
    synth_d5_path = OUT / f"synth_d5_seed{seed}.csv"
    synth_d6_path = OUT / f"synth_d6_seed{seed}.csv"
    synth_d5_df.to_csv(synth_d5_path, index=False)
    synth_d6_df.to_csv(synth_d6_path, index=False)

    # Compare distributions
    real_ret = np.diff(np.log(d5_base["typical_price"].values))
    synth_ret = np.diff(np.log(synth_d5))
    print(f"  Real d5 ret: μ={real_ret.mean():.6f} σ={real_ret.std():.6f}")
    print(f"  Synth d5 ret: μ={synth_ret.mean():.6f} σ={synth_ret.std():.6f}")
    print(f"  Synth d5 price range: [{synth_d5.min():.4f}, {synth_d5.max():.4f}]")
    print(f"  Real d5 price range: [{d5_base['typical_price'].min():.4f}, {d5_base['typical_price'].max():.4f}]")

    results.append({
        "seed": seed,
        "synth_d5_path": str(synth_d5_path),
        "synth_d6_path": str(synth_d6_path),
        "real_d5_ret_std": float(real_ret.std()),
        "synth_d5_ret_std": float(synth_ret.std()),
        "std_ratio": float(synth_ret.std() / real_ret.std()),
    })

# Save results summary
with open(OUT / "tolerance_regime_prep.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nGenerated {n_seeds} synthetic d5/d6 pairs. Ready for predictor eval.")
print("Next: train MIMO on real d4, eval on both real and synthetic d5/d6.")
