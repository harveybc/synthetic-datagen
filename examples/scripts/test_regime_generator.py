#!/usr/bin/env python3
"""Test regime-conditional generator: fit on d1-d3, generate, compare to real d4-d6."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
import pandas as pd
from sdg_plugins.generator.regime_conditional import fit_regime_model, generate_synthetic, save_model
from pathlib import Path

DATA = Path("/home/openclaw/predictor/examples/data_downsampled/phase_1")
OUT = Path(__file__).parent.parent / "results/regime_conditional"
OUT.mkdir(parents=True, exist_ok=True)

# Load training data (d1-d3) for fitting
print("Loading data...")
train_prices = []
for d in ["base_d1.csv", "base_d2.csv", "base_d3.csv"]:
    df = pd.read_csv(DATA / d)
    train_prices.append(df["typical_price"].values)
train_all = np.concatenate(train_prices)
print(f"  Training data: {len(train_all)} samples from d1-d3")

# Load real test data for comparison
d5 = pd.read_csv(DATA / "base_d5.csv")["typical_price"].values
d6 = pd.read_csv(DATA / "base_d6.csv")["typical_price"].values
print(f"  Real d5: {len(d5)}, d6: {len(d6)}")

# Fit regime model with different K values
for n_regimes in [3, 4, 5]:
    print(f"\n=== {n_regimes} Regimes ===")
    model = fit_regime_model(train_all, n_regimes=n_regimes)
    save_model(model, str(OUT / f"regime_model_k{n_regimes}.json"))

    # Generate multiple synthetic samples and compare stats
    real_ret = np.diff(np.log(d5))
    synth_stats = []
    for seed in range(10):
        synth = generate_synthetic(model, len(d5), seed=seed, initial_price=d5[0])
        synth_ret = np.diff(np.log(synth))
        synth_stats.append({
            "seed": seed,
            "mean_ret": synth_ret.mean(),
            "std_ret": synth_ret.std(),
            "skew": float(pd.Series(synth_ret).skew()),
            "autocorr": float(np.corrcoef(synth_ret[:-1], synth_ret[1:])[0, 1]),
            "price_mean": synth.mean(),
            "price_std": synth.std(),
        })

    real_stats = {
        "mean_ret": real_ret.mean(), "std_ret": real_ret.std(),
        "skew": float(pd.Series(real_ret).skew()),
        "autocorr": float(np.corrcoef(real_ret[:-1], real_ret[1:])[0, 1]),
        "price_mean": d5.mean(), "price_std": d5.std(),
    }

    # Compare
    avg_synth = {k: np.mean([s[k] for s in synth_stats]) for k in synth_stats[0] if k != "seed"}
    print(f"  {'Stat':<12} {'Real':>12} {'Synth(avg)':>12} {'Ratio':>8}")
    for k in ["mean_ret", "std_ret", "skew", "autocorr"]:
        r, s = real_stats[k], avg_synth[k]
        ratio = s / r if abs(r) > 1e-10 else float('inf')
        print(f"  {k:<12} {r:>12.6f} {s:>12.6f} {ratio:>8.2f}")

print("\nDone.")
