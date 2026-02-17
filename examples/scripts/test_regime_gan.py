#!/usr/bin/env python3
"""Test regime-GAN generator: fit on d1-d3, generate, compare stats to real."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
import pandas as pd
from sdg_plugins.generator.regime_gan import fit, generate
from pathlib import Path

DATA = Path("/home/openclaw/predictor/examples/data_downsampled/phase_1")
OUT = Path(__file__).parent.parent / "results/regime_gan"
OUT.mkdir(parents=True, exist_ok=True)

# Load training data
print("Loading d1-d3...")
prices = np.concatenate([
    pd.read_csv(DATA / f"base_d{i}.csv")["typical_price"].values for i in [1, 2, 3]
])
d5 = pd.read_csv(DATA / "base_d5.csv")["typical_price"].values
print(f"  Train: {len(prices)}, Real d5: {len(d5)}")

# Fit
print("\nFitting regime-GAN (K=4, seq=64, 300 epochs)...")
device = "cpu"  # CPU is fine for small GANs
model = fit(prices, n_regimes=4, seq_len=64, z_dim=8, gan_epochs=300, device=device)
model.save(str(OUT / "model_k4"))

# Generate & compare
real_ret = np.diff(np.log(d5))
print(f"\n{'Metric':<12} {'Real':>10} {'Synth(avg)':>10} {'Ratio':>8}")
synth_stds, synth_acs = [], []
for seed in range(10):
    s = generate(model, len(d5), seed=seed, initial_price=d5[0], device=device)
    sr = np.diff(np.log(s))
    synth_stds.append(sr.std())
    synth_acs.append(float(np.corrcoef(sr[:-1], sr[1:])[0, 1]) if len(sr) > 1 else 0)

print(f"{'std_ret':<12} {real_ret.std():>10.6f} {np.mean(synth_stds):>10.6f} {np.mean(synth_stds)/real_ret.std():>8.2f}")
print(f"{'autocorr':<12} {float(np.corrcoef(real_ret[:-1], real_ret[1:])[0,1]):>10.6f} {np.mean(synth_acs):>10.6f} {np.mean(synth_acs)/float(np.corrcoef(real_ret[:-1], real_ret[1:])[0,1]):>8.2f}")
print("Done.")
