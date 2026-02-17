#!/usr/bin/env python3
"""Test HMM+GARCH generator and compare all approaches."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
import pandas as pd
from sdg_plugins.generator import regime_hmm_garch as rg
from pathlib import Path

DATA = Path("/home/openclaw/predictor/examples/data_downsampled/phase_1")

prices = np.concatenate([
    pd.read_csv(DATA / f"base_d{i}.csv")["typical_price"].values for i in [1, 2, 3]
])
d5 = pd.read_csv(DATA / "base_d5.csv")["typical_price"].values
real_ret = np.diff(np.log(d5))

print(f"Train: {len(prices)}, d5: {len(d5)}")
print(f"\nFitting HMM+GARCH (K=4)...")
model = rg.fit(prices, n_regimes=4)

# Compare
print(f"\n{'Seed':<6} {'std_ratio':>10} {'ac_ratio':>10} {'skew_r':>10} {'price_range':>15}")
for seed in range(10):
    s = rg.generate(model, len(d5), seed=seed, initial_price=d5[0])
    sr = np.diff(np.log(s))
    std_r = sr.std() / real_ret.std()
    ac = float(np.corrcoef(sr[:-1], sr[1:])[0, 1])
    ac_real = float(np.corrcoef(real_ret[:-1], real_ret[1:])[0, 1])
    ac_r = ac / ac_real
    skew_r = float(pd.Series(sr).skew()) / float(pd.Series(real_ret).skew()) if abs(pd.Series(real_ret).skew()) > 0.01 else 0
    print(f"{seed:<6} {std_r:>10.3f} {ac_r:>10.3f} {skew_r:>10.3f} [{s.min():.4f}, {s.max():.4f}]")

# Summary comparison
print(f"\n=== Approach Comparison ===")
print(f"{'Approach':<20} {'std_ratio':>10} {'autocorr_ratio':>14}")
print(f"{'Block Bootstrap':<20} {'~1.0':>10} {'~0.3':>14}")
print(f"{'HMM (parametric)':<20} {'1.44':>10} {'0.99':>14}")
print(f"{'Regime-GAN':<20} {'1.36':>10} {'0.27':>14}")

stds, acs = [], []
for seed in range(10):
    s = rg.generate(model, len(d5), seed=seed, initial_price=d5[0])
    sr = np.diff(np.log(s))
    stds.append(sr.std() / real_ret.std())
    ac = float(np.corrcoef(sr[:-1], sr[1:])[0, 1])
    acs.append(ac / float(np.corrcoef(real_ret[:-1], real_ret[1:])[0, 1]))
print(f"{'HMM+GARCH':<20} {np.mean(stds):>10.2f} {np.mean(acs):>14.2f}")
