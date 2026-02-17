#!/usr/bin/env python3
"""Optimize hybrid generator hyperparameters using statistical metrics as fitness.
No GPU needed — runs in seconds per config."""
import sys, os, json, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon

from sdg_plugins.generator import regime_bootstrap_hybrid as hybrid

DATA = Path("/home/openclaw/predictor/examples/data_downsampled/phase_1")
OUT = Path(__file__).parent.parent / "results/optimization"
OUT.mkdir(parents=True, exist_ok=True)

# Load data
train_prices = np.concatenate([
    pd.read_csv(DATA / f"base_d{i}.csv")["typical_price"].values for i in [1, 2, 3]
])
d5 = pd.read_csv(DATA / "base_d5.csv")["typical_price"].values

def log_returns(p):
    return np.diff(np.log(np.clip(p, 1e-10, None)))

def autocorrelation(x, lag):
    if len(x) <= lag: return 0.0
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])

def vol_clustering(ret):
    return autocorrelation(np.abs(ret), 1)

def hurst(prices, max_lag=100):
    lags = range(2, min(max_lag, len(prices) // 4))
    rs = []
    for lag in lags:
        segs = [prices[i:i+lag] for i in range(0, len(prices) - lag, lag)]
        rv = []
        for s in segs:
            if len(s) < 2: continue
            r = np.diff(s); m = r.mean(); cd = np.cumsum(r - m)
            R = cd.max() - cd.min(); S = r.std()
            if S > 0: rv.append(R / S)
        if rv: rs.append(np.mean(rv))
    if len(rs) < 3: return 0.5
    return float(np.clip(np.polyfit(np.log(list(lags)[:len(rs)]), np.log(rs), 1)[0], 0, 1))

real_ret = log_returns(d5)
real_ac1 = autocorrelation(real_ret, 1)
real_ac5 = autocorrelation(real_ret, 5)
real_vc = vol_clustering(real_ret)
real_hurst = hurst(d5)

def score_config(n_regimes, block_size, smooth_weight, vol_scale, n_seeds=5):
    """Compute composite fitness for a config. Lower = better."""
    try:
        model = hybrid.fit(train_prices, n_regimes=n_regimes, block_size=block_size)
        # Override smooth_weight in generate
        old_code = hybrid.generate
    except: return 999, {}

    metrics = []
    for seed in range(n_seeds):
        synth = hybrid.generate(model, len(d5), seed=seed, initial_price=d5[0])
        sr = log_returns(synth)
        # Apply vol scaling
        if vol_scale != 1.0:
            sr = sr * vol_scale
            synth = d5[0] * np.exp(np.concatenate([[0], np.cumsum(sr)]))[:len(d5)]
            sr = log_returns(synth)

        std_r = sr.std() / real_ret.std()
        ac1_r = autocorrelation(sr, 1) / real_ac1 if abs(real_ac1) > 1e-6 else 0
        ac5_r = autocorrelation(sr, 5) / real_ac5 if abs(real_ac5) > 1e-6 else 0
        bins = np.linspace(min(real_ret.min(), sr.min()), max(real_ret.max(), sr.max()), 100)
        rh, _ = np.histogram(real_ret, bins=bins, density=True)
        sh, _ = np.histogram(sr, bins=bins, density=True)
        js = float(jensenshannon(rh + 1e-10, sh + 1e-10))
        wass = float(wasserstein_distance(real_ret, sr))
        ks = float(ks_2samp(real_ret, sr)[0])
        vc_r = vol_clustering(sr) / real_vc if abs(real_vc) > 1e-6 else 0
        h_diff = abs(hurst(synth) - real_hurst)

        metrics.append({
            "std_ratio": std_r, "ac1_ratio": ac1_r, "ac5_ratio": ac5_r,
            "js": js, "wass": wass, "ks": ks, "vc_ratio": vc_r, "hurst_diff": h_diff,
        })

    avg = {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}

    # Composite score: penalize deviation from ideal
    # Ratios: ideal=1.0, distances: ideal=0.0
    score = (
        3.0 * abs(avg["std_ratio"] - 1.0) +   # vol match (high weight)
        2.0 * abs(avg["ac1_ratio"] - 1.0) +    # short-term temporal
        1.5 * abs(avg["ac5_ratio"] - 1.0) +    # medium-term temporal
        2.0 * avg["js"] +                       # distributional
        500 * avg["wass"] +                      # distributional (scaled)
        1.0 * avg["ks"] +                       # distributional
        1.5 * abs(avg["vc_ratio"] - 1.0) +     # stylized fact
        2.0 * avg["hurst_diff"]                  # structural
    )
    return score, avg

# Grid search
configs = []
print(f"{'n_reg':>5} {'bs':>4} {'sw':>5} {'vs':>5} | {'score':>6} {'std':>5} {'ac1':>5} {'ac5':>5} {'js':>6} {'wass':>8} {'ks':>5} {'vc':>5} {'hurst':>5}")
print("-" * 95)

for n_reg in [3, 4, 5, 6]:
    for bs in [15, 24, 30, 48, 60]:
        for vs in [0.70, 0.75, 0.80, 0.85, 1.0]:
            sw = 0.3  # keep smooth_weight fixed
            score, avg = score_config(n_reg, bs, sw, vs, n_seeds=5)
            if score < 999:
                configs.append({
                    "n_regimes": n_reg, "block_size": bs,
                    "smooth_weight": sw, "vol_scale": vs,
                    "score": score, **avg
                })
                print(f"{n_reg:>5} {bs:>4} {sw:>5.2f} {vs:>5.2f} | {score:>6.3f} "
                      f"{avg['std_ratio']:>5.3f} {avg['ac1_ratio']:>5.3f} {avg['ac5_ratio']:>5.3f} "
                      f"{avg['js']:>6.4f} {avg['wass']:>8.6f} {avg['ks']:>5.3f} "
                      f"{avg['vc_ratio']:>5.3f} {avg['hurst_diff']:>5.3f}")

# Sort and save
configs.sort(key=lambda x: x["score"])
print(f"\n{'='*95}")
print(f"TOP 5 CONFIGS:")
for i, c in enumerate(configs[:5]):
    print(f"  #{i+1}: n_reg={c['n_regimes']}, bs={c['block_size']}, vs={c['vol_scale']:.2f} "
          f"→ score={c['score']:.3f}, std={c['std_ratio']:.3f}, ac1={c['ac1_ratio']:.3f}")

# Save to OLAP
df = pd.DataFrame(configs)
db_path = OUT / "hybrid_optimization.db"
conn = sqlite3.connect(db_path)
df.to_sql("configs", conn, if_exists="replace", index=False)
conn.close()
df.to_csv(OUT / "hybrid_optimization.csv", index=False)

with open(OUT / "best_config.json", "w") as f:
    json.dump(configs[0], f, indent=2)
print(f"\nBest config saved to {OUT / 'best_config.json'}")
print(f"OLAP: {db_path}")
