#!/usr/bin/env python3
"""Generator v3: Address fundamental weaknesses of block bootstrap.

Key insight: Block bootstrap destroys AC at block boundaries.
Solutions to try:
1. MUCH larger blocks (128-512) with fewer regime transitions
2. Regime-aware overlap-add (crossfade blocks instead of hard stitch)
3. AR(1) bridge between blocks (generate short transition using AR model)
4. Full-sequence regime generation (generate entire regime segments, don't sub-sample)

This script tests architectural variants, not just hyperparameters.
"""
import sys, os, json, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
warnings.filterwarnings("ignore")
import logging
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from pathlib import Path
from sdg_plugins.generator import regime_bootstrap_hybrid as hybrid
from sdg_plugins.evaluator.composite_metric import composite_score, composite_score_multi_seed

DATA = Path("/home/openclaw/predictor/examples/data_downsampled/phase_1")
OUT = Path(__file__).parent.parent / "results/optimization_v3"
OUT.mkdir(parents=True, exist_ok=True)

print("Loading data...", flush=True)
train_prices = np.concatenate([
    pd.read_csv(DATA / f"base_d{i}.csv")["typical_price"].values for i in [1, 2, 3, 4]
])
d5 = pd.read_csv(DATA / "base_d5.csv")["typical_price"].values
real_ret = np.diff(np.log(np.clip(d5, 1e-10, None)))

print(f"Real d5: AC1={np.corrcoef(real_ret[:-1], real_ret[1:])[0,1]:.4f}, "
      f"AC5={np.corrcoef(real_ret[:-5], real_ret[5:])[0,1]:.4f}, "
      f"AC24={np.corrcoef(real_ret[:-24], real_ret[24:])[0,1]:.4f}", flush=True)


def eval_generator(gen_fn, name, n_seeds=10):
    """Evaluate a generator function."""
    scores = []
    all_details = []
    for seed in range(n_seeds):
        try:
            synth = gen_fn(seed)
            synth = synth[:len(d5)]
            s, d = composite_score(d5, synth)
            scores.append(s)
            all_details.append(d)
        except Exception as e:
            print(f"  {name} seed {seed} failed: {e}", flush=True)
    
    if not scores:
        return None
    
    mean_details = {k: np.mean([d[k] for d in all_details]) for k in all_details[0]}
    result = {
        "name": name,
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "details": mean_details,
    }
    print(f"  {name}: {result['score_mean']:.4f}±{result['score_std']:.4f} | "
          f"AC5={mean_details['ac_lag5']:.3f} AC24={mean_details['ac_lag24']:.3f} "
          f"VC={mean_details['vol_cluster']:.3f} JS={mean_details['js_divergence']:.3f}", flush=True)
    return result


# ── Variant 1: Very large blocks ────────────────────────────────
print("\n=== Variant 1: Very large blocks ===", flush=True)
results = []
for bs in [64, 96, 128, 192, 256, 384, 512]:
    model = hybrid.fit(train_prices, n_regimes=4, block_size=bs,
                       smooth_weight=0.2, min_block_length=1,
                       covariance_type="diag", vol_short_window=8,
                       vol_mid_window=24, vol_long_window=48, quiet=True)
    def gf(seed, m=model): return hybrid.generate(m, len(d5), seed=seed, initial_price=d5[0])
    r = eval_generator(gf, f"bs={bs}")
    if r: results.append(r)


# ── Variant 2: Crossfade stitching ──────────────────────────────
print("\n=== Variant 2: Crossfade stitching ===", flush=True)

def generate_crossfade(model, n_steps, seed=0, initial_price=1.0, crossfade_len=12):
    """Generate with crossfade between blocks instead of hard stitch."""
    rng = np.random.RandomState(seed)
    trans = model["transition_matrix"]
    n_regimes = model["n_regimes"]
    regime_blocks = model["regime_blocks"]
    block_size = model["block_size"]
    
    raw_blocks = []
    regime = rng.choice(n_regimes, p=model["stationary_dist"])
    total_len = 0
    
    while total_len < n_steps + block_size * 2:
        blocks = regime_blocks[regime]
        if not blocks:
            rets = model["regime_returns"][regime]
            sampled = rng.choice(rets, size=max(3, block_size), replace=True)
        else:
            block = blocks[rng.randint(len(blocks))]
            if len(block) > block_size:
                start = rng.randint(0, max(1, len(block) - block_size))
                sampled = block[start:start + block_size]
            else:
                sampled = block
        raw_blocks.append(sampled)
        total_len += len(sampled)
        regime = rng.choice(n_regimes, p=trans[regime])
    
    # Crossfade stitch
    log_returns = list(raw_blocks[0])
    for i in range(1, len(raw_blocks)):
        block = raw_blocks[i]
        if len(block) < crossfade_len * 2 or len(log_returns) < crossfade_len:
            log_returns.extend(block)
            continue
        
        # Crossfade: linear blend of tail of previous with head of new block
        overlap = min(crossfade_len, len(block), len(log_returns))
        for j in range(overlap):
            alpha = j / overlap  # 0→1
            log_returns[-(overlap - j)] = (1 - alpha) * log_returns[-(overlap - j)] + alpha * block[j]
        log_returns.extend(block[overlap:])
    
    log_returns = np.array(log_returns[:n_steps])
    log_prices = np.log(initial_price) + np.cumsum(log_returns)
    prices = np.exp(log_prices)
    return np.concatenate([[initial_price], prices])[:n_steps]

for cf_len in [6, 12, 24, 48]:
    model = hybrid.fit(train_prices, n_regimes=4, block_size=96,
                       smooth_weight=0.0, min_block_length=1,
                       covariance_type="diag", vol_short_window=8,
                       vol_mid_window=24, vol_long_window=48, quiet=True)
    def gf(seed, m=model, cfl=cf_len): return generate_crossfade(m, len(d5), seed=seed, initial_price=d5[0], crossfade_len=cfl)
    r = eval_generator(gf, f"crossfade={cf_len}")
    if r: results.append(r)


# ── Variant 3: AR(1) bridge ─────────────────────────────────────
print("\n=== Variant 3: AR(1) bridge between blocks ===", flush=True)

def generate_ar_bridge(model, n_steps, seed=0, initial_price=1.0, bridge_len=6):
    """Generate with AR(1) transition between blocks."""
    rng = np.random.RandomState(seed)
    trans = model["transition_matrix"]
    n_regimes = model["n_regimes"]
    regime_blocks = model["regime_blocks"]
    block_size = model["block_size"]
    
    # Compute AR(1) coef from training returns
    train_ret = np.diff(np.log(np.clip(train_prices, 1e-10, None)))
    ar1_coef = np.corrcoef(train_ret[:-1], train_ret[1:])[0, 1]
    noise_std = np.std(train_ret) * np.sqrt(1 - ar1_coef**2)
    
    raw_blocks = []
    regime = rng.choice(n_regimes, p=model["stationary_dist"])
    total_len = 0
    
    while total_len < n_steps + block_size * 2:
        blocks = regime_blocks[regime]
        if not blocks:
            rets = model["regime_returns"][regime]
            sampled = rng.choice(rets, size=max(3, block_size), replace=True)
        else:
            block = blocks[rng.randint(len(blocks))]
            if len(block) > block_size:
                start = rng.randint(0, max(1, len(block) - block_size))
                sampled = block[start:start + block_size]
            else:
                sampled = block
        raw_blocks.append(sampled)
        total_len += len(sampled)
        regime = rng.choice(n_regimes, p=trans[regime])
    
    # Stitch with AR(1) bridge
    log_returns = list(raw_blocks[0])
    for i in range(1, len(raw_blocks)):
        block = raw_blocks[i]
        # Generate AR(1) bridge from last value to first value of next block
        if len(log_returns) > 0 and len(block) > 0 and bridge_len > 0:
            start_val = log_returns[-1]
            end_val = block[0]
            bridge = []
            val = start_val
            for j in range(bridge_len):
                target = start_val + (end_val - start_val) * (j + 1) / (bridge_len + 1)
                val = ar1_coef * val + (1 - ar1_coef) * target + rng.normal(0, noise_std * 0.3)
                bridge.append(val)
            log_returns.extend(bridge)
        log_returns.extend(block)
    
    log_returns = np.array(log_returns[:n_steps])
    log_prices = np.log(initial_price) + np.cumsum(log_returns)
    prices = np.exp(log_prices)
    return np.concatenate([[initial_price], prices])[:n_steps]

for bl in [3, 6, 12, 24]:
    model = hybrid.fit(train_prices, n_regimes=4, block_size=96,
                       smooth_weight=0.0, min_block_length=1,
                       covariance_type="diag", vol_short_window=8,
                       vol_mid_window=24, vol_long_window=48, quiet=True)
    def gf(seed, m=model, b=bl): return generate_ar_bridge(m, len(d5), seed=seed, initial_price=d5[0], bridge_len=b)
    r = eval_generator(gf, f"ar_bridge={bl}")
    if r: results.append(r)


# ── Variant 4: Full regime segments (no sub-sampling) ───────────
print("\n=== Variant 4: Full regime segments ===", flush=True)

def generate_full_segments(model, n_steps, seed=0, initial_price=1.0):
    """Use entire regime blocks without sub-sampling. More realistic regime durations."""
    rng = np.random.RandomState(seed)
    trans = model["transition_matrix"]
    n_regimes = model["n_regimes"]
    regime_blocks = model["regime_blocks"]
    smooth_weight = model.get("smooth_weight", 0.3)
    
    log_returns = []
    regime = rng.choice(n_regimes, p=model["stationary_dist"])
    
    while len(log_returns) < n_steps + 100:
        blocks = regime_blocks[regime]
        if not blocks:
            rets = model["regime_returns"][regime]
            sampled = rng.choice(rets, size=max(10, 30), replace=True)
        else:
            # Use FULL block (no sub-sampling)
            block = blocks[rng.randint(len(blocks))]
            sampled = block
        
        # Smooth boundary
        if log_returns and len(sampled) > 0:
            sampled = list(sampled)
            sampled[0] = (1 - smooth_weight) * sampled[0] + smooth_weight * log_returns[-1]
        
        log_returns.extend(sampled)
        regime = rng.choice(n_regimes, p=trans[regime])
    
    log_returns = np.array(log_returns[:n_steps])
    log_prices = np.log(initial_price) + np.cumsum(log_returns)
    prices = np.exp(log_prices)
    return np.concatenate([[initial_price], prices])[:n_steps]

for nr in [3, 4, 5, 6]:
    model = hybrid.fit(train_prices, n_regimes=nr, block_size=9999,  # irrelevant for full segments
                       smooth_weight=0.3, min_block_length=1,
                       covariance_type="diag", vol_short_window=8,
                       vol_mid_window=24, vol_long_window=48, quiet=True)
    def gf(seed, m=model): return generate_full_segments(m, len(d5), seed=seed, initial_price=d5[0])
    r = eval_generator(gf, f"full_seg_K={nr}")
    if r: results.append(r)


# ── Variant 5: Crossfade + large blocks + full segments combined ─
print("\n=== Variant 5: Best combo ===", flush=True)
# Take best performing approach and combine with best hyperparams

# Sort by score
results.sort(key=lambda x: x["score_mean"])

print(f"\n{'='*70}", flush=True)
print("RANKINGS", flush=True)
print(f"{'='*70}", flush=True)
for i, r in enumerate(results):
    print(f"  {i+1}. {r['name']:25s} score={r['score_mean']:.4f}±{r['score_std']:.4f} | "
          f"AC5={r['details']['ac_lag5']:.3f} AC24={r['details']['ac_lag24']:.3f} "
          f"VC={r['details']['vol_cluster']:.3f}", flush=True)

# Save
with open(OUT / "variant_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nBest: {results[0]['name']} = {results[0]['score_mean']:.4f}", flush=True)
print(f"Previous best: 0.1984", flush=True)
print(f"Saved to {OUT}/", flush=True)
