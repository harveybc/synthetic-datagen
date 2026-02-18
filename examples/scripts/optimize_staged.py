#!/usr/bin/env python3
"""Multi-stage iterative optimizer for hybrid generator.

Like predictor's staged optimization: each stage refines parameters
focusing on the weakest metrics. Uses tournament selection + mutation.

Stage 1: Broad search (all params, equal weights)
Stage 2: Focus on AC structure (increase ac_lag5, ac_lag24, spectral weights)
Stage 3: Focus on distribution (js, ks, skew, kurt weights)
Stage 4: Fine-tune around best (small mutations)
Stage 5: Final polish (tiny mutations, more seeds for stability)
"""
import sys, os, time, json, subprocess, warnings, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
warnings.filterwarnings("ignore")
import logging
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
EVAL_SCRIPT = SCRIPT_DIR / "eval_single_config.py"
OUT = SCRIPT_DIR.parent / "results/optimization_staged"
OUT.mkdir(parents=True, exist_ok=True)

DATA = Path("/home/openclaw/predictor/examples/data_downsampled/phase_1")

# ── Load data ────────────────────────────────────────────────────
print("Loading data (d1-d4 train, d5 validation)...", flush=True)
train_prices = np.concatenate([
    pd.read_csv(DATA / f"base_d{i}.csv")["typical_price"].values for i in [1, 2, 3, 4]
])
d5 = pd.read_csv(DATA / "base_d5.csv")["typical_price"].values
print(f"  Train: {len(train_prices)}, d5 (val): {len(d5)}", flush=True)

# ── Parameter space ──────────────────────────────────────────────
PARAM_RANGES = {
    "n_regimes": (2, 10),       # int
    "block_size": (5, 120),     # int
    "smooth_weight": (0.0, 1.0),  # float
    "min_block_length": (1, 15),  # int
    "vol_short_window": (3, 12),  # int
    "vol_mid_window": (8, 48),    # int
    "vol_long_window": (24, 96),  # int
}
COVARIANCE_TYPES = ["diag", "tied"]

INT_PARAMS = {"n_regimes", "block_size", "min_block_length",
              "vol_short_window", "vol_mid_window", "vol_long_window"}

# ── Stage definitions ────────────────────────────────────────────
STAGES = [
    {
        "name": "Broad search",
        "pop_size": 40,
        "generations": 8,
        "n_seeds": 3,
        "mutation_scale": 0.3,   # fraction of range
        "elite_keep": 4,
        "tournament_size": 3,
        "weights": None,  # default equal weights
    },
    {
        "name": "AC structure focus",
        "pop_size": 30,
        "generations": 8,
        "n_seeds": 3,
        "mutation_scale": 0.2,
        "elite_keep": 4,
        "tournament_size": 3,
        "weights": {
            "js_divergence": 1.0, "ks_statistic": 1.0,
            "ac_lag1": 3.0, "ac_lag5": 5.0, "ac_lag24": 5.0,
            "hurst_diff": 2.0, "vol_cluster": 4.0,
            "spectral": 4.0, "skew_diff": 1.0, "kurt_diff": 1.0, "std_ratio": 1.0,
        },
    },
    {
        "name": "Distribution focus",
        "pop_size": 30,
        "generations": 6,
        "n_seeds": 3,
        "mutation_scale": 0.15,
        "elite_keep": 4,
        "tournament_size": 3,
        "weights": {
            "js_divergence": 5.0, "ks_statistic": 4.0,
            "ac_lag1": 2.0, "ac_lag5": 2.0, "ac_lag24": 2.0,
            "hurst_diff": 2.0, "vol_cluster": 2.0,
            "spectral": 2.0, "skew_diff": 4.0, "kurt_diff": 3.0, "std_ratio": 4.0,
        },
    },
    {
        "name": "Fine-tune",
        "pop_size": 25,
        "generations": 8,
        "n_seeds": 4,
        "mutation_scale": 0.08,
        "elite_keep": 5,
        "tournament_size": 3,
        "weights": None,  # back to balanced
    },
    {
        "name": "Final polish",
        "pop_size": 20,
        "generations": 6,
        "n_seeds": 5,
        "mutation_scale": 0.03,
        "elite_keep": 5,
        "tournament_size": 2,
        "weights": None,
    },
]


def random_config(rng):
    cfg = {}
    for p, (lo, hi) in PARAM_RANGES.items():
        if p in INT_PARAMS:
            cfg[p] = int(rng.randint(lo, hi + 1))
        else:
            cfg[p] = float(rng.uniform(lo, hi))
    cfg["covariance_type"] = rng.choice(COVARIANCE_TYPES)
    cfg["vol_windows"] = [cfg.pop("vol_short_window"), cfg.pop("vol_mid_window"), cfg.pop("vol_long_window")]
    # Ensure window ordering
    cfg["vol_windows"].sort()
    return cfg


def mutate(cfg, rng, scale):
    child = copy.deepcopy(cfg)
    # Extract vol windows back to individual params
    vw = child.pop("vol_windows", [6, 24, 48])
    child["vol_short_window"] = vw[0]
    child["vol_mid_window"] = vw[1]
    child["vol_long_window"] = vw[2]
    
    # Mutate 1-3 params
    n_mutations = rng.randint(1, 4)
    mutable = list(PARAM_RANGES.keys()) + ["covariance_type"]
    params_to_mutate = rng.choice(mutable, size=min(n_mutations, len(mutable)), replace=False)
    
    for p in params_to_mutate:
        if p == "covariance_type":
            child[p] = rng.choice(COVARIANCE_TYPES)
        elif p in INT_PARAMS:
            lo, hi = PARAM_RANGES[p]
            delta = max(1, int((hi - lo) * scale))
            child[p] = int(np.clip(child[p] + rng.randint(-delta, delta + 1), lo, hi))
        else:
            lo, hi = PARAM_RANGES[p]
            delta = (hi - lo) * scale
            child[p] = float(np.clip(child[p] + rng.normal(0, delta), lo, hi))
    
    # Re-pack vol windows
    child["vol_windows"] = sorted([child.pop("vol_short_window"),
                                     child.pop("vol_mid_window"),
                                     child.pop("vol_long_window")])
    return child


def crossover(parent1, parent2, rng):
    child = copy.deepcopy(parent1)
    # Extract windows
    for cfg in [child, parent2]:
        vw = cfg.get("vol_windows", [6, 24, 48])
        cfg["_vol_short"] = vw[0]
        cfg["_vol_mid"] = vw[1]
        cfg["_vol_long"] = vw[2]
    
    all_params = list(PARAM_RANGES.keys()) + ["covariance_type"]
    # Map vol_X_window to _vol_X
    param_map = {}
    for p in all_params:
        if p == "vol_short_window": param_map[p] = "_vol_short"
        elif p == "vol_mid_window": param_map[p] = "_vol_mid"
        elif p == "vol_long_window": param_map[p] = "_vol_long"
        else: param_map[p] = p
    
    for p in all_params:
        if rng.random() < 0.5:
            key = param_map[p]
            child_key = param_map[p] if param_map[p] in child else p
            child[child_key] = parent2.get(key, parent2.get(p))
    
    # Re-pack
    child["vol_windows"] = sorted([child.pop("_vol_short", 6),
                                     child.pop("_vol_mid", 24),
                                     child.pop("_vol_long", 48)])
    parent2.pop("_vol_short", None)
    parent2.pop("_vol_mid", None)
    parent2.pop("_vol_long", None)
    child.pop("_vol_short", None)
    child.pop("_vol_mid", None)
    child.pop("_vol_long", None)
    
    # Remove any PARAM_RANGES keys that shouldn't be in final config
    for k in ["vol_short_window", "vol_mid_window", "vol_long_window"]:
        child.pop(k, None)
    
    return child


def eval_config(cfg, n_seeds, weights=None):
    """Evaluate a config using subprocess."""
    eval_cfg = copy.deepcopy(cfg)
    if weights:
        eval_cfg["_weights"] = weights
    
    try:
        result = subprocess.run(
            [sys.executable, str(EVAL_SCRIPT), json.dumps(eval_cfg), str(n_seeds)],
            capture_output=True, text=True, timeout=90,
            cwd=str(SCRIPT_DIR.parent.parent)
        )
        if result.returncode != 0:
            return float('inf'), 0, {}
        data = json.loads(result.stdout.strip().split('\n')[-1])
        return data["score"], data["std"], data["details"]
    except:
        return float('inf'), 0, {}


def tournament_select(population, scores, rng, k=3):
    indices = rng.choice(len(population), size=k, replace=False)
    best_idx = indices[np.argmin([scores[i] for i in indices])]
    return population[best_idx]


# ── Main optimization loop ───────────────────────────────────────
rng = np.random.RandomState(42)
global_best_cfg = None
global_best_score = float('inf')
all_results = []

# Seed with previous sweep's top configs
prev_sweep = SCRIPT_DIR.parent / "results/optimization/sweep_results.csv"
seed_configs = []
if prev_sweep.exists():
    prev_df = pd.read_csv(prev_sweep).head(10)
    for _, row in prev_df.iterrows():
        seed_configs.append({
            "n_regimes": int(row["n_regimes"]),
            "block_size": int(row["block_size"]),
            "smooth_weight": float(row["smooth_weight"]),
            "min_block_length": int(row["min_block_length"]),
            "covariance_type": row["covariance_type"],
            "vol_windows": sorted([int(row["vol_short"]), int(row["vol_mid"]), int(row["vol_long"])]),
        })
    print(f"Seeded with {len(seed_configs)} configs from previous sweep", flush=True)

total_evals = 0
start_time = time.time()

for stage_idx, stage in enumerate(STAGES):
    stage_start = time.time()
    print(f"\n{'='*60}", flush=True)
    print(f"STAGE {stage_idx+1}/{len(STAGES)}: {stage['name']}", flush=True)
    print(f"  pop={stage['pop_size']}, gens={stage['generations']}, "
          f"seeds={stage['n_seeds']}, mutation={stage['mutation_scale']}", flush=True)
    print(f"{'='*60}", flush=True)
    
    weights = stage.get("weights")
    
    # Initialize population
    pop = []
    if stage_idx == 0:
        # First stage: seed configs + random
        pop.extend(seed_configs[:stage["pop_size"] // 2])
        while len(pop) < stage["pop_size"]:
            pop.append(random_config(rng))
    else:
        # Subsequent stages: carry over elite + mutate from global best
        pop.append(copy.deepcopy(global_best_cfg))
        for c in seed_configs[:stage["elite_keep"] - 1]:
            pop.append(copy.deepcopy(c))
        while len(pop) < stage["pop_size"]:
            pop.append(mutate(global_best_cfg, rng, stage["mutation_scale"]))
    
    # Evaluate initial population
    scores = []
    for i, cfg in enumerate(pop):
        score, std, details = eval_config(cfg, stage["n_seeds"], weights)
        scores.append(score)
        total_evals += 1
        if score < global_best_score:
            global_best_score = score
            global_best_cfg = copy.deepcopy(cfg)
            print(f"  ★ New global best: {score:.4f}", flush=True)
    
    print(f"  Gen 0: best={min(scores):.4f}, mean={np.mean([s for s in scores if s < float('inf')]):.4f}", flush=True)
    
    # Evolutionary loop
    for gen in range(stage["generations"]):
        new_pop = []
        new_scores = []
        
        # Keep elite
        elite_idx = np.argsort(scores)[:stage["elite_keep"]]
        for idx in elite_idx:
            new_pop.append(pop[idx])
            new_scores.append(scores[idx])
        
        # Generate offspring
        while len(new_pop) < stage["pop_size"]:
            if rng.random() < 0.7:
                # Crossover + mutate
                p1 = tournament_select(pop, scores, rng, stage["tournament_size"])
                p2 = tournament_select(pop, scores, rng, stage["tournament_size"])
                child = crossover(p1, p2, rng)
                child = mutate(child, rng, stage["mutation_scale"])
            else:
                # Pure mutation from elite
                parent = pop[rng.choice(elite_idx)]
                child = mutate(parent, rng, stage["mutation_scale"])
            
            score, std, details = eval_config(child, stage["n_seeds"], weights)
            total_evals += 1
            new_pop.append(child)
            new_scores.append(score)
            
            if score < global_best_score:
                global_best_score = score
                global_best_cfg = copy.deepcopy(child)
                print(f"  ★ New global best: {score:.4f} (gen {gen+1})", flush=True)
        
        pop = new_pop
        scores = new_scores
        valid_scores = [s for s in scores if s < float('inf')]
        if valid_scores:
            print(f"  Gen {gen+1}: best={min(valid_scores):.4f}, "
                  f"mean={np.mean(valid_scores):.4f} ({total_evals} evals total)", flush=True)
    
    # Update seed configs for next stage
    best_idx = np.argsort(scores)[:min(10, len(scores))]
    seed_configs = [pop[i] for i in best_idx if scores[i] < float('inf')]
    
    stage_time = time.time() - stage_start
    print(f"  Stage {stage_idx+1} done in {stage_time:.0f}s, global best: {global_best_score:.4f}", flush=True)
    
    # Save intermediate results
    with open(OUT / f"stage_{stage_idx+1}_best.json", "w") as f:
        json.dump({"config": global_best_cfg, "score": global_best_score, "stage": stage_idx + 1}, f, indent=2)

# ── Final evaluation with more seeds ─────────────────────────────
print(f"\n{'='*60}", flush=True)
print("FINAL EVALUATION (10 seeds)", flush=True)
print(f"{'='*60}", flush=True)

final_score, final_std, final_details = eval_config(global_best_cfg, n_seeds=10)
print(f"  Score: {final_score:.4f}±{final_std:.4f}", flush=True)
print(f"  Details:", flush=True)
for k, v in final_details.items():
    print(f"    {k}: {v:.4f}", flush=True)

# ── Save ─────────────────────────────────────────────────────────
result = {
    "best_config": global_best_cfg,
    "final_score": final_score,
    "final_std": final_std,
    "final_details": final_details,
    "total_evals": total_evals,
    "total_time_s": time.time() - start_time,
}
with open(OUT / "final_result.json", "w") as f:
    json.dump(result, f, indent=2)

# Compare with previous best
prev_best = SCRIPT_DIR.parent / "results/optimization/best_config.json"
if prev_best.exists():
    with open(prev_best) as f:
        prev = json.load(f)
    print(f"\n  Previous best (random sweep): {prev['composite_score']:.4f}", flush=True)
    print(f"  New best (staged GA):         {final_score:.4f}", flush=True)
    improvement = (prev['composite_score'] - final_score) / prev['composite_score'] * 100
    print(f"  Improvement: {improvement:.1f}%", flush=True)

print(f"\n  Total evaluations: {total_evals}", flush=True)
print(f"  Total time: {time.time() - start_time:.0f}s", flush=True)
print(f"  Results: {OUT}/", flush=True)
