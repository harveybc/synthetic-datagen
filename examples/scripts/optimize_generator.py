#!/usr/bin/env python3
"""Parameter sweep optimizer for hybrid generator. Uses subprocess per config for crash isolation."""
import sys, os, time, json, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
EVAL_SCRIPT = SCRIPT_DIR / "eval_single_config.py"
OUT = SCRIPT_DIR.parent / "results/optimization"
OUT.mkdir(parents=True, exist_ok=True)

N_SEEDS = 3
MAX_CONFIG_TIME = 60

# ── Parameter grid ───────────────────────────────────────────────
param_grid = {
    "n_regimes": [2, 3, 4, 5, 6, 7, 8],
    "block_size": [10, 15, 20, 25, 30, 40, 50, 60, 80],
    "smooth_weight": [0.0, 0.1, 0.2, 0.3, 0.5, 0.7],
    "min_block_length": [2, 3, 5, 8],
    "covariance_type": ["diag", "tied"],
    "vol_windows": [[6, 24, 48], [4, 12, 36], [8, 32, 64], [6, 16, 48]],
}

total_full = 1
for v in param_grid.values():
    total_full *= len(v)
print(f"Full grid: {total_full} combos. Using random sample of 300.", flush=True)

MAX_CONFIGS = 300
rng = np.random.RandomState(42)

def random_config():
    return {k: v[rng.randint(len(v))] for k, v in param_grid.items()}

configs = []
seen = set()
while len(configs) < MAX_CONFIGS:
    c = random_config()
    key = tuple(sorted((k, str(v)) for k, v in c.items()))
    if key not in seen:
        seen.add(key)
        configs.append(c)

# Default config first
default_config = {
    "n_regimes": 4, "block_size": 30, "smooth_weight": 0.3,
    "min_block_length": 3, "covariance_type": "diag",
    "vol_windows": [6, 24, 48],
}
configs.insert(0, default_config)

# ── Incremental CSV ──────────────────────────────────────────────
csv_path = OUT / "sweep_results_incremental.csv"
csv_header = "config_id,n_regimes,block_size,smooth_weight,min_block_length,covariance_type,vol_short,vol_mid,vol_long,composite_score,score_std,fit_time_s\n"
with open(csv_path, "w") as f:
    f.write(csv_header)

# ── Sweep ────────────────────────────────────────────────────────
results = []
start_time = time.time()
best_score = float('inf')

for i, cfg in enumerate(configs):
    t0 = time.time()
    vw = cfg["vol_windows"]
    
    try:
        result = subprocess.run(
            [sys.executable, str(EVAL_SCRIPT), json.dumps(cfg), str(N_SEEDS)],
            capture_output=True, text=True, timeout=MAX_CONFIG_TIME,
            cwd=str(SCRIPT_DIR.parent.parent)
        )
        if result.returncode != 0:
            err = result.stderr.strip().split('\n')[-1] if result.stderr else "unknown"
            print(f"  [{i+1}/{len(configs)}] FAILED: {err}", flush=True)
            continue
        
        data = json.loads(result.stdout.strip().split('\n')[-1])
        mean_score = data["score"]
        std_score = data["std"]
        details = data["details"]
        
    except subprocess.TimeoutExpired:
        print(f"  [{i+1}/{len(configs)}] TIMEOUT ({MAX_CONFIG_TIME}s)", flush=True)
        continue
    except Exception as e:
        print(f"  [{i+1}/{len(configs)}] ERROR: {e}", flush=True)
        continue
    
    elapsed = time.time() - t0
    row = {
        "config_id": i,
        "n_regimes": cfg["n_regimes"],
        "block_size": cfg["block_size"],
        "smooth_weight": cfg["smooth_weight"],
        "min_block_length": cfg["min_block_length"],
        "covariance_type": cfg["covariance_type"],
        "vol_short": vw[0],
        "vol_mid": vw[1],
        "vol_long": vw[2],
        "composite_score": mean_score,
        "score_std": std_score,
        "fit_time_s": elapsed,
    }
    row.update({f"detail_{k}": v for k, v in details.items()})
    results.append(row)
    
    # Incremental save
    with open(csv_path, "a") as f:
        f.write(f"{i},{cfg['n_regimes']},{cfg['block_size']},{cfg['smooth_weight']},"
                f"{cfg['min_block_length']},{cfg['covariance_type']},{vw[0]},{vw[1]},{vw[2]},"
                f"{mean_score},{std_score},{elapsed}\n")
    
    if mean_score < best_score:
        best_score = mean_score
        best_cfg = cfg.copy()
        best_cfg["score"] = mean_score
    
    is_default = (i == 0)
    tag = " [DEFAULT]" if is_default else ""
    total_elapsed = time.time() - start_time
    eta = total_elapsed / (i + 1) * (len(configs) - i - 1)
    print(f"  [{i+1}/{len(configs)}] score={mean_score:.4f}±{std_score:.4f} "
          f"K={cfg['n_regimes']} bs={cfg['block_size']} sw={cfg['smooth_weight']} "
          f"mbl={cfg['min_block_length']} cov={cfg['covariance_type']} "
          f"vw={vw} ({elapsed:.1f}s, ETA {eta:.0f}s){tag}", flush=True)

# ── Save final results ───────────────────────────────────────────
df = pd.DataFrame(results).sort_values("composite_score")
final_csv = OUT / "sweep_results.csv"
df.to_csv(final_csv, index=False)

db_path = OUT / "optimization_olap.db"
conn = sqlite3.connect(db_path)
df.to_sql("sweep_results", conn, if_exists="replace", index=False)
conn.commit()
conn.close()

# ── Report ───────────────────────────────────────────────────────
print(f"\n{'='*80}", flush=True)
print("TOP 10 CONFIGURATIONS", flush=True)
print("="*80, flush=True)
for _, row in df.head(10).iterrows():
    tag = " ★ DEFAULT" if row["config_id"] == 0 else ""
    print(f"  Score: {row['composite_score']:.4f}±{row['score_std']:.4f} | "
          f"K={int(row['n_regimes'])} bs={int(row['block_size'])} "
          f"sw={row['smooth_weight']:.1f} mbl={int(row['min_block_length'])} "
          f"cov={row['covariance_type']} "
          f"vw=({int(row['vol_short'])},{int(row['vol_mid'])},{int(row['vol_long'])}){tag}", flush=True)

default_row = df[df["config_id"] == 0]
best_row = df.iloc[0]
print(f"\n{'='*80}", flush=True)
if len(default_row):
    print(f"DEFAULT config score: {default_row.iloc[0]['composite_score']:.4f}", flush=True)
print(f"BEST config score:    {best_row['composite_score']:.4f}", flush=True)

# Save best config
json_path = OUT / "best_config.json"
with open(json_path, "w") as f:
    json.dump({
        "n_regimes": int(best_row["n_regimes"]),
        "block_size": int(best_row["block_size"]),
        "smooth_weight": float(best_row["smooth_weight"]),
        "min_block_length": int(best_row["min_block_length"]),
        "covariance_type": best_row["covariance_type"],
        "vol_short_window": int(best_row["vol_short"]),
        "vol_mid_window": int(best_row["vol_mid"]),
        "vol_long_window": int(best_row["vol_long"]),
        "composite_score": float(best_row["composite_score"]),
    }, f, indent=2)

print(f"\nSaved: {final_csv}, {db_path}, {json_path}", flush=True)
