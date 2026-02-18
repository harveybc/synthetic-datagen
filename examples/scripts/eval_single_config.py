#!/usr/bin/env python3
"""Evaluate a single generator config. Called by optimize_generator.py."""
import sys, os, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
warnings.filterwarnings("ignore")
import logging
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from sdg_plugins.generator import regime_bootstrap_hybrid as hybrid
from sdg_plugins.evaluator.composite_metric import composite_score_multi_seed

cfg = json.loads(sys.argv[1])
n_seeds = int(sys.argv[2])

DATA = "/home/openclaw/predictor/examples/data_downsampled/phase_1"
train = np.concatenate([pd.read_csv(f"{DATA}/base_d{i}.csv")["typical_price"].values for i in [1,2,3,4]])
d5 = pd.read_csv(f"{DATA}/base_d5.csv")["typical_price"].values

vw = cfg["vol_windows"]
model = hybrid.fit(train, n_regimes=cfg["n_regimes"], block_size=cfg["block_size"],
    smooth_weight=cfg["smooth_weight"], min_block_length=cfg["min_block_length"],
    covariance_type=cfg["covariance_type"], vol_short_window=vw[0],
    vol_mid_window=vw[1], vol_long_window=vw[2], quiet=True)

def gen_fn(seed, m=model):
    return hybrid.generate(m, len(d5), seed=seed, initial_price=d5[0])

weights = cfg.pop("_weights", None)
kwargs = {"weights": weights} if weights else {}
mean_score, std_score, details = composite_score_multi_seed(d5, gen_fn, n_seeds=n_seeds, **kwargs)
print(json.dumps({"score": mean_score, "std": std_score, "details": details}))
