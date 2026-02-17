#!/usr/bin/env python3
"""Comprehensive evaluation of all synthetic generators using SOTA metrics.

Metrics from: CLSGAN (2023), TimeGAN (2019), TSBenchmark, finance literature.

Categories:
1. DISTRIBUTIONAL FIDELITY — do synth returns match real distribution?
   - Mean, Std, Skewness, Kurtosis (moments)
   - KS test statistic (Kolmogorov-Smirnov)
   - Jensen-Shannon divergence
   - Wasserstein distance

2. TEMPORAL FIDELITY — do synth dynamics match real dynamics?
   - Autocorrelation (lag 1, 5, 10, 24)
   - Partial autocorrelation (lag 1)
   - Spectral density (power spectrum distance)
   - Volatility clustering (GARCH residual autocorrelation)

3. STYLIZED FACTS — does synth exhibit known financial properties?
   - Fat tails (kurtosis > 3)
   - Volatility clustering (|returns| autocorrelation)
   - Leverage effect (negative correlation between returns and future vol)
   - Mean reversion tendency

4. STRUCTURAL FIDELITY
   - Max drawdown similarity
   - Hurst exponent (long-range dependence)
   - Price range ratio (synth vs real)

All results → SQLite OLAP for Metabase.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from scipy import stats as sp_stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, ks_2samp

# Generators
from sdg_plugins.generator import regime_hmm_garch as hmm_garch
from sdg_plugins.generator.regime_conditional import fit_regime_model, generate_synthetic
from sdg_plugins.generator.block_bootstrap_generator import BlockBootstrapGenerator
from sdg_plugins.generator import regime_bootstrap_hybrid as hybrid

DATA = Path("/home/openclaw/predictor/examples/data_downsampled/phase_1")
OUT = Path(__file__).parent.parent / "results/evaluation"
OUT.mkdir(parents=True, exist_ok=True)

# ── Metric Functions ──────────────────────────────────────────────

def log_returns(prices):
    return np.diff(np.log(np.clip(prices, 1e-10, None)))

def autocorrelation(x, lag):
    if len(x) <= lag:
        return 0.0
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])

def spectral_distance(real_ret, synth_ret):
    """Distance between power spectra (lower = better)."""
    n = min(len(real_ret), len(synth_ret))
    real_psd = np.abs(np.fft.fft(real_ret[:n]))**2
    synth_psd = np.abs(np.fft.fft(synth_ret[:n]))**2
    # Normalize
    real_psd = real_psd / real_psd.sum()
    synth_psd = synth_psd / synth_psd.sum()
    return float(np.sqrt(np.mean((real_psd - synth_psd)**2)))

def volatility_clustering(ret, lag=1):
    """Autocorrelation of absolute returns (measures vol clustering)."""
    abs_ret = np.abs(ret)
    return autocorrelation(abs_ret, lag)

def leverage_effect(ret, lag=1):
    """Correlation between returns and future volatility."""
    if len(ret) <= lag:
        return 0.0
    abs_future = np.abs(ret[lag:])
    return float(np.corrcoef(ret[:-lag], abs_future)[0, 1])

def hurst_exponent(prices, max_lag=100):
    """Simplified R/S Hurst exponent."""
    lags = range(2, min(max_lag, len(prices) // 4))
    rs = []
    for lag in lags:
        segments = [prices[i:i+lag] for i in range(0, len(prices) - lag, lag)]
        rs_vals = []
        for seg in segments:
            if len(seg) < 2:
                continue
            ret = np.diff(seg)
            mean_ret = ret.mean()
            cum_dev = np.cumsum(ret - mean_ret)
            R = cum_dev.max() - cum_dev.min()
            S = ret.std()
            if S > 0:
                rs_vals.append(R / S)
        if rs_vals:
            rs.append(np.mean(rs_vals))
    if len(rs) < 3:
        return 0.5
    log_lags = np.log(list(lags)[:len(rs)])
    log_rs = np.log(rs)
    H = np.polyfit(log_lags, log_rs, 1)[0]
    return float(np.clip(H, 0, 1))

def max_drawdown(prices):
    peak = np.maximum.accumulate(prices)
    dd = (prices - peak) / peak
    return float(dd.min())

def compute_all_metrics(real_prices, synth_prices, label=""):
    """Compute all metrics comparing synth to real."""
    real_ret = log_returns(real_prices)
    synth_ret = log_returns(synth_prices)

    # Distributional
    ks_stat, ks_pval = ks_2samp(real_ret, synth_ret)
    
    # Histogram-based JS divergence
    bins = np.linspace(min(real_ret.min(), synth_ret.min()),
                       max(real_ret.max(), synth_ret.max()), 100)
    real_hist, _ = np.histogram(real_ret, bins=bins, density=True)
    synth_hist, _ = np.histogram(synth_ret, bins=bins, density=True)
    real_hist = real_hist + 1e-10
    synth_hist = synth_hist + 1e-10
    js_div = float(jensenshannon(real_hist, synth_hist))
    wass = float(wasserstein_distance(real_ret, synth_ret))

    m = {
        "label": label,
        # Moments (ratio synth/real, 1.0 = perfect)
        "mean_ratio": float(synth_ret.mean() / real_ret.mean()) if abs(real_ret.mean()) > 1e-10 else 0,
        "std_ratio": float(synth_ret.std() / real_ret.std()),
        "skew_real": float(sp_stats.skew(real_ret)),
        "skew_synth": float(sp_stats.skew(synth_ret)),
        "skew_diff": float(abs(sp_stats.skew(synth_ret) - sp_stats.skew(real_ret))),
        "kurt_real": float(sp_stats.kurtosis(real_ret, fisher=False)),
        "kurt_synth": float(sp_stats.kurtosis(synth_ret, fisher=False)),
        "kurt_diff": float(abs(sp_stats.kurtosis(synth_ret, fisher=False) - sp_stats.kurtosis(real_ret, fisher=False))),
        # Distributional distance
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pval),
        "js_divergence": float(js_div),
        "wasserstein": wass,
        # Temporal
        "ac_lag1_real": autocorrelation(real_ret, 1),
        "ac_lag1_synth": autocorrelation(synth_ret, 1),
        "ac_lag1_ratio": autocorrelation(synth_ret, 1) / autocorrelation(real_ret, 1) if abs(autocorrelation(real_ret, 1)) > 1e-6 else 0,
        "ac_lag5_ratio": autocorrelation(synth_ret, 5) / autocorrelation(real_ret, 5) if abs(autocorrelation(real_ret, 5)) > 1e-6 else 0,
        "ac_lag10_ratio": autocorrelation(synth_ret, 10) / autocorrelation(real_ret, 10) if abs(autocorrelation(real_ret, 10)) > 1e-6 else 0,
        "ac_lag24_ratio": autocorrelation(synth_ret, 24) / autocorrelation(real_ret, 24) if abs(autocorrelation(real_ret, 24)) > 1e-6 else 0,
        "spectral_distance": spectral_distance(real_ret, synth_ret),
        # Stylized facts
        "vol_cluster_real": volatility_clustering(real_ret),
        "vol_cluster_synth": volatility_clustering(synth_ret),
        "vol_cluster_ratio": volatility_clustering(synth_ret) / volatility_clustering(real_ret) if abs(volatility_clustering(real_ret)) > 1e-6 else 0,
        "leverage_real": leverage_effect(real_ret),
        "leverage_synth": leverage_effect(synth_ret),
        # Structural
        "hurst_real": hurst_exponent(real_prices),
        "hurst_synth": hurst_exponent(synth_prices),
        "hurst_diff": abs(hurst_exponent(synth_prices) - hurst_exponent(real_prices)),
        "max_dd_real": max_drawdown(real_prices),
        "max_dd_synth": max_drawdown(synth_prices),
        "price_range_ratio": float((synth_prices.max() - synth_prices.min()) / (real_prices.max() - real_prices.min())),
    }
    return m


# ── Load Data ────────────────────────────────────────────────────

print("Loading data...")
train_prices = np.concatenate([
    pd.read_csv(DATA / f"base_d{i}.csv")["typical_price"].values for i in [1, 2, 3]
])
d5 = pd.read_csv(DATA / "base_d5.csv")["typical_price"].values
d6 = pd.read_csv(DATA / "base_d6.csv")["typical_price"].values
print(f"  Train: {len(train_prices)}, d5: {len(d5)}, d6: {len(d6)}")

# ── Fit Generators ───────────────────────────────────────────────

print("\nFitting generators...")

# 1. HMM+GARCH
print("  HMM+GARCH (K=4)...")
hmm_model = hmm_garch.fit(train_prices, n_regimes=4)

# 2. HMM parametric
print("  HMM parametric (K=4)...")
hmm_param_model = fit_regime_model(train_prices, n_regimes=4)

# 3. Block Bootstrap
print("  Block Bootstrap...")
bb_train_files = [str(DATA / f"base_d{i}.csv") for i in [1, 2, 3]]
bb_gen = BlockBootstrapGenerator({"block_size": 30, "train_data": bb_train_files, "target_column": "typical_price"})

# 4. Hybrid: HMM + Block Bootstrap
print("  Hybrid (HMM + Block Bootstrap)...")
hybrid_model = hybrid.fit(train_prices, n_regimes=4, block_size=30)

# ── Evaluate ─────────────────────────────────────────────────────

all_results = []
n_seeds = 10

for gen_name, gen_fn in [
    ("HMM+GARCH_K4", lambda seed: hmm_garch.generate(hmm_model, len(d5), seed=seed, initial_price=d5[0])),
    ("HMM_Parametric_K4", lambda seed: generate_synthetic(hmm_param_model, len(d5), seed=seed, initial_price=d5[0])),
    ("Block_Bootstrap_bs30", lambda seed: bb_gen.generate(seed=seed, n_samples=len(d5))["typical_price"].values[:len(d5)]),
    ("Hybrid_HMM_BB_K4", lambda seed: hybrid.generate(hybrid_model, len(d5), seed=seed, initial_price=d5[0])),
]:
    print(f"\n  Evaluating {gen_name} ({n_seeds} seeds)...")
    for seed in range(n_seeds):
        try:
            synth = gen_fn(seed)
            if len(synth) < len(d5):
                # Pad if short
                synth = np.concatenate([synth, np.full(len(d5) - len(synth), synth[-1])])
            synth = synth[:len(d5)]
            m = compute_all_metrics(d5, synth, label=gen_name)
            m["generator"] = gen_name
            m["seed"] = seed
            m["target"] = "d5"
            all_results.append(m)
        except Exception as e:
            print(f"    Seed {seed} failed: {e}")

print(f"\nTotal: {len(all_results)} evaluations")

# ── Save to OLAP ─────────────────────────────────────────────────

df = pd.DataFrame(all_results)
csv_path = OUT / "generator_metrics.csv"
df.to_csv(csv_path, index=False)

db_path = OUT / "generator_eval_olap.db"
conn = sqlite3.connect(db_path)
df.to_sql("metrics", conn, if_exists="replace", index=False)

# Create summary view
conn.execute("""
CREATE VIEW IF NOT EXISTS v_generator_summary AS
SELECT 
    generator,
    COUNT(*) as n_seeds,
    ROUND(AVG(std_ratio), 3) as avg_std_ratio,
    ROUND(AVG(ac_lag1_ratio), 3) as avg_ac1_ratio,
    ROUND(AVG(ac_lag5_ratio), 3) as avg_ac5_ratio,
    ROUND(AVG(js_divergence), 4) as avg_js_div,
    ROUND(AVG(wasserstein), 6) as avg_wasserstein,
    ROUND(AVG(ks_statistic), 3) as avg_ks_stat,
    ROUND(AVG(skew_diff), 3) as avg_skew_diff,
    ROUND(AVG(kurt_diff), 3) as avg_kurt_diff,
    ROUND(AVG(vol_cluster_ratio), 3) as avg_vol_cluster_ratio,
    ROUND(AVG(hurst_diff), 3) as avg_hurst_diff,
    ROUND(AVG(spectral_distance), 6) as avg_spectral_dist,
    ROUND(AVG(price_range_ratio), 3) as avg_price_range_ratio
FROM metrics
GROUP BY generator
""")

# Per-metric ranking view
conn.execute("""
CREATE VIEW IF NOT EXISTS v_metric_rankings AS
SELECT 
    generator,
    ROUND(AVG(std_ratio), 3) as std_ratio,
    ROUND(AVG(ac_lag1_ratio), 3) as ac1_ratio,
    ROUND(AVG(js_divergence), 4) as js_div,
    ROUND(AVG(wasserstein), 6) as wass,
    ROUND(AVG(vol_cluster_ratio), 3) as vol_clust,
    ROUND(AVG(hurst_diff), 3) as hurst_diff,
    ROUND(AVG(spectral_distance), 6) as spectral
FROM metrics
GROUP BY generator
ORDER BY js_div ASC
""")

conn.commit()
conn.close()

# Print summary
print(f"\n{'Generator':<25} {'Std':>6} {'AC1':>6} {'AC5':>6} {'JS':>8} {'Wass':>10} {'KS':>6} {'VolCl':>6} {'Hurst':>6}")
for _, row in df.groupby("generator").mean(numeric_only=True).iterrows():
    g = _
    print(f"{g:<25} {row['std_ratio']:>6.3f} {row['ac_lag1_ratio']:>6.3f} {row['ac_lag5_ratio']:>6.3f} "
          f"{row['js_divergence']:>8.4f} {row['wasserstein']:>10.6f} {row['ks_statistic']:>6.3f} "
          f"{row['vol_cluster_ratio']:>6.3f} {row['hurst_diff']:>6.3f}")

print(f"\nOLAP: {db_path}")
print(f"CSV: {csv_path}")
