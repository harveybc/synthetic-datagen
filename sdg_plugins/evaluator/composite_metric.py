#!/usr/bin/env python3
"""Composite quality metric for scoring synthetic vs real data.

Returns a single scalar (0=perfect, higher=worse) combining:
- JS divergence, KS statistic, spectral distance (lower=better)
- AC ratios at lags 1/5/24, vol clustering ratio, std ratio (closer to 1=better)
- Hurst exponent diff, skewness diff, kurtosis diff (closer to 0=better)
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, skew, kurtosis


def log_returns(prices):
    return np.diff(np.log(np.clip(prices, 1e-10, None)))


def autocorrelation(x, lag):
    if len(x) <= lag:
        return 0.0
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def hurst_exponent(prices, max_lag=100):
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
    return float(np.clip(np.polyfit(log_lags, log_rs, 1)[0], 0, 1))


def spectral_distance(real_ret, synth_ret):
    n = min(len(real_ret), len(synth_ret))
    real_psd = np.abs(np.fft.fft(real_ret[:n]))**2
    synth_psd = np.abs(np.fft.fft(synth_ret[:n]))**2
    real_psd = real_psd / (real_psd.sum() + 1e-20)
    synth_psd = synth_psd / (synth_psd.sum() + 1e-20)
    return float(np.sqrt(np.mean((real_psd - synth_psd)**2)))


def volatility_clustering_ac(ret, lag=1):
    abs_ret = np.abs(ret)
    if len(abs_ret) <= lag:
        return 0.0
    return float(np.corrcoef(abs_ret[:-lag], abs_ret[lag:])[0, 1])


def composite_score(real_prices, synth_prices, weights=None):
    """Compute composite quality score.
    
    Args:
        real_prices: Real price array
        synth_prices: Synthetic price array
        weights: Optional dict of metric weights
        
    Returns:
        (score, details) where score is 0=perfect, higher=worse
    """
    if weights is None:
        weights = {
            "js_divergence": 3.0,
            "ks_statistic": 2.0,
            "ac_lag1": 2.0,
            "ac_lag5": 1.5,
            "ac_lag24": 1.0,
            "hurst_diff": 2.0,
            "vol_cluster": 1.5,
            "spectral": 1.5,
            "skew_diff": 1.0,
            "kurt_diff": 1.0,
            "std_ratio": 2.0,
        }
    
    real_ret = log_returns(real_prices)
    synth_ret = log_returns(synth_prices)
    
    # JS divergence
    bins = np.linspace(min(real_ret.min(), synth_ret.min()),
                       max(real_ret.max(), synth_ret.max()), 100)
    rh, _ = np.histogram(real_ret, bins=bins, density=True)
    sh, _ = np.histogram(synth_ret, bins=bins, density=True)
    js = float(jensenshannon(rh + 1e-10, sh + 1e-10))
    
    # KS
    ks, _ = ks_2samp(real_ret, synth_ret)
    
    # AC ratios (deviation from 1.0)
    ac_scores = {}
    for lag, key in [(1, "ac_lag1"), (5, "ac_lag5"), (24, "ac_lag24")]:
        ac_r = autocorrelation(real_ret, lag)
        ac_s = autocorrelation(synth_ret, lag)
        if abs(ac_r) > 1e-6:
            ac_scores[key] = abs(ac_s / ac_r - 1.0)  # 0=perfect ratio
        else:
            ac_scores[key] = abs(ac_s)
    
    # Hurst
    h_diff = abs(hurst_exponent(synth_prices) - hurst_exponent(real_prices))
    
    # Vol clustering ratio deviation from 1
    vc_r = volatility_clustering_ac(real_ret)
    vc_s = volatility_clustering_ac(synth_ret)
    vc_dev = abs(vc_s / vc_r - 1.0) if abs(vc_r) > 1e-6 else abs(vc_s)
    
    # Spectral
    spec = spectral_distance(real_ret, synth_ret)
    
    # Skewness/kurtosis diff
    sk_diff = abs(float(skew(synth_ret)) - float(skew(real_ret)))
    kt_diff = abs(float(kurtosis(synth_ret, fisher=False)) - float(kurtosis(real_ret, fisher=False)))
    # Normalize kurtosis diff (can be large)
    kt_diff_norm = kt_diff / max(float(kurtosis(real_ret, fisher=False)), 1.0)
    
    # Std ratio deviation from 1
    std_dev = abs(synth_ret.std() / real_ret.std() - 1.0)
    
    details = {
        "js_divergence": js,
        "ks_statistic": float(ks),
        "ac_lag1": ac_scores["ac_lag1"],
        "ac_lag5": ac_scores["ac_lag5"],
        "ac_lag24": ac_scores["ac_lag24"],
        "hurst_diff": h_diff,
        "vol_cluster": vc_dev,
        "spectral": spec,
        "skew_diff": sk_diff,
        "kurt_diff": kt_diff_norm,
        "std_ratio": std_dev,
    }
    
    # Weighted sum
    total_weight = sum(weights.values())
    score = sum(weights[k] * details[k] for k in weights) / total_weight
    
    return score, details


def composite_score_multi_seed(real_prices, generate_fn, n_seeds=3, **kwargs):
    """Average composite score over multiple seeds.
    
    Args:
        real_prices: Real validation prices
        generate_fn: callable(seed) -> synth_prices array
        n_seeds: Number of seeds to average
        
    Returns:
        (mean_score, std_score, mean_details)
    """
    scores = []
    all_details = []
    for seed in range(n_seeds):
        try:
            synth = generate_fn(seed)
            synth = synth[:len(real_prices)]
            if len(synth) < len(real_prices):
                synth = np.concatenate([synth, np.full(len(real_prices) - len(synth), synth[-1])])
            s, d = composite_score(real_prices, synth, **kwargs)
            scores.append(s)
            all_details.append(d)
        except Exception:
            continue
    
    if not scores:
        return float('inf'), float('inf'), {}
    
    mean_details = {}
    for k in all_details[0]:
        mean_details[k] = np.mean([d[k] for d in all_details])
    
    return float(np.mean(scores)), float(np.std(scores)), mean_details
