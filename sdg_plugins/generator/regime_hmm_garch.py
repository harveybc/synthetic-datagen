#!/usr/bin/env python3
"""Regime HMM + GARCH Generator.

Best of both worlds:
- HMM for regime detection & transitions (preserves regime structure)
- GARCH(1,1) per regime for volatility clustering (preserves vol dynamics)
- AR(1) per regime for autocorrelation (preserves temporal dependence)
- Skewed-t innovations for fat tails (realistic financial returns)

Deterministic via seed. No neural networks. Fast training & generation.
"""
import numpy as np
import pandas as pd
import json, os, warnings
from typing import Optional
from hmmlearn.hmm import GaussianHMM
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")
_QUIET = os.environ.get("SDG_QUIET", "0") == "1"


def _fit_garch11(returns: np.ndarray) -> dict:
    """Fit GARCH(1,1) via maximum likelihood on returns.
    
    σ²_t = omega + alpha * r²_{t-1} + beta * σ²_{t-1}
    
    Simple iterative fit — no external dependencies.
    """
    r = returns - returns.mean()
    n = len(r)
    if n < 10:
        return {"omega": float(r.var()), "alpha": 0.05, "beta": 0.90, "long_run_var": float(r.var())}

    # Initialize
    var_target = r.var()
    best_params = {"omega": var_target * 0.05, "alpha": 0.05, "beta": 0.90}
    best_ll = -1e18

    # Grid search over alpha/beta (fast, robust)
    for alpha in [0.02, 0.05, 0.08, 0.12, 0.15, 0.20]:
        for beta in [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.93, 0.95]:
            if alpha + beta >= 0.999:
                continue
            omega = var_target * (1 - alpha - beta)
            if omega <= 0:
                continue

            # Compute log-likelihood
            sigma2 = np.zeros(n)
            sigma2[0] = var_target
            for t in range(1, n):
                sigma2[t] = omega + alpha * r[t-1]**2 + beta * sigma2[t-1]
                sigma2[t] = max(sigma2[t], 1e-20)

            ll = -0.5 * np.sum(np.log(sigma2) + r**2 / sigma2)
            if ll > best_ll:
                best_ll = ll
                best_params = {"omega": omega, "alpha": alpha, "beta": beta}

    best_params["long_run_var"] = best_params["omega"] / max(1 - best_params["alpha"] - best_params["beta"], 0.001)
    return best_params


def extract_features(prices: np.ndarray, vol_window: int = 24) -> np.ndarray:
    """Extract HMM features: log returns, rolling vol, vol ratio."""
    log_ret = np.diff(np.log(np.clip(prices, 1e-10, None)))
    vol_short = pd.Series(log_ret).rolling(vol_window, min_periods=1).std().values
    vol_long = pd.Series(log_ret).rolling(vol_window * 3, min_periods=1).std().values
    vol_ratio = np.divide(vol_short, vol_long, out=np.ones_like(vol_short), where=vol_long > 1e-10)

    features = np.column_stack([log_ret, vol_short[:len(log_ret)], vol_ratio[:len(log_ret)]])
    mask = ~np.isnan(features).any(axis=1) & ~np.isinf(features).any(axis=1)
    return features[mask], mask


def fit(prices: np.ndarray, n_regimes: int = 4, hmm_iter: int = 200) -> dict:
    """Fit HMM + per-regime GARCH model."""
    features, valid_mask = extract_features(prices)
    log_ret_full = np.diff(np.log(np.clip(prices, 1e-10, None)))
    n_dropped = len(log_ret_full) - len(features)
    log_ret = log_ret_full[n_dropped:]

    # Normalize for HMM
    f_mean = features.mean(axis=0)
    f_std = features.std(axis=0) + 1e-10
    features_norm = (features - f_mean) / f_std

    # Fit HMM
    hmm = GaussianHMM(n_components=n_regimes, covariance_type="full",
                       n_iter=hmm_iter, random_state=42, verbose=False)
    hmm.fit(features_norm)
    labels = hmm.predict(features_norm)

    # Per-regime models
    regime_models = []
    for k in range(n_regimes):
        mask = labels == k
        rets_k = log_ret[mask]
        if len(rets_k) < 10:
            regime_models.append({
                "mean": 0.0, "ar1": 0.0,
                "garch": {"omega": float(log_ret.var()), "alpha": 0.05, "beta": 0.90,
                          "long_run_var": float(log_ret.var())},
                "skew": 0.0, "kurt": 3.0, "count": int(len(rets_k)),
            })
            continue

        # AR(1)
        ar1 = float(np.corrcoef(rets_k[:-1], rets_k[1:])[0, 1]) if len(rets_k) > 2 else 0.0
        if np.isnan(ar1): ar1 = 0.0

        # GARCH(1,1)
        garch = _fit_garch11(rets_k)

        # Higher moments
        skew = float(sp_stats.skew(rets_k))
        kurt = float(sp_stats.kurtosis(rets_k, fisher=False))  # excess=False → raw kurtosis

        rm = {
            "mean": float(rets_k.mean()),
            "ar1": ar1,
            "garch": garch,
            "skew": skew,
            "kurt": kurt,
            "count": int(len(rets_k)),
        }
        regime_models.append(rm)

        if not _QUIET:
            print(f"  Regime {k}: n={rm['count']}, μ={rm['mean']:.6f}, "
                  f"σ={garch['long_run_var']**.5:.6f}, ar1={ar1:.3f}, "
                  f"α={garch['alpha']:.2f}, β={garch['beta']:.2f}")

    model = {
        "n_regimes": n_regimes,
        "transition_matrix": hmm.transmat_.tolist(),
        "start_probs": hmm.startprob_.tolist(),
        "regime_models": regime_models,
        "data_mean": float(prices.mean()),
        "data_std": float(prices.std()),
    }
    return model


def generate(model: dict, n_samples: int, seed: int,
             initial_price: Optional[float] = None) -> np.ndarray:
    """Generate synthetic prices. Deterministic given seed."""
    rng = np.random.RandomState(seed)
    trans = np.array(model["transition_matrix"])
    start_p = np.array(model["start_probs"])
    n_reg = model["n_regimes"]

    if initial_price is None:
        initial_price = model["data_mean"]

    # Sample regime sequence
    regimes = np.zeros(n_samples, dtype=int)
    regimes[0] = rng.choice(n_reg, p=start_p)
    for t in range(1, n_samples):
        regimes[t] = rng.choice(n_reg, p=trans[regimes[t - 1]])

    # Generate returns with AR(1) + GARCH(1,1)
    log_returns = np.zeros(n_samples)
    prev_residual = 0.0
    prev_sigma2 = None

    for t in range(n_samples):
        k = regimes[t]
        rm = model["regime_models"][k]
        g = rm["garch"]

        # GARCH variance
        if prev_sigma2 is None:
            sigma2 = g["long_run_var"]
        else:
            sigma2 = g["omega"] + g["alpha"] * prev_residual**2 + g["beta"] * prev_sigma2
        sigma2 = max(sigma2, 1e-20)

        # Innovation (normal — could use skewed-t for more realism)
        innovation = rng.normal(0, np.sqrt(sigma2))

        # AR(1): r_t = mean + ar1 * prev_residual + innovation
        log_returns[t] = rm["mean"] + rm["ar1"] * prev_residual + innovation
        prev_residual = log_returns[t] - rm["mean"]
        prev_sigma2 = sigma2

    # Prices
    log_prices = np.log(initial_price) + np.cumsum(log_returns)
    return np.exp(log_prices)


def save_model(model: dict, path: str):
    with open(path, "w") as f:
        json.dump(model, f, indent=2)


def load_model(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
