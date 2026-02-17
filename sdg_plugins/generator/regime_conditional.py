#!/usr/bin/env python3
"""Regime-Conditional Synthetic Data Generator.

1. Detect market regimes via HMM on features (returns, volatility, vol ratio)
2. Learn per-regime return distributions + GARCH volatility
3. Generate synthetic data by sampling regime sequences + per-regime returns
4. Deterministic via seed control
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from hmmlearn.hmm import GaussianHMM
from scipy import stats
import json, os, warnings
warnings.filterwarnings("ignore")

_QUIET = os.environ.get("SDG_QUIET", "0") == "1"

@dataclass
class RegimeModel:
    """Fitted regime model — serializable."""
    n_regimes: int
    transition_matrix: np.ndarray  # (K, K)
    start_probs: np.ndarray        # (K,)
    regime_params: list            # per-regime: {mean, std, skew, ar1, vol_mean, vol_std}
    feature_means: np.ndarray      # for normalization
    feature_stds: np.ndarray
    hmm_means: np.ndarray          # HMM emission means
    hmm_covars: np.ndarray         # HMM emission covariances
    data_mean: float = 0.0         # original price level
    data_std: float = 1.0

    def to_dict(self):
        return {
            "n_regimes": self.n_regimes,
            "transition_matrix": self.transition_matrix.tolist(),
            "start_probs": self.start_probs.tolist(),
            "regime_params": self.regime_params,
            "feature_means": self.feature_means.tolist(),
            "feature_stds": self.feature_stds.tolist(),
            "hmm_means": self.hmm_means.tolist(),
            "hmm_covars": self.hmm_covars.tolist(),
            "data_mean": self.data_mean,
            "data_std": self.data_std,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            n_regimes=d["n_regimes"],
            transition_matrix=np.array(d["transition_matrix"]),
            start_probs=np.array(d["start_probs"]),
            regime_params=d["regime_params"],
            feature_means=np.array(d["feature_means"]),
            feature_stds=np.array(d["feature_stds"]),
            hmm_means=np.array(d["hmm_means"]),
            hmm_covars=np.array(d["hmm_covars"]),
            data_mean=d.get("data_mean", 0.0),
            data_std=d.get("data_std", 1.0),
        )


def extract_features(prices: np.ndarray, vol_window: int = 24, vol_long: int = 72) -> np.ndarray:
    """Extract regime features: log returns, rolling vol, vol ratio."""
    log_ret = np.diff(np.log(np.clip(prices, 1e-10, None)))

    # Rolling volatility (short and long)
    vol_short = pd.Series(log_ret).rolling(vol_window, min_periods=1).std().values
    vol_long_arr = pd.Series(log_ret).rolling(vol_long, min_periods=1).std().values

    # Vol ratio (short/long) — regime transition indicator
    vol_ratio = np.divide(vol_short, vol_long_arr, out=np.ones_like(vol_short), where=vol_long_arr > 1e-10)

    # Stack features, trim to common length, drop NaN rows
    n = len(log_ret)
    features = np.column_stack([log_ret, vol_short[:n], vol_ratio[:n]])
    mask = ~np.isnan(features).any(axis=1) & ~np.isinf(features).any(axis=1)
    return features[mask]


def fit_regime_model(prices: np.ndarray, n_regimes: int = 4, n_iter: int = 200) -> RegimeModel:
    """Fit HMM regime model on price data."""
    features = extract_features(prices)

    # Also compute log_ret aligned with features (features already drops NaN rows)
    log_ret_full = np.diff(np.log(np.clip(prices, 1e-10, None)))

    # Normalize features for HMM
    f_mean = features.mean(axis=0)
    f_std = features.std(axis=0) + 1e-10
    features_norm = (features - f_mean) / f_std

    # Fit HMM
    hmm = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=n_iter,
        random_state=42,
        verbose=False,
    )
    hmm.fit(features_norm)
    regime_labels = hmm.predict(features_norm)

    # Align log_ret with features (features may have dropped leading NaN rows)
    n_dropped = len(log_ret_full) - len(features)
    log_ret = log_ret_full[n_dropped:]
    regime_params = []
    for k in range(n_regimes):
        mask = regime_labels == k
        if mask.sum() < 5:
            # Fallback for sparse regimes
            regime_params.append({"mean": 0.0, "std": float(log_ret.std()), "skew": 0.0, "ar1": 0.0,
                                  "vol_mean": float(log_ret.std()), "count": int(mask.sum())})
            continue

        rets_k = log_ret[mask]
        # AR(1) coefficient for autocorrelation within regime
        ar1 = 0.0
        if len(rets_k) > 2:
            ar1 = float(np.corrcoef(rets_k[:-1], rets_k[1:])[0, 1])
            if np.isnan(ar1):
                ar1 = 0.0

        regime_params.append({
            "mean": float(rets_k.mean()),
            "std": float(rets_k.std()),
            "skew": float(stats.skew(rets_k)),
            "ar1": float(ar1),
            "vol_mean": float(features[mask, 1].mean()),
            "count": int(mask.sum()),
        })

    if not _QUIET:
        for k, rp in enumerate(regime_params):
            print(f"  Regime {k}: n={rp['count']}, μ={rp['mean']:.6f}, σ={rp['std']:.6f}, ar1={rp['ar1']:.3f}")

    return RegimeModel(
        n_regimes=n_regimes,
        transition_matrix=hmm.transmat_,
        start_probs=hmm.startprob_,
        regime_params=regime_params,
        feature_means=f_mean,
        feature_stds=f_std,
        hmm_means=hmm.means_,
        hmm_covars=hmm.covars_,
        data_mean=float(prices.mean()),
        data_std=float(prices.std()),
    )


def generate_synthetic(model: RegimeModel, n_samples: int, seed: int,
                       initial_price: Optional[float] = None) -> np.ndarray:
    """Generate synthetic price series from fitted regime model.
    
    Deterministic given seed.
    """
    rng = np.random.RandomState(seed)

    if initial_price is None:
        initial_price = model.data_mean

    # Sample regime sequence from transition matrix
    regimes = np.zeros(n_samples, dtype=int)
    regimes[0] = rng.choice(model.n_regimes, p=model.start_probs)
    for t in range(1, n_samples):
        regimes[t] = rng.choice(model.n_regimes, p=model.transition_matrix[regimes[t - 1]])

    # Generate returns per regime with AR(1) autocorrelation
    log_returns = np.zeros(n_samples)
    prev_ret = 0.0
    for t in range(n_samples):
        k = regimes[t]
        rp = model.regime_params[k]
        # AR(1): r_t = ar1 * r_{t-1} + innovation
        innovation_std = rp["std"] * np.sqrt(max(1 - rp["ar1"] ** 2, 0.01))
        innovation = rng.normal(0, innovation_std)
        log_returns[t] = rp["ar1"] * prev_ret + rp["mean"] + innovation
        prev_ret = log_returns[t] - rp["mean"]  # residual for AR

    # Cumulative sum → price path
    log_prices = np.log(initial_price) + np.cumsum(log_returns)
    prices = np.exp(log_prices)

    return prices


def save_model(model: RegimeModel, path: str):
    with open(path, "w") as f:
        json.dump(model.to_dict(), f, indent=2)


def load_model(path: str) -> RegimeModel:
    with open(path) as f:
        return RegimeModel.from_dict(json.load(f))
