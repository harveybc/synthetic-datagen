#!/usr/bin/env python3
"""Hybrid generator: HMM regime transitions + block bootstrap within each regime.

Combines strengths of both approaches:
- HMM: realistic regime switching (high AC lag1, good Hurst, structure)
- Block bootstrap: real return distribution (low JS, Wasserstein, KS)

Algorithm:
1. Fit HMM on training returns → learn regime means, transitions
2. Segment training data by detected regime
3. Generate regime sequence from HMM transition matrix
4. For each regime segment in generated sequence, sample contiguous blocks
   from real data of that same regime (block bootstrap)
5. Concatenate blocks, cumsum returns → synthetic prices
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    raise ImportError("pip install hmmlearn")


def _compute_features(prices: np.ndarray, 
                      vol_short_window: int = 6,
                      vol_mid_window: int = 24,
                      vol_long_window: int = 48) -> np.ndarray:
    """Compute features for HMM: log returns, rolling vol, vol ratio.
    
    Args:
        prices: Price series
        vol_short_window: Short volatility window (default 6)
        vol_mid_window: Mid volatility window for rolling vol feature (default 24)
        vol_long_window: Long volatility window (default 48)
    """
    log_ret = np.diff(np.log(np.clip(prices, 1e-10, None)))
    # Rolling volatility
    vol = pd.Series(log_ret).rolling(vol_mid_window, min_periods=max(1, vol_mid_window // 4)).std().bfill().values
    # Vol ratio (short/long)
    vol_short = pd.Series(log_ret).rolling(vol_short_window, min_periods=max(1, vol_short_window // 2)).std().bfill().values
    vol_long = pd.Series(log_ret).rolling(vol_long_window, min_periods=max(1, vol_long_window // 4)).std().bfill().values
    vol_ratio = vol_short / np.clip(vol_long, 1e-10, None)
    return np.column_stack([log_ret, vol, vol_ratio])


def fit(prices: np.ndarray, n_regimes: int = 4, block_size: int = 30,
        min_block_length: int = 3, covariance_type: str = "full",
        vol_short_window: int = 6, vol_mid_window: int = 24,
        vol_long_window: int = 48, smooth_weight: float = 0.3,
        quiet: bool = False, **kwargs) -> Dict[str, Any]:
    """Fit HMM on training data and segment returns by regime.
    
    Args:
        prices: Training price series
        n_regimes: Number of HMM states
        block_size: Block size for bootstrap sampling within regimes
        min_block_length: Minimum contiguous block length to keep
        covariance_type: HMM covariance type ("full", "diag", "tied")
        vol_short_window: Short vol window for features
        vol_mid_window: Mid vol window for features
        vol_long_window: Long vol window for features
        smooth_weight: Block boundary smoothing weight (stored for generate)
    
    Returns:
        Model dict with HMM + per-regime return segments
    """
    features = _compute_features(prices, vol_short_window, vol_mid_window, vol_long_window)
    log_ret = np.diff(np.log(np.clip(prices, 1e-10, None)))
    
    # Fit HMM
    hmm = GaussianHMM(n_components=n_regimes, covariance_type=covariance_type,
                       n_iter=200, random_state=42, tol=0.001)
    hmm.fit(features)
    states = hmm.predict(features)
    
    # Segment returns by regime
    regime_returns = {}
    regime_prices = {}
    for k in range(n_regimes):
        mask = states == k
        regime_returns[k] = log_ret[mask]
        # Also store contiguous blocks of returns for this regime
        blocks = []
        current_block = []
        for i, s in enumerate(states):
            if s == k:
                current_block.append(log_ret[i])
            else:
                if len(current_block) >= min_block_length:
                    blocks.append(np.array(current_block))
                current_block = []
        if len(current_block) >= min_block_length:
            blocks.append(np.array(current_block))
        regime_prices[k] = blocks
    
    # Print regime info
    for k in range(n_regimes):
        n = len(regime_returns[k])
        mu = regime_returns[k].mean()
        sigma = regime_returns[k].std()
        n_blocks = len(regime_prices[k])
        avg_block = np.mean([len(b) for b in regime_prices[k]]) if regime_prices[k] else 0
        if not quiet:
            print(f"  Regime {k}: n={n}, μ={mu:.6f}, σ={sigma:.6f}, "
                  f"blocks={n_blocks}, avg_len={avg_block:.1f}")
    
    return {
        "hmm": hmm,
        "n_regimes": n_regimes,
        "block_size": block_size,
        "min_block_length": min_block_length,
        "smooth_weight": smooth_weight,
        "covariance_type": covariance_type,
        "vol_windows": (vol_short_window, vol_mid_window, vol_long_window),
        "transition_matrix": hmm.transmat_,
        "stationary_dist": hmm.startprob_,
        "regime_returns": regime_returns,
        "regime_blocks": regime_prices,
        "train_mean": log_ret.mean(),
        "train_std": log_ret.std(),
    }


def generate(model: Dict[str, Any], n_steps: int, seed: int = 0,
             initial_price: float = 1.0) -> np.ndarray:
    """Generate synthetic prices using HMM transitions + block bootstrap.
    
    Algorithm:
    1. Sample initial regime from stationary distribution
    2. At each step, decide regime from transition matrix
    3. When in a regime, sample a random contiguous block from that regime's real data
    4. Use the entire block (preserves within-regime temporal structure)
    5. When regime switches, start new block from new regime
    6. Cumsum log returns → prices
    """
    rng = np.random.RandomState(seed)
    trans = model["transition_matrix"]
    n_regimes = model["n_regimes"]
    regime_blocks = model["regime_blocks"]
    block_size = model["block_size"]
    
    raw_blocks = []  # list of (block_returns, regime_id)
    
    # Initial regime
    regime = rng.choice(n_regimes, p=model["stationary_dist"])
    total_len = 0
    
    while total_len < n_steps + block_size:  # generate extra for blending
        # Sample a block from current regime
        blocks = regime_blocks[regime]
        if not blocks:
            rets = model["regime_returns"][regime]
            block_len = min(block_size, n_steps - total_len + block_size)
            sampled = rng.choice(rets, size=max(3, block_len), replace=True)
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
    
    # Stitch blocks with single-point boundary smoothing to reduce AC discontinuity
    log_returns = list(raw_blocks[0])
    smooth_weight = model.get("smooth_weight", 0.3)
    
    for i in range(1, len(raw_blocks)):
        block = list(raw_blocks[i])
        if len(log_returns) > 0 and len(block) > 0:
            # Smooth just the first point of the new block
            prev_val = log_returns[-1]
            block[0] = (1 - smooth_weight) * block[0] + smooth_weight * prev_val
        log_returns.extend(block)
    
    log_returns = np.array(log_returns[:n_steps])
    
    # Cumulative sum → prices
    log_prices = np.log(initial_price) + np.cumsum(log_returns)
    prices = np.exp(log_prices)
    
    # Prepend initial price
    prices = np.concatenate([[initial_price], prices])[:n_steps]
    
    return prices


def generate_df(model: Dict[str, Any], n_steps: int, seed: int = 0,
                initial_price: float = 1.0) -> pd.DataFrame:
    """Generate as DataFrame with typical_price column."""
    prices = generate(model, n_steps, seed=seed, initial_price=initial_price)
    return pd.DataFrame({"typical_price": prices})
