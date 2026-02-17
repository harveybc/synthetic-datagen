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


def _compute_features(prices: np.ndarray) -> np.ndarray:
    """Compute features for HMM: log returns, rolling vol, vol ratio."""
    log_ret = np.diff(np.log(np.clip(prices, 1e-10, None)))
    # Rolling volatility (window=24 = 4 days at 4h)
    vol = pd.Series(log_ret).rolling(24, min_periods=6).std().bfill().values
    # Vol ratio (short/long)
    vol_short = pd.Series(log_ret).rolling(6, min_periods=3).std().bfill().values
    vol_long = pd.Series(log_ret).rolling(48, min_periods=12).std().bfill().values
    vol_ratio = vol_short / np.clip(vol_long, 1e-10, None)
    return np.column_stack([log_ret, vol, vol_ratio])


def fit(prices: np.ndarray, n_regimes: int = 4, block_size: int = 30) -> Dict[str, Any]:
    """Fit HMM on training data and segment returns by regime.
    
    Args:
        prices: Training price series (d1+d2+d3 concatenated)
        n_regimes: Number of HMM states
        block_size: Block size for bootstrap sampling within regimes
    
    Returns:
        Model dict with HMM + per-regime return segments
    """
    features = _compute_features(prices)
    log_ret = np.diff(np.log(np.clip(prices, 1e-10, None)))
    
    # Fit HMM
    hmm = GaussianHMM(n_components=n_regimes, covariance_type="full",
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
                if len(current_block) >= 3:  # min block length
                    blocks.append(np.array(current_block))
                current_block = []
        if len(current_block) >= 3:
            blocks.append(np.array(current_block))
        regime_prices[k] = blocks
    
    # Print regime info
    for k in range(n_regimes):
        n = len(regime_returns[k])
        mu = regime_returns[k].mean()
        sigma = regime_returns[k].std()
        n_blocks = len(regime_prices[k])
        avg_block = np.mean([len(b) for b in regime_prices[k]]) if regime_prices[k] else 0
        print(f"  Regime {k}: n={n}, μ={mu:.6f}, σ={sigma:.6f}, "
              f"blocks={n_blocks}, avg_len={avg_block:.1f}")
    
    return {
        "hmm": hmm,
        "n_regimes": n_regimes,
        "block_size": block_size,
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
    
    log_returns = []
    
    # Initial regime
    regime = rng.choice(n_regimes, p=model["stationary_dist"])
    
    while len(log_returns) < n_steps:
        # Sample a block from current regime
        blocks = regime_blocks[regime]
        if not blocks:
            # Fallback: use regime returns directly
            rets = model["regime_returns"][regime]
            block_len = min(block_size, n_steps - len(log_returns))
            sampled = rng.choice(rets, size=block_len, replace=True)
        else:
            # Pick a random block
            block = blocks[rng.randint(len(blocks))]
            
            if len(block) > block_size:
                # Sample a sub-block of block_size from within the block
                start = rng.randint(0, max(1, len(block) - block_size))
                sampled = block[start:start + block_size]
            else:
                sampled = block
            
            # Trim to not exceed n_steps
            remaining = n_steps - len(log_returns)
            sampled = sampled[:remaining]
        
        log_returns.extend(sampled)
        
        # Transition to next regime
        regime = rng.choice(n_regimes, p=trans[regime])
    
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
