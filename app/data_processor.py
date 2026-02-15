"""
Data loading & preprocessing utilities.

Handles:
- Reading DATE_TIME,typical_price CSVs (already 4h periodicity)
- Computing log-returns and reconstructing prices
- Creating sliding windows for training
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Sequence


# ── I/O ─────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """Load a DATE_TIME,typical_price CSV."""
    df = pd.read_csv(path, parse_dates=["DATE_TIME"])
    if "typical_price" not in df.columns:
        raise ValueError(f"CSV {path} must have a 'typical_price' column")
    return df


def load_multiple_csv(paths: Sequence[str]) -> pd.DataFrame:
    """Concatenate several CSVs, sort by DATE_TIME, drop duplicates."""
    frames = [load_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df.sort_values("DATE_TIME", inplace=True)
    df.drop_duplicates(subset="DATE_TIME", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def save_csv(df: pd.DataFrame, path: str) -> None:
    """Save a DATE_TIME,typical_price DataFrame to CSV."""
    df.to_csv(path, index=False)


# ── Returns ─────────────────────────────────────────────────────────────────

def prices_to_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log-returns from price array.  Output length = len(prices) - 1."""
    prices = np.asarray(prices, dtype=np.float64)
    return np.diff(np.log(prices))


def returns_to_prices(returns: np.ndarray, initial_price: float) -> np.ndarray:
    """Reconstruct prices from log-returns and an initial price."""
    returns = np.asarray(returns, dtype=np.float64)
    log_prices = np.concatenate([[np.log(initial_price)], returns])
    return np.exp(np.cumsum(log_prices))


# ── Windowing ───────────────────────────────────────────────────────────────

def create_windows(data: np.ndarray, window_size: int) -> np.ndarray:
    """Create overlapping sliding windows.  Shape: (N - W + 1, W)."""
    n = len(data)
    if n < window_size:
        raise ValueError(f"Data length {n} < window_size {window_size}")
    indices = np.arange(window_size)[None, :] + np.arange(n - window_size + 1)[:, None]
    return data[indices]


# ── Prepare training data ──────────────────────────────────────────────────

def prepare_training_data(
    paths: Sequence[str],
    window_size: int,
    use_returns: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Load 4h CSVs → optional returns → sliding windows.

    Returns
    -------
    windows : ndarray of shape (N, window_size)
    initial_price : the first price (needed to reconstruct from returns)
    """
    df = load_multiple_csv(paths)
    prices = df["typical_price"].values.astype(np.float64)

    initial_price = float(prices[0])

    if use_returns:
        series = prices_to_returns(prices)
    else:
        series = prices

    windows = create_windows(series, window_size)
    return windows, initial_price
