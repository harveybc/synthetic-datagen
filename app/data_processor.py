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
from typing import Sequence, Optional


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


# ── Temporal features ──────────────────────────────────────────────────────

def extract_temporal_features(dates: pd.Series) -> np.ndarray:
    """
    Extract cyclical temporal features from a datetime Series.

    Returns ndarray of shape (len(dates), 6):
        hour sin/cos, day_of_week sin/cos, month sin/cos
    """
    dt = pd.to_datetime(dates)
    TWO_PI = 2.0 * np.pi

    hour = dt.dt.hour.values.astype(np.float64)
    dow = dt.dt.dayofweek.values.astype(np.float64)  # 0=Mon … 4=Fri
    month = dt.dt.month.values.astype(np.float64)

    features = np.column_stack([
        np.sin(TWO_PI * hour / 24.0),
        np.cos(TWO_PI * hour / 24.0),
        np.sin(TWO_PI * dow / 5.0),
        np.cos(TWO_PI * dow / 5.0),
        np.sin(TWO_PI * month / 12.0),
        np.cos(TWO_PI * month / 12.0),
    ])
    return features


# ── Prepare training data ──────────────────────────────────────────────────

def prepare_training_data(
    paths: Sequence[str],
    window_size: int,
    use_returns: bool = True,
    conditional: bool = False,
) -> tuple[np.ndarray, ...]:
    """
    Load 4h CSVs → optional returns → sliding windows.

    Returns
    -------
    If conditional=False:
        (windows, initial_price)
    If conditional=True:
        (windows, temporal_features, initial_price)
        where temporal_features shape = (n_windows, 6), using last timestamp per window.
    """
    df = load_multiple_csv(paths)
    prices = df["typical_price"].values.astype(np.float64)
    dates = df["DATE_TIME"]

    initial_price = float(prices[0])

    if use_returns:
        series = prices_to_returns(prices)
        # dates for returns align with the second price onward
        series_dates = dates.iloc[1:].reset_index(drop=True)
    else:
        series = prices
        series_dates = dates.reset_index(drop=True)

    windows = create_windows(series, window_size)

    if not conditional:
        return windows, initial_price

    # For each window, take temporal features of the LAST timestamp
    all_temporal = extract_temporal_features(series_dates)
    # Window i spans indices [i, i+window_size), last index = i+window_size-1
    last_indices = np.arange(window_size - 1, window_size - 1 + len(windows))
    temporal_features = all_temporal[last_indices]

    return windows, temporal_features, initial_price
