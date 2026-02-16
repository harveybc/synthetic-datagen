"""
Grasynda-inspired graph-based synthetic time series generator.

Based on: "Grasynda: Graph-based Synthetic Time Series Generation" (IDA 2026)

Approach:
1. Discretize the real price series into states (quantile bins)
2. Build a directed transition graph: P(state_j | state_i)
3. Generate new series by random walking on the transition matrix
4. Map states back to prices by sampling from real prices in each bin

Training-free — just builds the graph from real data.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from app.data_processor import load_csv, load_multiple_csv, prices_to_returns

log = logging.getLogger(__name__)

DEFAULT_N_BINS = 50


class GrasyndaGenerator:
    """Plugin: graph-based synthetic typical_price generator."""

    plugin_params: Dict[str, Any] = {}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg: Dict[str, Any] = {
            "n_bins": DEFAULT_N_BINS,
            "use_returns": True,
            "start_datetime": "2020-01-01 00:00:00",
            "interval_hours": 4,
        }
        if config:
            self.cfg.update(config)

    def configure(self, config: Dict[str, Any]) -> None:
        self.cfg.update(config)

    def set_params(self, **kw) -> None:
        self.cfg.update(kw)

    def generate(self, seed: int, n_samples: int,
                 start_datetime: Optional[str] = None) -> pd.DataFrame:
        """Generate synthetic data via transition graph random walk."""
        train_data = self.cfg.get("train_data")
        if not train_data:
            raise ValueError("train_data (list of CSV paths) required")

        if isinstance(train_data, str):
            train_data = [train_data]

        df = load_multiple_csv(train_data)
        prices = df["typical_price"].values.astype(np.float64)
        log.info(f"Loaded {len(prices)} real prices for Grasynda generation")

        n_bins = self.cfg.get("n_bins", DEFAULT_N_BINS)
        use_returns = self.cfg.get("use_returns", True)

        if use_returns:
            # Work in log-return space (stationary, better transitions)
            series = prices_to_returns(prices)
        else:
            series = prices.copy()

        # 1. Discretize into quantile bins
        bin_edges = np.quantile(series, np.linspace(0, 1, n_bins + 1))
        # Ensure unique edges
        bin_edges = np.unique(bin_edges)
        actual_bins = len(bin_edges) - 1
        if actual_bins < 3:
            raise ValueError(f"Too few unique bins ({actual_bins}), data may be constant")

        states = np.digitize(series, bin_edges[1:-1])  # 0 to actual_bins-1

        # 2. Build transition matrix
        trans = np.zeros((actual_bins, actual_bins), dtype=np.float64)
        for i in range(len(states) - 1):
            trans[states[i], states[i + 1]] += 1

        # Normalize rows to probabilities (add small epsilon for zero rows)
        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # avoid div by zero
        trans_prob = trans / row_sums

        # 3. Build per-bin value pools (real values that fell in each bin)
        bin_values = [[] for _ in range(actual_bins)]
        for i, s in enumerate(states):
            bin_values[s].append(series[i])
        # Convert to arrays
        bin_values = [np.array(bv) if bv else np.array([series.mean()])
                      for bv in bin_values]

        log.info(
            f"Grasynda graph: {actual_bins} states, "
            f"transition matrix density: {(trans > 0).sum() / trans.size:.2%}"
        )

        # 4. Random walk generation
        rng = np.random.default_rng(seed)

        # Start from a random state weighted by frequency
        state_freq = np.bincount(states, minlength=actual_bins).astype(np.float64)
        state_freq /= state_freq.sum()
        current_state = rng.choice(actual_bins, p=state_freq)

        synthetic_values = []
        for _ in range(n_samples if use_returns else n_samples):
            # Sample a value from the current bin
            val = rng.choice(bin_values[current_state])
            synthetic_values.append(val)
            # Transition to next state
            current_state = rng.choice(actual_bins, p=trans_prob[current_state])

        synthetic_values = np.array(synthetic_values)

        if use_returns:
            # Convert returns back to prices
            initial_price = prices[rng.integers(len(prices))]
            log_prices = np.concatenate([[np.log(initial_price)], synthetic_values])
            synthetic_prices = np.exp(np.cumsum(log_prices))[:n_samples]
        else:
            synthetic_prices = synthetic_values

        start_dt = pd.to_datetime(
            start_datetime or self.cfg.get("start_datetime", "2020-01-01 00:00:00")
        )
        interval = timedelta(hours=self.cfg.get("interval_hours", 4))
        dates = [start_dt + i * interval for i in range(n_samples)]

        result = pd.DataFrame({
            "DATE_TIME": dates,
            "typical_price": synthetic_prices[:n_samples],
        })
        log.info(
            f"Generated {n_samples} samples via Grasynda "
            f"(n_bins={actual_bins}, use_returns={use_returns}, seed={seed})"
        )
        return result
