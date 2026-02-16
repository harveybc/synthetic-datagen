"""
Block Bootstrap synthetic time series generator.

Generates synthetic data by sampling contiguous blocks from real training data
and concatenating them. Simple, training-free, preserves real transition dynamics
within blocks.

Blocks are sampled at trading-week boundaries (30 ticks = 5 days × 6 4h-ticks)
to maintain intra-week structure.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from app.data_processor import load_csv, load_multiple_csv

log = logging.getLogger(__name__)

# 1 trading week at 4h = 5 days × 6 ticks
DEFAULT_BLOCK_SIZE = 30


class BlockBootstrapGenerator:
    """Plugin: block bootstrap synthetic typical_price generator."""

    plugin_params: Dict[str, Any] = {}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg: Dict[str, Any] = {
            "block_size": DEFAULT_BLOCK_SIZE,
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
        """Generate synthetic data by sampling contiguous blocks from real data."""
        train_data = self.cfg.get("train_data")
        if not train_data:
            raise ValueError("train_data (list of CSV paths) required")

        if isinstance(train_data, str):
            train_data = [train_data]

        df = load_multiple_csv(train_data)
        prices = df["typical_price"].values.astype(np.float64)
        log.info(f"Loaded {len(prices)} real prices for block bootstrap")

        block_size = self.cfg.get("block_size", DEFAULT_BLOCK_SIZE)
        rng = np.random.default_rng(seed)

        # Sample blocks of contiguous prices
        max_start = len(prices) - block_size
        if max_start < 1:
            raise ValueError(
                f"Real data ({len(prices)}) too short for block_size={block_size}"
            )

        synthetic_prices = []
        while len(synthetic_prices) < n_samples:
            start_idx = rng.integers(0, max_start)
            block = prices[start_idx : start_idx + block_size].copy()

            if synthetic_prices:
                # Scale block so its first price matches the last synthetic price
                # This avoids price jumps between blocks
                scale = synthetic_prices[-1] / block[0]
                block = block * scale

            synthetic_prices.extend(block.tolist())

        synthetic_prices = synthetic_prices[:n_samples]

        start_dt = pd.to_datetime(
            start_datetime or self.cfg.get("start_datetime", "2020-01-01 00:00:00")
        )
        interval = timedelta(hours=self.cfg.get("interval_hours", 4))
        dates = [start_dt + i * interval for i in range(n_samples)]

        result = pd.DataFrame({
            "DATE_TIME": dates,
            "typical_price": synthetic_prices,
        })
        log.info(
            f"Generated {n_samples} samples via block bootstrap "
            f"(block_size={block_size}, seed={seed})"
        )
        return result
