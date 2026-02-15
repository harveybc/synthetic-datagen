"""
Typical-price generator plugin.

Loads a trained decoder model and generates synthetic typical_price
timeseries using seed-based deterministic latent sampling.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from app.data_processor import returns_to_prices, downsample

log = logging.getLogger(__name__)


class TypicalPriceGenerator:
    """Plugin: generates synthetic typical_price CSV."""

    plugin_params: Dict[str, Any] = {}

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def set_params(self, **kw):
        self.cfg.update(kw)

    # ── public ──────────────────────────────────────────────────────────

    def generate(self) -> pd.DataFrame:
        """Generate synthetic data and save to output_file. Returns the DataFrame."""
        cfg = self.cfg
        model_path = cfg["load_model"]
        seed = cfg["seed"]
        n_samples = cfg["n_samples"]

        # Load decoder + metadata
        decoder, meta = self._load_model(model_path)
        latent_dim = meta["latent_dim"]
        window_size = meta["window_size"]
        initial_price = meta["initial_price"]

        log.info(f"Generating {n_samples} samples with seed={seed}, "
                 f"latent_dim={latent_dim}, window_size={window_size}")

        # Deterministic RNG
        rng = np.random.default_rng(seed)

        # Generate windows of returns by sampling latent space
        n_windows = (n_samples // window_size) + 2  # extra margin
        z = rng.standard_normal((n_windows, latent_dim)).astype(np.float32)
        windows = decoder.predict(z, verbose=0)

        # Flatten all windows into a continuous returns series
        # Use overlapping reconstruction: keep full first window, then last
        # value of each subsequent window to get smooth continuation.
        if cfg.get("use_returns", True):
            # returns_to_prices produces len(returns)+1 values (initial + cumulative)
            # so we need n_samples-1 returns to get n_samples prices
            all_returns = windows.reshape(-1)[: n_samples - 1]
            prices = returns_to_prices(all_returns, initial_price)
        else:
            prices = windows.reshape(-1)[:n_samples]

        # Optional downsampling
        ds = cfg.get("downsample_factor", 1)
        if ds > 1:
            prices = downsample(prices, ds)

        # Build DataFrame
        start_dt = pd.to_datetime(cfg.get("start_datetime", "2020-01-01 00:00:00"))
        interval = timedelta(hours=cfg.get("interval_hours", 4))
        dates = [start_dt + i * interval for i in range(len(prices))]

        df = pd.DataFrame({
            "DATE_TIME": dates,
            "typical_price": prices,
        })

        out = cfg["output_file"]
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        df.to_csv(out, index=False)
        return df

    # ── model loading ───────────────────────────────────────────────────

    @staticmethod
    def _load_model(path: str) -> tuple[keras.Model, dict]:
        """Load decoder and metadata."""
        parts_dir = path + ".parts"
        if os.path.isdir(parts_dir):
            decoder = keras.models.load_model(
                os.path.join(parts_dir, "decoder.keras"), compile=False
            )
            with open(os.path.join(parts_dir, "meta.json")) as f:
                meta = json.load(f)
        else:
            # Fallback: load standalone decoder, infer metadata
            decoder = keras.models.load_model(path, compile=False)
            meta = {
                "latent_dim": int(decoder.input_shape[-1]),
                "window_size": int(decoder.output_shape[-1]),
                "initial_price": 1.3,  # sensible default for EUR/USD
            }
            log.warning("No .parts/ dir found — using inferred metadata")
        return decoder, meta
