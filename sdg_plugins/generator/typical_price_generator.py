"""
Typical-price generator plugin.

Supports conditional (temporal-aware) and non-conditional generation.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from tensorflow import keras

from app.data_processor import returns_to_prices, extract_temporal_features

log = logging.getLogger(__name__)


class TypicalPriceGenerator:
    """
    Plugin: seed-deterministic synthetic typical_price generator.
    """

    plugin_params: Dict[str, Any] = {}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg: Dict[str, Any] = {
            "use_returns": True,
            "start_datetime": "2020-01-01 00:00:00",
            "interval_hours": 4,
        }
        if config:
            self.cfg.update(config)
        self._decoder: Optional[keras.Model] = None
        self._meta: Optional[Dict[str, Any]] = None

    # ── Plugin interface ────────────────────────────────────────────────

    def configure(self, config: Dict[str, Any]) -> None:
        self.cfg.update(config)

    def set_params(self, **kw) -> None:
        self.cfg.update(kw)

    def load_model(self, model_path: str) -> None:
        self._decoder, self._meta = self._load_from_disk(model_path)
        log.info(
            f"Loaded model: latent_dim={self._meta['latent_dim']}, "
            f"window_size={self._meta['window_size']}, "
            f"initial_price={self._meta['initial_price']:.6f}, "
            f"conditional={self._meta.get('conditional', False)}"
        )

    def generate(self, seed: int, n_samples: int,
                 start_datetime: Optional[str] = None) -> pd.DataFrame:
        if self._decoder is None or self._meta is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        return self._generate_impl(self._decoder, self._meta, seed, n_samples,
                                   start_datetime=start_datetime)

    # ── CLI entry-point ─────────────────────────────────────────────────

    def run_generate(self) -> pd.DataFrame:
        model_path = self.cfg.get("load_model")
        if not model_path:
            raise ValueError("--load_model (--model) is required for generate mode")

        self.load_model(model_path)
        df = self.generate(
            seed=self.cfg.get("seed", 42),
            n_samples=self.cfg.get("n_samples", 1575),
            start_datetime=self.cfg.get("start_datetime"),
        )

        out = self.cfg.get("output_file", "synthetic_typical_price.csv")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        df.to_csv(out, index=False)
        log.info(f"Wrote {len(df)} rows → {out}")
        return df

    # ── Internal ────────────────────────────────────────────────────────

    def _generate_impl(
        self,
        decoder: keras.Model,
        meta: Dict[str, Any],
        seed: int,
        n_samples: int,
        start_datetime: Optional[str] = None,
    ) -> pd.DataFrame:
        latent_dim = meta["latent_dim"]
        window_size = meta["window_size"]
        initial_price = meta["initial_price"]
        conditional = meta.get("conditional", False)
        n_temporal = meta.get("n_temporal", 6) if conditional else 0

        start_dt = pd.to_datetime(
            start_datetime or self.cfg.get("start_datetime", "2020-01-01 00:00:00")
        )
        interval = timedelta(hours=self.cfg.get("interval_hours", 4))

        log.info(
            f"Generating {n_samples} samples with seed={seed}, "
            f"latent_dim={latent_dim}, window_size={window_size}, "
            f"conditional={conditional}"
        )

        rng = np.random.default_rng(seed)
        n_windows = (n_samples // window_size) + 2

        if conditional and n_temporal > 0:
            # Generate timestamps for all points, then extract temporal features
            # per window (last timestamp in each window)
            total_points = n_windows * window_size
            all_dates = pd.Series([
                start_dt + i * interval for i in range(total_points)
            ])
            all_temporal = extract_temporal_features(all_dates)

            windows_list = []
            for w in range(n_windows):
                z = rng.standard_normal((1, latent_dim)).astype(np.float32)
                # Last timestamp index for this window
                last_idx = (w + 1) * window_size - 1
                if last_idx >= len(all_temporal):
                    last_idx = len(all_temporal) - 1
                temp = all_temporal[last_idx:last_idx + 1].astype(np.float32)
                window = decoder.predict([z, temp], verbose=0)
                windows_list.append(window[0])
            windows = np.array(windows_list)
        else:
            z = rng.standard_normal((n_windows, latent_dim)).astype(np.float32)
            windows = decoder.predict(z, verbose=0)

        if self.cfg.get("use_returns", True):
            all_returns = windows.reshape(-1)[: n_samples - 1]
            prices = returns_to_prices(all_returns, initial_price)
        else:
            prices = windows.reshape(-1)[:n_samples]

        dates = [start_dt + i * interval for i in range(len(prices))]
        return pd.DataFrame({"DATE_TIME": dates, "typical_price": prices})

    @staticmethod
    def _load_from_disk(path: str) -> tuple[keras.Model, dict]:
        """Load decoder and metadata from disk."""
        parts_dir = path + ".parts"
        if os.path.isdir(parts_dir):
            decoder = keras.models.load_model(
                os.path.join(parts_dir, "decoder.keras"), compile=False
            )
            with open(os.path.join(parts_dir, "meta.json")) as f:
                meta = json.load(f)
        else:
            decoder = keras.models.load_model(path, compile=False)
            meta = {
                "latent_dim": int(decoder.input_shape[-1]),
                "window_size": int(decoder.output_shape[-1]),
                "initial_price": 1.3,
                "conditional": False,
                "n_temporal": 0,
            }
            log.warning("No .parts/ dir found — using inferred metadata")
        return decoder, meta
