"""
Typical-price generator plugin.

Programmatic API for DOIN evaluators and other callers:

    from sdg_plugins.generator.typical_price_generator import TypicalPriceGenerator
    gen = TypicalPriceGenerator()
    gen.configure({"use_returns": True, "interval_hours": 4})
    gen.load_model("path/to/model.keras")
    df = gen.generate(seed=42, n_samples=5000)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from tensorflow import keras

from app.data_processor import returns_to_prices

log = logging.getLogger(__name__)


class TypicalPriceGenerator:
    """
    Plugin: seed-deterministic synthetic typical_price generator.

    Discoverable via entry_points as ``sdg.generator → typical_price_generator``.
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
        """Update configuration parameters."""
        self.cfg.update(config)

    def set_params(self, **kw) -> None:
        """Alias for configure (Harvey's plugin convention)."""
        self.cfg.update(kw)

    def load_model(self, model_path: str) -> None:
        """Load a trained decoder model + metadata from disk."""
        self._decoder, self._meta = self._load_from_disk(model_path)
        log.info(
            f"Loaded model: latent_dim={self._meta['latent_dim']}, "
            f"window_size={self._meta['window_size']}, "
            f"initial_price={self._meta['initial_price']:.6f}"
        )

    def generate(self, seed: int, n_samples: int) -> pd.DataFrame:
        """
        Generate synthetic typical_price timeseries.

        Args:
            seed: RNG seed — same model + same seed = identical output.
            n_samples: Number of rows to generate.

        Returns:
            DataFrame with columns ``DATE_TIME`` and ``typical_price``.
        """
        if self._decoder is None or self._meta is None:
            raise RuntimeError(
                "No model loaded. Call load_model() first, or pass "
                "'load_model' in config for CLI usage."
            )
        return self._generate_impl(self._decoder, self._meta, seed, n_samples)

    # ── CLI entry-point (called by app/main.py) ─────────────────────────

    def run_generate(self) -> pd.DataFrame:
        """CLI wrapper: reads paths/params from self.cfg, writes output file."""
        model_path = self.cfg.get("load_model")
        if not model_path:
            raise ValueError("--load_model (--model) is required for generate mode")

        self.load_model(model_path)
        df = self.generate(
            seed=self.cfg.get("seed", 42),
            n_samples=self.cfg.get("n_samples", 5000),
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
    ) -> pd.DataFrame:
        latent_dim = meta["latent_dim"]
        window_size = meta["window_size"]
        initial_price = meta["initial_price"]

        log.info(
            f"Generating {n_samples} samples with seed={seed}, "
            f"latent_dim={latent_dim}, window_size={window_size}"
        )

        rng = np.random.default_rng(seed)
        n_windows = (n_samples // window_size) + 2
        z = rng.standard_normal((n_windows, latent_dim)).astype(np.float32)
        windows = decoder.predict(z, verbose=0)

        if self.cfg.get("use_returns", True):
            all_returns = windows.reshape(-1)[: n_samples - 1]
            prices = returns_to_prices(all_returns, initial_price)
        else:
            prices = windows.reshape(-1)[:n_samples]

        start_dt = pd.to_datetime(
            self.cfg.get("start_datetime", "2020-01-01 00:00:00")
        )
        interval = timedelta(hours=self.cfg.get("interval_hours", 4))
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
            }
            log.warning("No .parts/ dir found — using inferred metadata")
        return decoder, meta
