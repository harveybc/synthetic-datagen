"""Stationary block-bootstrap generator for OHLCV synthetic paths.

Loads a fitted state produced by :class:`StationaryBootstrapOhlcvTrainer`
and generates a synthetic OHLCV file by stitching geometric-length
blocks of transformed primitives (Politis & Romano stationary bootstrap)
and reconstructing OHLCV via :class:`OhlcvReconstructor`.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sdg_plugins.reconstructor.ohlcv_reconstructor import OhlcvReconstructor
from sdg_plugins.transformer.ohlcv_transformer import OhlcvTransformer


class StationaryBootstrapOhlcvGenerator:
    plugin_params: Dict[str, Any] = {
        "datetime_column": "DATE_TIME",
        "open_col": "OPEN",
        "high_col": "HIGH",
        "low_col": "LOW",
        "close_col": "CLOSE",
        "volume_col": "VOLUME",
        "load_model": "stationary_bootstrap_ohlcv.npz",
        "output_file": "synthetic_ohlcv.csv",
        "n_samples": 1000,
        "block_length_mean": None,   # overrides fitted state if set
        "seed": None,                # overrides fitted seed if set
        "initial_close": None,       # overrides fitted first_close if set
        "start_timestamp": None,
        "frequency": "1h",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.params: Dict[str, Any] = dict(self.plugin_params)
        if config:
            for k, v in config.items():
                if k in self.plugin_params:
                    self.params[k] = v

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.plugin_params:
                self.params[k] = v

    # ------------------------------------------------------------------
    def _load_state(self) -> Dict[str, Any]:
        p = self.params
        z = np.load(p["load_model"], allow_pickle=False)
        side = os.path.splitext(p["load_model"])[0] + ".transformer.json"
        with open(side, "r") as f:
            transformer_state = json.load(f)
        return {
            "matrix": z["matrix"],
            "block_length_mean": int(z["block_length_mean"]),
            "seed": int(z["seed"]),
            "first_close": float(z["first_close"]),
            "transformer_state": transformer_state,
        }

    @staticmethod
    def _stationary_bootstrap_indices(
        n_source: int, n_out: int, p_continue: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Politis-Romano stationary bootstrap index sequence.

        At each step, with probability ``p_continue`` continue the current
        block; otherwise restart from a uniformly-random new index.
        """
        idx = np.empty(n_out, dtype=np.int64)
        cur = int(rng.integers(0, n_source))
        for t in range(n_out):
            if t == 0 or rng.random() >= p_continue:
                cur = int(rng.integers(0, n_source))
            else:
                cur = (cur + 1) % n_source
            idx[t] = cur
        return idx

    # ------------------------------------------------------------------
    def run_generate(self) -> Dict[str, Any]:
        p = self.params
        st = self._load_state()
        block_mean = int(p["block_length_mean"] or st["block_length_mean"])
        seed = int(p["seed"]) if p["seed"] is not None else st["seed"]
        n = int(p["n_samples"])
        rng = np.random.default_rng(seed)
        p_continue = 1.0 - 1.0 / max(block_mean, 1)

        idx = self._stationary_bootstrap_indices(
            len(st["matrix"]), n, p_continue, rng
        )
        z = st["matrix"][idx]
        z_df = pd.DataFrame(z, columns=["r_close", "r_open", "d_high", "d_low", "v"])

        # Rehydrate transformer for unscaling.
        transformer = OhlcvTransformer().load_state_dict(st["transformer_state"])

        initial_close = float(p["initial_close"] or st["first_close"])
        timestamps = self._build_timestamps(n)
        recon = OhlcvReconstructor(self.config)
        recon.set_params(**{
            k: p[k] for k in (
                "datetime_column", "open_col", "high_col", "low_col",
                "close_col", "volume_col",
            )
        })
        out = recon.reconstruct(
            z_df, initial_close=initial_close,
            transformer=transformer, timestamps=timestamps,
        )

        os.makedirs(os.path.dirname(os.path.abspath(p["output_file"])) or ".", exist_ok=True)
        if str(p["output_file"]).lower().endswith(".parquet"):
            out.to_parquet(p["output_file"], index=False)
        else:
            out.to_csv(p["output_file"], index=False)
        return {
            "output_file": p["output_file"],
            "n_rows": int(len(out)),
            "seed": seed,
            "block_length_mean": block_mean,
        }

    def _build_timestamps(self, n: int) -> Optional[pd.Series]:
        p = self.params
        if not p["start_timestamp"]:
            return None
        return pd.Series(pd.date_range(
            start=pd.Timestamp(p["start_timestamp"]),
            periods=n,
            freq=p["frequency"],
        ))


__all__ = ["StationaryBootstrapOhlcvGenerator"]
