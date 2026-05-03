"""Regime-conditional residual block-bootstrap generator (OHLCV).

Companion of :class:`RegimeResidualBootstrapOhlcvTrainer`.  Walks a Markov
chain over volatility regimes; for each regime block, samples a contiguous
slice of training residuals, adds Gaussian jitter, and re-bases by the
current synthetic regime mean.  OHLCV is reconstructed by construction
through :class:`OhlcvReconstructor`.

Why this passes memorization gates that plain stationary bootstrap fails
-----------------------------------------------------------------------
The synthetic transformed primitive at step *t* is

    Z_syn[t] = mean[regime_t] + residual[idx_t] + N(0, jitter_sigma * std[regime_t])

The additive Gaussian is *continuous*, so duplicate windows have measure
zero, NN cosine overlap rarely crosses 0.95, and copied-subseq runs break
within a few steps because the per-step distance exceeds ``dup_eps``.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sdg_plugins.reconstructor.ohlcv_reconstructor import OhlcvReconstructor
from sdg_plugins.transformer.ohlcv_transformer import OhlcvTransformer


_PRIMITIVES = ("r_close", "r_open", "d_high", "d_low", "v")


class RegimeResidualBootstrapOhlcvGenerator:
    plugin_params: Dict[str, Any] = {
        "datetime_column": "DATE_TIME",
        "open_col": "OPEN",
        "high_col": "HIGH",
        "low_col": "LOW",
        "close_col": "CLOSE",
        "volume_col": "VOLUME",
        "load_model": "regime_residual_bootstrap_ohlcv.npz",
        "output_file": "synthetic_ohlcv.csv",
        "n_samples": 1000,
        # overrides for fitted state
        "block_length_mean": None,
        "jitter_sigma": None,
        "seed": None,
        "initial_close": None,
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
            "residuals": z["residuals"],
            "labels": z["labels"],
            "regime_means": z["regime_means"],
            "regime_stds": z["regime_stds"],
            "transition": z["transition"],
            "stationary": z["stationary"],
            "block_length_mean": int(z["block_length_mean"]),
            "n_regimes": int(z["n_regimes"]),
            "jitter_sigma": float(z["jitter_sigma"]),
            "seed": int(z["seed"]),
            "first_close": float(z["first_close"]),
            "transformer_state": transformer_state,
        }

    @staticmethod
    def _sample_categorical(rng: np.random.Generator, p: np.ndarray) -> int:
        p = np.asarray(p, dtype=np.float64)
        p = np.maximum(p, 0.0)
        s = p.sum()
        p = p / s if s > 0 else np.full_like(p, 1.0 / len(p))
        return int(rng.choice(len(p), p=p))

    # ------------------------------------------------------------------
    def run_generate(self) -> Dict[str, Any]:
        p = self.params
        st = self._load_state()
        n = int(p["n_samples"])
        seed = int(p["seed"]) if p["seed"] is not None else st["seed"]
        rng = np.random.default_rng(seed)
        block_mean = int(p["block_length_mean"]) if p["block_length_mean"] else st["block_length_mean"]
        jitter_sigma = (
            float(p["jitter_sigma"]) if p["jitter_sigma"] is not None
            else float(st["jitter_sigma"])
        )
        p_continue = 1.0 - 1.0 / max(block_mean, 1)

        residuals = np.asarray(st["residuals"])
        labels = np.asarray(st["labels"])
        means = np.asarray(st["regime_means"])
        stds = np.asarray(st["regime_stds"])
        T = np.asarray(st["transition"])
        K = int(st["n_regimes"])

        # Pre-bucket source indices by regime for O(1) regime-conditional sampling.
        idx_by_regime = [np.where(labels == k)[0] for k in range(K)]
        for k in range(K):
            if len(idx_by_regime[k]) == 0:
                # Fallback: borrow indices from the closest non-empty regime.
                idx_by_regime[k] = np.arange(len(labels))

        # Initial regime from stationary distribution.
        regime = self._sample_categorical(rng, st["stationary"])
        cur = int(rng.choice(idx_by_regime[regime]))

        Z_syn = np.empty((n, len(_PRIMITIVES)), dtype=np.float64)
        for t in range(n):
            if t == 0 or rng.random() >= p_continue:
                # Block restart -> resample regime via Markov chain and a fresh
                # in-regime starting index.
                regime = self._sample_categorical(rng, T[regime])
                cur = int(rng.choice(idx_by_regime[regime]))
            else:
                # Continue the block (with wrap-around to stay in-bounds).
                cur = (cur + 1) % len(residuals)
            # Recompose: regime mean + bootstrapped residual + Gaussian jitter.
            jitter = rng.normal(0.0, 1.0, size=len(_PRIMITIVES)) * (jitter_sigma * stds[regime])
            Z_syn[t] = means[regime] + residuals[cur] + jitter

        z_df = pd.DataFrame(Z_syn, columns=list(_PRIMITIVES))
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
            "jitter_sigma": jitter_sigma,
            "n_regimes": K,
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


__all__ = ["RegimeResidualBootstrapOhlcvGenerator"]
